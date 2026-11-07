from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.core.errors import PromptInjectionDetectedError
from app.db.models import DocumentChunk
from app.db.session import get_db
from app.rag.generator import RAGResponse, rag_generator
from app.rag.retrieval import HybridRetriever, RetrievedChunk
from app.rag.verifier import ClaimVerifier, VerificationReport, claim_verifier

router = APIRouter(prefix="/rag", tags=["Hybrid RAG"])


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query string")
    top_k: Optional[int] = Field(default=5, ge=1, le=25)
    document_id: Optional[str] = None
    file_type: Optional[str] = None


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    content: str
    token_count: int
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: float
    metadata: Dict[str, Any]

    model_config = ConfigDict(from_attributes=True)


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question for the RAG platform")
    top_k: Optional[int] = Field(default=5, ge=1, le=25)
    document_id: Optional[str] = None
    file_type: Optional[str] = None


class CitationResponse(BaseModel):
    document_id: str
    document_title: str
    chunk_id: str
    page_number: Optional[int] = None
    row_number: Optional[int] = None
    section_header: Optional[str] = None
    snippet: str

    model_config = ConfigDict(from_attributes=True)


class ClaimVerificationResponse(BaseModel):
    is_verified: bool
    confidence_score: float
    total_claims: int
    supported_claims: int
    partially_supported_claims: int
    unsupported_claims: int
    hallucination_detected: bool
    rejection_reason: Optional[str] = None
    claims: List[Dict[str, Any]]

    model_config = ConfigDict(from_attributes=True)


class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationResponse]
    provider: str
    model: str
    is_insufficient_evidence: bool
    verification: ClaimVerificationResponse
    latency_ms: float
    tokens: Dict[str, int]

    model_config = ConfigDict(from_attributes=True)


class VerifyRequest(BaseModel):
    answer: str = Field(..., min_length=1)
    chunk_ids: Optional[List[str]] = None
    evidence_texts: Optional[List[str]] = None


@router.post("/retrieve", response_model=List[RetrievedChunkResponse])
def retrieve_chunks(request: RetrieveRequest, db: Session = Depends(get_db)):
    """Executes hybrid retrieval (Dense pgvector + Sparse Keyword Search -> RRF fusion)."""
    retriever = HybridRetriever()
    results = retriever.retrieve(
        db=db,
        query=request.query,
        top_k=request.top_k,
        document_id=request.document_id,
        file_type=request.file_type,
    )
    return [
        RetrievedChunkResponse(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            document_title=c.document_title,
            content=c.content,
            token_count=c.token_count,
            dense_rank=c.dense_rank,
            sparse_rank=c.sparse_rank,
            rrf_score=c.rrf_score,
            metadata=c.metadata,
        )
        for c in results
    ]


@router.post("/query", response_model=RAGQueryResponse)
def execute_rag_query(request: RAGQueryRequest, db: Session = Depends(get_db)):
    """Executes end-to-end RAG query with prompt injection defense, multi-provider fallback,
    citation attribution, and claim-level response verification.
    """
    rag_res = rag_generator.query(
        db=db,
        query_text=request.query,
        top_k=request.top_k,
        document_id=request.document_id,
        file_type=request.file_type,
    )

    ver = rag_res.verification
    verification_payload = ClaimVerificationResponse(
        is_verified=ver.is_verified,
        confidence_score=ver.confidence_score,
        total_claims=ver.total_claims,
        supported_claims=ver.supported_claims,
        partially_supported_claims=ver.partially_supported_claims,
        unsupported_claims=ver.unsupported_claims,
        hallucination_detected=ver.hallucination_detected,
        rejection_reason=ver.rejection_reason,
        claims=[
            {
                "claim_text": c.claim_text,
                "status": c.status.value,
                "confidence_score": c.confidence_score,
                "supporting_chunk_id": c.supporting_chunk_id,
                "document_title": c.document_title,
                "evidence_snippet": c.evidence_snippet,
                "rationale": c.rationale,
            }
            for c in ver.claims
        ],
    )

    return RAGQueryResponse(
        query=request.query,
        answer=rag_res.answer,
        citations=[
            CitationResponse(
                document_id=c.document_id,
                document_title=c.document_title,
                chunk_id=c.chunk_id,
                page_number=c.page_number,
                row_number=c.row_number,
                section_header=c.section_header,
                snippet=c.snippet,
            )
            for c in rag_res.citations
        ],
        provider=rag_res.provider,
        model=rag_res.model,
        is_insufficient_evidence=rag_res.is_insufficient_evidence,
        verification=verification_payload,
        latency_ms=rag_res.latency_ms,
        tokens=rag_res.tokens,
    )


@router.post("/verify", response_model=ClaimVerificationResponse)
def verify_answer(request: VerifyRequest, db: Session = Depends(get_db)):
    """Standalone claim-level answer verification against provided chunk IDs or evidence snippets."""
    evidence_chunks: List[RetrievedChunk] = []

    if request.chunk_ids:
        chunks = db.query(DocumentChunk).filter(DocumentChunk.id.in_(request.chunk_ids)).all()
        for c in chunks:
            doc_title = c.document.title if c.document else "Document"
            evidence_chunks.append(
                RetrievedChunk(
                    chunk_id=c.id,
                    document_id=c.document_id,
                    document_title=doc_title,
                    content=c.content,
                    token_count=c.token_count,
                    metadata=c.chunk_metadata,
                )
            )

    if request.evidence_texts:
        for idx, text_snippet in enumerate(request.evidence_texts):
            evidence_chunks.append(
                RetrievedChunk(
                    chunk_id=f"snippet_{idx}",
                    document_id="manual",
                    document_title="Manual Evidence",
                    content=text_snippet,
                    token_count=len(text_snippet.split()),
                )
            )

    report = claim_verifier.verify_answer(request.answer, evidence_chunks)

    return ClaimVerificationResponse(
        is_verified=report.is_verified,
        confidence_score=report.confidence_score,
        total_claims=report.total_claims,
        supported_claims=report.supported_claims,
        partially_supported_claims=report.partially_supported_claims,
        unsupported_claims=report.unsupported_claims,
        hallucination_detected=report.hallucination_detected,
        rejection_reason=report.rejection_reason,
        claims=[
            {
                "claim_text": c.claim_text,
                "status": c.status.value,
                "confidence_score": c.confidence_score,
                "supporting_chunk_id": c.supporting_chunk_id,
                "document_title": c.document_title,
                "evidence_snippet": c.evidence_snippet,
                "rationale": c.rationale,
            }
            for c in report.claims
        ],
    )
