"""Evaluation execution engine supporting offline deterministic and live RAG modes."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

from sqlalchemy.orm import Session

from app.config import settings
from app.core.errors import PromptInjectionDetectedError
from app.db.models import Document, DocumentChunk
from app.evaluation.metrics import (
    aggregate_results,
    compute_case_pass,
    detect_abstention,
    evaluate_citations,
    evaluate_correctness,
    evaluate_hallucination_and_safety,
    evaluate_retrieval,
)
from app.evaluation.schemas import (
    CaseEvalResult,
    CitationEvalMetrics,
    CorrectnessEvalMetrics,
    EvalCase,
    EvaluationReport,
    HallucinationEvalMetrics,
    LatencyTokenMetrics,
    RetrievalEvalMetrics,
)
from app.llm.base import BaseLLMProvider, LLMResponse
from app.llm.fallback_router import LLMRouter
from app.rag.generator import RAGGenerator
from app.rag.retrieval import HybridRetriever, RetrievedChunk
from app.rag.verifier import ClaimVerifier

logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent / "datasets" / "benchmark_qa.json"
)

_STOPWORDS = {
    "what", "which", "when", "where", "who", "whom", "whose", "why", "how",
    "does", "did", "with", "from", "into", "about", "that", "this", "these",
    "those", "their", "there", "under", "between", "must", "should", "would",
    "could", "during", "versus", "related", "listed", "specified", "described",
    "required", "according", "section", "module", "rule", "challenge",
}


def load_eval_dataset(dataset_path: Optional[str] = None) -> Tuple[List[EvalCase], str]:
    """Load and validate evaluation benchmark cases from JSON."""
    resolved = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
    with resolved.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    cases = [EvalCase.model_validate(item) for item in raw]
    return cases, str(resolved)


class TimedRetrieverWrapper:
    """Transparent HybridRetriever wrapper capturing per-call retrieval latency and chunks."""

    def __init__(self, retriever: Optional[HybridRetriever] = None):
        self._retriever = retriever or HybridRetriever()
        self.last_retrieval_ms: float = 0.0
        self.last_chunks: List[RetrievedChunk] = []

    def retrieve(
        self,
        db: Session,
        query: str,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
        file_type: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        self.last_chunks = []
        t0 = time.perf_counter()
        chunks = self._retriever.retrieve(
            db=db,
            query=query,
            top_k=top_k,
            document_id=document_id,
            file_type=file_type,
        )
        self.last_retrieval_ms = round((time.perf_counter() - t0) * 1000, 2)
        self.last_chunks = list(chunks)
        return chunks


class OfflineExtractiveProvider(BaseLLMProvider):
    """Deterministic extractive test synthesizer used strictly in offline evaluation mode."""

    def __init__(self, chunks_getter: Callable[[], List[RetrievedChunk]]):
        self._chunks_getter = chunks_getter
        self.model = "offline-extractive-harness"

    def is_configured(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        chunks = self._chunks_getter()
        question_match = re.search(
            r"User Query:\s*(.+?)(?:\n\s*\nRetrieved Evidence:|$)",
            prompt,
            re.DOTALL,
        ) or re.search(r"Question:\s*(.+?)(?:\n|$)", prompt, re.DOTALL)
        question_text = question_match.group(1).strip() if question_match else ""

        q_tokens = [
            tok
            for tok in re.findall(r"[a-zA-Z0-9]{4,}", question_text.lower())
            if tok not in _STOPWORDS
        ]

        if not chunks or not q_tokens:
            content = "INSUFFICIENT_EVIDENCE: No relevant context chunks were retrieved."
        else:
            scored_chunks: List[Tuple[int, RetrievedChunk]] = []
            for ch in chunks:
                lowered = ch.content.lower()
                overlap = sum(1 for tok in set(q_tokens) if tok in lowered)
                scored_chunks.append((overlap, ch))
            scored_chunks.sort(key=lambda item: item[0], reverse=True)
            best_overlap = scored_chunks[0][0]
            min_required = 1 if len(set(q_tokens)) <= 2 else 2

            if best_overlap < min_required:
                content = (
                    "INSUFFICIENT_EVIDENCE: The retrieved context does not contain "
                    "sufficient information to answer this question."
                )
            else:
                selected = [ch for score, ch in scored_chunks[:3] if score > 0]
                snippets = []
                for ch in selected:
                    page = ch.metadata.get("page_number")
                    row = ch.metadata.get("row_number")
                    if page is not None:
                        tag = f"[Doc: {ch.document_title}, Page: {page}]"
                    elif row is not None:
                        tag = f"[Doc: {ch.document_title}, Row: {row}]"
                    else:
                        tag = f"[Doc: {ch.document_title}]"
                    snippets.append(f"{ch.content.strip()} {tag}")
                content = " ".join(snippets)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        prompt_toks = max(1, len(prompt.split()))
        comp_toks = max(1, len(content.split()))
        return LLMResponse(
            content=content,
            provider="offline_deterministic",
            model=self.model,
            prompt_tokens=prompt_toks,
            completion_tokens=comp_toks,
            total_tokens=prompt_toks + comp_toks,
            latency_ms=latency_ms,
        )


class EvaluationRunner:
    """Executes RAG benchmark cases in either 'offline' or 'live' mode."""

    def __init__(
        self,
        mode: str = "offline",
        top_k: int = 5,
        delay_seconds: float = 0.0,
        rate_limit_retry_delay: float = 20.0,
    ):
        if mode not in ("offline", "live"):
            raise ValueError(f"Unsupported evaluation mode '{mode}'. Must be 'offline' or 'live'.")
        self.mode = mode
        self.top_k = top_k
        self.delay_seconds = max(0.0, float(delay_seconds))
        self.rate_limit_retry_delay = max(0.0, float(rate_limit_retry_delay))
        self.verifier = ClaimVerifier()

    def _build_chunk_index_map(self, db: Session) -> Dict[str, int]:
        rows = db.query(DocumentChunk.id, DocumentChunk.chunk_index).all()
        return {str(chunk_id): int(chunk_idx) for chunk_id, chunk_idx in rows}

    def _collect_environment_metadata(self, db: Session) -> Dict[str, Any]:
        docs = db.query(Document).filter(Document.processing_status == "COMPLETED").all()
        dialect = str(db.bind.dialect.name) if db.bind is not None else "unknown"
        return {
            "mode": self.mode,
            "database_dialect": dialect,
            "indexed_documents": [
                {
                    "id": str(d.id),
                    "title": d.title,
                    "file_type": d.file_type,
                    "total_chunks": d.total_chunks,
                }
                for d in docs
            ],
            "top_k": self.top_k,
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
            "primary_llm_model": (
                settings.GROQ_MODEL if self.mode == "live" else "offline-extractive-harness"
            ),
            "fallback_llm_model": (
                settings.GEMINI_MODEL if self.mode == "live" else "offline-extractive-harness"
            ),
            "inter_request_delay_seconds": self.delay_seconds if self.mode == "live" else 0.0,
        }

    def _build_generator(self) -> Tuple[RAGGenerator, TimedRetrieverWrapper]:
        timed_retriever = TimedRetrieverWrapper(HybridRetriever())
        if self.mode == "offline":
            offline_provider = OfflineExtractiveProvider(
                chunks_getter=lambda: timed_retriever.last_chunks
            )
            router = LLMRouter(
                primary_provider=offline_provider,
                fallback_provider=offline_provider,
                max_retries=1,
            )
        else:
            router = LLMRouter()

        generator = RAGGenerator(
            retriever=timed_retriever,
            router=router,
            verifier=self.verifier,
        )
        return generator, timed_retriever

    def _evaluate_single_case(
        self,
        db: Session,
        case: EvalCase,
        generator: RAGGenerator,
        timed_retriever: TimedRetrieverWrapper,
        chunk_index_map: Dict[str, int],
    ) -> Tuple[CaseEvalResult, bool]:
        """Run one evaluation case and return (CaseEvalResult, called_llm)."""
        t_start = time.perf_counter()
        timed_retriever.last_chunks = []
        timed_retriever.last_retrieval_ms = 0.0

        called_llm = False
        attempt = 0
        max_attempts = 2 if self.mode == "live" else 1

        while attempt < max_attempts:
            attempt += 1
            try:
                rag_resp = generator.query(
                    db=db,
                    query_text=case.question,
                    top_k=self.top_k,
                )
                called_llm = True
                total_ms = round((time.perf_counter() - t_start) * 1000, 2)

                retrieval_metrics = evaluate_retrieval(
                    case=case,
                    retrieved_chunks=rag_resp.retrieved_chunks,
                    top_k=self.top_k,
                    chunk_index_map=chunk_index_map,
                )
                abstained = detect_abstention(rag_resp.answer, rag_resp.is_insufficient_evidence)
                correctness_metrics = evaluate_correctness(
                    case=case,
                    generated_answer=rag_resp.answer,
                    abstained=abstained,
                )
                citation_metrics = evaluate_citations(
                    case=case,
                    generated_answer=rag_resp.answer,
                    retrieved_chunks=rag_resp.retrieved_chunks,
                    structured_citations=rag_resp.citations,
                    abstained=abstained,
                )
                hallucination_metrics = evaluate_hallucination_and_safety(
                    case=case,
                    generated_answer=rag_resp.answer,
                    retrieved_chunks=rag_resp.retrieved_chunks,
                    verifier=self.verifier,
                    is_insufficient_flag=rag_resp.is_insufficient_evidence,
                    blocked_by_injection_guard=False,
                )
                passed = compute_case_pass(
                    case=case,
                    correctness=correctness_metrics,
                    citation=citation_metrics,
                    hallucination=hallucination_metrics,
                    status="completed",
                )

                retrieval_ms = timed_retriever.last_retrieval_ms
                generation_ms = max(0.0, round(total_ms - retrieval_ms, 2))

                perf = LatencyTokenMetrics(
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    total_ms=total_ms,
                    prompt_tokens=int(
                        rag_resp.tokens.get("prompt", rag_resp.tokens.get("prompt_tokens", 0))
                    ),
                    completion_tokens=int(
                        rag_resp.tokens.get("completion", rag_resp.tokens.get("completion_tokens", 0))
                    ),
                    total_tokens=int(
                        rag_resp.tokens.get("total", rag_resp.tokens.get("total_tokens", 0))
                    ),
                    provider=rag_resp.provider,
                    model=rag_resp.model,
                )
                return (
                    CaseEvalResult(
                        case_id=case.case_id,
                        category=case.category,
                        question=case.question,
                        expected_answer=case.expected_answer,
                        generated_answer=rag_resp.answer,
                        status="completed",
                        passed=passed,
                        retrieval=retrieval_metrics,
                        correctness=correctness_metrics,
                        citation=citation_metrics,
                        hallucination=hallucination_metrics,
                        performance=perf,
                    ),
                    called_llm,
                )

            except PromptInjectionDetectedError as sec_err:
                total_ms = round((time.perf_counter() - t_start) * 1000, 2)
                hallucination_metrics = evaluate_hallucination_and_safety(
                    case=case,
                    generated_answer=f"[BLOCKED_BY_SECURITY_GUARD] {sec_err}",
                    retrieved_chunks=[],
                    verifier=self.verifier,
                    blocked_by_injection_guard=True,
                )
                passed = compute_case_pass(
                    case=case,
                    correctness=CorrectnessEvalMetrics(applicable=False),
                    citation=CitationEvalMetrics(applicable=False),
                    hallucination=hallucination_metrics,
                    status="blocked_injection",
                )
                return (
                    CaseEvalResult(
                        case_id=case.case_id,
                        category=case.category,
                        question=case.question,
                        expected_answer=case.expected_answer,
                        generated_answer=f"[BLOCKED_BY_SECURITY_GUARD] {sec_err}",
                        status="blocked_injection",
                        passed=passed,
                        retrieval=RetrievalEvalMetrics(applicable=False, top_k=self.top_k),
                        correctness=CorrectnessEvalMetrics(applicable=False),
                        citation=CitationEvalMetrics(applicable=False),
                        hallucination=hallucination_metrics,
                        performance=LatencyTokenMetrics(
                            retrieval_ms=0.0,
                            generation_ms=0.0,
                            total_ms=total_ms,
                            provider="prompt_injection_defender",
                            model="regex-guard",
                        ),
                    ),
                    False,
                )

            except Exception as err:
                err_str = str(err)
                is_rate_limit = any(
                    hint in err_str.lower()
                    for hint in ("429", "rate_limit", "rate limit", "tokens per minute", "tpm")
                )
                if is_rate_limit and attempt < max_attempts and self.rate_limit_retry_delay > 0:
                    logger.warning(
                        f"Case {case.case_id} encountered rate limit; waiting {self.rate_limit_retry_delay}s before retry..."
                    )
                    time.sleep(self.rate_limit_retry_delay)
                    continue

                total_ms = round((time.perf_counter() - t_start) * 1000, 2)
                retrieval_metrics = (
                    evaluate_retrieval(
                        case=case,
                        retrieved_chunks=timed_retriever.last_chunks,
                        top_k=self.top_k,
                        chunk_index_map=chunk_index_map,
                    )
                    if timed_retriever.last_chunks
                    else RetrievalEvalMetrics(applicable=False, top_k=self.top_k)
                )
                return (
                    CaseEvalResult(
                        case_id=case.case_id,
                        category=case.category,
                        question=case.question,
                        expected_answer=case.expected_answer,
                        generated_answer="",
                        status="error",
                        error_message=err_str,
                        passed=False,
                        retrieval=retrieval_metrics,
                        correctness=CorrectnessEvalMetrics(applicable=False),
                        citation=CitationEvalMetrics(applicable=False),
                        hallucination=HallucinationEvalMetrics(claim_verifier_is_verified=False),
                        performance=LatencyTokenMetrics(
                            retrieval_ms=timed_retriever.last_retrieval_ms,
                            generation_ms=0.0,
                            total_ms=total_ms,
                        ),
                    ),
                    True,
                )

        # Unreachable safeguard
        raise RuntimeError("Evaluation retry loop exited unexpectedly")

    def run(
        self,
        db: Session,
        cases: List[EvalCase],
        dataset_path: str = str(DEFAULT_DATASET_PATH),
        progress_callback: Optional[Callable[[int, int, CaseEvalResult], None]] = None,
    ) -> EvaluationReport:
        """Execute all evaluation cases and return a structured EvaluationReport."""
        chunk_index_map = self._build_chunk_index_map(db)
        env_meta = self._collect_environment_metadata(db)
        generator, timed_retriever = self._build_generator()

        results: List[CaseEvalResult] = []
        total = len(cases)

        for idx, case in enumerate(cases, start=1):
            res, called_llm = self._evaluate_single_case(
                db=db,
                case=case,
                generator=generator,
                timed_retriever=timed_retriever,
                chunk_index_map=chunk_index_map,
            )
            results.append(res)
            if progress_callback:
                progress_callback(idx, total, res)

            if self.mode == "live" and called_llm and idx < total and self.delay_seconds > 0:
                time.sleep(self.delay_seconds)

        aggregate, by_category = aggregate_results(results)

        methodology = [
            "Retrieval Relevance: Measured via Hit@K, Precision@K, Recall@K, Mean Reciprocal Rank (MRR), NDCG@K, and Page Hit@K against ground-truth chunk indices using HybridRetriever (dense ONNX pgvector + PostgreSQL full-text search with RRF k=60; portable cosine + keyword in SQLite).",
            "Answer Correctness: Measured on non-abstaining answerable, multi-hop, and numerical cases using key-fact recall, exact number/word-alias match accuracy, and ONNX BAAI/bge-small-en-v1.5 cosine similarity against reference answers.",
            "Source Grounding & Citation Accuracy: Inline [Doc: ..., Page: ...] and [Doc: ..., Row: ...] tags are parsed and verified against actual retrieved_chunks metadata to measure citation presence, precision, and fabricated-citation rate.",
            "Hallucination & Safety: Reuses ClaimVerifier sentence-level lexical/numeric claim verification on retrieved evidence, abstention detection (INSUFFICIENT_EVIDENCE / refusal phrasing) on unanswerable queries, and PromptInjectionDefender inspection on adversarial inputs.",
            "Latency & Token Usage: Retrieval latency, total end-to-end latency (mean, p50, p95, max), and prompt/completion/total token counts are collected directly from timed HybridRetriever and RAGGenerator execution.",
        ]

        limitations = [
            "Single-Document Corpus: The live knowledge base currently contains 1 completed document (AI_Engineer_Technical_Assessment.pdf, 27 chunks across 5 pages), so cross-document multi-hop interference is not yet represented.",
            "Automated Proxy Correctness: Answer correctness relies on key-fact phrase matching, exact numerical checks, and bi-encoder embedding similarity rather than human grading or an independent LLM-as-judge.",
            "Rule-Based Claim Verification & Injection Defense: ClaimVerifier uses lexical/numerical overlap heuristics rather than NLI entailment, and PromptInjectionDefender uses static regex confidence thresholds (>=0.80).",
            "Free-Tier Provider Constraints: Groq openai/gpt-oss-120b enforces an 8,000 TPM limit and Gemini fallback can experience transient HTTP 503 unavailability under load.",
        ]
        if self.mode == "offline":
            limitations.append(
                "Offline Deterministic Harness: This run used OfflineExtractiveProvider instead of live Groq/Gemini inference; metrics verify deterministic pipeline mechanics and retrieval rather than live LLM generation quality."
            )

        return EvaluationReport(
            run_id=str(uuid.uuid4())[:8],
            mode=self.mode,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            dataset_path=dataset_path,
            top_k=self.top_k,
            environment=env_meta,
            aggregate=aggregate,
            by_category=by_category,
            methodology_notes=methodology,
            limitations=limitations,
            results=results,
        )
