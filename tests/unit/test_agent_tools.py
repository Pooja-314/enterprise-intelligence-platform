"""Unit tests for Stage 4 AI Agent Tools:
1. Knowledge Base Search Tool
2. Document Analysis Tool
3. Structured Report Generation Tool
"""
import pytest
from sqlalchemy.orm import Session

from app.agent.base import ToolExecutionError
from app.agent.tools.doc_analysis import DocumentAnalysisTool
from app.agent.tools.kb_search import KnowledgeBaseSearchTool
from app.agent.tools.report_gen import StructuredReportGenerationTool
from app.core.errors import EntityNotFoundError
from app.db.models import AuditLog, Document, DocumentChunk, ProcessingStatus
from app.llm.base import BaseLLMProvider, LLMResponse
from app.llm.fallback_router import LLMRouter
from app.rag.embeddings import embedding_service


class DummyLLMProvider(BaseLLMProvider):
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.model = "dummy-model"

    def is_configured(self) -> bool:
        return True

    def generate(self, prompt: str, system_prompt=None, temperature=0.2, max_tokens=1024) -> LLMResponse:
        return LLMResponse(
            content=self.response_text,
            provider="dummy_provider",
            model=self.model,
            prompt_tokens=40,
            completion_tokens=20,
            total_tokens=60,
            latency_ms=45.0,
        )


@pytest.fixture
def sample_indexed_documents(db_session: Session):
    # Doc 1: Financial PDF
    doc1 = Document(
        id="doc_fin_100",
        title="Q3 Financial Overview",
        filename="fin_q3.pdf",
        file_type="pdf",
        file_size_bytes=2048,
        content_hash="hash_fin_100",
        processing_status=ProcessingStatus.COMPLETED.value,
        doc_metadata={},
    )
    # Doc 2: Security TXT
    doc2 = Document(
        id="doc_sec_200",
        title="Zero Trust Guidelines",
        filename="security.txt",
        file_type="txt",
        file_size_bytes=1024,
        content_hash="hash_sec_200",
        processing_status=ProcessingStatus.COMPLETED.value,
        doc_metadata={},
    )
    db_session.add_all([doc1, doc2])
    db_session.commit()

    c1_text = "Operating revenue increased by 23 percent year over year to 150 million dollars."
    c2_text = "All employee endpoints must require biometric multi-factor authentication."

    emb1 = embedding_service.embed_text(c1_text)
    emb2 = embedding_service.embed_text(c2_text)

    chunk1 = DocumentChunk(
        id="chk_fin_1",
        document_id="doc_fin_100",
        chunk_index=0,
        content=c1_text,
        token_count=14,
        embedding=emb1,
        chunk_metadata={"page_number": 4, "header": "Revenue Performance"},
    )
    chunk2 = DocumentChunk(
        id="chk_sec_2",
        document_id="doc_sec_200",
        chunk_index=0,
        content=c2_text,
        token_count=10,
        embedding=emb2,
        chunk_metadata={"page_number": 1, "header": "MFA Policy"},
    )
    db_session.add_all([chunk1, chunk2])
    db_session.commit()

    return doc1, doc2


# ------------------------------------------------------------------------------
# Tool A: Knowledge Base Search
# ------------------------------------------------------------------------------

def test_kb_search_tool_execution(sample_indexed_documents, db_session: Session):
    tool = KnowledgeBaseSearchTool()
    output = tool.run(db=db_session, query="What was the operating revenue increase?", top_k=3)

    assert output["total_retrieved"] >= 1
    top_hit = output["results"][0]
    assert "23 percent" in top_hit["content"]
    assert top_hit["document_id"] == "doc_fin_100"
    assert "Page: 4" in top_hit["citation"]
    assert "Doc: Q3 Financial Overview" in top_hit["citation"]
    assert top_hit["rrf_score"] > 0.0


def test_kb_search_tool_filtering(sample_indexed_documents, db_session: Session):
    tool = KnowledgeBaseSearchTool()
    # Filter strictly by file_type = txt (should only match security document)
    txt_output = tool.run(db=db_session, query="authentication", file_type="txt", top_k=5)
    for r in txt_output["results"]:
        assert r["document_id"] == "doc_sec_200"

    # Filter strictly by document_id
    fin_output = tool.run(db=db_session, query="revenue", document_id="doc_fin_100", top_k=5)
    assert all(r["document_id"] == "doc_fin_100" for r in fin_output["results"])


def test_kb_search_tool_validation_error(db_session: Session):
    tool = KnowledgeBaseSearchTool()
    # top_k > 20 violates schema constraint (le=20)
    with pytest.raises(ToolExecutionError) as exc_info:
        tool.run(db=db_session, query="test", top_k=99)
    assert "Validation error" in str(exc_info.value)


def test_kb_search_records_audit_log(sample_indexed_documents, db_session: Session):
    tool = KnowledgeBaseSearchTool()
    tool.run(db=db_session, user_id=42, query="revenue metrics", top_k=2)

    log = (
        db_session.query(AuditLog)
        .filter(AuditLog.action == "tool_execution:knowledge_base_search")
        .first()
    )
    assert log is not None
    assert log.user_id == 42
    assert log.details["status"] == "SUCCESS"
    assert log.details["inputs"]["query"] == "revenue metrics"
    assert log.details["latency_ms"] >= 0.0


# ------------------------------------------------------------------------------
# Tool B: Document Analysis
# ------------------------------------------------------------------------------

def test_document_analysis_tool_execution(sample_indexed_documents, db_session: Session):
    doc1, _ = sample_indexed_documents
    mock_llm = DummyLLMProvider(
        "Executive Summary: Revenue surged by 23% in Q3. [Page: 4, Chunk: chk_fin_1]\n"
        "Key Findings: Annualized revenue hit 150M. [Page: 4, Chunk: chk_fin_1]"
    )
    router = LLMRouter(primary_provider=mock_llm, max_retries=0)
    tool = DocumentAnalysisTool(router=router)

    output = tool.run(db=db_session, document_id=doc1.id, focus_area="growth metrics")
    assert output["document_id"] == "doc_fin_100"
    assert output["total_chunks"] == 1
    assert "Revenue surged by 23%" in output["analysis"]
    assert output["provider"] == "dummy_provider"
    assert len(output["available_chunk_references"]) >= 1


def test_document_analysis_tool_nonexistent_document(db_session: Session):
    tool = DocumentAnalysisTool()
    with pytest.raises(EntityNotFoundError):
        tool.run(db=db_session, document_id="non_existent_doc_id_999")


def test_document_analysis_tool_empty_document(db_session: Session):
    empty_doc = Document(
        id="doc_empty_99",
        title="Empty Doc",
        filename="empty.txt",
        file_type="txt",
        file_size_bytes=0,
        content_hash="empty_hash_99",
        processing_status=ProcessingStatus.COMPLETED.value,
        doc_metadata={},
    )
    db_session.add(empty_doc)
    db_session.commit()

    tool = DocumentAnalysisTool()
    res = tool.run(db=db_session, document_id="doc_empty_99")
    assert res["total_chunks"] == 0
    assert "zero processed chunks" in res["summary"]


# ------------------------------------------------------------------------------
# Tool C: Structured Report Generation
# ------------------------------------------------------------------------------

def test_report_generation_tool_execution(sample_indexed_documents, db_session: Session):
    mock_llm = DummyLLMProvider(
        "# Intelligence Report: Q3 Performance\n"
        "## 1. Executive Summary\n"
        "Operating revenue increased by 23 percent year over year to 150 million dollars. [Doc: Q3 Financial Overview, Page: 4, Chunk: chk_fin_1]\n\n"
        "## 2. Missing Evidence & Information Gaps\n"
        "- Net profit margins and operating costs were not mentioned in the source documents."
    )
    router = LLMRouter(primary_provider=mock_llm, max_retries=0)
    tool = StructuredReportGenerationTool(router=router)

    output = tool.run(db=db_session, topic="Q3 Performance Summary", report_format="executive_summary")
    assert output["topic"] == "Q3 Performance Summary"
    assert output["evidence_chunks_count"] >= 1
    assert "150 million dollars" in output["report_markdown"]
    assert "Missing Evidence & Information Gaps" in output["report_markdown"]
    assert output["verification"]["is_verified"] is True
    assert output["verification"]["confidence_score"] >= 0.70


def test_report_generation_tool_empty_knowledge_base(db_session: Session):
    tool = StructuredReportGenerationTool()
    # In an empty database
    output = tool.run(db=db_session, topic="Quantum Computing Roadmap")
    assert output["is_insufficient_evidence"] is True
    assert "No relevant documents were found" in output["report_markdown"]
    assert "Missing Evidence & Information Gaps" in output["report_markdown"]
    assert len(output["citations"]) == 0
