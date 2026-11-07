"""Unit tests for the Verified Source Citations UI Component (frontend/components/citations.py).

Verifies:
1. Every citation renders with Document name, Page number, Chunk ID.
2. Separate Streamlit expander is created for each individual citation.
3. Actual retrieved chunk text is displayed with a readable preview and expandable full text.
4. Short chunk text displays directly without unnecessary truncation.
5. Missing or empty chunk text is handled gracefully.
6. Legacy string citations ([Doc: ..., Page: ..., Chunk: ...]) are parsed accurately.
7. Claim verification banners render properly.
8. Insufficient evidence notifications render without crashing.
"""
from unittest.mock import MagicMock, call, patch
import pytest

from frontend.components.citations import render_citations


@pytest.fixture
def mock_st(monkeypatch):
    """Mocks streamlit rendering functions."""
    m_expander_ctx = MagicMock()
    m_expander_ctx.__enter__ = MagicMock(return_value=m_expander_ctx)
    m_expander_ctx.__exit__ = MagicMock(return_value=None)

    m_expander = MagicMock(return_value=m_expander_ctx)
    m_markdown = MagicMock()
    m_info = MagicMock()
    m_caption = MagicMock()

    import frontend.components.citations as cit_module
    monkeypatch.setattr(cit_module.st, "expander", m_expander)
    monkeypatch.setattr(cit_module.st, "markdown", m_markdown)
    monkeypatch.setattr(cit_module.st, "info", m_info)
    monkeypatch.setattr(cit_module.st, "caption", m_caption)

    return {
        "expander": m_expander,
        "markdown": m_markdown,
        "info": m_info,
        "caption": m_caption,
    }


def test_render_citations_with_long_chunk_text(mock_st):
    """Test citation with retrieved content exceeding preview length creates separate expander and preview."""
    long_content = (
        "Module A: Document Ingestion Pipeline. Implements multi-format parsing for PDF, Markdown, TXT, and CSV. "
        "Enforces document deduplication via SHA-256 hashes and transactional metadata persistence in Neon PostgreSQL. "
        "Chunks documents into 500-token boundaries with 50-token overlap."
    )
    citations = [
        {
            "document_id": "doc_mod_a",
            "document_title": "AI_Engineer_Technical_Assessment.pdf",
            "chunk_id": "chk_663ed3b5",
            "page_number": 2,
            "snippet": long_content[:150],
            "content": long_content,
        }
    ]

    render_citations(citations=citations)

    # 1. Main expander for citation 1
    expander_calls = [c.args[0] for c in mock_st["expander"].call_args_list]
    assert any("AI_Engineer_Technical_Assessment.pdf" in c and "Page 2" in c and "chk_663e" in c for c in expander_calls)

    # 2. Document name, Location, and Chunk ID
    markdown_texts = [str(c.args[0]) for c in mock_st["markdown"].call_args_list]
    assert any("- **Document Name:** `AI_Engineer_Technical_Assessment.pdf`" in t for t in markdown_texts)
    assert any("- **Page / Location:** Page 2" in t for t in markdown_texts)
    assert any("- **Chunk ID:** `chk_663ed3b5`" in t for t in markdown_texts)

    # 3. Preview text displayed in st.info
    info_texts = [str(c.args[0]) for c in mock_st["info"].call_args_list]
    assert any("Module A: Document Ingestion Pipeline" in t for t in info_texts)

    # 4. Inner expander for full text
    assert any("View Full Retrieved Text" in c for c in expander_calls)


def test_render_citations_with_short_chunk_text(mock_st):
    """Test citation with short retrieved content renders directly without inner expander."""
    short_content = "AES-256 encryption is strictly required at rest."
    citations = [
        {
            "document_id": "doc_sec",
            "document_title": "Security_Guidelines.pdf",
            "chunk_id": "chk_sec_01",
            "page_number": 5,
            "content": short_content,
        }
    ]

    render_citations(citations=citations)

    # Expander called for citation
    expander_calls = [c.args[0] for c in mock_st["expander"].call_args_list]
    assert any("Security_Guidelines.pdf" in c for c in expander_calls)

    # Full text rendered in info
    info_texts = [str(c.args[0]) for c in mock_st["info"].call_args_list]
    assert any("AES-256 encryption is strictly required at rest." in t for t in info_texts)

    # No inner "View Full Retrieved Text" expander needed for short content
    assert not any("View Full Retrieved Text" in c for c in expander_calls)


def test_render_citations_missing_content_handled_gracefully(mock_st):
    """Test citation with missing or empty content renders metadata and fallback caption."""
    citations = [
        {
            "document_id": "doc_empty",
            "document_title": "Empty_Ref.pdf",
            "chunk_id": "chk_empty_01",
            "page_number": 1,
            "content": "",
        }
    ]

    render_citations(citations=citations)

    # Expander called
    expander_calls = [c.args[0] for c in mock_st["expander"].call_args_list]
    assert any("Empty_Ref.pdf" in c for c in expander_calls)

    # Caption called for unavailable preview
    caption_texts = [str(c.args[0]) for c in mock_st["caption"].call_args_list]
    assert any("Actual chunk text preview unavailable" in t for t in caption_texts)


def test_render_multiple_citations_separate_expanders(mock_st):
    """Test multiple citations create separate expanders for each citation."""
    citations = [
        {
            "document_id": "doc_1",
            "document_title": "Doc_One.pdf",
            "chunk_id": "chk_001",
            "page_number": 1,
            "content": "Content of chunk 1",
        },
        {
            "document_id": "doc_2",
            "document_title": "Doc_Two.pdf",
            "chunk_id": "chk_002",
            "page_number": 4,
            "content": "Content of chunk 2",
        },
    ]

    render_citations(citations=citations)

    expander_calls = [c.args[0] for c in mock_st["expander"].call_args_list]
    # Expect 2 separate citation expanders
    c1_expanders = [c for c in expander_calls if "Doc_One.pdf" in c]
    c2_expanders = [c for c in expander_calls if "Doc_Two.pdf" in c]
    assert len(c1_expanders) == 1
    assert len(c2_expanders) == 1


def test_render_legacy_string_citation_parsing(mock_st):
    """Test legacy string citations ([Doc: Title, Page: N, Chunk: ID]) parse accurately."""
    citations = [
        "[Doc: AI_Engineer_Technical_Assessment.pdf, Page: 2, Chunk: 663ed3b5-4152-860e-5d1fc3ee5b9e]"
    ]

    render_citations(citations=citations)

    expander_calls = [c.args[0] for c in mock_st["expander"].call_args_list]
    assert any("AI_Engineer_Technical_Assessment.pdf" in c and "Page 2" in c for c in expander_calls)

    markdown_texts = [str(c.args[0]) for c in mock_st["markdown"].call_args_list]
    assert any("AI_Engineer_Technical_Assessment.pdf" in t for t in markdown_texts)
    assert any("663ed3b5-4152-860e-5d1fc3ee5b9e" in t for t in markdown_texts)


def test_render_verification_status_banner(mock_st):
    """Test claim verification status banner displays grounding confidence and status."""
    verification_data = {
        "is_verified": True,
        "confidence_score": 0.95,
        "supported_claims": 5,
        "total_claims": 5,
    }
    citations = ["[Doc: Policy.pdf, Page: 1, Chunk: chk_1]"]

    render_citations(citations=citations, verification_data=verification_data)

    markdown_texts = [str(c.args[0]) for c in mock_st["markdown"].call_args_list]
    assert any("GROUNDED & VERIFIED" in t for t in markdown_texts)
    assert any("95.0%" in t for t in markdown_texts)
    assert any("5/5 claims grounded" in t for t in markdown_texts)


def test_render_insufficient_evidence_notice(mock_st):
    """Test insufficient evidence displays notification and skips citations."""
    render_citations(citations=[], is_insufficient_evidence=True)

    info_texts = [str(c.args[0]) for c in mock_st["info"].call_args_list]
    assert any("Insufficient factual evidence" in t for t in info_texts)
    assert mock_st["expander"].call_count == 0
