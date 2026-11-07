import pytest
from sqlalchemy.orm import Session

from app.db.models import Document, DocumentChunk, ProcessingStatus
from app.rag.retrieval import HybridRetriever, RetrievedChunk


def test_rrf_mathematical_precision_and_deduplication():
    # Chunk A: Rank 1 in dense, Rank 2 in sparse
    chunk_a = RetrievedChunk(chunk_id="A", document_id="D1", document_title="Doc 1", content="Text A", token_count=10)
    # Chunk B: Rank 2 in dense, absent in sparse
    chunk_b = RetrievedChunk(chunk_id="B", document_id="D1", document_title="Doc 1", content="Text B", token_count=10)
    # Chunk C: Absent in dense, Rank 1 in sparse
    chunk_c = RetrievedChunk(chunk_id="C", document_id="D2", document_title="Doc 2", content="Text C", token_count=10)

    dense = [chunk_a, chunk_b]
    sparse = [chunk_c, RetrievedChunk(chunk_id="A", document_id="D1", document_title="Doc 1", content="Text A", token_count=10)]

    k = 60
    fused = HybridRetriever.reciprocal_rank_fusion(dense_results=dense, sparse_results=sparse, k=k)

    # Expected count = 3 (A deduplicated)
    assert len(fused) == 3

    # Expected score for A: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.0163934 + 0.0161290 = 0.032522
    a_fused = next(c for c in fused if c.chunk_id == "A")
    expected_score_a = round(1.0 / 61.0 + 1.0 / 62.0, 6)
    assert a_fused.rrf_score == expected_score_a
    assert a_fused.dense_rank == 1
    assert a_fused.sparse_rank == 2

    # A should be rank 1 overall
    assert fused[0].chunk_id == "A"


def test_hybrid_retrieval_and_filtering(db_session: Session):
    # Setup 2 documents in database
    doc1 = Document(
        id="doc_fin",
        title="Financial Q3 Report",
        filename="fin_q3.pdf",
        file_type="pdf",
        file_size_bytes=1000,
        content_hash="h1",
        processing_status=ProcessingStatus.COMPLETED.value,
        doc_metadata={},
    )
    doc2 = Document(
        id="doc_sec",
        title="Security Guidelines",
        filename="sec_guide.txt",
        file_type="txt",
        file_size_bytes=500,
        content_hash="h2",
        processing_status=ProcessingStatus.COMPLETED.value,
        doc_metadata={},
    )
    db_session.add_all([doc1, doc2])
    db_session.commit()

    # Add chunks with embeddings generated from actual content
    from app.rag.embeddings import embedding_service
    c1_content = "Q3 Net profit increased by 18 percent year over year."
    c2_content = "Authentication requires multi-factor confirmation across all admin access points."

    vec_alpha = embedding_service.embed_text(c1_content)
    vec_beta = embedding_service.embed_text(c2_content)

    c1 = DocumentChunk(
        id="c1",
        document_id="doc_fin",
        chunk_index=0,
        content=c1_content,
        token_count=12,
        embedding=vec_alpha,
        chunk_metadata={"page_number": 1},
    )
    c2 = DocumentChunk(
        id="c2",
        document_id="doc_sec",
        chunk_index=0,
        content=c2_content,
        token_count=14,
        embedding=vec_beta,
        chunk_metadata={"page_number": 1},
    )
    db_session.add_all([c1, c2])
    db_session.commit()

    retriever = HybridRetriever()

    # Query targeting finance
    results = retriever.retrieve(db=db_session, query="What is the net profit growth?", top_k=2)
    assert len(results) >= 1
    # Financial chunk should be top
    top_chunk = results[0]
    assert "profit" in top_chunk.content.lower()
    assert top_chunk.document_id == "doc_fin"
    assert top_chunk.rrf_score > 0.0

    # Test filtering by document_id
    sec_only = retriever.retrieve(db=db_session, query="profit", document_id="doc_sec", top_k=5)
    # doc_sec contains no profit keywords and has opposite vector
    assert all(c.document_id == "doc_sec" for c in sec_only)

    # Test filtering by file_type
    pdf_only = retriever.retrieve(db=db_session, query="authentication", file_type="pdf", top_k=5)
    assert all(c.document_id == "doc_fin" for c in pdf_only)


def test_retriever_empty_query_returns_empty(db_session: Session):
    retriever = HybridRetriever()
    assert retriever.retrieve(db=db_session, query="") == []
    assert retriever.retrieve(db=db_session, query="   ") == []
