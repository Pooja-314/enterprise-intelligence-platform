import io
import pytest
import pandas as pd
import pymupdf as fitz
from sqlalchemy.orm import Session

from app.db.models import Document, DocumentChunk, ProcessingStatus
from app.ingestion.pipeline import IngestionPipeline, IngestionResult


def create_sample_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), text, fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_pipeline_ingest_txt_success(db_session: Session):
    pipeline = IngestionPipeline()
    content = "Enterprise Intelligence Platform documentation: Chapter 1 Introduction to Advanced Hybrid Retrieval."
    file_bytes = content.encode("utf-8")

    result = pipeline.ingest_bytes(db=db_session, filename="intro.txt", file_bytes=file_bytes)

    assert isinstance(result, IngestionResult)
    assert result.status == ProcessingStatus.COMPLETED.value
    assert not result.is_duplicate
    assert result.total_chunks >= 1
    assert result.total_tokens > 0

    # Query database directly to verify real persistence
    doc = db_session.query(Document).filter(Document.id == result.document_id).first()
    assert doc is not None
    assert doc.processing_status == ProcessingStatus.COMPLETED.value
    assert doc.filename == "intro.txt"
    assert doc.content_hash == result.content_hash

    chunks = db_session.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).all()
    assert len(chunks) == result.total_chunks
    assert chunks[0].embedding is not None
    assert len(chunks[0].embedding) == 384


def test_pipeline_ingest_pdf_with_page_metadata(db_session: Session):
    pipeline = IngestionPipeline()
    pdf_bytes = create_sample_pdf("Page 1: Artificial Intelligence systems evaluation standards.")

    result = pipeline.ingest_bytes(db=db_session, filename="standards.pdf", file_bytes=pdf_bytes)

    assert result.status == ProcessingStatus.COMPLETED.value
    assert result.file_type == "pdf"

    chunks = db_session.query(DocumentChunk).filter(DocumentChunk.document_id == result.document_id).all()
    assert len(chunks) >= 1
    assert chunks[0].chunk_metadata.get("page_number") == 1


def test_pipeline_ingest_csv_with_row_metadata(db_session: Session):
    pipeline = IngestionPipeline()
    df = pd.DataFrame({
        "Model": ["Groq Llama-3.3", "Gemini 2.0 Flash"],
        "LatencyMs": [210, 480],
        "CostPer1M": [0.59, 0.10],
    })
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode("utf-8")

    result = pipeline.ingest_bytes(db=db_session, filename="llm_benchmarks.csv", file_bytes=csv_bytes)
    assert result.status == ProcessingStatus.COMPLETED.value
    assert result.file_type == "csv"

    chunks = db_session.query(DocumentChunk).filter(DocumentChunk.document_id == result.document_id).all()
    assert len(chunks) == 2
    assert "Row 1" in chunks[0].content
    assert chunks[0].chunk_metadata.get("row_number") == 1


def test_pipeline_sha256_duplicate_detection(db_session: Session):
    pipeline = IngestionPipeline()
    content = "Unique content for SHA-256 duplicate detection validation."
    file_bytes = content.encode("utf-8")

    # First ingestion
    result_1 = pipeline.ingest_bytes(db=db_session, filename="doc_v1.txt", file_bytes=file_bytes)
    assert not result_1.is_duplicate
    assert result_1.status == ProcessingStatus.COMPLETED.value

    # Second ingestion with identical content bytes
    result_2 = pipeline.ingest_bytes(db=db_session, filename="doc_copy.txt", file_bytes=file_bytes)
    assert result_2.is_duplicate
    assert result_2.document_id == result_1.document_id
    assert result_2.content_hash == result_1.content_hash

    # Ensure no duplicate chunks were created in database
    total_docs = db_session.query(Document).filter(Document.content_hash == result_1.content_hash).count()
    assert total_docs == 1


def test_pipeline_processing_failure_records_failed_status(db_session: Session):
    pipeline = IngestionPipeline()
    corrupted_pdf = b"%PDF-invalid truncated data"

    result = pipeline.ingest_bytes(db=db_session, filename="broken.pdf", file_bytes=corrupted_pdf)

    assert result.status == ProcessingStatus.FAILED.value
    assert result.error_message is not None

    # Check that failed status was committed to DB for dashboard inspection
    doc = db_session.query(Document).filter(Document.id == result.document_id).first()
    assert doc is not None
    assert doc.processing_status == ProcessingStatus.FAILED.value
    assert "broken.pdf" in doc.filename
    assert doc.error_message is not None

    # No orphan chunks for failed document
    chunks = db_session.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).all()
    assert len(chunks) == 0


def test_cascade_delete_removes_chunks(db_session: Session):
    pipeline = IngestionPipeline()
    file_bytes = b"Sample text for cascade deletion verification."
    result = pipeline.ingest_bytes(db=db_session, filename="delete_me.txt", file_bytes=file_bytes)

    doc_id = result.document_id
    doc = db_session.query(Document).filter(Document.id == doc_id).first()
    assert doc is not None

    chunks_count = db_session.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).count()
    assert chunks_count > 0

    # Delete parent document
    db_session.delete(doc)
    db_session.commit()

    # Verify both document and chunks are removed
    assert db_session.query(Document).filter(Document.id == doc_id).first() is None
    assert db_session.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).count() == 0


def test_pipeline_ingest_markdown_with_headers(db_session: Session):
    pipeline = IngestionPipeline()
    md_content = """# Executive Summary
The intelligence platform automates document understanding.

## Security Controls
All uploads are scanned for prompt injection and hashed with SHA-256.
"""
    result = pipeline.ingest_bytes(db=db_session, filename="security.md", file_bytes=md_content.encode("utf-8"))

    assert result.status == ProcessingStatus.COMPLETED.value
    assert result.file_type == "md"
    assert result.total_chunks >= 2

    chunks = db_session.query(DocumentChunk).filter(DocumentChunk.document_id == result.document_id).all()
    headers = [c.chunk_metadata.get("header") for c in chunks]
    assert "Executive Summary" in headers or "Security Controls" in headers


def test_pipeline_transaction_rollback_on_db_error(db_session: Session, monkeypatch):
    pipeline = IngestionPipeline()
    file_bytes = b"Valid content that will encounter a database failure during bulk save."

    # Simulate database failure during chunk insertion
    def fail_bulk_save(*args, **kwargs):
        raise RuntimeError("Simulated database write timeout")

    monkeypatch.setattr(db_session, "bulk_save_objects", fail_bulk_save)

    result = pipeline.ingest_bytes(db=db_session, filename="rollback_test.txt", file_bytes=file_bytes)

    assert result.status == ProcessingStatus.FAILED.value
    assert "Simulated database write timeout" in result.error_message

    # Confirm document record reflects FAILED state in DB
    doc = db_session.query(Document).filter(Document.id == result.document_id).first()
    assert doc is not None
    assert doc.processing_status == ProcessingStatus.FAILED.value
    assert "Simulated database write timeout" in doc.error_message

    # Confirm 0 orphan chunks were persisted
    orphan_chunks = db_session.query(DocumentChunk).filter(DocumentChunk.document_id == result.document_id).count()
    assert orphan_chunks == 0
