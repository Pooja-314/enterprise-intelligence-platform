import io
import pytest
from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.errors import DocumentProcessingError, EntityNotFoundError
from app.api.documents import (
    upload_document,
    list_documents,
    get_document,
    get_document_chunks,
    delete_document,
)


@pytest.mark.asyncio
async def test_api_upload_document_success(db_session: Session):
    file_content = b"Enterprise Intelligence Platform API integration test content."
    upload_file = UploadFile(filename="api_test.txt", file=io.BytesIO(file_content))

    # Test POST /api/v1/documents/upload
    result = await upload_document(file=upload_file, allow_duplicate=False, db=db_session)
    assert result.status == "COMPLETED"
    assert result.filename == "api_test.txt"
    assert result.total_chunks >= 1
    assert not result.is_duplicate
    doc_id = result.document_id

    # Test GET /api/v1/documents/{id}
    doc = get_document(document_id=doc_id, db=db_session)
    assert doc.id == doc_id
    assert doc.processing_status == "COMPLETED"

    # Test GET /api/v1/documents/{id}/chunks
    chunks = get_document_chunks(document_id=doc_id, db=db_session)
    assert len(chunks) == result.total_chunks
    assert "Enterprise Intelligence" in chunks[0].content

    # Test GET /api/v1/documents (listing)
    doc_list = list_documents(limit=10, offset=0, status_filter=None, db=db_session)
    assert len(doc_list) >= 1
    assert any(d.id == doc_id for d in doc_list)

    # Test DELETE /api/v1/documents/{id}
    del_res = delete_document(document_id=doc_id, db=db_session)
    assert del_res["success"] is True

    # Confirm 404 EntityNotFoundError after deletion
    with pytest.raises(EntityNotFoundError):
        get_document(document_id=doc_id, db=db_session)


@pytest.mark.asyncio
async def test_api_upload_duplicate_detection(db_session: Session):
    file_content = b"Content repeated for API duplicate detection endpoint test."

    upload_file_1 = UploadFile(filename="dup1.txt", file=io.BytesIO(file_content))
    res1 = await upload_document(file=upload_file_1, allow_duplicate=False, db=db_session)
    assert not res1.is_duplicate

    # Upload duplicate bytes
    upload_file_2 = UploadFile(filename="dup2.txt", file=io.BytesIO(file_content))
    res2 = await upload_document(file=upload_file_2, allow_duplicate=False, db=db_session)
    assert res2.is_duplicate
    assert res2.document_id == res1.document_id


@pytest.mark.asyncio
async def test_api_upload_empty_document_rejected(db_session: Session):
    upload_file = UploadFile(filename="empty.txt", file=io.BytesIO(b""))
    with pytest.raises(DocumentProcessingError):
        await upload_document(file=upload_file, allow_duplicate=False, db=db_session)


def test_api_get_nonexistent_document_raises_404(db_session: Session):
    with pytest.raises(EntityNotFoundError):
        get_document(document_id="nonexistent-id-12345", db=db_session)


@pytest.mark.asyncio
async def test_api_list_documents_pagination_and_filtering(db_session: Session):
    # Ingest 3 distinct documents
    for i in range(3):
        upload_file = UploadFile(filename=f"doc_{i}.txt", file=io.BytesIO(f"Unique content for document {i}.".encode("utf-8")))
        await upload_document(file=upload_file, allow_duplicate=False, db=db_session)

    # Test limit=2
    page_1 = list_documents(limit=2, offset=0, status_filter=None, db=db_session)
    assert len(page_1) == 2

    # Test offset=2, limit=2
    page_2 = list_documents(limit=2, offset=2, status_filter=None, db=db_session)
    assert len(page_2) >= 1
    assert page_1[0].id != page_2[0].id

    # Test status filter matching COMPLETED
    completed_docs = list_documents(limit=10, offset=0, status_filter="COMPLETED", db=db_session)
    assert len(completed_docs) >= 3
    assert all(d.processing_status == "COMPLETED" for d in completed_docs)

    # Test status filter matching FAILED (should be 0)
    failed_docs = list_documents(limit=10, offset=0, status_filter="FAILED", db=db_session)
    assert len(failed_docs) == 0


def test_api_get_chunks_and_delete_nonexistent_document(db_session: Session):
    with pytest.raises(EntityNotFoundError):
        get_document_chunks(document_id="missing-doc-id", db=db_session)

    with pytest.raises(EntityNotFoundError):
        delete_document(document_id="missing-doc-id", db=db_session)
