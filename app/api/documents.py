from typing import List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.core.errors import DocumentProcessingError, EntityNotFoundError
from app.db.session import get_db
from app.db.models import Document, DocumentChunk, ProcessingStatus
from app.ingestion.pipeline import IngestionPipeline, IngestionResult

router = APIRouter(prefix="/documents", tags=["Documents"])
pipeline = IngestionPipeline()


class DocumentResponse(BaseModel):
    id: str
    title: str
    filename: str
    file_type: str
    file_size_bytes: int
    content_hash: str
    processing_status: str
    error_message: Optional[str] = None
    total_chunks: int
    total_tokens: int
    created_at: str

    model_config = ConfigDict(from_attributes=True)


class ChunkResponse(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    content: str
    token_count: int
    chunk_metadata: dict

    model_config = ConfigDict(from_attributes=True)


class UploadResultResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    content_hash: str
    status: str
    is_duplicate: bool
    total_chunks: int
    total_tokens: int
    error_message: Optional[str] = None
    duration_ms: float


@router.post("/upload", response_model=UploadResultResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    allow_duplicate: bool = Query(default=False),
    db: Session = Depends(get_db),
):
    """Upload a PDF, TXT, Markdown, or CSV file for parsing, deduplication, chunking, and embedding."""
    if not file.filename:
        raise DocumentProcessingError("Filename must not be empty.")

    # Read underlying stream directly
    if hasattr(file, "file") and hasattr(file.file, "read"):
        file_bytes = file.file.read()
    else:
        file_bytes = await file.read()

    if not file_bytes:
        raise DocumentProcessingError(f"Uploaded file '{file.filename}' is 0 bytes.")

    result = pipeline.ingest_bytes(
        db=db,
        filename=file.filename,
        file_bytes=file_bytes,
        allow_duplicate=allow_duplicate,
    )

    return UploadResultResponse(
        document_id=result.document_id,
        filename=result.filename,
        file_type=result.file_type,
        content_hash=result.content_hash,
        status=result.status,
        is_duplicate=result.is_duplicate,
        total_chunks=result.total_chunks,
        total_tokens=result.total_tokens,
        error_message=result.error_message,
        duration_ms=result.duration_ms,
    )


@router.get("", response_model=List[DocumentResponse])
def list_documents(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status_filter: Optional[str] = Query(default=None, alias="status"),
    db: Session = Depends(get_db),
):
    """Retrieve list of ingested documents with optional status filtering."""
    query = db.query(Document)
    if status_filter:
        query = query.filter(Document.processing_status == status_filter.upper())
    docs = query.order_by(Document.created_at.desc()).offset(offset).limit(limit).all()

    return [
        DocumentResponse(
            id=d.id,
            title=d.title,
            filename=d.filename,
            file_type=d.file_type,
            file_size_bytes=d.file_size_bytes,
            content_hash=d.content_hash,
            processing_status=d.processing_status,
            error_message=d.error_message,
            total_chunks=d.total_chunks,
            total_tokens=d.total_tokens,
            created_at=d.created_at.isoformat(),
        )
        for d in docs
    ]


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str, db: Session = Depends(get_db)):
    """Retrieve details and processing status of a specific document."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise EntityNotFoundError(f"Document with ID '{document_id}' not found.")
    return DocumentResponse(
        id=doc.id,
        title=doc.title,
        filename=doc.filename,
        file_type=doc.file_type,
        file_size_bytes=doc.file_size_bytes,
        content_hash=doc.content_hash,
        processing_status=doc.processing_status,
        error_message=doc.error_message,
        total_chunks=doc.total_chunks,
        total_tokens=doc.total_tokens,
        created_at=doc.created_at.isoformat(),
    )


@router.get("/{document_id}/chunks", response_model=List[ChunkResponse])
def get_document_chunks(document_id: str, db: Session = Depends(get_db)):
    """Retrieve all chunks belonging to a document."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise EntityNotFoundError(f"Document with ID '{document_id}' not found.")

    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index.asc())
        .all()
    )
    return [
        ChunkResponse(
            id=c.id,
            document_id=c.document_id,
            chunk_index=c.chunk_index,
            content=c.content,
            token_count=c.token_count,
            chunk_metadata=c.chunk_metadata,
        )
        for c in chunks
    ]


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Deletes a document and cascades deletion of all associated chunks."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise EntityNotFoundError(f"Document with ID '{document_id}' not found.")

    db.delete(doc)
    db.commit()
    return {"success": True, "message": f"Document '{document_id}' and all its chunks were deleted."}
