import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.core.errors import DocumentProcessingError
from app.db.models import Document, DocumentChunk, ProcessingStatus
from app.ingestion.chunker import TextChunker
from app.ingestion.parser import DocumentParser, ParsedDocument
from app.rag.embeddings import EmbeddingService, embedding_service


@dataclass
class IngestionResult:
    document_id: str
    filename: str
    file_type: str
    content_hash: str
    status: str
    is_duplicate: bool
    total_chunks: int
    total_tokens: int
    error_message: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class IngestionPipeline:
    """Document Ingestion Pipeline handling validation, parsing, deduplication,
    chunking, vector embedding, and state tracking.
    """

    def __init__(
        self,
        chunker: Optional[TextChunker] = None,
        embedder: Optional[EmbeddingService] = None,
    ):
        self.chunker = chunker or TextChunker()
        self.embedder = embedder or embedding_service

    def ingest_bytes(
        self,
        db: Session,
        filename: str,
        file_bytes: bytes,
        allow_duplicate: bool = False,
    ) -> IngestionResult:
        """Processes raw file bytes from start to finish within a database transaction."""
        start_time = time.perf_counter()
        content_hash = DocumentParser.calculate_sha256(file_bytes)
        file_size = len(file_bytes)

        # 1. SHA-256 Duplicate Check
        existing_doc = (
            db.query(Document)
            .filter(Document.content_hash == content_hash, Document.processing_status == ProcessingStatus.COMPLETED.value)
            .first()
        )

        if existing_doc and not allow_duplicate:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return IngestionResult(
                document_id=existing_doc.id,
                filename=existing_doc.filename,
                file_type=existing_doc.file_type,
                content_hash=content_hash,
                status=existing_doc.processing_status,
                is_duplicate=True,
                total_chunks=existing_doc.total_chunks,
                total_tokens=existing_doc.total_tokens,
                duration_ms=round(duration_ms, 2),
                metadata=existing_doc.doc_metadata,
            )

        # 2. Create Initial Document Record in PENDING state
        doc_id = str(uuid.uuid4())
        ext = filename.split(".")[-1].lower() if "." in filename else "unknown"

        document = Document(
            id=doc_id,
            title=filename,
            filename=filename,
            file_type=ext,
            file_size_bytes=file_size,
            content_hash=content_hash,
            processing_status=ProcessingStatus.PENDING.value,
            doc_metadata={},
        )
        db.add(document)
        db.commit()

        # 3. Transition to PROCESSING state
        document.processing_status = ProcessingStatus.PROCESSING.value
        db.commit()

        try:
            # 4. Parse document
            parsed_doc = DocumentParser.parse(filename=filename, file_bytes=file_bytes)
            document.file_type = parsed_doc.file_type
            document.doc_metadata = parsed_doc.metadata

            # 5. Chunk sections
            chunks = self.chunker.chunk_sections(parsed_doc.sections)
            if not chunks:
                raise DocumentProcessingError(f"Document '{filename}' produced 0 chunks after parsing.")

            # 6. Generate embeddings
            texts_to_embed = [c.content for c in chunks]
            embeddings = self.embedder.embed_batch(texts_to_embed)

            # 7. Persist chunks
            total_tokens = sum(c.token_count for c in chunks)
            chunk_records = []
            for i, chunk in enumerate(chunks):
                emb = embeddings[i] if i < len(embeddings) else None
                chunk_rec = DocumentChunk(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    token_count=chunk.token_count,
                    embedding=emb,
                    chunk_metadata=chunk.metadata,
                )
                chunk_records.append(chunk_rec)

            db.bulk_save_objects(chunk_records)

            # 8. Mark document as COMPLETED
            document.processing_status = ProcessingStatus.COMPLETED.value
            document.total_chunks = len(chunks)
            document.total_tokens = total_tokens
            document.updated_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(document)

            duration_ms = (time.perf_counter() - start_time) * 1000
            return IngestionResult(
                document_id=doc_id,
                filename=filename,
                file_type=document.file_type,
                content_hash=content_hash,
                status=ProcessingStatus.COMPLETED.value,
                is_duplicate=False,
                total_chunks=len(chunks),
                total_tokens=total_tokens,
                duration_ms=round(duration_ms, 2),
                metadata=document.doc_metadata,
            )

        except Exception as exc:
            db.rollback()
            # In case of failure, record status FAILED with exact error message
            error_msg = str(exc)
            document.processing_status = ProcessingStatus.FAILED.value
            document.error_message = error_msg
            document.updated_at = datetime.now(timezone.utc)
            db.commit()

            duration_ms = (time.perf_counter() - start_time) * 1000
            return IngestionResult(
                document_id=doc_id,
                filename=filename,
                file_type=ext,
                content_hash=content_hash,
                status=ProcessingStatus.FAILED.value,
                is_duplicate=False,
                total_chunks=0,
                total_tokens=0,
                error_message=error_msg,
                duration_ms=round(duration_ms, 2),
            )
