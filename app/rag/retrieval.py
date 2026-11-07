import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Document, DocumentChunk
from app.rag.embeddings import EmbeddingService, embedding_service

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_title: str
    content: str
    token_count: int
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    """Hybrid Retrieval engine combining dense vector search (pgvector)
    and sparse keyword search (Full-Text Search) using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        embedder: Optional[EmbeddingService] = None,
        rrf_k: Optional[int] = None,
        top_k_vector: Optional[int] = None,
        top_k_keyword: Optional[int] = None,
        top_k_final: Optional[int] = None,
    ):
        self.embedder = embedder or embedding_service
        self.rrf_k = rrf_k or settings.RRF_K
        self.top_k_vector = top_k_vector or settings.TOP_K_VECTOR
        self.top_k_keyword = top_k_keyword or settings.TOP_K_KEYWORD
        self.top_k_final = top_k_final or settings.TOP_K_FINAL

    def retrieve(
        self,
        db: Session,
        query: str,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
        file_type: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Executes full hybrid retrieval: vector search + keyword search -> RRF fusion."""
        query = query.strip()
        if not query:
            return []

        final_limit = top_k or self.top_k_final
        query_embedding = self.embedder.embed_text(query)

        # 1. Dense Vector Retrieval
        dense_results = self.vector_search(
            db=db,
            query_embedding=query_embedding,
            top_k=self.top_k_vector,
            document_id=document_id,
            file_type=file_type,
        )

        # 2. Sparse Keyword Retrieval
        sparse_results = self.keyword_search(
            db=db,
            query_text=query,
            top_k=self.top_k_keyword,
            document_id=document_id,
            file_type=file_type,
        )

        # 3. Reciprocal Rank Fusion
        fused = self.reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=self.rrf_k,
        )

        return fused[:final_limit]

    def vector_search(
        self,
        db: Session,
        query_embedding: List[float],
        top_k: int = 10,
        document_id: Optional[str] = None,
        file_type: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Dense vector retrieval using pgvector cosine distance on PostgreSQL
        or in-memory vector cosine similarity on SQLite.
        """
        is_postgres = db.bind.dialect.name == "postgresql" if db.bind else False

        if is_postgres:
            return self._vector_search_postgres(db, query_embedding, top_k, document_id, file_type)
        else:
            return self._vector_search_portable(db, query_embedding, top_k, document_id, file_type)

    def _vector_search_postgres(
        self,
        db: Session,
        query_embedding: List[float],
        top_k: int,
        document_id: Optional[str],
        file_type: Optional[str],
    ) -> List[RetrievedChunk]:
        """PostgreSQL native vector search using the <=> cosine distance operator."""
        emb_str = str(query_embedding)
        sql = """
            SELECT 
                c.id AS chunk_id,
                c.document_id,
                d.title AS document_title,
                c.content,
                c.token_count,
                c.chunk_metadata,
                (c.embedding <=> :query_emb) AS distance
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.embedding IS NOT NULL
        """
        params: Dict[str, Any] = {"query_emb": emb_str, "top_k": top_k}

        if document_id:
            sql += " AND c.document_id = :doc_id"
            params["doc_id"] = document_id
        if file_type:
            sql += " AND d.file_type = :file_type"
            params["file_type"] = file_type

        sql += " ORDER BY distance ASC LIMIT :top_k"

        results = db.execute(text(sql), params).fetchall()
        chunks: List[RetrievedChunk] = []

        for row in results:
            distance = float(row.distance) if row.distance is not None else 1.0
            similarity = max(0.0, 1.0 - distance)
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(row.chunk_id),
                    document_id=str(row.document_id),
                    document_title=str(row.document_title),
                    content=str(row.content),
                    token_count=int(row.token_count),
                    dense_score=round(similarity, 4),
                    metadata=dict(row.chunk_metadata or {}),
                )
            )
        return chunks

    def _vector_search_portable(
        self,
        db: Session,
        query_embedding: List[float],
        top_k: int,
        document_id: Optional[str],
        file_type: Optional[str],
    ) -> List[RetrievedChunk]:
        """Portable in-memory cosine similarity search for SQLite test environments."""
        query = db.query(DocumentChunk, Document).join(Document, DocumentChunk.document_id == Document.id)
        if document_id:
            query = query.filter(DocumentChunk.document_id == document_id)
        if file_type:
            query = query.filter(Document.file_type == file_type)

        rows = query.all()
        if not rows:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        scored: List[tuple[float, DocumentChunk, Document]] = []

        for chunk, doc in rows:
            if not chunk.embedding:
                continue
            c_vec = np.array(chunk.embedding, dtype=np.float32)
            c_norm = np.linalg.norm(c_vec)
            if q_norm > 0 and c_norm > 0:
                sim = float(np.dot(q_vec, c_vec) / (q_norm * c_norm))
            else:
                sim = 0.0
            scored.append((sim, chunk, doc))

        # Sort descending by cosine similarity
        scored.sort(key=lambda x: x[0], reverse=True)
        top_rows = scored[:top_k]

        return [
            RetrievedChunk(
                chunk_id=c.id,
                document_id=c.document_id,
                document_title=d.title,
                content=c.content,
                token_count=c.token_count,
                dense_score=round(sim, 4),
                metadata=dict(c.chunk_metadata or {}),
            )
            for sim, c, d in top_rows
        ]

    def keyword_search(
        self,
        db: Session,
        query_text: str,
        top_k: int = 10,
        document_id: Optional[str] = None,
        file_type: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Sparse keyword search using PostgreSQL Full-Text Search or word-token matching."""
        is_postgres = db.bind.dialect.name == "postgresql" if db.bind else False

        if is_postgres:
            return self._keyword_search_postgres(db, query_text, top_k, document_id, file_type)
        else:
            return self._keyword_search_portable(db, query_text, top_k, document_id, file_type)

    def _keyword_search_postgres(
        self,
        db: Session,
        query_text: str,
        top_k: int,
        document_id: Optional[str],
        file_type: Optional[str],
    ) -> List[RetrievedChunk]:
        """PostgreSQL Full-Text Search using plainto_tsquery and ts_rank."""
        sql = """
            SELECT 
                c.id AS chunk_id,
                c.document_id,
                d.title AS document_title,
                c.content,
                c.token_count,
                c.chunk_metadata,
                ts_rank(to_tsvector('english', c.content), plainto_tsquery('english', :query)) AS rank_score
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', :query)
        """
        params: Dict[str, Any] = {"query": query_text, "top_k": top_k}

        if document_id:
            sql += " AND c.document_id = :doc_id"
            params["doc_id"] = document_id
        if file_type:
            sql += " AND d.file_type = :file_type"
            params["file_type"] = file_type

        sql += " ORDER BY rank_score DESC LIMIT :top_k"

        results = db.execute(text(sql), params).fetchall()

        # Step 1 Fallback: If strict conjunction returns zero results, retry with sanitized keywords and OR matching
        if not results:
            stopwords = {
                "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
                "by", "is", "are", "was", "were", "what", "which", "who", "whom", "this", "that",
                "these", "those", "how", "why", "when", "where", "can", "could", "should", "would",
                "do", "does", "did", "have", "has", "had", "as", "explain", "describe", "cite",
                "relevant", "source", "document", "sections", "pages", "its", "each", "per", "from"
            }
            words = [w for w in re.findall(r"\b[a-zA-Z0-9]{3,}\b", query_text.lower()) if w not in stopwords]
            if words:
                unique_words = list(dict.fromkeys(words))
                tsquery_or = " | ".join(unique_words)
                sql_or = """
                    SELECT 
                        c.id AS chunk_id,
                        c.document_id,
                        d.title AS document_title,
                        c.content,
                        c.token_count,
                        c.chunk_metadata,
                        ts_rank(to_tsvector('english', c.content), to_tsquery('english', :tsquery)) AS rank_score
                    FROM document_chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE to_tsvector('english', c.content) @@ to_tsquery('english', :tsquery)
                """
                params_or: Dict[str, Any] = {"tsquery": tsquery_or, "top_k": top_k}
                if document_id:
                    sql_or += " AND c.document_id = :doc_id"
                    params_or["doc_id"] = document_id
                if file_type:
                    sql_or += " AND d.file_type = :file_type"
                    params_or["file_type"] = file_type
                sql_or += " ORDER BY rank_score DESC LIMIT :top_k"
                try:
                    results = db.execute(text(sql_or), params_or).fetchall()
                    if results:
                        logger.info(f"FTS strict query yielded 0 results; recovered {len(results)} chunks via OR tsquery fallback.")
                except Exception as e:
                    logger.warning(f"Fallback tsquery search failed: {e}")

        chunks: List[RetrievedChunk] = []

        for row in results:
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(row.chunk_id),
                    document_id=str(row.document_id),
                    document_title=str(row.document_title),
                    content=str(row.content),
                    token_count=int(row.token_count),
                    sparse_score=round(float(row.rank_score), 4),
                    metadata=dict(row.chunk_metadata or {}),
                )
            )
        return chunks

    def _keyword_search_portable(
        self,
        db: Session,
        query_text: str,
        top_k: int,
        document_id: Optional[str],
        file_type: Optional[str],
    ) -> List[RetrievedChunk]:
        """Portable token-frequency keyword matching for SQLite test environments."""
        query = db.query(DocumentChunk, Document).join(Document, DocumentChunk.document_id == Document.id)
        if document_id:
            query = query.filter(DocumentChunk.document_id == document_id)
        if file_type:
            query = query.filter(Document.file_type == file_type)

        rows = query.all()
        if not rows:
            return []

        # Tokenize query into alphanumeric keywords (min length 2) and filter stopwords
        stopwords = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "is", "are", "was", "were", "what", "which", "who", "whom", "this", "that",
            "these", "those", "how", "why", "when", "where", "can", "could", "should", "would",
            "do", "does", "did", "have", "has", "had", "as"
        }
        all_words = set(re.findall(r"\b\w{2,}\b", query_text.lower()))
        keywords = all_words - stopwords
        if not keywords:
            keywords = all_words
        if not keywords:
            return []

        scored: List[tuple[float, DocumentChunk, Document]] = []

        for chunk, doc in rows:
            content_lower = chunk.content.lower()
            matched_count = 0
            for kw in keywords:
                # Count keyword occurrences
                occurrences = content_lower.count(kw)
                if occurrences > 0:
                    matched_count += 1 + math.log(occurrences)

            if matched_count > 0:
                # BM25-like length normalization
                length_norm = 1.0 + math.log(max(1, len(content_lower.split()) / 50.0))
                score = matched_count / length_norm
                scored.append((score, chunk, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_rows = scored[:top_k]

        return [
            RetrievedChunk(
                chunk_id=c.id,
                document_id=c.document_id,
                document_title=d.title,
                content=c.content,
                token_count=c.token_count,
                sparse_score=round(s, 4),
                metadata=dict(c.chunk_metadata or {}),
            )
            for s, c, d in top_rows
        ]

    @classmethod
    def reciprocal_rank_fusion(
        cls,
        dense_results: List[RetrievedChunk],
        sparse_results: List[RetrievedChunk],
        k: int = 60,
    ) -> List[RetrievedChunk]:
        """Combines dense and sparse ranked results using Reciprocal Rank Fusion (RRF).
        
        Formula: RRF_score(d) = sum_{m in {dense, sparse}} 1 / (k + rank_m(d))
        """
        fused_map: Dict[str, RetrievedChunk] = {}
        dense_ranks: Dict[str, int] = {}
        sparse_ranks: Dict[str, int] = {}

        # 1. Map dense rankings (1-indexed)
        for rank_idx, chunk in enumerate(dense_results):
            dense_ranks[chunk.chunk_id] = rank_idx + 1
            chunk.dense_rank = rank_idx + 1
            fused_map[chunk.chunk_id] = chunk

        # 2. Map sparse rankings (1-indexed)
        for rank_idx, chunk in enumerate(sparse_results):
            sparse_ranks[chunk.chunk_id] = rank_idx + 1
            chunk.sparse_rank = rank_idx + 1
            if chunk.chunk_id in fused_map:
                existing = fused_map[chunk.chunk_id]
                existing.sparse_rank = rank_idx + 1
                existing.sparse_score = chunk.sparse_score
            else:
                fused_map[chunk.chunk_id] = chunk

        # 3. Calculate fused RRF score
        for chunk_id, chunk in fused_map.items():
            rrf_score = 0.0
            if chunk_id in dense_ranks:
                rrf_score += 1.0 / (k + dense_ranks[chunk_id])
            if chunk_id in sparse_ranks:
                rrf_score += 1.0 / (k + sparse_ranks[chunk_id])

            chunk.rrf_score = round(rrf_score, 6)

        # 4. Sort descending by RRF score
        fused_list = list(fused_map.values())
        fused_list.sort(key=lambda c: c.rrf_score, reverse=True)

        return fused_list
