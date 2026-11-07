"""Knowledge Base Search Tool for AI Agent architecture.
Executes hybrid vector + full-text search using existing HybridRetriever.
Returns ranked chunks with RRF scores, page metadata, and verified citations.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agent.base import BaseTool
from app.rag.retrieval import HybridRetriever, RetrievedChunk


class KBSearchArgs(BaseModel):
    query: str = Field(..., min_length=1, description="Semantic or keyword search query string.")
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum number of relevant chunks to retrieve.")
    document_id: Optional[str] = Field(default=None, description="Filter chunks strictly by specific document ID.")
    file_type: Optional[str] = Field(default=None, description="Filter chunks by document file extension (e.g. pdf, csv, txt).")


class KnowledgeBaseSearchTool(BaseTool):
    """Searches the enterprise vector database using dense pgvector + sparse keyword retrieval (RRF)."""

    name = "knowledge_base_search"
    description = (
        "Search the enterprise knowledge base across uploaded documents (PDFs, CSVs, TXT, Markdown). "
        "Returns ranked passages, document metadata, page numbers, and exact citation references."
    )
    args_schema = KBSearchArgs

    def __init__(self, retriever: Optional[HybridRetriever] = None):
        self.retriever = retriever or HybridRetriever()

    def _execute(self, db: Session, args: KBSearchArgs) -> Dict[str, Any]:
        chunks: List[RetrievedChunk] = self.retriever.retrieve(
            db=db,
            query=args.query,
            top_k=args.top_k,
            document_id=args.document_id,
            file_type=args.file_type,
        )

        formatted_results = []
        for c in chunks:
            page = c.metadata.get("page_number")
            row = c.metadata.get("row_number")
            section = c.metadata.get("header")

            citation_parts = [f"Doc: {c.document_title}"]
            if page is not None:
                citation_parts.append(f"Page: {page}")
            if row is not None:
                citation_parts.append(f"Row: {row}")
            if section:
                citation_parts.append(f"Section: {section}")
            citation_parts.append(f"Chunk: {c.chunk_id}")
            citation_str = f"[{', '.join(citation_parts)}]"

            formatted_results.append({
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "document_title": c.document_title,
                "content": c.content,
                "rrf_score": c.rrf_score,
                "dense_rank": c.dense_rank,
                "sparse_rank": c.sparse_rank,
                "metadata": c.metadata,
                "citation": citation_str,
            })

        return {
            "query": args.query,
            "total_retrieved": len(formatted_results),
            "results": formatted_results,
        }
