"""API package for Enterprise Intelligence Platform."""
from app.api.documents import router as documents_router
from app.api.rag import router as rag_router

__all__ = ["documents_router", "rag_router"]
