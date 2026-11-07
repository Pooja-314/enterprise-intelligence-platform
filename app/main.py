import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.core.errors import AppException, app_exception_handler, generic_exception_handler
from app.core.telemetry import telemetry
from app.db.session import check_db_connection
from app.rag.embeddings import embedding_service
from app.api.documents import router as documents_router
from app.api.rag import router as rag_router
from app.api.agent import router as agent_router
from app.api.conversations import router as conversations_router
from app.api.auth import router as auth_router
from app.api.dashboard import router as dashboard_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    description="Enterprise-grade Document Intelligence and RAG platform with LLM fallback and multi-tool agents.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Exception handlers
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# CORS Middleware
origins = settings.CORS_ORIGINS if isinstance(settings.CORS_ORIGINS, list) else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pure ASGI Middleware for Request ID & Process Time (prevents BaseHTTPMiddleware multipart deadlocks)
class RequestContextMiddleware:
    def __init__(self, asgi_app):
        self.app = asgi_app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        req_id = str(uuid.uuid4())
        for name, value in scope.get("headers", []):
            if name.lower() == b"x-request-id":
                req_id = value.decode("latin-1")
                break

        start_time = time.perf_counter()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                latency_ms = (time.perf_counter() - start_time) * 1000
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", req_id.encode("latin-1")))
                headers.append((b"x-process-time-ms", f"{latency_ms:.2f}".encode("latin-1")))
                message["headers"] = headers
            await send(message)

        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["request_id"] = req_id

        await self.app(scope, receive, send_wrapper)


app.add_middleware(RequestContextMiddleware)

# Register Routers
app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(dashboard_router, prefix=settings.API_V1_STR)
app.include_router(documents_router, prefix=settings.API_V1_STR)
app.include_router(rag_router, prefix=settings.API_V1_STR)
app.include_router(agent_router, prefix=settings.API_V1_STR)
app.include_router(conversations_router, prefix=settings.API_V1_STR)


@app.get("/health", tags=["System"])
async def health_check():
    """System health check endpoint reporting status of core services."""
    db_health = check_db_connection()
    return {
        "status": "healthy" if db_health["connected"] else "degraded",
        "app_name": settings.APP_NAME,
        "environment": settings.ENVIRONMENT,
        "database": db_health,
        "llm_providers": {
            "primary": settings.PRIMARY_LLM_PROVIDER,
            "fallback": settings.FALLBACK_LLM_PROVIDER,
            "groq_configured": settings.has_groq,
            "gemini_configured": settings.has_gemini,
        },
        "embeddings": embedding_service.get_diagnostic(),
        "security": {
            "prompt_injection_defense": settings.PROMPT_INJECTION_PROTECTION_ENABLED,
        },
    }


@app.get(f"{settings.API_V1_STR}/telemetry", tags=["Analytics"])
async def get_telemetry_summary():
    """Retrieve operational telemetry, token metrics, and latency statistics."""
    return {
        "summary": telemetry.get_summary(),
        "recent_executions": telemetry.get_recent_records(limit=25),
    }
