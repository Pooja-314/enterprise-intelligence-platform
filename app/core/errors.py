from datetime import datetime, timezone
from typing import Any, Dict, Optional
from fastapi import Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    success: bool = Field(default=False)
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    request_id: Optional[str] = None


class AppException(Exception):
    """Base application exception."""
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "INTERNAL_SERVER_ERROR"

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DatabaseConnectionError(AppException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "DATABASE_CONNECTION_ERROR"


class DocumentProcessingError(AppException):
    status_code = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)
    error_code = "DOCUMENT_PROCESSING_ERROR"


class LLMProviderError(AppException):
    status_code = status.HTTP_502_BAD_GATEWAY
    error_code = "LLM_PROVIDER_ERROR"


class LLMFallbackExhaustedError(AppException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "LLM_FALLBACK_EXHAUSTED"


class PromptInjectionDetectedError(AppException):
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "PROMPT_INJECTION_DETECTED"


class AuthenticationError(AppException):
    status_code = status.HTTP_401_UNAUTHORIZED
    error_code = "AUTHENTICATION_FAILED"


class AuthorizationError(AppException):
    status_code = status.HTTP_403_FORBIDDEN
    error_code = "AUTHORIZATION_FAILED"


class EntityNotFoundError(AppException):
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "ENTITY_NOT_FOUND"


class InsufficientEvidenceError(AppException):
    status_code = status.HTTP_200_OK  # Handled gracefully in RAG
    error_code = "INSUFFICIENT_EVIDENCE"


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
            request_id=request_id,
        ).model_dump(),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error_code="INTERNAL_UNHANDLED_ERROR",
            message="An unexpected server error occurred.",
            details={"error_type": exc.__class__.__name__, "description": str(exc)},
            request_id=request_id,
        ).model_dump(),
    )
