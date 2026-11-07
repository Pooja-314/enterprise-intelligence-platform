"""Base Tool abstraction for AI Agent architecture.
Enforces Pydantic argument validation, execution telemetry, and structured audit logging.
"""
import abc
import time
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from app.core.errors import AppException
from app.db.models import AuditLog


class ToolExecutionError(AppException):
    """Raised when a tool encounters an unrecoverable execution failure."""
    status_code: int = 400
    error_code: str = "TOOL_EXECUTION_ERROR"

    def __init__(self, message: str, tool_name: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, details=details or {})
        self.tool_name = tool_name


class BaseTool(abc.ABC):
    """Abstract base class for all agent tools."""

    name: str
    description: str
    args_schema: Type[BaseModel]

    def run(self, db: Session, user_id: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Validates arguments, executes tool logic, measures latency, and records audit telemetry."""
        # 1. Strict argument validation using Pydantic schema
        try:
            validated_args = self.args_schema(**kwargs)
        except ValidationError as e:
            raise ToolExecutionError(
                message=f"Validation error for tool '{self.name}': {e.errors()}",
                tool_name=self.name,
                details={"errors": e.errors()},
            )

        start_time = time.perf_counter()
        status = "SUCCESS"
        error_msg = None

        try:
            result = self._execute(db=db, args=validated_args)
            return result
        except Exception as exc:
            status = "FAILED"
            error_msg = str(exc)
            raise
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._record_audit_log(
                db=db,
                user_id=user_id,
                validated_inputs=validated_args.model_dump(),
                status=status,
                latency_ms=round(latency_ms, 2),
                error=error_msg,
            )

    @abc.abstractmethod
    def _execute(self, db: Session, args: BaseModel) -> Dict[str, Any]:
        """Core tool implementation."""
        pass

    def _record_audit_log(
        self,
        db: Session,
        user_id: Optional[int],
        validated_inputs: Dict[str, Any],
        status: str,
        latency_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """Safely records tool execution telemetry to audit_logs without exposing hidden chains of thought."""
        try:
            sanitized_inputs = dict(validated_inputs)
            # Remove any raw sensitive content if present
            if "content" in sanitized_inputs:
                sanitized_inputs["content"] = f"<{len(str(sanitized_inputs['content']))} chars>"

            log_entry = AuditLog(
                user_id=user_id,
                action=f"tool_execution:{self.name}",
                resource_type="agent_tool",
                resource_id=self.name,
                details={
                    "status": status,
                    "latency_ms": latency_ms,
                    "inputs": sanitized_inputs,
                    "error": error,
                },
            )
            db.add(log_entry)
            db.commit()
        except Exception:
            # Audit logging failure must never crash the primary execution pipeline
            db.rollback()

    def get_schema(self) -> Dict[str, Any]:
        """Returns JSON schema definition for LLM tool selection."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_schema.model_json_schema(),
        }
