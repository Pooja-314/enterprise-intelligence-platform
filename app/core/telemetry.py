import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionRecord:
    timestamp: str
    provider: str
    model: str
    operation: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TelemetryTracker:
    """Thread-safe telemetry and usage tracking engine."""

    def __init__(self, max_history: int = 1000):
        self._lock = threading.Lock()
        self._max_history = max_history
        self._records: List[ExecutionRecord] = []
        self._provider_tokens: Dict[str, Dict[str, int]] = defaultdict(lambda: {"prompt": 0, "completion": 0, "total": 0})
        self._model_tokens: Dict[str, Dict[str, int]] = defaultdict(lambda: {"prompt": 0, "completion": 0, "total": 0})
        self._operation_counts: Dict[str, int] = defaultdict(int)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._latencies: List[float] = []

    def record_execution(
        self,
        provider: str,
        model: str,
        operation: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExecutionRecord:
        total_tokens = prompt_tokens + completion_tokens
        record = ExecutionRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            provider=provider,
            model=model,
            operation=operation,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            success=success,
            error=error,
            metadata=metadata or {},
        )

        with self._lock:
            self._records.append(record)
            if len(self._records) > self._max_history:
                self._records.pop(0)

            # Update provider aggregations
            self._provider_tokens[provider]["prompt"] += prompt_tokens
            self._provider_tokens[provider]["completion"] += completion_tokens
            self._provider_tokens[provider]["total"] += total_tokens

            # Update model aggregations
            self._model_tokens[model]["prompt"] += prompt_tokens
            self._model_tokens[model]["completion"] += completion_tokens
            self._model_tokens[model]["total"] += total_tokens

            self._operation_counts[operation] += 1
            if not success:
                self._error_counts[provider] += 1

            self._latencies.append(latency_ms)
            if len(self._latencies) > self._max_history:
                self._latencies.pop(0)

        return record

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            total_requests = len(self._records)
            successful_requests = sum(1 for r in self._records if r.success)
            failed_requests = total_requests - successful_requests
            avg_latency = (sum(self._latencies) / len(self._latencies)) if self._latencies else 0.0

            total_prompt = sum(v["prompt"] for v in self._provider_tokens.values())
            total_completion = sum(v["completion"] for v in self._provider_tokens.values())
            total_tokens = sum(v["total"] for v in self._provider_tokens.values())

            return {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "average_latency_ms": round(avg_latency, 2),
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_tokens": total_tokens,
                "tokens_by_provider": dict(self._provider_tokens),
                "tokens_by_model": dict(self._model_tokens),
                "operations": dict(self._operation_counts),
                "errors_by_provider": dict(self._error_counts),
            }

    def get_recent_records(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            records = self._records[-limit:]
            return [
                {
                    "timestamp": r.timestamp,
                    "provider": r.provider,
                    "model": r.model,
                    "operation": r.operation,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "total_tokens": r.total_tokens,
                    "latency_ms": round(r.latency_ms, 2),
                    "success": r.success,
                    "error": r.error,
                    "metadata": r.metadata,
                }
                for r in reversed(records)
            ]

    def reset(self) -> None:
        """Reset all in-memory telemetry data (primarily for testing)."""
        with self._lock:
            self._records.clear()
            self._provider_tokens.clear()
            self._model_tokens.clear()
            self._operation_counts.clear()
            self._error_counts.clear()
            self._latencies.clear()


# Global telemetry singleton
telemetry = TelemetryTracker()
