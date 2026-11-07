import logging
import time
from typing import Optional
from app.config import settings
from app.core.errors import LLMProviderError
from app.core.telemetry import telemetry
from app.llm.base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class GroqProvider(BaseLLMProvider):
    """Primary LLM provider using Groq API for high-throughput, low-latency inference."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key if api_key is not None else settings.GROQ_API_KEY
        self.model = model or settings.GROQ_MODEL
        self._client = None

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip() and not self.api_key.startswith("gsk_your_"))

    def _get_client(self):
        if self._client is None:
            if not self.is_configured():
                raise LLMProviderError("Groq API key is not configured.")
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key, timeout=settings.LLM_TIMEOUT_SECONDS)
            except Exception as e:
                raise LLMProviderError(f"Failed to initialize Groq client: {str(e)}")
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency_ms = (time.perf_counter() - start) * 1000

            content = completion.choices[0].message.content or ""
            usage = completion.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else (prompt_tokens + completion_tokens)

            # Record telemetry
            telemetry.record_execution(
                provider="groq",
                model=self.model,
                operation="generate",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                success=True,
            )

            return LLMResponse(
                content=content,
                provider="groq",
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=round(latency_ms, 2),
            )
        except Exception as exc:
            # Handle Groq's special 'tool_use_failed' when model emits tool call syntax without tools schema
            err_body = getattr(exc, "body", None)
            failed_gen = None
            if isinstance(err_body, dict):
                failed_gen = err_body.get("error", {}).get("failed_generation")
            if not failed_gen and "failed_generation" in str(exc):
                import re
                match = re.search(r"['\"]failed_generation['\"]\s*:\s*['\"](\{[\s\S]*?\})['\"]", str(exc))
                if match:
                    try:
                        failed_gen = match.group(1).encode().decode("unicode-escape")
                    except Exception:
                        failed_gen = match.group(1)

            if failed_gen:
                if isinstance(failed_gen, dict):
                    import json
                    failed_gen = json.dumps(failed_gen)
                latency_ms = (time.perf_counter() - start) * 1000
                logger.info(f"Recovered tool call from Groq tool_use_failed: {failed_gen}")
                telemetry.record_execution(
                    provider="groq",
                    model=self.model,
                    operation="generate_recovered_tool_call",
                    prompt_tokens=100,
                    completion_tokens=50,
                    latency_ms=latency_ms,
                    success=True,
                )
                return LLMResponse(
                    content=failed_gen,
                    provider="groq",
                    model=self.model,
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                    latency_ms=round(latency_ms, 2),
                )

            latency_ms = (time.perf_counter() - start) * 1000
            telemetry.record_execution(
                provider="groq",
                model=self.model,
                operation="generate",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=latency_ms,
                success=False,
                error=str(exc),
            )
            logger.warning(f"Groq API error on model {self.model}: {exc}")
            raise LLMProviderError(f"Groq provider error: {str(exc)}")
