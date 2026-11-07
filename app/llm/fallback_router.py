import logging
import time
from typing import Dict, List, Optional
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings
from app.core.errors import LLMFallbackExhaustedError, LLMProviderError
from app.llm.base import BaseLLMProvider, LLMResponse
from app.llm.gemini_provider import GeminiProvider
from app.llm.groq_provider import GroqProvider

logger = logging.getLogger(__name__)


class LLMRouter:
    """Intelligent Multi-Provider LLM Router with exponential backoff and automatic failover."""

    def __init__(
        self,
        primary_provider: Optional[BaseLLMProvider] = None,
        fallback_provider: Optional[BaseLLMProvider] = None,
        max_retries: Optional[int] = None,
    ):
        self.primary = primary_provider or GroqProvider()
        self.fallback = fallback_provider or GeminiProvider()
        self.max_retries = max_retries or settings.LLM_MAX_RETRIES

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Attempts primary provider with retries, then automatically fails over to secondary provider."""
        primary_error: Optional[str] = None

        # 1. Attempt Primary Provider if configured
        if self.primary.is_configured():
            try:
                logger.info(f"Attempting Primary LLM Provider: {getattr(self.primary, 'model', 'default')}")
                return self._invoke_with_retry(
                    provider=self.primary,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                primary_error = str(e)
                logger.warning(
                    f"Primary LLM provider failed ({primary_error}). Initiating automatic fallback to secondary provider."
                )
        else:
            primary_error = "Primary provider not configured (missing or invalid API key)."
            logger.info("Primary LLM provider is not configured. Routing directly to fallback provider.")

        # 2. Attempt Fallback Provider if configured
        if self.fallback.is_configured():
            try:
                logger.info(f"Attempting Fallback LLM Provider: {getattr(self.fallback, 'model', 'default')}")
                response = self._invoke_with_retry(
                    provider=self.fallback,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                response.metadata["fallback_activated"] = True
                response.metadata["primary_error"] = primary_error
                return response
            except Exception as fallback_err:
                logger.error(f"Fallback LLM provider also failed: {fallback_err}")
                raise LLMFallbackExhaustedError(
                    message="All configured LLM providers failed to generate an answer.",
                    details={"primary_error": primary_error, "fallback_error": str(fallback_err)},
                )

        # 3. Both unconfigured
        raise LLMFallbackExhaustedError(
            message="No functional LLM provider available. Please configure GROQ_API_KEY or GEMINI_API_KEY in .env.",
            details={"primary_error": primary_error, "fallback_error": "Fallback provider not configured."},
        )

    def _invoke_with_retry(
        self,
        provider: BaseLLMProvider,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Invokes provider with exponential backoff on retryable transient errors."""
        attempt = 0
        last_exc = None

        while attempt <= self.max_retries:
            try:
                return provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                last_exc = exc
                attempt += 1
                if attempt > self.max_retries:
                    break
                sleep_time = min(4.0, 0.5 * (2 ** (attempt - 1)))
                logger.warning(f"Transient error on provider attempt {attempt}: {exc}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)

        raise last_exc or LLMProviderError("Retry attempts exhausted.")


# Global router singleton
llm_router = LLMRouter()
