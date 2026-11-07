import logging
import time
from typing import Optional
from app.config import settings
from app.core.errors import LLMProviderError
from app.core.telemetry import telemetry
from app.llm.base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Secondary / Fallback LLM provider using Google Gemini API."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key if api_key is not None else settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_MODEL
        self._client = None

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip() and not self.api_key.startswith("your_"))

    def _get_client(self):
        if self._client is None:
            if not self.is_configured():
                raise LLMProviderError("Gemini API key is not configured.")
            try:
                # Use Google GenAI official SDK
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                # Fallback to google.generativeai if genai fails
                try:
                    import google.generativeai as gai
                    gai.configure(api_key=self.api_key)
                    self._client = gai.GenerativeModel(self.model)
                except Exception as ex:
                    raise LLMProviderError(f"Failed to initialize Gemini client: {str(ex)}")
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        client = self._get_client()
        start = time.perf_counter()

        try:
            full_content = ""
            prompt_tokens = 0
            completion_tokens = 0

            # 1. New Google GenAI SDK (Client)
            if hasattr(client, "models"):
                config = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
                if system_prompt:
                    config["system_instruction"] = system_prompt

                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                full_content = response.text or ""
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                    completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            # 2. Legacy google.generativeai GenerativeModel
            else:
                formatted_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                response = client.generate_content(
                    formatted_prompt,
                    generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
                )
                full_content = response.text or ""

            latency_ms = (time.perf_counter() - start) * 1000
            total_tokens = prompt_tokens + completion_tokens

            # Record telemetry
            telemetry.record_execution(
                provider="gemini",
                model=self.model,
                operation="generate",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                success=True,
            )

            return LLMResponse(
                content=full_content,
                provider="gemini",
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=round(latency_ms, 2),
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            telemetry.record_execution(
                provider="gemini",
                model=self.model,
                operation="generate",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=latency_ms,
                success=False,
                error=str(exc),
            )
            logger.warning(f"Gemini API error on model {self.model}: {exc}")
            raise LLMProviderError(f"Gemini provider error: {str(exc)}")
