import pytest
from app.core.errors import LLMFallbackExhaustedError, LLMProviderError
from app.core.telemetry import telemetry
from app.llm.base import BaseLLMProvider, LLMResponse
from app.llm.fallback_router import LLMRouter


class DummyProvider(BaseLLMProvider):
    def __init__(self, name: str, should_fail: bool = False, fail_message: str = "Provider error"):
        self.name = name
        self.model = f"{name}-model"
        self.should_fail = should_fail
        self.fail_message = fail_message
        self.call_count = 0

    def is_configured(self) -> bool:
        return True

    def generate(self, prompt: str, system_prompt=None, temperature=0.2, max_tokens=1024) -> LLMResponse:
        self.call_count += 1
        if self.should_fail:
            raise LLMProviderError(self.fail_message)
        return LLMResponse(
            content=f"Response from {self.name}",
            provider=self.name,
            model=self.model,
            prompt_tokens=25,
            completion_tokens=15,
            total_tokens=40,
            latency_ms=120.0,
        )


class UnconfiguredProvider(BaseLLMProvider):
    def is_configured(self) -> bool:
        return False

    def generate(self, prompt: str, system_prompt=None, temperature=0.2, max_tokens=1024) -> LLMResponse:
        raise LLMProviderError("Unconfigured")


def test_router_primary_success():
    primary = DummyProvider("primary_groq")
    fallback = DummyProvider("fallback_gemini")
    router = LLMRouter(primary_provider=primary, fallback_provider=fallback, max_retries=1)

    res = router.generate("What is RAG?")
    assert res.provider == "primary_groq"
    assert res.content == "Response from primary_groq"
    assert primary.call_count == 1
    assert fallback.call_count == 0


def test_router_fallback_when_primary_fails():
    primary = DummyProvider("primary_groq", should_fail=True, fail_message="Rate limit 429")
    fallback = DummyProvider("fallback_gemini")
    router = LLMRouter(primary_provider=primary, fallback_provider=fallback, max_retries=1)

    res = router.generate("Explain pgvector.")
    assert res.provider == "fallback_gemini"
    assert res.content == "Response from fallback_gemini"
    assert res.metadata.get("fallback_activated") is True
    assert "Rate limit 429" in res.metadata.get("primary_error")
    assert primary.call_count == 2  # 1 initial + 1 retry
    assert fallback.call_count == 1


def test_router_fallback_when_primary_unconfigured():
    primary = UnconfiguredProvider()
    fallback = DummyProvider("fallback_gemini")
    router = LLMRouter(primary_provider=primary, fallback_provider=fallback, max_retries=1)

    res = router.generate("Explain vector indexing.")
    assert res.provider == "fallback_gemini"
    assert fallback.call_count == 1


def test_router_exhausted_when_both_fail():
    primary = DummyProvider("primary_groq", should_fail=True, fail_message="Connection timed out")
    fallback = DummyProvider("fallback_gemini", should_fail=True, fail_message="Service unavailable 503")
    router = LLMRouter(primary_provider=primary, fallback_provider=fallback, max_retries=1)

    with pytest.raises(LLMFallbackExhaustedError) as exc_info:
        router.generate("Summarize report.")
    assert "All configured LLM providers failed" in str(exc_info.value)


def test_groq_provider_model_configuration():
    from app.config import Settings, settings
    from app.llm.groq_provider import GroqProvider

    # Default model in Settings class definition matches updated configuration
    s = Settings(_env_file=None)
    assert s.GROQ_MODEL == "openai/gpt-oss-120b"
    assert settings.GROQ_MODEL == "openai/gpt-oss-120b"

    provider = GroqProvider()
    assert provider.model == "openai/gpt-oss-120b"

    # Custom model override is supported
    custom_provider = GroqProvider(model="custom-groq-model")
    assert custom_provider.model == "custom-groq-model"


def test_gemini_provider_model_configuration():
    from app.config import Settings, settings
    from app.llm.gemini_provider import GeminiProvider

    # Default model in Settings class definition matches updated configuration
    s = Settings(_env_file=None)
    assert s.GEMINI_MODEL == "gemini-3.8-flash"
    assert settings.GEMINI_MODEL == "gemini-3.8-flash"

    provider = GeminiProvider()
    assert provider.model == "gemini-3.8-flash"

    # Custom model override is supported
    custom_provider = GeminiProvider(model="custom-gemini-model")
    assert custom_provider.model == "custom-gemini-model"
