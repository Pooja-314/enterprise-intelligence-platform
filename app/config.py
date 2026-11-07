import json
from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    APP_NAME: str = "Enterprise Intelligence Platform"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "change-this-to-a-super-secret-random-key-in-production-min-32-chars"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    # Database
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/enterprise_intel"
    DIRECT_URL: str = ""
    TEST_DATABASE_URL: str = "sqlite:///./test.db"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_ECHO: bool = False

    # LLM Providers
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-120b"

    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.8-flash"

    PRIMARY_LLM_PROVIDER: str = "groq"
    FALLBACK_LLM_PROVIDER: str = "gemini"
    LLM_TIMEOUT_SECONDS: int = 30
    LLM_MAX_RETRIES: int = 2

    # Embeddings
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 32

    # Ingestion & Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MAX_UPLOAD_SIZE_MB: int = 25

    # RAG & Retrieval
    RRF_K: int = 60
    TOP_K_VECTOR: int = 15
    TOP_K_KEYWORD: int = 15
    TOP_K_FINAL: int = 8
    MIN_RELEVANCE_SCORE: float = 0.35

    # Security
    PROMPT_INJECTION_PROTECTION_ENABLED: bool = True
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "change_this_admin_password_123!"
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:8501",
        "http://localhost:3000",
        "http://127.0.0.1:8501",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, list):
            return v
        return ["*"]

    @property
    def has_groq(self) -> bool:
        return bool(self.GROQ_API_KEY and self.GROQ_API_KEY.strip() and not self.GROQ_API_KEY.startswith("gsk_your_"))

    @property
    def has_gemini(self) -> bool:
        return bool(self.GEMINI_API_KEY and self.GEMINI_API_KEY.strip() and not self.GEMINI_API_KEY.startswith("your_"))

    @property
    def is_postgres(self) -> bool:
        return self.DATABASE_URL.startswith("postgresql")

    def safe_dict(self) -> dict:
        """Returns configuration dictionary with all secrets masked for logging/telemetry."""
        data = self.model_dump()
        masked_keys = {"SECRET_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "ADMIN_PASSWORD"}
        for k in masked_keys:
            if k in data and data[k]:
                data[k] = "***MASKED***"
        # Mask database passwords if present
        for url_key in ("DATABASE_URL", "DIRECT_URL"):
            if url_key in data and data[url_key] and "@" in data[url_key]:
                prefix, rest = data[url_key].split("@", 1)
                if ":" in prefix:
                    proto_user = prefix.split(":", 2)
                    data[url_key] = f"{proto_user[0]}:{proto_user[1]}:***@{rest}"
        return data


# Global singleton instance
settings = Settings()
