import os
import time
from typing import Generator, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.config import settings

def get_normalized_database_url(url: str) -> str:
    """Normalizes database URL for SQLAlchemy compatibility."""
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg2://", 1)
    if url.startswith("postgresql://") and not url.startswith("postgresql+"):
        return url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url


def get_migration_database_url() -> str:
    """Resolves target database URL for Alembic migrations.
    
    Prefers DIRECT_URL when provided (e.g. direct PostgreSQL port 5432 for schema updates
    and extension initialization), otherwise falls back to DATABASE_URL for backward compatibility.
    Normalizes the connection scheme to postgresql+psycopg2 for SQLAlchemy.
    """
    raw_url = (
        os.getenv("DIRECT_URL")
        or getattr(settings, "DIRECT_URL", "")
        or os.getenv("DATABASE_URL", settings.DATABASE_URL)
    )
    return get_normalized_database_url(raw_url)

DATABASE_URL = get_normalized_database_url(settings.DATABASE_URL)

engine_kwargs: Dict[str, Any] = {
    "echo": settings.DB_ECHO,
}

if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs["pool_size"] = settings.DB_POOL_SIZE
    engine_kwargs["max_overflow"] = settings.DB_MAX_OVERFLOW
    engine_kwargs["pool_pre_ping"] = True

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency providing a database session with guaranteed closure."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connection() -> Dict[str, Any]:
    """Verifies database connectivity and returns latency and dialect info."""
    start = time.perf_counter()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "connected": True,
            "dialect": engine.dialect.name,
            "latency_ms": round(latency_ms, 2),
            "error": None,
        }
    except Exception as exc:
        return {
            "connected": False,
            "dialect": engine.dialect.name,
            "latency_ms": None,
            "error": str(exc),
        }
