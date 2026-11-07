"""Secure Integration Pre-flight Check.
Verifies live connectivity to Neon (pooled and direct), pgvector availability,
runs Alembic migrations, verifies schema, runs integration tests, and verifies Groq & Gemini APIs.

SECURITY DIRECTIVE:
Never prints secrets, passwords, connection credentials, or full API keys.
"""
import os
import re
import sys
import time
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import create_engine, inspect, text
from app.config import settings
from app.db.session import get_migration_database_url, get_normalized_database_url


def mask_url(url: Optional[str]) -> str:
    """Masks credentials in database URL for safe diagnostic printing."""
    if not url:
        return "[NOT CONFIGURED]"
    # Mask password
    masked = re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", url)
    return masked


def scrub_error(msg: str) -> str:
    """Removes sensitive credentials from database or API error messages."""
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", str(msg))


def run_preflight():
    results = {}
    print("=" * 70)
    print("ENTERPRISE INTELLIGENCE PLATFORM - PRE-FLIGHT INTEGRATION AUDIT")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # 1. DATABASE_URL Connectivity (Neon Pooled Connection)
    # --------------------------------------------------------------------------
    raw_db_url = settings.DATABASE_URL
    norm_db_url = get_normalized_database_url(raw_db_url)
    print(f"\n[1] Testing Neon Pooled Connection (DATABASE_URL)...")
    print(f"    Target endpoint: {mask_url(norm_db_url)}")

    if not norm_db_url.startswith("postgresql"):
        results["1_pooled_connection"] = {
            "status": "FAIL",
            "error": f"Dialect is '{norm_db_url.split(':')[0]}', expected PostgreSQL."
        }
    else:
        try:
            start = time.perf_counter()
            engine_pooled = create_engine(norm_db_url, connect_args={"connect_timeout": 8})
            with engine_pooled.connect() as conn:
                version = conn.execute(text("SELECT version();")).scalar()
            latency = (time.perf_counter() - start) * 1000
            print(f"    Status: PASS (Latency: {latency:.2f}ms)")
            print(f"    Engine: {version.split(',')[0] if version else 'PostgreSQL'}")
            results["1_pooled_connection"] = {
                "status": "PASS",
                "latency_ms": round(latency, 2),
                "server": version.split(",")[0] if version else "PostgreSQL"
            }
        except Exception as exc:
            err = scrub_error(str(exc))
            print(f"    Status: FAIL")
            print(f"    Error: {err}")
            results["1_pooled_connection"] = {"status": "FAIL", "error": err}

    # --------------------------------------------------------------------------
    # 2. DIRECT_URL Connectivity (Neon Direct Connection)
    # --------------------------------------------------------------------------
    raw_direct_url = os.getenv("DIRECT_URL") or getattr(settings, "DIRECT_URL", "")
    norm_direct_url = get_migration_database_url()
    print(f"\n[2] Testing Neon Direct Connection (DIRECT_URL)...")
    print(f"    Target endpoint: {mask_url(norm_direct_url)}")

    if not raw_direct_url:
        print("    Status: WARNING - DIRECT_URL is absent. Falling back to DATABASE_URL.")
    
    direct_engine = None
    try:
        start = time.perf_counter()
        direct_engine = create_engine(norm_direct_url, connect_args={"connect_timeout": 8})
        with direct_engine.connect() as conn:
            version = conn.execute(text("SELECT version();")).scalar()
        latency = (time.perf_counter() - start) * 1000
        print(f"    Status: PASS (Latency: {latency:.2f}ms)")
        results["2_direct_connection"] = {
            "status": "PASS",
            "latency_ms": round(latency, 2),
            "is_direct_url_configured": bool(raw_direct_url)
        }
    except Exception as exc:
        err = scrub_error(str(exc))
        print(f"    Status: FAIL")
        print(f"    Error: {err}")
        results["2_direct_connection"] = {"status": "FAIL", "error": err}

    # --------------------------------------------------------------------------
    # 3. Check pgvector extension availability & installation
    # --------------------------------------------------------------------------
    print(f"\n[3] Checking pgvector extension status in PostgreSQL catalog...")
    pgvector_ready = False
    if direct_engine and results.get("2_direct_connection", {}).get("status") == "PASS":
        try:
            with direct_engine.connect() as conn:
                # Check if already installed
                installed = conn.execute(
                    text("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
                ).fetchone()

                # Check available in catalog
                available = conn.execute(
                    text("SELECT name, default_version, installed_version, comment FROM pg_available_extensions WHERE name = 'vector';")
                ).fetchone()

            if installed:
                print(f"    Status: PASS (pgvector extension is INSTALLED, version: {installed[1]})")
                results["3_pgvector_status"] = {
                    "status": "PASS",
                    "installed": True,
                    "version": installed[1]
                }
                pgvector_ready = True
            elif available:
                print(f"    Status: AVAILABLE (pgvector is available in catalog v{available[1]}, not yet enabled in current DB)")
                results["3_pgvector_status"] = {
                    "status": "AVAILABLE",
                    "installed": False,
                    "available_version": available[1]
                }
            else:
                print(f"    Status: FAIL (pgvector extension is NOT available on this PostgreSQL instance)")
                results["3_pgvector_status"] = {
                    "status": "FAIL",
                    "installed": False,
                    "error": "pgvector is not listed in pg_available_extensions."
                }
        except Exception as exc:
            err = scrub_error(str(exc))
            print(f"    Status: FAIL ({err})")
            results["3_pgvector_status"] = {"status": "FAIL", "error": err}
    else:
        print("    Status: NOT VERIFIED (Direct connection unavailable)")
        results["3_pgvector_status"] = {"status": "NOT VERIFIED", "reason": "No direct DB connection"}

    # --------------------------------------------------------------------------
    # 4. Enable pgvector if available but not installed
    # --------------------------------------------------------------------------
    if results.get("3_pgvector_status", {}).get("status") == "AVAILABLE":
        print(f"\n[4] Enabling pgvector extension (CREATE EXTENSION IF NOT EXISTS vector;)...")
        try:
            with direct_engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                conn.commit()
            print("    Status: PASS (pgvector extension successfully enabled)")
            pgvector_ready = True
            results["4_enable_pgvector"] = {"status": "PASS"}
        except Exception as exc:
            err = scrub_error(str(exc))
            print(f"    Status: FAIL ({err})")
            results["4_enable_pgvector"] = {"status": "FAIL", "error": err}
    else:
        results["4_enable_pgvector"] = {"status": "SKIPPED", "detail": "Already installed or unavailable"}

    # --------------------------------------------------------------------------
    # 5. Run Alembic migrations using DIRECT_URL
    # --------------------------------------------------------------------------
    print(f"\n[5] Executing Alembic migrations using DIRECT_URL...")
    if direct_engine and results.get("2_direct_connection", {}).get("status") == "PASS":
        try:
            from alembic import command
            from alembic.config import Config
            alembic_cfg = Config("alembic.ini")
            command.upgrade(alembic_cfg, "head")
            print("    Status: PASS (Alembic schema migrated to 'head')")
            results["5_alembic_migrations"] = {"status": "PASS"}
        except Exception as exc:
            err = scrub_error(str(exc))
            print(f"    Status: FAIL ({err})")
            results["5_alembic_migrations"] = {"status": "FAIL", "error": err}
    else:
        print("    Status: NOT VERIFIED (Direct connection unavailable)")
        results["5_alembic_migrations"] = {"status": "NOT VERIFIED"}

    # --------------------------------------------------------------------------
    # 6. Verify tables and vector columns exist
    # --------------------------------------------------------------------------
    print(f"\n[6] Verifying database schema tables and vector columns...")
    expected_tables = {"users", "documents", "document_chunks", "conversations", "messages", "audit_logs"}
    if direct_engine and results.get("5_alembic_migrations", {}).get("status") == "PASS":
        try:
            insp = inspect(direct_engine)
            existing_tables = set(insp.get_table_names())
            missing = expected_tables - existing_tables

            if missing:
                print(f"    Status: FAIL (Missing tables: {missing})")
                results["6_schema_verification"] = {"status": "FAIL", "missing_tables": list(missing)}
            else:
                # Check vector column on document_chunks
                cols = {c["name"]: str(c["type"]) for c in insp.get_columns("document_chunks")}
                emb_type = cols.get("embedding", "MISSING")
                print(f"    Tables verified: {len(existing_tables)} tables present.")
                print(f"    document_chunks.embedding column type: {emb_type}")
                if "vector" in emb_type.lower() or "user-defined" in emb_type.lower():
                    print("    Status: PASS (All schema tables and vector column verified)")
                    results["6_schema_verification"] = {
                        "status": "PASS",
                        "tables_found": sorted(list(existing_tables)),
                        "embedding_column_type": emb_type
                    }
                else:
                    print(f"    Status: FAIL (embedding column type is {emb_type}, expected vector(384))")
                    results["6_schema_verification"] = {"status": "FAIL", "embedding_column_type": emb_type}
        except Exception as exc:
            err = scrub_error(str(exc))
            print(f"    Status: FAIL ({err})")
            results["6_schema_verification"] = {"status": "FAIL", "error": err}
    else:
        print("    Status: NOT VERIFIED (Migrations not executed)")
        results["6_schema_verification"] = {"status": "NOT VERIFIED"}

    # --------------------------------------------------------------------------
    # 7. Run PostgreSQL + pgvector Integration Test
    # --------------------------------------------------------------------------
    print(f"\n[7] Running tests/integration/test_postgres_pgvector.py...")
    if direct_engine and results.get("5_alembic_migrations", {}).get("status") == "PASS":
        try:
            import pytest
            test_exit_code = pytest.main([
                "-v",
                "-s",
                "tests/integration/test_postgres_pgvector.py",
                "-k", "test_real_postgres_connection_and_pgvector_extension"
            ])
            if test_exit_code == 0:
                print("    Status: PASS (Integration test passed against live database)")
                results["7_pgvector_integration_test"] = {"status": "PASS"}
            else:
                print(f"    Status: FAIL (Pytest exited with code {test_exit_code})")
                results["7_pgvector_integration_test"] = {"status": "FAIL", "exit_code": int(test_exit_code)}
        except Exception as exc:
            err = scrub_error(str(exc))
            print(f"    Status: FAIL ({err})")
            results["7_pgvector_integration_test"] = {"status": "FAIL", "error": err}
    else:
        print("    Status: NOT VERIFIED (Schema/migrations not verified)")
        results["7_pgvector_integration_test"] = {"status": "NOT VERIFIED"}

    # --------------------------------------------------------------------------
    # 8. Test Groq API connectivity
    # --------------------------------------------------------------------------
    print(f"\n[8] Testing Groq API connectivity (Model: {settings.GROQ_MODEL})...")
    from app.llm.groq_provider import GroqProvider
    groq = GroqProvider()
    if not groq.is_configured():
        print("    Status: FAIL (GROQ_API_KEY is not configured or has default placeholder)")
        results["8_groq_api"] = {"status": "FAIL", "error": "GROQ_API_KEY missing or placeholder"}
    else:
        try:
            start = time.perf_counter()
            resp = groq.generate(prompt="Reply with the exact word: ONLINE", max_tokens=10)
            latency = (time.perf_counter() - start) * 1000
            print(f"    Status: PASS (Model: {resp.model}, Latency: {latency:.2f}ms, Output: '{resp.content.strip()[:40]}')")
            results["8_groq_api"] = {
                "status": "PASS",
                "model": resp.model,
                "latency_ms": round(latency, 2),
                "tokens": resp.total_tokens
            }
        except Exception as exc:
            err = scrub_error(str(exc))
            print(f"    Status: FAIL ({err})")
            results["8_groq_api"] = {"status": "FAIL", "error": err}

    # --------------------------------------------------------------------------
    # 9. Test Gemini API connectivity
    # --------------------------------------------------------------------------
    print(f"\n[9] Testing Gemini API connectivity (Model: {settings.GEMINI_MODEL})...")
    from app.llm.gemini_provider import GeminiProvider
    gemini = GeminiProvider()
    if not gemini.is_configured():
        print("    Status: FAIL (GEMINI_API_KEY is not configured or has default placeholder)")
        results["9_gemini_api"] = {"status": "FAIL", "error": "GEMINI_API_KEY missing or placeholder"}
    else:
        try:
            start = time.perf_counter()
            resp = gemini.generate(prompt="Reply with the exact word: ONLINE", max_tokens=10)
            latency = (time.perf_counter() - start) * 1000
            print(f"    Status: PASS (Model: {resp.model}, Latency: {latency:.2f}ms, Output: '{resp.content.strip()[:40]}')")
            results["9_gemini_api"] = {
                "status": "PASS",
                "model": resp.model,
                "latency_ms": round(latency, 2),
                "tokens": resp.total_tokens
            }
        except Exception as exc:
            err = scrub_error(str(exc))
            print(f"    Status: FAIL ({err})")
            results["9_gemini_api"] = {"status": "FAIL", "error": err}

    # --------------------------------------------------------------------------
    # 10. Verify LLM Fallback Router Identification
    # --------------------------------------------------------------------------
    print(f"\n[10] Verifying LLM Fallback Router configuration...")
    from app.llm.fallback_router import llm_router
    primary_ok = llm_router.primary.is_configured()
    fallback_ok = llm_router.fallback.is_configured()
    primary_name = getattr(llm_router.primary, "provider_name", "groq")
    fallback_name = getattr(llm_router.fallback, "provider_name", "gemini")
    print(f"    Primary provider ({primary_name}): {'CONFIGURED' if primary_ok else 'UNCONFIGURED'}")
    print(f"    Fallback provider ({fallback_name}): {'CONFIGURED' if fallback_ok else 'UNCONFIGURED'}")

    if primary_ok and fallback_ok:
        print("    Status: PASS (Both providers recognized and active)")
        results["10_router_status"] = {"status": "PASS", "primary": True, "fallback": True}
    elif primary_ok or fallback_ok:
        print("    Status: PARTIAL (Only one provider is active)")
        results["10_router_status"] = {"status": "PARTIAL", "primary": primary_ok, "fallback": fallback_ok}
    else:
        print("    Status: FAIL (Neither provider is configured)")
        results["10_router_status"] = {"status": "FAIL", "primary": False, "fallback": False}

    print("\n" + "=" * 70)
    print("PRE-FLIGHT AUDIT COMPLETE")
    print("=" * 70)
    return results


if __name__ == "__main__":
    run_preflight()
