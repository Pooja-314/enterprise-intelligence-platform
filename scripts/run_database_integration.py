"""Database Integration Phase Script.
Executes Step 1 (pgvector), Step 2 (Alembic upgrade), Step 3 (Schema verification), and Step 4 (Integration tests).

SECURITY DIRECTIVE:
Never prints connection credentials, passwords, or API keys.
Does not touch or mutate application data.
"""
import os
import re
import sys
import time
import urllib.parse
from typing import Any, Dict

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv(override=True)

from sqlalchemy import create_engine, inspect, text
from app.db.session import get_migration_database_url, get_normalized_database_url


def scrub_error(msg: str) -> str:
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", str(msg))


def main():
    print("=" * 70)
    print("ENTERPRISE INTELLIGENCE PLATFORM - DATABASE INTEGRATION PHASE")
    print("=" * 70)

    direct_url = get_migration_database_url()
    parsed = urllib.parse.urlparse(direct_url)
    target_host = parsed.hostname or "unknown"
    print(f"Target Neon Host: {target_host} (Port: {parsed.port or 5432})\n")

    engine = create_engine(direct_url, connect_args={"connect_timeout": 8})

    results = {}

    # --------------------------------------------------------------------------
    # STEP 1: pgvector Check & Enablement
    # --------------------------------------------------------------------------
    print("--- [STEP 1] pgvector EXTENSION VERIFICATION & ENABLEMENT ---")
    try:
        with engine.connect() as conn:
            # Check existing extension
            installed = conn.execute(
                text("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
            ).fetchone()

            if not installed:
                print("  Vector extension is not yet enabled. Checking catalog availability...")
                avail = conn.execute(
                    text("SELECT name, default_version FROM pg_available_extensions WHERE name = 'vector';")
                ).fetchone()
                if avail:
                    print(f"  Extension is available in catalog (Default version: {avail[1]}). Enabling...")
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                    conn.commit()
                    print("  Executed: CREATE EXTENSION IF NOT EXISTS vector;")
                else:
                    raise RuntimeError("pgvector is not listed in pg_available_extensions on this database server.")

            # Verify installed version
            installed = conn.execute(
                text("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
            ).fetchone()

            if installed:
                print(f"  Status: PASS (pgvector extension active, version: {installed[1]})")
                results["step_1_pgvector"] = {"status": "PASS", "version": installed[1]}
            else:
                print("  Status: FAIL (pgvector extension was not found after enablement)")
                results["step_1_pgvector"] = {"status": "FAIL"}

    except Exception as exc:
        err = scrub_error(str(exc))
        print(f"  Status: FAIL ({err})")
        results["step_1_pgvector"] = {"status": "FAIL", "error": err}
        return results

    # --------------------------------------------------------------------------
    # STEP 2: Alembic Migrations
    # --------------------------------------------------------------------------
    print("\n--- [STEP 2] ALEMBIC SCHEMA UPGRADE (upgrade head) ---")
    try:
        from alembic.config import Config
        from alembic import command
        from alembic.script import ScriptDirectory

        alembic_cfg = Config("alembic.ini")
        script = ScriptDirectory.from_config(alembic_cfg)
        head_rev = script.get_current_head()

        print(f"  Target migration revision: {head_rev}")

        # 1. Inspect existing tables and current Alembic revision
        insp = inspect(engine)
        pre_existing_tables = set(insp.get_table_names())
        print(f"  Pre-migration tables in database: {sorted(list(pre_existing_tables)) if pre_existing_tables else 'None (empty database)'}")

        current_rev = None
        if "alembic_version" in pre_existing_tables:
            with engine.connect() as conn:
                current_rev = conn.execute(text("SELECT version_num FROM alembic_version;")).scalar()
            print(f"  Current recorded Alembic revision: '{current_rev}'")
        else:
            print("  No 'alembic_version' table found (clean/unmanaged database).")

        # 2. Check for conflicting pre-existing application tables
        app_tables = {"users", "documents", "document_chunks", "conversations", "messages", "audit_logs"}
        conflicting = pre_existing_tables.intersection(app_tables)

        if conflicting and current_rev != head_rev:
            print(f"  SAFETY HALT: Detected pre-existing application tables {sorted(list(conflicting))} without matching migration revision '{head_rev}'.")
            print("  Halting migration to prevent overwriting or conflicting with pre-existing schema.")
            results["step_2_alembic"] = {
                "status": "HALTED",
                "reason": "Conflicting pre-existing application tables without matching migration history.",
                "conflicting_tables": list(conflicting)
            }
            return results

        if current_rev == head_rev:
            print(f"  Database schema is already at head revision '{head_rev}'.")
            applied_rev = current_rev
        else:
            print("  Pre-migration checks passed: No conflicting tables. Proceeding with upgrade head...")
            start_mig = time.perf_counter()
            command.upgrade(alembic_cfg, "head")
            dur_mig = (time.perf_counter() - start_mig) * 1000

            # Verify current revision in alembic_version table
            with engine.connect() as conn:
                applied_rev = conn.execute(text("SELECT version_num FROM alembic_version;")).scalar()
            print(f"  Alembic upgrade completed in {dur_mig:.1f}ms.")

        if applied_rev == head_rev:
            print(f"  Status: PASS (Alembic verified at revision '{applied_rev}')")
            results["step_2_alembic"] = {"status": "PASS", "revision": applied_rev}
        else:
            print(f"  Status: FAIL (Expected revision '{head_rev}', found '{applied_rev}')")
            results["step_2_alembic"] = {"status": "FAIL", "revision": applied_rev}

    except Exception as exc:
        err = scrub_error(str(exc))
        print(f"  Status: FAIL ({err})")
        results["step_2_alembic"] = {"status": "FAIL", "error": err}
        return results

    # --------------------------------------------------------------------------
    # STEP 3: Schema Verification
    # --------------------------------------------------------------------------
    print("\n--- [STEP 3] SCHEMA VERIFICATION ---")
    try:
        insp = inspect(engine)
        existing_tables = set(insp.get_table_names())
        required_tables = ["users", "documents", "document_chunks", "conversations", "messages", "audit_logs"]

        missing_tables = [t for t in required_tables if t not in existing_tables]
        if missing_tables:
            print(f"  Tables Check: FAIL (Missing: {missing_tables})")
            results["step_3_tables"] = {"status": "FAIL", "missing": missing_tables}
        else:
            print(f"  Tables Check: PASS (All {len(required_tables)} expected tables present: {', '.join(required_tables)})")
            results["step_3_tables"] = {"status": "PASS", "tables": required_tables}

        # Verify document_chunks.embedding column is vector(384)
        with engine.connect() as conn:
            # 1. Check UDT type
            col_info = conn.execute(text("""
                SELECT data_type, udt_name 
                FROM information_schema.columns 
                WHERE table_name = 'document_chunks' AND column_name = 'embedding';
            """)).fetchone()

            # 2. Check dimension from pg_attribute (atttypmod for vector represents dimension)
            dim_info = conn.execute(text("""
                SELECT atttypmod 
                FROM pg_attribute 
                WHERE attrelid = 'document_chunks'::regclass AND attname = 'embedding';
            """)).scalar()

            # 3. Check GIN full-text index
            gin_idx = conn.execute(text("""
                SELECT indexname, indexdef 
                FROM pg_indexes 
                WHERE tablename = 'document_chunks' AND indexname = 'ix_document_chunks_content_fts';
            """)).fetchone()

        if col_info and col_info[1] == "vector" and dim_info == 384:
            print(f"  Vector Column: PASS (document_chunks.embedding is native PostgreSQL vector(384))")
            results["step_3_vector_column"] = {"status": "PASS", "type": "vector(384)"}
        else:
            print(f"  Vector Column: FAIL (Found udt_name={col_info[1] if col_info else 'none'}, dimension={dim_info})")
            results["step_3_vector_column"] = {"status": "FAIL"}

        if gin_idx and "gin" in gin_idx[1].lower():
            print(f"  Full-Text GIN Index: PASS (Index 'ix_document_chunks_content_fts' verified)")
            results["step_3_gin_index"] = {"status": "PASS", "indexname": gin_idx[0]}
        else:
            print("  Full-Text GIN Index: FAIL (GIN index not found on document_chunks)")
            results["step_3_gin_index"] = {"status": "FAIL"}

    except Exception as exc:
        err = scrub_error(str(exc))
        print(f"  Schema Verification: FAIL ({err})")
        results["step_3_schema"] = {"status": "FAIL", "error": err}
        return results

    # --------------------------------------------------------------------------
    # STEP 4: Live PostgreSQL + pgvector Integration Tests
    # --------------------------------------------------------------------------
    print("\n--- [STEP 4] LIVE POSTGRESQL + PGVECTOR INTEGRATION TEST ---")
    try:
        import pytest

        # Verify environment before running
        print(f"  Verifying target environment: Engine dialect is '{engine.dialect.name}', host is '{target_host}'.")
        if "sqlite" in direct_url.lower() or ":memory:" in direct_url.lower() or target_host == "localhost":
            raise RuntimeError(f"Refusing test execution: Expected live Neon database, but found '{target_host}'.")

        test_exit_code = pytest.main([
            "-v",
            "-s",
            "tests/integration/test_postgres_pgvector.py"
        ])

        if test_exit_code == 0:
            print("  Integration Test: PASS (Real Neon PostgreSQL + native pgvector cosine search validated)")
            results["step_4_integration_test"] = {"status": "PASS"}
        else:
            print(f"  Integration Test: FAIL (pytest exited with code {test_exit_code})")
            results["step_4_integration_test"] = {"status": "FAIL", "exit_code": int(test_exit_code)}

    except Exception as exc:
        err = scrub_error(str(exc))
        print(f"  Integration Test: FAIL ({err})")
        results["step_4_integration_test"] = {"status": "FAIL", "error": err}

    print("\n" + "=" * 70)
    print("DATABASE INTEGRATION PHASE COMPLETE")
    print("=" * 70)
    return results


if __name__ == "__main__":
    main()
