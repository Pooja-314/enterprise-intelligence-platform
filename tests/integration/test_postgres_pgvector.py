import pytest
from sqlalchemy import create_engine, text
from app.config import settings
from app.db.session import get_migration_database_url, get_normalized_database_url


def is_live_postgres_available() -> tuple[bool, str]:
    """Checks if a real live PostgreSQL database is accessible via DIRECT_URL or DATABASE_URL."""
    db_url = get_migration_database_url()
    if not db_url.startswith("postgresql"):
        return False, f"Database dialect is '{db_url.split(':')[0]}', not 'postgresql'."

    try:
        eng = create_engine(db_url, connect_args={"connect_timeout": 5})
        with eng.connect() as conn:
            conn.execute(text("SELECT 1;"))
        return True, "Connected successfully."
    except Exception as exc:
        return False, f"PostgreSQL connection failed: {str(exc)}"


# Check live postgres status at module level
postgres_ok, postgres_reason = is_live_postgres_available()


@pytest.mark.postgres_integration
@pytest.mark.skipif(not postgres_ok, reason=f"Requires real PostgreSQL + pgvector database. Status: {postgres_reason}")
def test_real_postgres_connection_and_pgvector_extension():
    """Integration Test: Requires real PostgreSQL instance.
    
    Verifies:
    1. Direct connection to hosted or local PostgreSQL instance.
    2. pgvector extension initialization (CREATE EXTENSION IF NOT EXISTS vector).
    3. Creation of vector table with 384 dimensions.
    4. Insertion of real dense vector embeddings.
    5. Native vector similarity query execution using cosine distance operator (<=>).
    """
    db_url = get_migration_database_url()
    eng = create_engine(db_url)

    import uuid
    unique_table = f"test_probe_{uuid.uuid4().hex[:8]}"

    with eng.connect() as conn:
        # 1. Enable extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()

        try:
            # 2. Create isolated uniquely-named test table
            conn.execute(
                text(
                    f"""
                    CREATE TABLE {unique_table} (
                        id SERIAL PRIMARY KEY,
                        title VARCHAR(100),
                        embedding vector(384)
                    );
                    """
                )
            )
            conn.commit()

            # 3. Insert vectors
            v1 = [0.1] * 384
            v2 = [0.9] * 384
            conn.execute(
                text(f"INSERT INTO {unique_table} (title, embedding) VALUES (:title, :emb);"),
                {"title": "Doc Alpha", "emb": str(v1)},
            )
            conn.execute(
                text(f"INSERT INTO {unique_table} (title, embedding) VALUES (:title, :emb);"),
                {"title": "Doc Beta", "emb": str(v2)},
            )
            conn.commit()

            # 4. Perform cosine distance search (<=> operator)
            query_vec = [0.12] * 384
            result = conn.execute(
                text(
                    f"""
                    SELECT title, embedding <=> :q AS distance
                    FROM {unique_table}
                    ORDER BY distance ASC
                    LIMIT 1;
                    """
                ),
                {"q": str(query_vec)},
            ).fetchone()

            assert result is not None
            assert result[0] == "Doc Alpha"
            assert result[1] >= 0.0

        finally:
            # Guaranteed cleanup in finally block
            conn.execute(text(f"DROP TABLE IF EXISTS {unique_table};"))
            conn.commit()
