# Enterprise Intelligence Platform - Guidelines & Architecture

## System Overview
The Enterprise Intelligence Platform is an enterprise-grade document intelligence and hybrid RAG system engineered with:
- Dual-engine database portability (PostgreSQL + pgvector for production; isolated SQLite for local tests).
- Multi-format ingestion pipeline (PyMuPDF for PDF, Pandas for CSV, TXT, Markdown) with SHA-256 deduplication.
- Hybrid Retrieval: Dense vector retrieval (384-dim) + Full-Text Search (tsvector/GIN) combined via Reciprocal Rank Fusion (RRF, k=60).
- Multi-provider LLM integration with intelligent fallback: Primary (Groq / Llama-3.3) -> Fallback (Google Gemini / Gemini-2.0).
- Claim-level AI response verification: Atomic claim extraction, citation grounding, and hallucination gating.
- Layered prompt injection defense: Input query scanning and untrusted context isolation in prompts.

## Development & Test Commands
```powershell
# Virtual Environment
.\.venv\Scripts\Activate.ps1

# Run full test suite
.\.venv\Scripts\python.exe -m pytest tests/ -v

# Run Stage 2 & 3 production verification
.\.venv\Scripts\python.exe scripts/verify_stage2.py

# Run Alembic migrations (requires active DATABASE_URL)
.\.venv\Scripts\alembic.exe upgrade head

# Generate static SQL migrations (offline check)
.\.venv\Scripts\alembic.exe upgrade head --sql
```

## Key Architectural Principles
1. **Zero Mocks as Real Functionality**: Real database transactions, parsers, token counters, and status lifecycles are executed. Unit tests only mock third-party network APIs when testing network failure conditions.
2. **Database Portability**: Use `PortableVector` in SQLAlchemy models to allow both PostgreSQL native `VECTOR(384)` and SQLite compatibility.
3. **No Secrets in Repository**: `.env` is gitignored. All configuration reads from `app.config.settings` with safe masking in logs and telemetry.
4. **Context Untrusted Isolation**: All retrieved document text is treated as untrusted user data and enclosed in `<untrusted_retrieved_context>` XML blocks with explicit instructions forbidding command execution.
5. **Explainable Verification**: Claim verification calculates an explainable evidence grounding ratio rather than treating LLM confidence as a probability.
