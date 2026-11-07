# Enterprise Intelligence Platform

An enterprise-grade document intelligence and hybrid RAG platform engineered with resilient multi-provider LLM fallback, multi-tool AI agents, conversational memory, claim verification, and real-time telemetry.

---

## Architecture Overview

```
                                  +-----------------------------+
                                  |   Streamlit Web Dashboard   |
                                  |   (Auth, Chat, Analytics)   |
                                  +--------------+--------------+
                                                 | (HTTP / REST)
                                                 v
                                  +-----------------------------+
                                  |    FastAPI Backend Server   |
                                  +--------------+--------------+
                                                 |
             +--------------------+--------------+--------------------+
             |                    |                                   |
             v                    v                                   v
+------------------------+ +-------------------------+ +-------------------------+
|  Ingestion Pipeline    | | Hybrid Retrieval (RRF)  | | LLM Router & Fallback   |
|  - PDF (PyMuPDF)       | | - Dense Vector Search   | | - Primary: Groq         |
|  - TXT, Markdown, CSV  | |   (pgvector / Cosine)   | | - Fallback: Gemini      |
|  - SHA-256 Deduplication| | - Full-Text Search (FTS)| | - Token & Latency Meter |
|  - Recursive Chunker   | | - Reciprocal Rank Fusion| | - Structured Outputs    |
+-----------+------------+ +------------+------------+ +------------+------------+
            |                           |                           |
            +---------------------------+---------------------------+
                                        |
                                        v
                         +-----------------------------+
                         |    Database Layer           |
                         | - PostgreSQL + pgvector     |
                         | - SQLAlchemy 2 + Alembic    |
                         +-----------------------------+
```

---

## Execution Roadmap & Progress

- [x] **Stage 1: Project Foundation & Configuration**
  - Dedicated virtual environment (`.venv`) initialized.
  - Pydantic Settings configuration (`app/config.py`) reading from environment variables with credential masking.
  - Security engine: Prompt injection defense & jailbreak detector (`app/core/security.py`).
  - Telemetry engine: Thread-safe token usage & latency meter (`app/core/telemetry.py`).
  - Structured error hierarchy & exception handlers (`app/core/errors.py`).
  - FastAPI core server with health and telemetry routes (`app/main.py`).
  - Containerization: `docker-compose.yml`, `Dockerfile.backend`, `Dockerfile.frontend`.

- [x] **Stage 2: Database & Document Ingestion**
  - **SQLAlchemy Models** (`app/db/models.py`):
    - `User`: RBAC (admin, analyst, user), PBKDF2 hashed credentials.
    - `Document`: Metadata, content hash, processing status lifecycle (`PENDING` -> `PROCESSING` -> `COMPLETED` / `FAILED`), token and chunk counts.
    - `DocumentChunk`: Text content, token budgeting, sectional metadata (`page_number`, `row_number`, `header`), dense vector embeddings (`VECTOR(384)`).
    - `Conversation` & `Message`: Multi-turn conversational persistence with citation and model tracking.
    - `AuditLog`: Security audit trails for all operations.
  - **Database Portability**:
    - `PortableVector`: Resolves to native `Vector(384)` on PostgreSQL and structured `JSON` on SQLite.
    - Zero mocks: Runs real database operations on both hosted PostgreSQL (e.g. Neon, Supabase) and local test environments.
  - **Alembic Migrations** (`alembic/`):
    - Auto-enables `CREATE EXTENSION IF NOT EXISTS vector` on PostgreSQL.
    - Migration `001_initial_schema.py` establishes complete schema and performance indexes.
  - **Document Parsing Engine** (`app/ingestion/parser.py`):
    - **PDF**: PyMuPDF (`pymupdf`) page-by-page extraction, page numbering metadata, empty/scanned page detection.
    - **TXT**: Plaintext parser with multi-encoding support (`utf-8`, `utf-8-sig`, `latin-1`, `cp1252`).
    - **Markdown**: Header-aware section parser (`#`, `##`, `###`).
    - **CSV**: Pandas tabular data parser converting rows to structured semantic key-value representations.
  - **Recursive Chunker** (`app/ingestion/chunker.py`):
    - Recursive splitting (`\n\n` -> `\n` -> sentence -> word) preserving contextual overlap and section metadata.
  - **Ingestion Pipeline** (`app/ingestion/pipeline.py`):
    - Cryptographic SHA-256 duplicate content detection.
    - Status tracking (`PENDING` -> `PROCESSING` -> `COMPLETED`/`FAILED`).
    - Transactional error recording with orphan chunk cleanup.
  - **REST API Endpoints** (`app/api/documents.py`):
    - `POST /api/v1/documents/upload`: Upload file with deduplication check.
    - `GET /api/v1/documents`: List documents with pagination and status filter.
    - `GET /api/v1/documents/{id}`: Detailed document status.
    - `GET /api/v1/documents/{id}/chunks`: Retrieve chunks with embeddings and metadata.
    - `DELETE /api/v1/documents/{id}`: Cascade deletion of documents and chunks.

- [x] **Stage 3: Hybrid RAG & AI Response Verification**
  - **Hybrid Retrieval Engine** (`app/rag/retrieval.py`):
    - Dense vector retrieval using pgvector cosine distance (`<=>` operator) and portable cosine similarity fallback.
    - Sparse keyword retrieval using PostgreSQL Full-Text Search (`to_tsvector`, `plainto_tsquery`, `ts_rank`) and portable token matching with stopword filtration.
    - Reciprocal Rank Fusion (RRF, $k=60$) combining dense and sparse results with mathematical rank deduplication.
    - Metadata filtering by `document_id` and `file_type`.
  - **Prompt Injection Defense & Context Hardening** (`app/rag/generator.py`, `app/core/security.py`):
    - Input query inspection rejecting prompt override and jailbreak attacks (`PromptInjectionDefender`).
    - Layered context boundary isolation: retrieved content delimited inside `<untrusted_retrieved_context>` XML tags.
    - Strict system prompt directives treating document content as passive untrusted data that cannot issue commands or alter platform instructions.
  - **Intelligent Multi-Provider LLM Integration** (`app/llm/`):
    - **Primary Provider**: Groq API (`Llama-3.3-70b-versatile`) via official Groq SDK with custom timeouts and telemetry.
    - **Secondary Provider**: Google Gemini API (`gemini-2.0-flash`) via `google-genai` SDK with intelligent failover.
    - **Resilient Router** (`app/llm/fallback_router.py`): Tenacity exponential backoff retries on rate limits, server 5xx errors, and timeouts; transparent failover to secondary provider; provider attribution and token/latency metrics.
  - **Claim-Level AI Response Verification** (`app/rag/verifier.py`):
    - Extracts atomic claims from generated responses (filtering boilerplate and citation tags).
    - Evaluates claim-level semantic overlap and quantitative metric matching against candidate chunks.
    - Categorizes claims into `SUPPORTED`, `PARTIALLY_SUPPORTED`, and `UNSUPPORTED`.
    - Computes an explainable grounding confidence ratio: $\frac{\text{supported} + 0.5 \times \text{partially supported}}{\text{total}}$.
    - Enforces verification gating: detects hallucinations, rejects low-grounding answers, and provides `INSUFFICIENT_EVIDENCE` responses when evidence is lacking.
  - **REST API Endpoints** (`app/api/rag.py`):
    - `POST /api/v1/rag/retrieve`: Hybrid dense + sparse chunk retrieval with RRF rank scoring.
    - `POST /api/v1/rag/query`: End-to-end question answering with citations, provider tracking, and verification reports.
    - `POST /api/v1/rag/verify`: Standalone claim verification for answers against chunk IDs or text snippets.

- [x] **Stage 4: Autonomous Multi-Tool AI Agent Architecture**
  - **Tool Abstraction & Telemetry Engine** (`app/agent/base.py`):
    - Abstract base class `BaseTool` enforcing Pydantic argument schemas, latency measurement, and automatic `AuditLog` database persistence.
  - **3 Functional Agent Tools** (`app/agent/tools/`):
    - `KnowledgeBaseSearchTool` (`app/agent/tools/kb_search.py`): Hybrid dense + sparse search with RRF scoring, document filters, and formatted citations `[Doc: <title>, Page: <page>, Chunk: <id>]`.
    - `DocumentAnalysisTool` (`app/agent/tools/doc_analysis.py`): Sequential chunk assembly and structured LLM extraction for specific documents.
    - `StructuredReportGenerationTool` (`app/agent/tools/report_gen.py`): Multi-source evidence synthesis with claim verification and explicit missing evidence separation.
  - **Agent Orchestrator** (`app/agent/orchestrator.py`):
    - Multi-step ReAct reasoning loop with tool allowlisting, argument self-correction feedback, step trace recording, and bounded execution (max 5 iterations).
  - **REST API Endpoints** (`app/api/agent.py`):
    - `POST /api/v1/agent/run`: Multi-step reasoning workflow.
    - `GET /api/v1/agent/tools`: JSON schema definitions of registered tools.
    - Direct tool invocation routes under `/api/v1/agent/tools/*`.

- [x] **Stage 5: Conversational Memory & Multi-Turn Intelligence**
  - **Persistent Conversation Lifecycle** (`app/memory/manager.py`):
    - Database models `Conversation` and `Message` mapped to Neon PostgreSQL with UUID primary keys.
    - Automatic title generation from initial user queries.
    - User-specific history filtering with strict cross-user isolation raising `AuthorizationError` (HTTP 403).
  - **Sliding-Window & Context Compaction**:
    - `get_relevant_context`: Dynamically retrieves recent dialogue turns and token-efficient formatted history.
    - `summarize_and_compact`: Automated LLM background summarization of older dialogue turns into a compact system memory briefing when exceeding turn thresholds.
  - **REST API Endpoints** (`app/api/conversations.py`):
    - `POST /api/v1/conversations`: Create conversation.
    - `GET /api/v1/conversations`: List user's conversations.
    - `GET /api/v1/conversations/{id}`: Detailed conversation history.
    - `POST /api/v1/conversations/{id}/messages`: Continue conversation with agent memory and citations.
    - `DELETE /api/v1/conversations/{id}`: Soft/hard delete conversation.
    - `POST /api/v1/conversations/{id}/summarize`: Explicit memory compaction.

- [x] **Stage 6: Frontend & Streamlit Dashboard**
  - **Enterprise Streamlit Interface** (`frontend/app.py`):
    - Clean, modern, responsive multi-page dashboard.
  - **Authentication & Session Guards** (`frontend/state.py`, `frontend/components/auth_view.py`):
    - JWT Bearer token login and user registration interface.
    - Session state protection guarding dashboard pages from unauthenticated access.
  - **Executive Dashboard** (`frontend/pages_views/dashboard_view.py`):
    - Real database KPI metric cards (documents, chunks, active sessions, query latency).
    - Ingestion status breakdown and LLM token distribution.
    - Persisted audit activity feed.
  - **Document Management** (`frontend/pages_views/documents_view.py`):
    - File uploader supporting PDF, TXT, MD, CSV with duplicate detection toggles.
    - Document table with chunk inspection expander and cascade deletion.
  - **Conversational AI Chat** (`frontend/pages_views/chat_view.py`):
    - Conversation selector with creation and deletion.
    - Persistent chat message history with follow-up memory and auto-summarization indicators.
    - Verified source citation cards displaying document titles, pages, and chunk IDs.
  - **AI Agent Interface** (`frontend/pages_views/agent_view.py`):
    - Tool catalog inspector with parameter schemas.
    - Step-by-step ReAct execution trace accordion (Thought, Action, Observation, Latency).
    - Grounded final report presentation.
  - **Observability & Telemetry** (`frontend/pages_views/observability_view.py`):
    - Real-time token consumption breakdown (prompt vs completion).
    - Live infrastructure health status (PostgreSQL connection, configured LLMs).
    - Interactive database audit log explorer.

- [ ] **Stage 7: Production Evaluation, Hardening & Deployment** `[PENDING USER APPROVAL]`

---

## Running the Application (Backend & Frontend)

### 1. Start the FastAPI Backend Server
```powershell
.\.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Docs (Swagger UI): `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/health`

### 2. Start the Streamlit Frontend Dashboard
In a separate terminal:
```powershell
.\.venv\Scripts\streamlit.exe run frontend/app.py --server.port 8501
```
- Open your browser at: `http://localhost:8501`

### 3. Default Login Credentials
- **Username:** `admin`
- **Password:** `change_this_admin_password_123!` (or the configured `ADMIN_PASSWORD` in your `.env` file)
- *Alternatively, click the **Create Account** tab to register a new user.*

---

## Database Configuration & Portability

The platform supports both hosted cloud PostgreSQL (Neon, Supabase, AWS RDS) and Docker-based local PostgreSQL without modifying application code:

```powershell
# Option A: Hosted PostgreSQL with pgvector (Free tier Neon/Supabase)
DATABASE_URL=postgresql+psycopg2://user:password@ep-host.region.aws.neon.tech/neondb?sslmode=require

# Option B: Local Docker PostgreSQL
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/enterprise_intel

# Option C: Test SQLite Database (Used in automated local test runs)
DATABASE_URL=sqlite:///./test.db
```

### Running Migrations
```powershell
.\.venv\Scripts\alembic.exe upgrade head
```

---

## Test Suite Execution

Run all unit and integration tests:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

### Tests Requiring a Live PostgreSQL Connection
The following integration test is specifically designed to run against a real PostgreSQL instance with `pgvector`:
- `tests/integration/test_postgres_pgvector.py::test_real_postgres_connection_and_pgvector_extension`
  - Marker: `@pytest.mark.postgres_integration`
  - When `DATABASE_URL` points to an active PostgreSQL database, it tests `CREATE EXTENSION vector`, creates a vector table, inserts 384-dim embeddings, and executes native cosine distance searches (`<=>`).
  - When a live PostgreSQL instance is not connected, the test is cleanly skipped with an explicit diagnostic message explaining that a real PostgreSQL connection is required.
