# Development Report — Enterprise Intelligence Platform

## 1. Architecture Overview

The **Enterprise Intelligence Platform** uses a decoupled, containerized client-server architecture deployed as two Docker web services (`render.yaml`) backed by managed PostgreSQL (`pgvector`) and external LLM inference APIs:

- **Architecture Diagram:** [`docs/architecture.svg`](docs/architecture.svg) | [`docs/architecture.png`](docs/architecture.png)
- **Presentation Tier (`frontend/`):** Multi-page Streamlit web application (`app.py`, port `8501`) communicating exclusively with the backend via [`APIClient`](frontend/api_client.py) (`httpx` with JWT Bearer authentication, `60s` standard timeout, and `150s` read timeout for multi-step agent execution).
- **API & Orchestration Tier (`app/`):** FastAPI REST service (`app/main.py`, port `10000`, `/api/v1`) coordinating document ingestion, ONNX embedding generation, hybrid vector + full-text retrieval, claim verification, multi-step AI agents, conversational memory, and telemetry.
- **Persistence Tier (`app/db/`, `alembic/`):** Hosted Neon PostgreSQL with `pgvector` cosine similarity (`<=>`) and PostgreSQL `to_tsvector`/`ts_rank` full-text search, abstracted via `PortableVector(384)` for SQLite compatibility in automated tests.

---

## 2. Technologies Used

| Layer | Technologies & Models |
| :--- | :--- |
| **Backend API** | Python 3.11/3.13, FastAPI, Pydantic Settings, SQLAlchemy, Alembic |
| **Frontend UI** | Streamlit, `httpx`, `requests`, `pandas` |
| **Database & Search** | Neon PostgreSQL + `pgvector` (384-dim cosine) + PostgreSQL FTS (`tsvector`) + RRF (`k=60`); SQLite (`:memory:`) for offline tests |
| **Embeddings** | Pure `onnxruntime` + Hugging Face `tokenizers` (`BAAI/bge-small-en-v1.5`, 384-dim L2-normalized vectors) |
| **LLM Providers** | Primary: Groq (`openai/gpt-oss-120b`); Fallback: Google Gemini (`gemini-3.8-flash`) via `LLMRouter` with Tenacity exponential backoff |
| **Document Parsing** | PyMuPDF (`fitz`) for PDF, UTF-8/fallback decoding for TXT & Markdown, Python `csv` parser for CSV |
| **Evaluation & Testing** | Custom evaluation framework (`app/evaluation/`), ONNX cosine similarity, `ClaimVerifier`, `pytest`, `httpx` |
| **Deployment** | Docker (`Dockerfile.backend`, `Dockerfile.frontend`), Render Web Services blueprint (`render.yaml`) |

---

## 3. Implemented Modules

- **Document Intelligence (`app/ingestion/`, `app/api/documents.py`):** Supports `.pdf`, `.txt`, `.md`/`.markdown`, and `.csv` uploads with SHA-256 duplicate detection, processing status lifecycle (`COMPLETED` / failure tracking), 500-token chunking with 50-token overlap, page/row metadata preservation, and endpoints for upload, list, chunk inspection, and deletion.
- **Advanced RAG Pipeline (`app/rag/`, `app/api/rag.py`):**
  - `HybridRetriever`: Combines top-15 dense cosine vector search and top-15 keyword/FTS search (with `OR tsquery` fallback recovery) via Reciprocal Rank Fusion (`k=60`).
  - `RAGGenerator`: Applies input prompt-injection checks, hardened context delimiters, `[Doc: <title>, Page/Row]` citation directives, and `INSUFFICIENT_EVIDENCE` abstention gating.
  - `ClaimVerifier`: Sentence-level lexical and numeric verification (`min_support_ratio=0.60`), rejecting poorly grounded responses (`confidence < 0.40`).
- **Multi-Provider LLM Integration (`app/llm/`):** Unified `BaseLLMProvider` / `LLMResponse` interface across `GroqProvider` and `GeminiProvider`, orchestrated by `LLMRouter` with retry logic, token-budget controls, and automatic failover.
- **Autonomous AI Agent (`app/agent/`, `app/api/agent.py`):** Intent classifier (`QueryRouter`) routing queries across `CONVERSATIONAL`, `KNOWLEDGE_BASE_RAG`, `DOCUMENT_ANALYSIS`, and `MULTI_STEP_AGENT`. `AgentOrchestrator` executes validated tool workflows using three tools: `knowledge_base_search`, `document_analysis`, and `generate_report`.
- **Conversational Memory (`app/memory/`, `app/api/conversations.py`):** Per-user conversation CRUD, citation persistence, sliding/bounded history context (`context.py`), and `POST /conversations/{id}/summarize`.
- **Auth, Security & Observability (`app/core/`, `app/api/{auth,dashboard}.py`, `frontend/`):** JWT authentication (`/auth/login`, `/register`, `/me`), `PromptInjectionDefender` input sanitizer, audit logging, `TelemetryTracker`, and Streamlit views for Dashboard Stats, Documents, Chat, Autonomous Agent, and Observability.
- **AI Evaluation Module (`app/evaluation/`, `scripts/run_evaluation.py`):** 30-case benchmark dataset (`benchmark_qa.json`), retrieval/correctness/citation/hallucination/safety/latency/token metrics, `offline` and `live` CLI modes, unit tests, and Markdown/JSON report generation (`reports/evaluation_report.{md,json}`).

---

## 4. AI Tools Usage Disclosure (Rule 2)

- **Tools Used:** Antigravity AI coding assistant (used during development/debugging phases of this project).
- **Tasks Delegated:** Boilerplate scaffolding for modules, evaluation schemas/reporter formatting, unit test drafting, root-cause diagnostics on logs/tracebacks, and generating SVG/Markdown documentation from repository state.
- **Independent Engineering Decisions:** Choosing pure ONNX Runtime + Hugging Face Tokenizers over PyTorch to stay within Render Free's 512 MB RAM ceiling; implementing RRF (`k=60`) hybrid search with OR-tsquery recovery; separating `QueryRouter` routes from multi-step agent execution; designing Groq 8,000 TPM throttling/budgeting and dedicated 150s agent timeouts; authoring the 30-case ground-truth dataset directly from the indexed PDF; requiring dual `offline` and real `live` evaluation without fabricated metrics.
- **AI-Generated Bugs Caught & Fixed During Validation:**
  - `RAGGenerator.query()` returns token keys `"prompt"`, `"completion"`, `"total"` (initial evaluation runner expected `"prompt_tokens"` -> recorded `0` tokens until caught and fixed).
  - `OfflineExtractiveProvider` initially matched `Question:` instead of `RAGGenerator`'s `User Query:` header, breaking unanswerable abstention in unit tests until fixed.
  - LLM output uses Unicode CJK lenticular brackets `【Doc: ..., Page: ...】` and non-breaking hyphens (`\u2011`) / narrow spaces (`\u202f`), which initially caused false `0%` citation precision and false missed-fact penalties until Unicode normalization was added and verified in unit tests.
  - `scripts/run_evaluation.py` offline SQLite test fixtures initially omitted required NOT NULL `file_size_bytes` and relied on truthy return values from `check_db_connection()`, caught immediately by pytest and offline CLI verification.

---

## 5. Key Engineering Decisions & Errors Resolved

### Architectural Decisions
1. **ONNX Runtime + HF Tokenizers Embeddings (`0ea8432`):** Replaced PyTorch and `sentence-transformers` with pure `onnxruntime` + `tokenizers` loading `BAAI/bge-small-en-v1.5` (pre-cached in `Dockerfile.backend`), preserving exact 384-dim L2-normalized pgvector compatibility without database re-indexing.
2. **Hybrid Retrieval with RRF & OR Fallback:** Dense pgvector cosine retrieval handles semantic paraphrase while PostgreSQL `tsvector` + OR `tsquery` fallback preserves exact technical terms and numbers, merged cleanly with `RRF k=60`.
3. **Evaluation Fidelity:** Separated `offline` mode (`OfflineExtractiveProvider`) from `live` mode (real Neon Postgres + Groq `openai/gpt-oss-120b`), saving separate outputs (`evaluation_report_offline.*` vs `evaluation_report.*`) and explicitly pacing live requests (`--delay 10.0`) below Groq's 8,000 TPM cap.

### Production Errors Resolved (Verified Git History)
- **Render Free 512 MB OOM (`5cb89ae` -> `0ea8432`):** BF16 PyTorch weights still exceeded the 512 MB Render Free container limit (~193 MB baseline RSS + Torch runtime/inference). Migrating completely to ONNX Runtime + single-thread env vars (`OMP_NUM_THREADS=1`, `MALLOC_ARENA_MAX=2`) brought runtime RSS down to ~231–258 MB locally after model load and resolved container crashes.
- **Agent Groq 8,000 TPM Overflow & Gemini 503 (`ad00150`):** Bounded retrieved passages and summarized intermediate tool outputs prior to agent synthesis while retaining citation IDs and surfacing provider fallback failures cleanly.
- **Streamlit Agent Read Timeout (`44d8c4d`):** Raised `APIClient.run_agent()` timeout in `frontend/api_client.py` selectively from `60s` to `150s` (keeping `10s` connect timeout and leaving other endpoints at `60s`) and added request/duration/exception logging in `app/api/agent.py`.
- **Early Deployment & Auth Issues (`e60245d`, `2cbc846`, `6c5ae34`, `e7d78f4`):** Normalized frontend `BACKEND_API_URL`, simplified Render uvicorn port binding, migrated deprecated LLM model IDs, and fixed Swagger HTTP Bearer authentication.

---

## 6. Testing & Deployment Summary

- **Automated Unit Tests:** `.\.venv\Scripts\python.exe -m pytest tests/unit/ --tb=short` passed **159 tests in 196.21s** (`0` failures, `0` skipped) across 27 test modules covering auth, ingestion parsers/chunking, ONNX embeddings, hybrid retrieval, RAG API/citations, prompt-injection defense, verification, agent tools/orchestrator, memory/routing, frontend client/views, and the new evaluation module (`tests/unit/test_evaluation.py`).
- **Integration Test Suite:** `tests/integration/test_postgres_pgvector.py` is present in the repository for live Postgres/pgvector checks (not re-executed separately in this final pass).
- **Cloud Deployment Configuration (`render.yaml`):**
  - `enterprise-intelligence-backend` (`Dockerfile.backend`, health check `/health`, port `10000`, ONNX model pre-baked into Docker image layer).
  - `enterprise-intelligence-frontend` (`Dockerfile.frontend`, health check `/_stcore/health`, port `8501`).

---

## 7. Actual Evaluation Results

Executed live against Neon PostgreSQL (`AI_Engineer_Technical_Assessment.pdf`, 27 chunks), ONNX `BAAI/bge-small-en-v1.5`, and Groq `openai/gpt-oss-120b` (`Run ID: 83884353`, `--top-k 5`, `--delay 10.0s`, source: [`reports/evaluation_report.md`](reports/evaluation_report.md) & [`reports/evaluation_report.json`](reports/evaluation_report.json)):

| Metric | Measured Live Result | Notes / Breakdown |
| :--- | :---: | :--- |
| **Cases Executed / Errors** | **30 / 30 (0 errors)** | `groq` handled 26 queries; `prompt_injection_defender` blocked 4 locally |
| **Overall Pass Rate** | **76.7% (23/30)** | `numerical` 100%, `unanswerable` 100%, `prompt_injection` 100%, `answerable` 70%, `multi_hop` 33.3% |
| **Retrieval HitRate@5 / Page Hit@5** | **100.0% / 100.0%** | Evaluated across all 21 applicable KB queries |
| **Mean Recall@5 / Precision@5** | **0.8095 / 0.2667** | — |
| **MRR / NDCG@5** | **0.8770 / 0.7670** | `numerical` achieved MRR `1.0000` |
| **Answer Correctness (Composite)** | **0.7258** | Fact Recall **73.1%**, Numeric Accuracy **91.7%**, ONNX Similarity **0.7108** |
| **Inline Citation Presence / Precision** | **100.0% / 100.0%** | Across all 18 non-abstaining grounded responses (`0.0%` fabricated citations) |
| **ClaimVerifier Confidence / Flag Rate** | **0.8278 / 11.5%** | Mean unsupported claim rate **1.1%** |
| **Unanswerable Abstention Accuracy** | **100.0% (5/5)** | `Q22`–`Q26` correctly returned `INSUFFICIENT_EVIDENCE` |
| **Prompt-Injection Resistance** | **100.0% (4/4)** | `Q27`–`Q30` intercepted pre-LLM in `0.1–2.4 ms` (`0` tokens) |
| **Latency (Retrieval / End-to-End Mean)** | **3,307.7 ms / 4,558.2 ms** | E2E p50: `3,919.1 ms`, p95: `8,856.5 ms`, max: `19,632.4 ms` |
| **Token Consumption** | **35,445 total** | `28,457` prompt + `6,988` completion (`1,181.5` mean/case across 30 cases) |

*(An offline deterministic verification run `c1698183` using an ephemeral 7-chunk SQLite fixture is recorded separately in [`reports/evaluation_report_offline.md`](reports/evaluation_report_offline.md): 30/30 executed, 70.0% pass rate, 60.0% abstention, 100.0% injection guard, 12,178 harness tokens.)*

---

## 8. Known Limitations

1. **Single-Document Live Corpus:** Only one completed document (`AI_Engineer_Technical_Assessment.pdf`, 27 chunks) is currently indexed in the live knowledge base, so cross-document multi-hop interference is untested.
2. **Active `TOP_K_FINAL=5` Override:** While `app/config.py` and `render.yaml` default `TOP_K_FINAL=8`, the local `.env` sets `TOP_K_FINAL=5`, reducing multi-chunk recall on `Q09`, `Q13`, and `Q21` when spanning multiple distant sections.
3. **Rule-Based `ClaimVerifier` & Injection Guard:** `ClaimVerifier` relies on sentence-level lexical/numeric overlap heuristics (producing strict verification rejections or flags on valid paraphrased synthesis in `Q04`, `Q06`, `Q10`, and `Q12`), and `PromptInjectionDefender` uses static regex confidence thresholds (`>= 0.80`).
4. **Proxy Correctness Metrics & Free-Tier Rate Limits:** Answer correctness relies on fact-string recall, numeric matching, and ONNX embedding similarity rather than human or LLM-as-judge grading. Groq `openai/gpt-oss-120b` enforces an 8,000 TPM limit, and `TelemetryTracker` uses an in-memory 1,000-record ring buffer reset on backend restart.
5. **Embedding Config Label Discrepancy:** In local `.env`, `settings.EMBEDDING_MODEL_NAME` reports `sentence-transformers/all-MiniLM-L6-v2` in report environment metadata while runtime logs confirm ONNX loaded `BAAI/bge-small-en-v1.5`.
