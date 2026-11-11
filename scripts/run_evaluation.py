"""CLI evaluation runner supporting deterministic offline and real live RAG modes."""

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure project root is importable when running as script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import settings
from app.db.models import Base, Document, DocumentChunk
from app.db.session import SessionLocal
from app.evaluation.reporter import write_evaluation_reports
from app.evaluation.runner import DEFAULT_DATASET_PATH, EvaluationRunner, load_eval_dataset
from app.evaluation.schemas import CaseEvalResult
from app.rag.embeddings import embedding_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_evaluation")


def _create_ephemeral_offline_session() -> Session:
    """Create an isolated in-memory SQLite KB for offline runs when Postgres is unavailable."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    db = session_factory()

    doc = Document(
        id="offline-assessment-doc",
        title="AI_Engineer_Technical_Assessment.pdf",
        filename="AI_Engineer_Technical_Assessment.pdf",
        file_type="pdf",
        file_size_bytes=4096,
        content_hash="offline-ephemeral-fixture",
        processing_status="COMPLETED",
        total_chunks=7,
        total_tokens=450,
    )
    db.add(doc)
    db.flush()

    sample_sections = [
        (
            0,
            1,
            "1. Assessment Objective: design, develop, test, and deploy intelligent applications using LLMs, RAG, AI agents, embeddings, and vector databases. Expected duration: 24 hours. Section 2 example query: identify reasons for system failures with evidence and recommend possible solutions.",
        ),
        (
            5,
            2,
            "Module A Document Intelligence: support PDF, TXT, Markdown, and CSV uploads. Validate files, extract content, generate embeddings, store vectors and metadata, and handle duplicates.",
        ),
        (
            6,
            2,
            "Module B Advanced RAG Architecture: retrieve chunks, provide source citations, handle insufficient evidence, and apply retrieval improvements such as hybrid search, reranking, query rewriting, metadata filtering, or contextual retrieval.",
        ),
        (
            7,
            2,
            "Module C LLM Integration & Module D AI Agents: integrate at least two LLM models or providers with structured output, prompt templates, model selection, token tracking, error handling, rate-limit handling, model fallback, and response validation. Module D requires at least three functional tools: Knowledge Base Search, Document Analysis, and Structured Report Generation.",
        ),
        (
            9,
            2,
            "Module E Conversational Memory & Module F Application Dashboard: support multiple conversations, user conversation history, follow-up context management referencing previously discussed documents, authentication, protected routes, usage statistics, document management, source references, agent execution visibility, token usage, latency, errors, and activity logs.",
        ),
        (
            10,
            3,
            "Module G AI Evaluation & Section 4 Challenges: create an evaluation dataset of at least 20 questions assessing retrieval relevance, answer correctness, source grounding, hallucination handling, latency, and token consumption. Complete at least three of seven challenges: Prompt Injection Protection, Retrieval Optimization, AI Response Verification, and Agent Reliability (execution limits, parameter validation, timeouts, recovery). Section 5 suggestions: OpenAI, Hugging Face, Pinecone, pgvector, PostgreSQL, Supabase, RAGAS, DeepEval.",
        ),
        (
            18,
            4,
            "Rules 2-6 & Section 8 Live Verification: Claude Code, ChatGPT, and Cursor are permitted during development with disclosure of delegated tasks, independent decisions, errors, and validation. During supervised live evaluation (approximately 60 minutes, screen sharing), AI coding agents are not permitted. Live tasks include secondary LLM provider fallback, conflicting information, implementing a new tool, preventing duplicate jobs, fixing unsupported answers, and security vulnerability fixes. Rule 6 prohibits hardcoded responses, mock search, fake tool execution, and fabricated evaluation results. Submission rules prohibit local-only apps and committing API keys (provide .env.example), plus a 5 to 10 minute walkthrough demo recording and Section 11 Final Expectations (AI architectures, RAG pipelines, AI agents, evaluate AI output quality, debug, production-ready).",
        ),
    ]

    for chunk_idx, page_num, content in sample_sections:
        db.add(
            DocumentChunk(
                id=f"offline-chunk-{chunk_idx}",
                document_id=doc.id,
                chunk_index=chunk_idx,
                content=content,
                token_count=len(content.split()),
                embedding=embedding_service.embed_text(content),
                chunk_metadata={
                    "page_number": page_num,
                    "source_file": doc.filename,
                    "chunk_index": chunk_idx,
                },
            )
        )
    db.commit()
    return db


def _print_progress(idx: int, total: int, result: CaseEvalResult) -> None:
    verdict = "PASS" if result.passed else "FAIL"
    print(
        f"[{idx:02d}/{total:02d}] {result.case_id} ({result.category.value}) -> "
        f"{verdict} | status={result.status} | latency={result.performance.total_ms:.0f}ms | "
        f"tokens={result.performance.total_tokens}"
    )


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Enterprise Intelligence Platform RAG Evaluation"
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "live"],
        default="offline",
        help="Execution mode: 'offline' (deterministic extractive test harness) or 'live' (actual DB + Groq/Gemini LLM).",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to evaluation dataset JSON.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.TOP_K_FINAL,
        help=f"Number of chunks to retrieve per query (default: {settings.TOP_K_FINAL}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of evaluation cases to execute.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Inter-request delay in seconds for live LLM pacing (defaults to 9.0 in live mode, 0.0 offline).",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory to write evaluation_report.md and evaluation_report.json.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional output filename stem (default: evaluation_report for live, evaluation_report_offline for offline).",
    )
    args = parser.parse_args(argv)

    delay_seconds = (
        args.delay
        if args.delay is not None
        else (9.0 if args.mode == "live" else 0.0)
    )

    cases, resolved_dataset = load_eval_dataset(args.dataset)
    if args.limit is not None and args.limit > 0:
        cases = cases[: args.limit]

    using_ephemeral_sqlite = False
    db: Session
    if args.mode == "live":
        if not (settings.has_groq or settings.has_gemini):
            print(
                "ERROR: Neither GROQ_API_KEY nor GEMINI_API_KEY is configured for live evaluation.",
                file=sys.stderr,
            )
            return 1
        db = SessionLocal()
        try:
            completed_docs = (
                db.query(Document).filter(Document.processing_status == "COMPLETED").count()
            )
        except Exception as db_err:
            db.close()
            print(
                f"ERROR: Live database connection failed ({type(db_err).__name__}). Live mode requires an active PostgreSQL knowledge base.",
                file=sys.stderr,
            )
            return 1
        if completed_docs == 0:
            db.close()
            print(
                "ERROR: Live database has 0 COMPLETED documents indexed.",
                file=sys.stderr,
            )
            return 1
    else:
        candidate_db = SessionLocal()
        try:
            completed_docs = (
                candidate_db.query(Document)
                .filter(Document.processing_status == "COMPLETED")
                .count()
            )
        except Exception:
            candidate_db.close()
            completed_docs = 0

        if completed_docs > 0:
            db = candidate_db
        else:
            try:
                candidate_db.close()
            except Exception:
                pass
            using_ephemeral_sqlite = True
            db = _create_ephemeral_offline_session()

    try:
        print(
            f"Starting evaluation run: mode={args.mode}, cases={len(cases)}, "
            f"top_k={args.top_k}, delay={delay_seconds}s"
        )
        if using_ephemeral_sqlite:
            print(
                "Note: Configured DB unreachable/empty in offline mode; using ephemeral SQLite sample fixture."
            )

        runner = EvaluationRunner(
            mode=args.mode,
            top_k=args.top_k,
            delay_seconds=delay_seconds,
        )
        report = runner.run(
            db=db,
            cases=cases,
            dataset_path=resolved_dataset,
            progress_callback=_print_progress,
        )
        if using_ephemeral_sqlite:
            report.environment["database_source"] = "ephemeral_sqlite_sample_fixture"
            report.limitations.append(
                "Database Fallback: Configured PostgreSQL was unreachable in offline mode, so retrieval ran against an ephemeral in-memory SQLite sample fixture."
            )

        md_path, json_path = write_evaluation_reports(
            report=report,
            output_dir=args.output_dir,
            prefix=args.prefix,
        )

        agg = report.aggregate
        print("\n=== Evaluation Summary ===")
        print(f"Run ID          : {report.run_id} ({report.mode})")
        print(f"Cases           : {agg.executed_cases}/{agg.total_cases} executed ({agg.error_cases} errors)")
        print(f"Pass Rate       : {agg.overall_pass_rate * 100:.1f}%")
        print(f"Retrieval Hit@K : {agg.hit_rate_at_k * 100:.1f}% | MRR={agg.mean_mrr:.4f} | NDCG@K={agg.mean_ndcg_at_k:.4f}")
        print(f"Correctness     : {agg.mean_correctness_score:.4f} | Fact Recall={agg.mean_fact_recall * 100:.1f}%")
        print(f"Citations       : Presence={agg.citation_presence_rate * 100:.1f}% | Precision={agg.mean_citation_precision * 100:.1f}%")
        print(f"Safety          : Abstention={agg.abstention_accuracy} | Injection Guard={agg.injection_resistance_rate}")
        print(f"Tokens / Meanms : {agg.total_tokens:,} total tokens | Mean latency={agg.total_latency.mean_ms:.1f}ms")
        print(f"Markdown Report : {Path(md_path).resolve()}")
        print(f"JSON Report     : {Path(json_path).resolve()}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
