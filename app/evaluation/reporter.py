"""Markdown and JSON report generators for evaluation runs."""

from pathlib import Path
from typing import Optional, Tuple

from app.evaluation.schemas import EvaluationReport


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def _num(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def render_markdown_report(report: EvaluationReport) -> str:
    """Render an EvaluationReport into a readable Markdown report."""
    agg = report.aggregate
    lines = [
        "# AI Evaluation Report — Enterprise Intelligence Platform",
        "",
        f"- **Run ID:** `{report.run_id}`",
        f"- **Mode:** `{report.mode}`",
        f"- **Timestamp (UTC):** `{report.timestamp_utc}`",
        f"- **Dataset:** `{report.dataset_path}` ({agg.total_cases} cases)",
        f"- **Retrieval Top-K:** `{report.top_k}`",
        f"- **Database Dialect:** `{report.environment.get('database_dialect', 'unknown')}`",
        f"- **Embedding Model:** `{report.environment.get('embedding_model', 'unknown')}` ({report.environment.get('embedding_dimension', 0)}d)",
        f"- **Primary / Fallback LLM:** `{report.environment.get('primary_llm_model', 'N/A')}` / `{report.environment.get('fallback_llm_model', 'N/A')}`",
        "",
        "## 1. Indexed Knowledge Base Scope",
        "",
    ]

    docs = report.environment.get("indexed_documents", [])
    if docs:
        lines.append("| Document Title | File Type | Chunks | Document ID |")
        lines.append("| :--- | :--- | ---: | :--- |")
        for d in docs:
            lines.append(
                f"| `{d['title']}` | `{d['file_type']}` | {d['total_chunks']} | `{d['id']}` |"
            )
    else:
        lines.append("_No completed documents found in database metadata._")

    lines.extend(
        [
            "",
            "## 2. Executive Summary (Actual Measured Metrics)",
            "",
            "| Metric Category | Metric | Measured Value |",
            "| :--- | :--- | ---: |",
            f"| **Execution** | Total / Executed / Errors | {agg.total_cases} / {agg.executed_cases} / {agg.error_cases} |",
            f"| **Execution** | Overall Pass Rate | {_pct(agg.overall_pass_rate)} |",
            f"| **Retrieval** (n={agg.retrieval_applicable_cases}) | HitRate@{report.top_k} (Chunk) | {_pct(agg.hit_rate_at_k)} |",
            f"| **Retrieval** (n={agg.retrieval_applicable_cases}) | Page Hit@{report.top_k} | {_pct(agg.mean_page_hit_at_k)} |",
            f"| **Retrieval** (n={agg.retrieval_applicable_cases}) | Mean Recall@{report.top_k} | {_num(agg.mean_recall_at_k)} |",
            f"| **Retrieval** (n={agg.retrieval_applicable_cases}) | Mean Precision@{report.top_k} | {_num(agg.mean_precision_at_k)} |",
            f"| **Retrieval** (n={agg.retrieval_applicable_cases}) | Mean Reciprocal Rank (MRR) | {_num(agg.mean_mrr)} |",
            f"| **Retrieval** (n={agg.retrieval_applicable_cases}) | Mean NDCG@{report.top_k} | {_num(agg.mean_ndcg_at_k)} |",
            f"| **Correctness** (n={agg.correctness_applicable_cases}) | Mean Fact Recall | {_pct(agg.mean_fact_recall)} |",
            f"| **Correctness** (numerical) | Mean Numeric Accuracy | {_pct(agg.mean_numeric_accuracy)} |",
            f"| **Correctness** (n={agg.correctness_applicable_cases}) | ONNX Semantic Similarity | {_num(agg.mean_semantic_similarity)} |",
            f"| **Correctness** (n={agg.correctness_applicable_cases}) | Composite Correctness Score | {_num(agg.mean_correctness_score)} |",
            f"| **Citations** (n={agg.citation_applicable_cases}) | Citation Presence Rate | {_pct(agg.citation_presence_rate)} |",
            f"| **Citations** (n={agg.citation_applicable_cases}) | Inline Citation Precision | {_pct(agg.mean_citation_precision)} |",
            f"| **Citations** (n={agg.citation_applicable_cases}) | Fabricated Citation Rate | {_pct(agg.fabricated_citation_rate)} |",
            f"| **Grounding & Safety** | Mean ClaimVerifier Confidence | {_num(agg.mean_claim_confidence)} |",
            f"| **Grounding & Safety** | Mean Unsupported Claim Rate | {_pct(agg.mean_unsupported_claim_rate)} |",
            f"| **Grounding & Safety** | Hallucination Flag Rate | {_pct(agg.hallucination_rate)} |",
            f"| **Grounding & Safety** (n={agg.unanswerable_cases}) | Unanswerable Abstention Accuracy | {_pct(agg.abstention_accuracy)} |",
            f"| **Grounding & Safety** (n={agg.prompt_injection_cases}) | Prompt-Injection Resistance Rate | {_pct(agg.injection_resistance_rate)} |",
            f"| **Performance** | Retrieval Latency (mean / p50 / p95 / max) | {agg.retrieval_latency.mean_ms:.1f} / {agg.retrieval_latency.p50_ms:.1f} / {agg.retrieval_latency.p95_ms:.1f} / {agg.retrieval_latency.max_ms:.1f} ms |",
            f"| **Performance** | End-to-End Latency (mean / p50 / p95 / max) | {agg.total_latency.mean_ms:.1f} / {agg.total_latency.p50_ms:.1f} / {agg.total_latency.p95_ms:.1f} / {agg.total_latency.max_ms:.1f} ms |",
            f"| **Tokens** | Prompt / Completion / Total Tokens | {agg.total_prompt_tokens:,} / {agg.total_completion_tokens:,} / {agg.total_tokens:,} |",
            f"| **Tokens** | Mean Tokens per Executed Case | {agg.mean_tokens_per_case:.1f} |",
            "",
            "## 3. Breakdown by Question Category",
            "",
            "| Category | Total | Completed | Errors | Pass Rate | Correctness | Recall@K | MRR | Cit Precision | Claim Conf | Safety/Abstention |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for cat in report.by_category:
        safety_col = (
            _pct(cat.abstention_accuracy)
            if cat.abstention_accuracy is not None
            else (_pct(cat.injection_resistance_rate) if cat.injection_resistance_rate is not None else "N/A")
        )
        lines.append(
            f"| `{cat.category.value}` | {cat.total_cases} | {cat.completed_cases} | {cat.error_cases} | "
            f"{_pct(cat.pass_rate)} | {_num(cat.mean_correctness)} | {_num(cat.mean_recall_at_k)} | "
            f"{_num(cat.mean_mrr)} | {_pct(cat.mean_citation_precision)} | {_num(cat.mean_claim_confidence)} | {safety_col} |"
        )

    lines.extend(["", "## 4. Provider Usage", ""])
    if agg.providers_used:
        for prov, count in sorted(agg.providers_used.items()):
            lines.append(f"- `{prov}`: {count} case(s)")
    else:
        lines.append("- None recorded.")

    lines.extend(["", "## 5. Per-Case Execution Results", ""])
    lines.append(
        "| ID | Category | Pass | Status | Recall@K | MRR | Correctness | Cit Prec | Claim Conf | Abstained / Blocked | Latency (ms) | Tokens | Provider |"
    )
    lines.append(
        "| :--- | :--- | :---: | :--- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | :--- |"
    )

    for r in report.results:
        ret_rec = _num(r.retrieval.recall_at_k, 2) if r.retrieval.applicable else "N/A"
        ret_mrr = _num(r.retrieval.mrr, 2) if r.retrieval.applicable else "N/A"
        corr = _num(r.correctness.overall_score, 2) if r.correctness.applicable else "N/A"
        cit = _pct(r.citation.citation_precision) if r.citation.applicable else "N/A"
        conf = _num(r.hallucination.claim_confidence_score, 2)
        guard_flag = (
            "Blocked"
            if r.hallucination.blocked_by_injection_guard
            else ("Abstained" if r.hallucination.abstained else "Answered")
        )
        verdict = "PASS" if r.passed else "FAIL"
        lines.append(
            f"| `{r.case_id}` | `{r.category.value}` | {verdict} | `{r.status}` | "
            f"{ret_rec} | {ret_mrr} | {corr} | {cit} | {conf} | {guard_flag} | "
            f"{r.performance.total_ms:.1f} | {r.performance.total_tokens} | `{r.performance.provider or '-'}` |"
        )

    failed = [r for r in report.results if not r.passed]
    lines.extend(["", f"## 6. Failed / Error Case Diagnostics ({len(failed)} cases)", ""])
    if not failed:
        lines.append("All cases passed evaluation thresholds.")
    else:
        for r in failed:
            lines.append(f"### `{r.case_id}` ({r.category.value}) — `{r.status}`")
            lines.append(f"- **Question:** {r.question}")
            if r.error_message:
                lines.append(f"- **Error:** `{r.error_message[:300]}`")
            if r.correctness.applicable and r.correctness.missed_facts:
                lines.append(f"- **Missed Facts:** {r.correctness.missed_facts}")
            if r.correctness.applicable and r.correctness.missed_numbers:
                lines.append(f"- **Missed Numbers:** {r.correctness.missed_numbers}")
            if r.citation.applicable and r.citation.fabricated_citation_detected:
                bad_cits = [c.failure_reason for c in r.citation.parsed_citations if not c.valid]
                lines.append(f"- **Invalid Citations:** {bad_cits}")
            if r.hallucination.hallucination_detected:
                lines.append(
                    f"- **ClaimVerifier:** unsupported={r.hallucination.unsupported_claims}/{r.hallucination.total_claims}, confidence={r.hallucination.claim_confidence_score:.2f}"
                )
            lines.append("")

    lines.extend(["## 7. Evaluation Methodology", ""])
    for item in report.methodology_notes:
        lines.append(f"- {item}")

    lines.extend(["", "## 8. Limitations & Methodological Tradeoffs", ""])
    for item in report.limitations:
        lines.append(f"- {item}")

    lines.append("")
    return "\n".join(lines)


def write_evaluation_reports(
    report: EvaluationReport,
    output_dir: str = "reports",
    prefix: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Write evaluation report to both JSON and Markdown files in output_dir."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    stem = prefix or ("evaluation_report" if report.mode == "live" else f"evaluation_report_{report.mode}")
    json_path = out_path / f"{stem}.json"
    md_path = out_path / f"{stem}.md"

    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    return md_path, json_path
