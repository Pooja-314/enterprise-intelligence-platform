"""Pydantic schemas for RAG evaluation datasets, per-case metrics, and reports."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvalCategory(str, Enum):
    """Evaluation case taxonomy."""

    ANSWERABLE = "answerable"
    MULTI_HOP = "multi_hop"
    NUMERICAL = "numerical"
    UNANSWERABLE = "unanswerable"
    PROMPT_INJECTION = "prompt_injection"


class EvalCase(BaseModel):
    """Single benchmark evaluation item grounded in indexed knowledge base documents."""

    case_id: str
    question: str
    category: EvalCategory
    expected_answer: str
    expected_facts: List[str] = Field(default_factory=list)
    expected_numbers: List[str] = Field(default_factory=list)
    relevant_document_titles: List[str] = Field(default_factory=list)
    relevant_chunk_indices: List[int] = Field(default_factory=list)
    relevant_chunk_ids: List[str] = Field(default_factory=list)
    expected_pages: List[int] = Field(default_factory=list)
    should_abstain: bool = False
    is_prompt_injection: bool = False
    notes: str = ""


class RetrievalEvalMetrics(BaseModel):
    """Per-case retrieval relevance metrics."""

    applicable: bool = True
    top_k: int = 0
    retrieved_count: int = 0
    hit_at_k: float = 0.0
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    page_hit_at_k: float = 0.0
    matched_chunk_indices: List[int] = Field(default_factory=list)
    retrieved_chunk_indices: List[int] = Field(default_factory=list)
    retrieved_pages: List[int] = Field(default_factory=list)
    mean_rrf_score: float = 0.0
    mean_dense_score: float = 0.0


class CorrectnessEvalMetrics(BaseModel):
    """Per-case answer correctness metrics."""

    applicable: bool = True
    fact_recall: float = 0.0
    matched_facts: List[str] = Field(default_factory=list)
    missed_facts: List[str] = Field(default_factory=list)
    numeric_accuracy: Optional[float] = None
    matched_numbers: List[str] = Field(default_factory=list)
    missed_numbers: List[str] = Field(default_factory=list)
    semantic_similarity: float = 0.0
    overall_score: float = 0.0


class ParsedCitation(BaseModel):
    """Structured inline citation extracted from generated answer text."""

    raw_text: str
    document_title: str
    page_number: Optional[int] = None
    row_number: Optional[int] = None
    chunk_id: Optional[str] = None
    valid: bool = False
    failure_reason: Optional[str] = None


class CitationEvalMetrics(BaseModel):
    """Per-case citation and source grounding accuracy."""

    applicable: bool = True
    inline_citation_count: int = 0
    structured_citation_count: int = 0
    valid_inline_citation_count: int = 0
    invalid_inline_citation_count: int = 0
    citation_precision: float = 0.0
    has_required_citation: bool = False
    fabricated_citation_detected: bool = False
    parsed_citations: List[ParsedCitation] = Field(default_factory=list)


class HallucinationEvalMetrics(BaseModel):
    """Per-case grounding, abstention, and injection resistance metrics."""

    claim_verifier_is_verified: bool = True
    claim_confidence_score: float = 1.0
    total_claims: int = 0
    supported_claims: int = 0
    partially_supported_claims: int = 0
    unsupported_claims: int = 0
    unsupported_claim_rate: float = 0.0
    hallucination_detected: bool = False
    abstained: bool = False
    abstention_correct: Optional[bool] = None
    blocked_by_injection_guard: bool = False
    injection_resisted: Optional[bool] = None


class LatencyTokenMetrics(BaseModel):
    """Per-case timing and token usage metrics."""

    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    provider: str = ""
    model: str = ""


class CaseEvalResult(BaseModel):
    """Complete evaluation outcome for a single benchmark question."""

    case_id: str
    category: EvalCategory
    question: str
    expected_answer: str
    generated_answer: str = ""
    status: str = "completed"  # completed | blocked_injection | error
    error_message: Optional[str] = None
    passed: bool = False
    retrieval: RetrievalEvalMetrics = Field(default_factory=RetrievalEvalMetrics)
    correctness: CorrectnessEvalMetrics = Field(default_factory=CorrectnessEvalMetrics)
    citation: CitationEvalMetrics = Field(default_factory=CitationEvalMetrics)
    hallucination: HallucinationEvalMetrics = Field(default_factory=HallucinationEvalMetrics)
    performance: LatencyTokenMetrics = Field(default_factory=LatencyTokenMetrics)


class LatencySummary(BaseModel):
    """Latency distribution summary in milliseconds."""

    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    max_ms: float = 0.0


class CategorySummary(BaseModel):
    """Aggregated summary metrics for a single category."""

    category: EvalCategory
    total_cases: int = 0
    completed_cases: int = 0
    error_cases: int = 0
    pass_rate: float = 0.0
    mean_correctness: Optional[float] = None
    mean_recall_at_k: Optional[float] = None
    mean_mrr: Optional[float] = None
    mean_citation_precision: Optional[float] = None
    mean_claim_confidence: Optional[float] = None
    abstention_accuracy: Optional[float] = None
    injection_resistance_rate: Optional[float] = None


class AggregateMetrics(BaseModel):
    """Overall benchmark summary across all executed cases."""

    total_cases: int = 0
    executed_cases: int = 0
    error_cases: int = 0
    overall_pass_rate: float = 0.0

    # Retrieval (computed on cases with ground-truth chunks)
    retrieval_applicable_cases: int = 0
    hit_rate_at_k: float = 0.0
    mean_precision_at_k: float = 0.0
    mean_recall_at_k: float = 0.0
    mean_mrr: float = 0.0
    mean_ndcg_at_k: float = 0.0
    mean_page_hit_at_k: float = 0.0

    # Answer correctness (answerable / multi_hop / numerical)
    correctness_applicable_cases: int = 0
    mean_fact_recall: float = 0.0
    mean_numeric_accuracy: Optional[float] = None
    mean_semantic_similarity: float = 0.0
    mean_correctness_score: float = 0.0

    # Citation & grounding
    citation_applicable_cases: int = 0
    citation_presence_rate: float = 0.0
    mean_citation_precision: float = 0.0
    fabricated_citation_rate: float = 0.0

    # Hallucination, abstention, prompt-injection defense
    mean_claim_confidence: float = 0.0
    mean_unsupported_claim_rate: float = 0.0
    hallucination_rate: float = 0.0
    unanswerable_cases: int = 0
    abstention_accuracy: Optional[float] = None
    prompt_injection_cases: int = 0
    injection_resistance_rate: Optional[float] = None

    # Performance & tokens
    retrieval_latency: LatencySummary = Field(default_factory=LatencySummary)
    total_latency: LatencySummary = Field(default_factory=LatencySummary)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    mean_tokens_per_case: float = 0.0
    providers_used: Dict[str, int] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """Top-level evaluation report artifact written to JSON and Markdown."""

    run_id: str
    mode: str  # "offline" or "live"
    timestamp_utc: str
    dataset_path: str
    dataset_version: str = "1.0.0"
    top_k: int
    environment: Dict[str, Any] = Field(default_factory=dict)
    aggregate: AggregateMetrics
    by_category: List[CategorySummary] = Field(default_factory=list)
    methodology_notes: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    results: List[CaseEvalResult] = Field(default_factory=list)
