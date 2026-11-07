import pytest
from app.rag.retrieval import RetrievedChunk
from app.rag.verifier import ClaimResult, ClaimStatus, ClaimVerifier, VerificationReport


@pytest.fixture
def evidence_chunks():
    return [
        RetrievedChunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            document_title="Financial Report 2024",
            content="Q3 Net profit increased by 18.4% reaching a total of $45.2M in annual revenue.",
            token_count=20,
            metadata={"page_number": 3},
        ),
        RetrievedChunk(
            chunk_id="chunk_2",
            document_id="doc_2",
            document_title="Security Whitepaper",
            content="All API endpoints enforce HMAC-SHA256 authenticated JWT bearer tokens.",
            token_count=18,
            metadata={"page_number": 7},
        ),
    ]


@pytest.fixture
def multi_module_chunks():
    return [
        RetrievedChunk(
            chunk_id="chunk_mod1",
            document_id="doc_assessment",
            document_title="Technical Assessment Specification",
            content="Module 1: Document Ingestion and Preprocessing. Ingests PDFs, extracts text layout, and generates embeddings.",
            token_count=18,
            metadata={"page_number": 1},
        ),
        RetrievedChunk(
            chunk_id="chunk_mod2",
            document_id="doc_assessment",
            document_title="Technical Assessment Specification",
            content="Module 2: Hybrid RAG Engine. Combines dense pgvector semantic similarity with sparse keyword full-text search using RRF.",
            token_count=20,
            metadata={"page_number": 2},
        ),
        RetrievedChunk(
            chunk_id="chunk_mod3",
            document_id="doc_assessment",
            document_title="Technical Assessment Specification",
            content="Module 3: Autonomous Agent Orchestrator. Implements multi-step reasoning using ReAct loops and tool calling.",
            token_count=18,
            metadata={"page_number": 3},
        ),
        RetrievedChunk(
            chunk_id="chunk_mod4",
            document_id="doc_assessment",
            document_title="Technical Assessment Specification",
            content="Module 4: Explainable Verification Engine. Audits LLM responses at the claim level and detects hallucinations.",
            token_count=18,
            metadata={"page_number": 4},
        ),
        RetrievedChunk(
            chunk_id="chunk_mod5",
            document_id="doc_assessment",
            document_title="Technical Assessment Specification",
            content="Module 5: Observability and Telemetry. Provides Prometheus metrics, query latency tracking, and audit logging.",
            token_count=18,
            metadata={"page_number": 5},
        ),
    ]


def test_verify_supported_answer(evidence_chunks):
    verifier = ClaimVerifier(min_support_ratio=0.60)
    answer = (
        "According to the Financial Report 2024, Q3 Net profit increased by 18.4%. "
        "Total annual revenue reached $45.2M. "
        "In addition, API endpoints enforce HMAC-SHA256 authenticated JWT bearer tokens."
    )

    report = verifier.verify_answer(answer, evidence_chunks)
    assert report.is_verified is True
    assert report.confidence_score >= 0.80
    assert report.supported_claims >= 2
    assert report.unsupported_claims == 0
    assert not report.hallucination_detected


def test_verify_unsupported_hallucinated_answer(evidence_chunks):
    verifier = ClaimVerifier(min_support_ratio=0.60)
    hallucinated_answer = (
        "The company acquired a quantum computing facility in Zurich for $1.2B in December. "
        "The proprietary neural network model operates at 10,000 petaflops."
    )

    report = verifier.verify_answer(hallucinated_answer, evidence_chunks)
    assert report.is_verified is False
    assert report.confidence_score < 0.30
    assert report.unsupported_claims >= 1
    assert report.hallucination_detected is True
    assert report.rejection_reason is not None


def test_verify_partially_supported_answer(evidence_chunks):
    verifier = ClaimVerifier(min_support_ratio=0.70)
    mixed_answer = (
        "Q3 Net profit increased by 18.4%. "
        "The CEO announced that bonuses would double for all international contractors."
    )

    report = verifier.verify_answer(mixed_answer, evidence_chunks)
    assert report.total_claims == 2
    assert report.supported_claims == 1
    assert report.unsupported_claims == 1
    assert report.hallucination_detected is True


def test_verify_insufficient_evidence_response(evidence_chunks):
    verifier = ClaimVerifier()
    answer = "INSUFFICIENT_EVIDENCE: The uploaded files do not mention future expansion into aerospace."
    report = verifier.verify_answer(answer, evidence_chunks)
    assert report.is_verified is True
    assert report.confidence_score == 1.0
    assert not report.hallucination_detected


def test_verify_with_empty_evidence():
    verifier = ClaimVerifier()
    report = verifier.verify_answer("Some factual assertion.", [])
    assert report.is_verified is False
    assert report.confidence_score == 0.0
    assert report.hallucination_detected is True


def test_composite_evidence_synthesis_claim(multi_module_chunks):
    """Verifies that a multi-chunk synthesis claim combining facts across modules is properly supported."""
    verifier = ClaimVerifier(min_support_ratio=0.60)
    claim = (
        "The technical assessment requires Module 1 for document ingestion and preprocessing, "
        "Module 2 for hybrid RAG retrieval, and Module 3 for autonomous agent orchestration."
    )
    res = verifier._evaluate_single_claim(claim, multi_module_chunks)
    assert res.status == ClaimStatus.SUPPORTED
    assert res.confidence_score >= 0.65
    assert "Corroborated across multiple retrieved evidence chunks" in res.rationale
    assert res.supporting_chunk_id in {"chunk_mod1", "chunk_mod2", "chunk_mod3"}
    assert res.document_title == "Technical Assessment Specification"


def test_multi_module_synthesis_full_answer(multi_module_chunks):
    """Verifies that a comprehensive multi-module answer spanning 5 chunks verifies successfully."""
    verifier = ClaimVerifier(min_support_ratio=0.60)
    answer = (
        "The technical assessment defines five core modules:\n"
        "1. Module 1: Document Ingestion and Preprocessing for parsing PDFs and generating embeddings.\n"
        "2. Module 2: Hybrid RAG Engine combining dense pgvector search and sparse keyword search with RRF.\n"
        "3. Module 3: Autonomous Agent Orchestrator with ReAct reasoning and tool calling.\n"
        "4. Module 4: Explainable Verification Engine for claim-level hallucination detection.\n"
        "5. Module 5: Observability and Telemetry with Prometheus metrics and latency tracking."
    )
    report = verifier.verify_answer(answer, multi_module_chunks)
    assert report.is_verified is True
    assert report.confidence_score >= 0.80
    assert report.unsupported_claims == 0
    assert not report.hallucination_detected
    assert report.supported_claims >= 4


def test_unsupported_numerical_claim_strict_rejection(evidence_chunks):
    """Verifies that claims with hallucinated metrics/percentages are strictly rejected."""
    verifier = ClaimVerifier(min_support_ratio=0.60)
    # The words 'Q3', 'net', 'profit', 'annual', 'revenue' exist, but 99.9% and $999M are fabricated
    fabricated_claim = "Q3 Net profit increased by 99.9% reaching a total of $999M in annual revenue."
    res = verifier._evaluate_single_claim(fabricated_claim, evidence_chunks)
    assert res.status == ClaimStatus.UNSUPPORTED
    assert res.confidence_score < 0.35


def test_generic_word_overlap_rejection():
    """Verifies that generic words appearing singly across unrelated chunks do not inflate grounding score."""
    unrelated_chunks = [
        RetrievedChunk("u1", "d1", "Doc A", "The system architecture uses caching.", 6, metadata={"page_number": 1}),
        RetrievedChunk("u2", "d2", "Doc B", "Every user must provide credentials.", 6, metadata={"page_number": 2}),
        RetrievedChunk("u3", "d3", "Doc C", "Background processing handles queues.", 6, metadata={"page_number": 3}),
        RetrievedChunk("u4", "d4", "Doc D", "The database manages stored entities.", 6, metadata={"page_number": 4}),
    ]
    verifier = ClaimVerifier(min_support_ratio=0.60)
    scattered_claim = "The user database system processes background credentials."
    res = verifier._evaluate_single_claim(scattered_claim, unrelated_chunks)
    # Each chunk has only 1 word overlap, which must not qualify as composite evidence
    assert res.status != ClaimStatus.SUPPORTED
    assert res.confidence_score < 0.65


def test_clause_level_partial_hallucination_flagged(multi_module_chunks):
    """Verifies that a compound sentence containing one valid clause and one hallucinated clause is not marked supported."""
    verifier = ClaimVerifier(min_support_ratio=0.60)
    compound_claim = (
        "Module 1 handles document ingestion and preprocessing, "
        "while Module 9 operates a secret satellite communications relay in Antarctica."
    )
    res = verifier._evaluate_single_claim(compound_claim, multi_module_chunks)
    assert res.status != ClaimStatus.SUPPORTED
    assert res.confidence_score < 0.65


def test_citation_integrity_metadata(multi_module_chunks):
    """Verifies that supporting chunks preserve chunk IDs, document titles, and page numbers."""
    verifier = ClaimVerifier(min_support_ratio=0.60)
    claim = "Module 4 provides an explainable verification engine to detect hallucinations."
    res = verifier._evaluate_single_claim(claim, multi_module_chunks)
    assert res.status == ClaimStatus.SUPPORTED
    assert res.supporting_chunk_id == "chunk_mod4"
    assert res.document_title == "Technical Assessment Specification"
    assert res.page_number == 4
    assert res.evidence_snippet is not None
