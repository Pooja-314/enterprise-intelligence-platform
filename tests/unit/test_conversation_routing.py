"""Regression and unit tests for AI Chat and Conversation Query Routing.
Verifies:
1. Intent classification (CONVERSATIONAL, KNOWLEDGE_BASE_RAG, DOCUMENT_ANALYSIS, MULTI_STEP_AGENT).
2. Document questions invoke Knowledge Base Search (Hybrid RAG) and return verified citations.
3. Conversational greetings do not invoke the agent loop or execute unnecessary tools.
4. Ungrounded queries return explicit INSUFFICIENT_EVIDENCE responses.
5. Tool and inference failures return meaningful errors rather than misleading iteration limits.
6. Agent iteration limit is enforced only when iterations are genuinely exhausted.
7. Groq tool_use_failed recovery extracts tool calls from failed_generation.
"""
import json
import pytest
from unittest.mock import MagicMock
from sqlalchemy.orm import Session

from app.agent.orchestrator import AgentExecutionResult, AgentOrchestrator, AgentTraceStep
from app.agent.query_router import ConversationQueryRouter, QueryIntent, RoutedQueryResult
from app.core.errors import LLMProviderError
from app.db.models import User
from app.llm.base import BaseLLMProvider, LLMResponse
from app.llm.fallback_router import LLMRouter
from app.llm.groq_provider import GroqProvider
from app.rag.generator import Citation, RAGGenerator, RAGResponse
from app.rag.retrieval import RetrievedChunk
from app.rag.verifier import ClaimResult, ClaimStatus, VerificationReport


class MockLLM(BaseLLMProvider):
    def __init__(self, content: str = "Test response"):
        self.content = content
        self.model = "test-mock-llm"

    def is_configured(self) -> bool:
        return True

    def generate(self, prompt: str, system_prompt=None, temperature=0.2, max_tokens=1024) -> LLMResponse:
        return LLMResponse(
            content=self.content,
            provider="mock",
            model=self.model,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            latency_ms=12.0,
        )


# ------------------------------------------------------------------------------
# 1. Intent Classification Tests
# ------------------------------------------------------------------------------

def test_intent_classification_conversational():
    router = ConversationQueryRouter()

    greetings = [
        "hi",
        "Hello!",
        "hey there",
        "Good morning",
        "good afternoon",
        "who are you?",
        "what can you do",
        "What are your capabilities?",
        "thank you very much",
        "thanks",
        "bye",
        "Goodbye",
        "cool",
        "awesome",
    ]
    for q in greetings:
        assert router.classify_intent(q) == QueryIntent.CONVERSATIONAL, f"Failed for: {q}"


def test_intent_classification_document_questions():
    router = ConversationQueryRouter()

    questions = [
        "Explain the five core modules required by the technical assessment. For each module, describe its key functional requirements and cite the relevant source document sections or pages.",
        "What is the system architecture of the enterprise platform?",
        "What does the document say about pgvector distance metrics?",
        "How are JWT tokens validated?",
        "List all deliverables mentioned on page 3",
        "What are the requirements for Stage 4 AI Agent?",
    ]
    for q in questions:
        assert router.classify_intent(q) == QueryIntent.KNOWLEDGE_BASE_RAG, f"Failed for: {q}"


def test_intent_classification_multi_step_agent():
    router = ConversationQueryRouter()

    agent_queries = [
        "Generate a structured report on compliance risks across all documents",
        "Create a comprehensive report detailing the security posture",
        "Build a report on Q3 financial metrics",
        "Run autonomous agent to investigate and report findings",
        "Multi-step analysis and report generation",
    ]
    for q in agent_queries:
        assert router.classify_intent(q) == QueryIntent.MULTI_STEP_AGENT, f"Failed for: {q}"


def test_intent_classification_document_analysis():
    router = ConversationQueryRouter()

    uuid = "38981f72-3252-4152-860e-5d1fc3ee5b9e"
    analysis_queries = [
        f"Analyze document {uuid}",
        f"Summarize document {uuid}",
        f"Document {uuid} analysis",
        f"Please inspect document {uuid} and extract key findings",
    ]
    for q in analysis_queries:
        assert router.classify_intent(q) == QueryIntent.DOCUMENT_ANALYSIS, f"Failed for: {q}"


# ------------------------------------------------------------------------------
# 2. Conversational Greeting Execution (No Agent Loop)
# ------------------------------------------------------------------------------

def test_conversational_greeting_does_not_invoke_agent_loop(db_session: Session):
    mock_llm = MockLLM("Hello! I am your Enterprise Intelligence AI Assistant. How can I help you today?")
    llm_r = LLMRouter(primary_provider=mock_llm, max_retries=0)
    mock_orchestrator = MagicMock()
    mock_rag = MagicMock()

    router = ConversationQueryRouter(
        rag=mock_rag,
        orchestrator=mock_orchestrator,
        router=llm_r,
    )

    result = router.route_and_execute(db=db_session, query="Hello! How are you?")

    assert result.intent == QueryIntent.CONVERSATIONAL
    assert result.status == "COMPLETED"
    assert result.tools_used == []
    assert result.citations == []
    assert "Hello!" in result.answer

    # Orchestrator and RAG must NOT have been called
    mock_orchestrator.run.assert_not_called()
    mock_rag.query.assert_not_called()


# ------------------------------------------------------------------------------
# 3. Document Question Invokes Knowledge Base Search with Verified Citations
# ------------------------------------------------------------------------------

def test_document_question_invokes_knowledge_base_search_with_citations(db_session: Session):
    mock_rag = MagicMock()
    mock_rag.query.return_value = RAGResponse(
        answer=(
            "The five core modules are: 1. Foundational Architecture, 2. Secure Document Ingestion, "
            "3. Hybrid Retrieval & Grounded RAG, 4. Autonomous AI Agent, 5. Conversational Memory."
        ),
        citations=[
            Citation(
                document_id="doc_assessment",
                document_title="AI_Engineer_Technical_Assessment.pdf",
                chunk_id="chk_mod_1",
                page_number=1,
                snippet="Module 1: Architecture",
            ),
            Citation(
                document_id="doc_assessment",
                document_title="AI_Engineer_Technical_Assessment.pdf",
                chunk_id="chk_mod_2",
                page_number=2,
                snippet="Module 2: Ingestion",
            ),
        ],
        provider="groq",
        model="openai/gpt-oss-20b",
        verification=VerificationReport(
            is_verified=True,
            confidence_score=0.98,
            total_claims=5,
            supported_claims=5,
            partially_supported_claims=0,
            unsupported_claims=0,
            claims=[],
            hallucination_detected=False,
        ),
        retrieved_chunks=[],
        is_insufficient_evidence=False,
        latency_ms=850.0,
    )

    router = ConversationQueryRouter(rag=mock_rag)

    query = (
        "Explain the five core modules required by the technical assessment. "
        "For each module, describe its key functional requirements and cite the relevant source document sections or pages."
    )
    result = router.route_and_execute(db=db_session, query=query)

    assert result.intent == QueryIntent.KNOWLEDGE_BASE_RAG
    assert result.status == "COMPLETED"
    assert result.tools_used == ["knowledge_base_search"]
    assert len(result.citations) == 2
    assert "[Doc: AI_Engineer_Technical_Assessment.pdf, Page: 1, Chunk: chk_mod_1]" in result.citations
    assert "The five core modules are" in result.answer
    assert result.verification is not None
    assert result.verification["is_verified"] is True

    # Confirm RAG was called with the user query and sufficient chunk budget (>= 8)
    mock_rag.query.assert_called_once()
    assert mock_rag.query.call_args[1]["query_text"] == query
    assert mock_rag.query.call_args[1]["top_k"] >= 8


def test_rag_routing_enforces_minimum_chunk_budget_when_settings_low(db_session: Session, monkeypatch):
    """Verifies that RAG routing enforces at least top_k=8 even when TOP_K_FINAL=5 in settings."""
    from app.config import settings
    monkeypatch.setattr(settings, "TOP_K_FINAL", 5)

    mock_rag = MagicMock()
    mock_rag.query.return_value = RAGResponse(
        answer="Verified grounded answer.",
        citations=[],
        provider="groq",
        model="test-model",
        verification=VerificationReport(
            is_verified=True,
            confidence_score=0.9,
            total_claims=1,
            supported_claims=1,
            partially_supported_claims=0,
            unsupported_claims=0,
            claims=[],
            hallucination_detected=False,
        ),
        retrieved_chunks=[],
        is_insufficient_evidence=False,
        latency_ms=100.0,
    )
    router = ConversationQueryRouter(rag=mock_rag)
    router.route_and_execute(db=db_session, query="Explain the five core modules...")

    mock_rag.query.assert_called_once()
    assert mock_rag.query.call_args[1]["top_k"] >= 8


# ------------------------------------------------------------------------------
# 4. Ungrounded Document Question Returns Insufficient Evidence
# ------------------------------------------------------------------------------

def test_document_question_ungrounded_returns_insufficient_evidence(db_session: Session):
    mock_rag = MagicMock()
    mock_rag.query.return_value = RAGResponse(
        answer="INSUFFICIENT_EVIDENCE: The uploaded documents do not contain adequate information to answer this question.",
        citations=[],
        provider="groq",
        model="openai/gpt-oss-20b",
        verification=VerificationReport(
            is_verified=False,
            confidence_score=0.0,
            total_claims=0,
            supported_claims=0,
            partially_supported_claims=0,
            unsupported_claims=0,
            claims=[],
            hallucination_detected=False,
        ),
        retrieved_chunks=[],
        is_insufficient_evidence=True,
        latency_ms=250.0,
    )

    router = ConversationQueryRouter(rag=mock_rag)
    result = router.route_and_execute(db=db_session, query="What is the secret recipe for quantum teleportation?")

    assert result.intent == QueryIntent.KNOWLEDGE_BASE_RAG
    assert "INSUFFICIENT_EVIDENCE" in result.answer
    assert result.citations == []
    assert result.tools_used == ["knowledge_base_search"]


# ------------------------------------------------------------------------------
# 5. Tool & Provider Failure Returns Meaningful Error (Not Iteration Limit)
# ------------------------------------------------------------------------------

def test_rag_search_failure_returns_meaningful_error(db_session: Session):
    mock_rag = MagicMock()
    mock_rag.query.side_effect = RuntimeError("Database connection pool exhausted")

    router = ConversationQueryRouter(rag=mock_rag)
    result = router.route_and_execute(db=db_session, query="What are the five core modules?")

    assert result.status == "FAILED"
    assert "Database connection pool exhausted" in result.error
    assert "Knowledge base search encountered an error" in result.answer
    assert "maximum reasoning iterations" not in result.answer


def test_agent_orchestrator_step_failure_returns_meaningful_error(db_session: Session):
    class FailingLLM(BaseLLMProvider):
        def is_configured(self) -> bool:
            return True

        def generate(self, prompt, **kwargs):
            raise LLMProviderError("Groq upstream connection reset")

    failing_router = LLMRouter(primary_provider=FailingLLM(), fallback_provider=FailingLLM(), max_retries=0)
    orchestrator = AgentOrchestrator(router=failing_router, max_iterations=5)

    result = orchestrator.run(db=db_session, query="Analyze data and generate report")

    assert result.status == "FAILED"
    assert result.error is not None
    assert "Groq upstream connection reset" in result.error or "failed to generate" in result.error
    assert "maximum reasoning iterations" not in result.final_answer
    assert "Agent execution encountered an error" in result.final_answer


def test_agent_orchestrator_iteration_limit_enforced_only_when_exhausted(db_session: Session):
    """When an agent genuinely exhausts all allowed reasoning iterations without an error or final answer,
    it should report MAX_STEPS_REACHED.
    """
    class LoopingLLM(BaseLLMProvider):
        def is_configured(self) -> bool:
            return True

        def generate(self, prompt, **kwargs):
            # Model keeps calling tool without ever emitting final_answer
            return LLMResponse(
                content=json.dumps({
                    "thought": "I should search for more information.",
                    "action": "knowledge_base_search",
                    "action_input": {"query": "system architecture"},
                }),
                provider="mock",
                model="looping-llm",
                prompt_tokens=50,
                completion_tokens=20,
                total_tokens=70,
                latency_ms=10.0,
            )

    mock_tool = MagicMock()
    mock_tool.name = "knowledge_base_search"
    mock_tool.description = "Search KB"
    mock_tool.get_schema.return_value = {"name": "knowledge_base_search", "parameters": {}}
    from pydantic import BaseModel
    class EmptyArgs(BaseModel):
        query: str = ""
    mock_tool.args_schema = EmptyArgs
    mock_tool.run.return_value = {"results": []}

    looping_router = LLMRouter(primary_provider=LoopingLLM(), max_retries=0)
    orchestrator = AgentOrchestrator(router=looping_router, tools=[mock_tool], max_iterations=3)

    result = orchestrator.run(db=db_session, query="Continuous search inquiry", max_steps=3)

    assert result.status == "MAX_STEPS_REACHED"
    assert len(result.steps) == 3
    assert len(result.tools_used) == 1
    assert "maximum reasoning iterations (3)" in result.final_answer


# ------------------------------------------------------------------------------
# 6. Groq Tool Use Failed Recovery Test
# ------------------------------------------------------------------------------

def test_groq_tool_use_failed_recovery():
    provider = GroqProvider(api_key="gsk_test_key_mock", model="openai/gpt-oss-20b")

    mock_client = MagicMock()
    # Emulate Groq's tool_use_failed BadRequestError
    class MockBadRequest(Exception):
        def __init__(self):
            super().__init__("Error code: 400 - tool_use_failed")
            self.body = {
                "error": {
                    "message": "Tool choice is none, but model called a tool",
                    "type": "invalid_request_error",
                    "code": "tool_use_failed",
                    "failed_generation": '{"name": "knowledge_base_search", "arguments": {"query": "five core modules"}}',
                }
            }

    mock_client.chat.completions.create.side_effect = MockBadRequest()
    provider._client = mock_client

    resp = provider.generate("Explain the five core modules")

    assert resp.provider == "groq"
    assert "knowledge_base_search" in resp.content
    parsed = json.loads(resp.content)
    assert parsed["name"] == "knowledge_base_search"
    assert parsed["arguments"]["query"] == "five core modules"
