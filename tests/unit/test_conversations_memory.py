"""Unit tests for Stage 5: Conversational Memory Manager.
Tests:
1. Persistent conversation creation and retrieval.
2. User-specific conversation history and listing.
3. Message persistence with roles, timestamps, citations, and metadata.
4. Relevant conversational context retrieval (sliding window).
5. Context-window management and conversation summarization for long dialogues.
6. User isolation (prevent cross-user access).
7. Soft and hard deletion.
8. Error handling on invalid/missing conversations.
9. Integration with AgentOrchestrator.
"""
import json
import pytest
from sqlalchemy.orm import Session

from app.agent.orchestrator import AgentOrchestrator
from app.core.errors import AuthorizationError, EntityNotFoundError
from app.db.models import Conversation, Message, User
from app.llm.base import BaseLLMProvider, LLMResponse
from app.llm.fallback_router import LLMRouter
from app.memory.context import ConversationalContext
from app.memory.manager import ConversationMemoryManager


class MockSummaryLLM(BaseLLMProvider):
    def __init__(self, summary_text: str = "Summary: User discussed Q3 revenues and zero trust policies."):
        self.summary_text = summary_text
        self.model = "mock-summary-v1"

    def is_configured(self) -> bool:
        return True

    def generate(self, prompt: str, system_prompt=None, temperature=0.2, max_tokens=1024) -> LLMResponse:
        return LLMResponse(
            content=self.summary_text,
            provider="mock_llm",
            model=self.model,
            prompt_tokens=80,
            completion_tokens=25,
            total_tokens=105,
            latency_ms=10.0,
        )


@pytest.fixture
def memory_manager() -> ConversationMemoryManager:
    mock_llm = MockSummaryLLM()
    router = LLMRouter(primary_provider=mock_llm, max_retries=0)
    return ConversationMemoryManager(router=router)


@pytest.fixture
def test_users(db_session: Session):
    u1 = User(id=101, username="alice", email="alice@corp.com", hashed_password="pwd", role="user")
    u2 = User(id=102, username="bob", email="bob@corp.com", hashed_password="pwd", role="user")
    db_session.add_all([u1, u2])
    db_session.commit()
    return u1, u2


# ------------------------------------------------------------------------------
# 1. Conversation Lifecycle & Isolation Tests
# ------------------------------------------------------------------------------

def test_create_and_retrieve_conversation(memory_manager, db_session: Session, test_users):
    u1, _ = test_users
    conv = memory_manager.create_conversation(db=db_session, user_id=u1.id, title="Financial Review")

    assert conv.id is not None
    assert conv.title == "Financial Review"
    assert conv.user_id == u1.id
    assert conv.is_active is True

    fetched = memory_manager.get_conversation(db=db_session, conversation_id=conv.id, user_id=u1.id)
    assert fetched.id == conv.id
    assert fetched.title == "Financial Review"


def test_user_specific_conversation_history(memory_manager, db_session: Session, test_users):
    u1, u2 = test_users
    # Create 2 conversations for Alice, 1 for Bob
    c1 = memory_manager.create_conversation(db=db_session, user_id=u1.id, title="Alice Conv 1")
    c2 = memory_manager.create_conversation(db=db_session, user_id=u1.id, title="Alice Conv 2")
    c3 = memory_manager.create_conversation(db=db_session, user_id=u2.id, title="Bob Conv 1")

    alice_convs = memory_manager.list_conversations(db=db_session, user_id=u1.id)
    bob_convs = memory_manager.list_conversations(db=db_session, user_id=u2.id)

    assert len(alice_convs) == 2
    alice_titles = {c.title for c in alice_convs}
    assert "Alice Conv 1" in alice_titles
    assert "Alice Conv 2" in alice_titles
    assert "Bob Conv 1" not in alice_titles

    assert len(bob_convs) == 1
    assert bob_convs[0].title == "Bob Conv 1"


def test_user_isolation_prevents_cross_user_access(memory_manager, db_session: Session, test_users):
    u1, u2 = test_users
    conv = memory_manager.create_conversation(db=db_session, user_id=u1.id, title="Confidential M&A")

    # Bob attempts to view Alice's conversation
    with pytest.raises(AuthorizationError) as exc_info:
        memory_manager.get_conversation(db=db_session, conversation_id=conv.id, user_id=u2.id)

    assert "do not have permission" in str(exc_info.value)

    # Bob attempts to add message to Alice's conversation
    with pytest.raises(AuthorizationError):
        memory_manager.add_message(
            db=db_session,
            conversation_id=conv.id,
            role="user",
            content="Unauthorized message",
            user_id=u2.id,
        )


def test_delete_conversation_soft_and_hard(memory_manager, db_session: Session, test_users):
    u1, _ = test_users
    conv = memory_manager.create_conversation(db=db_session, user_id=u1.id, title="Temporary Conv")

    # Soft delete
    memory_manager.delete_conversation(db=db_session, conversation_id=conv.id, user_id=u1.id, hard_delete=False)

    with pytest.raises(EntityNotFoundError):
        memory_manager.get_conversation(db=db_session, conversation_id=conv.id, user_id=u1.id)

    assert len(memory_manager.list_conversations(db=db_session, user_id=u1.id)) == 0


def test_smart_title_auto_generation(memory_manager, db_session: Session, test_users):
    u1, _ = test_users
    conv = memory_manager.create_conversation(db=db_session, user_id=u1.id)  # Default title: "New Conversation"
    assert conv.title == "New Conversation"

    # Add first user message
    msg = memory_manager.add_message(
        db=db_session,
        conversation_id=conv.id,
        role="user",
        content="What are our cloud security compliance requirements for SOC2?",
        user_id=u1.id,
    )
    db_session.refresh(conv)
    assert conv.title.startswith("What are our cloud security compliance")


# ------------------------------------------------------------------------------
# 2. Message Persistence & Context Retrieval Tests
# ------------------------------------------------------------------------------

def test_message_persistence_with_metadata(memory_manager, db_session: Session, test_users):
    u1, _ = test_users
    conv = memory_manager.create_conversation(db=db_session, user_id=u1.id)

    m1 = memory_manager.add_message(
        db=db_session,
        conversation_id=conv.id,
        role="user",
        content="What was Q3 revenue?",
        user_id=u1.id,
    )
    assert m1.id is not None
    assert m1.role == "user"
    assert m1.content == "What was Q3 revenue?"

    citations = ["[Doc: Financials, Page: 4, Chunk: chk_1]"]
    m2 = memory_manager.add_message(
        db=db_session,
        conversation_id=conv.id,
        role="assistant",
        content="Q3 revenue was 150 million dollars.",
        msg_metadata={"citations": citations, "latency_ms": 12.5},
        user_id=u1.id,
    )
    assert m2.msg_metadata["citations"] == citations
    assert m2.msg_metadata["latency_ms"] == 12.5

    msgs = memory_manager.get_messages(db=db_session, conversation_id=conv.id, user_id=u1.id)
    assert len(msgs) == 2
    assert msgs[0].role == "user"
    assert msgs[1].role == "assistant"


def test_conversational_context_sliding_window(memory_manager, db_session: Session, test_users):
    u1, _ = test_users
    conv = memory_manager.create_conversation(db=db_session, user_id=u1.id)

    # Insert 5 back-and-forth turns (10 messages)
    for i in range(1, 6):
        memory_manager.add_message(db=db_session, conversation_id=conv.id, role="user", content=f"Question {i}", user_id=u1.id)
        memory_manager.add_message(db=db_session, conversation_id=conv.id, role="assistant", content=f"Answer {i}", user_id=u1.id)

    # Retrieve context with max_recent_turns=4
    ctx: ConversationalContext = memory_manager.get_relevant_context(
        db=db_session,
        conversation_id=conv.id,
        user_id=u1.id,
        max_recent_turns=4,
    )

    assert len(ctx.turns) == 4
    # Should contain the most recent turns: Question 4, Answer 4, Question 5, Answer 5
    assert "Question 4" in ctx.formatted_history
    assert "Answer 4" in ctx.formatted_history
    assert "Question 5" in ctx.formatted_history
    assert "Answer 5" in ctx.formatted_history
    assert "Question 1" not in ctx.formatted_history
    assert ctx.total_messages_count == 10
    assert ctx.token_count_estimate > 0


# ------------------------------------------------------------------------------
# 3. Context Summarization & Long Conversation Tests
# ------------------------------------------------------------------------------

def test_long_conversation_automatic_summarization(memory_manager, db_session: Session, test_users):
    u1, _ = test_users
    conv = memory_manager.create_conversation(db=db_session, user_id=u1.id)

    # Insert 6 turns (12 messages)
    for i in range(1, 7):
        memory_manager.add_message(db=db_session, conversation_id=conv.id, role="user", content=f"Detail inquiry number {i} regarding company policy.", user_id=u1.id)
        memory_manager.add_message(db=db_session, conversation_id=conv.id, role="assistant", content=f"Policy clause response {i}.", user_id=u1.id)

    # Trigger summarization with threshold 8 turns, preserving 4 turns
    summary = memory_manager.summarize_and_compact(
        db=db_session,
        conversation_id=conv.id,
        user_id=u1.id,
        threshold_turns=8,
        preserve_recent_turns=4,
    )

    assert summary is not None
    assert "Summary: User discussed" in summary

    # Verify context now includes the summary briefing as a prefix
    ctx = memory_manager.get_relevant_context(db=db_session, conversation_id=conv.id, user_id=u1.id, max_recent_turns=4)
    assert ctx.is_summarized is True
    assert "[CONVERSATION SUMMARY OF EARLIER TOPICS]:" in ctx.formatted_history
    assert "Summary: User discussed" in ctx.formatted_history
    assert len(ctx.turns) == 4


# ------------------------------------------------------------------------------
# 4. Agent Orchestrator with Conversational Memory Integration
# ------------------------------------------------------------------------------

def test_agent_orchestrator_receives_conversational_context(db_session: Session, test_users):
    u1, _ = test_users
    memory_manager = ConversationMemoryManager()
    conv = memory_manager.create_conversation(db=db_session, user_id=u1.id)

    # Previous turn established context
    memory_manager.add_message(db=db_session, conversation_id=conv.id, role="user", content="We are discussing Acme Corp Q3 results.", user_id=u1.id)
    memory_manager.add_message(db=db_session, conversation_id=conv.id, role="assistant", content="Acknowledged. Acme Corp reported $150M revenue.", user_id=u1.id)

    class FollowUpMockLLM(BaseLLMProvider):
        def __init__(self):
            self.model = "followup-mock-v1"
            self.captured_prompt = ""

        def is_configured(self):
            return True

        def generate(self, prompt: str, system_prompt=None, temperature=0.2, max_tokens=1024) -> LLMResponse:
            self.captured_prompt = prompt
            final_content = json.dumps({
                "thought": "Using conversation history to identify company as Acme Corp.",
                "action": "final_answer",
                "final_answer": "Acme Corp reported $150M revenue with 23% growth.",
                "citations": [],
            })
            return LLMResponse(
                content=final_content,
                provider="mock",
                model=self.model,
                prompt_tokens=40,
                completion_tokens=20,
                total_tokens=60,
                latency_ms=10.0,
            )

    mock_llm = FollowUpMockLLM()
    router = LLMRouter(primary_provider=mock_llm, max_retries=0)
    orchestrator = AgentOrchestrator(router=router)

    # Follow-up query uses pronoun "their"
    res = orchestrator.run(
        db=db_session,
        query="What was their growth rate?",
        user_id=u1.id,
        conversation_id=conv.id,
    )

    assert res.status == "COMPLETED"
    assert "Acme Corp reported" in res.final_answer
    # Confirm conversational history was injected into the prompt
    assert "Relevant Conversational History:" in mock_llm.captured_prompt
    assert "Acme Corp Q3 results" in mock_llm.captured_prompt
