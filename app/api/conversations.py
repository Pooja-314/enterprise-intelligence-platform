"""REST API endpoints for Conversational Memory, Session Management, and Multi-turn Dialogue."""
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.agent.orchestrator import AgentExecutionResult, agent_orchestrator
from app.agent.query_router import ConversationQueryRouter, RoutedQueryResult, conversation_router
from app.api.deps import get_current_user
from app.core.security import PromptInjectionDefender
from app.db.models import Conversation, Message, User
from app.db.session import get_db
from app.memory.manager import conversation_memory

router = APIRouter(prefix="/conversations", tags=["Conversational Memory"])


# ------------------------------------------------------------------------------
# Pydantic Request & Response Schemas
# ------------------------------------------------------------------------------

class ConversationCreateRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=255, description="Optional title for the conversation.")


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    citations: List[str] = Field(default_factory=list)
    msg_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ConversationResponse(BaseModel):
    id: str
    title: str
    user_id: Optional[int]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    messages_count: int = 0

    model_config = ConfigDict(from_attributes=True)


class ConversationDetailResponse(BaseModel):
    id: str
    title: str
    user_id: Optional[int]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse]
    summary: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class SendMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, description="User prompt or follow-up question.")
    max_steps: Optional[int] = Field(default=5, ge=1, le=10, description="Max reasoning iterations.")


class SendMessageResponse(BaseModel):
    conversation_id: str
    user_message: MessageResponse
    assistant_message: MessageResponse
    citations: List[str]
    tools_used: List[str]
    latency_ms: float
    status: str
    is_summarized: bool
    citation_details: List[Dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class SummarizeResponse(BaseModel):
    conversation_id: str
    summary: Optional[str]
    status: str


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------

@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(
    request: ConversationCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Creates a new persistent conversation isolated to the authenticated user."""
    conv = conversation_memory.create_conversation(
        db=db,
        user_id=current_user.id,
        title=request.title,
    )
    return ConversationResponse(
        id=conv.id,
        title=conv.title,
        user_id=conv.user_id,
        is_active=conv.is_active,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages_count=0,
    )


@router.get("", response_model=List[ConversationResponse])
def list_conversations(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Lists conversations belonging exclusively to the authenticated user."""
    conversations = conversation_memory.list_conversations(
        db=db,
        user_id=current_user.id,
        limit=limit,
        offset=offset,
    )

    results = []
    for c in conversations:
        count = db.query(Message).filter(Message.conversation_id == c.id).count()
        results.append(
            ConversationResponse(
                id=c.id,
                title=c.title,
                user_id=c.user_id,
                is_active=c.is_active,
                created_at=c.created_at,
                updated_at=c.updated_at,
                messages_count=count,
            )
        )
    return results


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation_details(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieves full conversation details and chronological message history, enforcing user isolation."""
    conv = conversation_memory.get_conversation(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id,
    )
    messages = conversation_memory.get_messages(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id,
    )

    summary_text = None
    msg_responses = []
    for m in messages:
        if m.role == "system" and m.msg_metadata.get("is_summary") is True:
            summary_text = m.content
        msg_responses.append(
            MessageResponse(
                id=m.id,
                conversation_id=m.conversation_id,
                role=m.role,
                content=m.content,
                citations=m.msg_metadata.get("citations", []),
                msg_metadata=m.msg_metadata,
                created_at=m.created_at,
            )
        )

    return ConversationDetailResponse(
        id=conv.id,
        title=conv.title,
        user_id=conv.user_id,
        is_active=conv.is_active,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=msg_responses,
        summary=summary_text,
    )


@router.post("/{conversation_id}/messages", response_model=SendMessageResponse)
def continue_conversation(
    conversation_id: str,
    request: SendMessageRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Submits a user message, runs the AI agent with conversational memory,
    persists both turns with telemetry and citations, and handles auto-summarization.
    """
    start_time = time.perf_counter()

    # 1. Verify access to conversation and sanitize input
    conv = conversation_memory.get_conversation(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id,
    )
    sanitized_prompt = PromptInjectionDefender.validate_or_raise(request.content)

    # 2. Persist the user message turn
    user_msg = conversation_memory.add_message(
        db=db,
        conversation_id=conv.id,
        role="user",
        content=sanitized_prompt,
        user_id=current_user.id,
    )

    # 3. Retrieve relevant conversational context (sliding window + summary)
    context = conversation_memory.get_relevant_context(
        db=db,
        conversation_id=conv.id,
        user_id=current_user.id,
    )

    # 4. Route query through ConversationQueryRouter
    routed_res: RoutedQueryResult = conversation_router.route_and_execute(
        db=db,
        query=sanitized_prompt,
        user_id=current_user.id,
        conversation_id=conv.id,
        conversation_context=context.formatted_history,
        max_steps=request.max_steps,
    )

    latency_ms = (time.perf_counter() - start_time) * 1000

    # 5. Persist assistant response turn with citation metadata and telemetry
    assistant_metadata = {
        "citations": routed_res.citations,
        "citation_details": getattr(routed_res, "citation_details", []),
        "tools_used": routed_res.tools_used,
        "tokens": routed_res.tokens,
        "latency_ms": round(latency_ms, 2),
        "status": routed_res.status,
        "intent": routed_res.intent.value if hasattr(routed_res.intent, "value") else str(routed_res.intent),
    }
    if routed_res.verification:
        assistant_metadata["verification"] = routed_res.verification

    assistant_msg = conversation_memory.add_message(
        db=db,
        conversation_id=conv.id,
        role="assistant",
        content=routed_res.answer,
        msg_metadata=assistant_metadata,
        user_id=current_user.id,
    )

    # 6. Check context window threshold and auto-summarize if dialogue has grown long
    summary = conversation_memory.summarize_and_compact(
        db=db,
        conversation_id=conv.id,
        user_id=current_user.id,
        threshold_turns=8,
        preserve_recent_turns=4,
    )

    return SendMessageResponse(
        conversation_id=conv.id,
        user_message=MessageResponse(
            id=user_msg.id,
            conversation_id=user_msg.conversation_id,
            role=user_msg.role,
            content=user_msg.content,
            citations=[],
            msg_metadata=user_msg.msg_metadata,
            created_at=user_msg.created_at,
        ),
        assistant_message=MessageResponse(
            id=assistant_msg.id,
            conversation_id=assistant_msg.conversation_id,
            role=assistant_msg.role,
            content=assistant_msg.content,
            citations=routed_res.citations,
            msg_metadata=assistant_msg.msg_metadata,
            created_at=assistant_msg.created_at,
        ),
        citations=routed_res.citations,
        citation_details=getattr(routed_res, "citation_details", []),
        tools_used=routed_res.tools_used,
        latency_ms=round(latency_ms, 2),
        status=routed_res.status,
        is_summarized=summary is not None or context.is_summarized,
    )


@router.delete("/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    hard_delete: bool = Query(default=False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Deletes or deactivates a conversation after enforcing user ownership."""
    success = conversation_memory.delete_conversation(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id,
        hard_delete=hard_delete,
    )
    return {"success": success, "conversation_id": conversation_id, "deleted": True}


@router.post("/{conversation_id}/summarize", response_model=SummarizeResponse)
def manually_summarize_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Explicitly triggers conversational memory summarization and context compaction."""
    summary = conversation_memory.summarize_and_compact(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id,
        threshold_turns=2,  # Compact if 2 or more dialogue turns exist
        preserve_recent_turns=2,
    )
    return SummarizeResponse(
        conversation_id=conversation_id,
        summary=summary,
        status="SUMMARIZED" if summary else "UNCHANGED",
    )
