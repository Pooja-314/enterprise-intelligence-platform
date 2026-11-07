"""Conversation Memory Manager for multi-turn dialogue persistence, user isolation,
recency-based context retrieval, and automatic summarization.
"""
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.core.errors import AuthorizationError, EntityNotFoundError
from app.db.models import Conversation, Message
from app.llm.fallback_router import LLMRouter, llm_router
from app.memory.context import ConversationalContext, ConversationTurn

logger = logging.getLogger(__name__)


class ConversationMemoryManager:
    """Manages persistent conversational memory across multi-turn sessions."""

    SUMMARIZE_SYSTEM_PROMPT = (
        "You are an enterprise conversation summarization engine. "
        "Distill the provided conversation history into a concise, factual briefing. "
        "Retain all specific business metrics, company names, document titles, questions resolved, "
        "and unresolved topics. Avoid conversational filler."
    )

    def __init__(self, router: Optional[LLMRouter] = None):
        self.router = router or llm_router

    # --------------------------------------------------------------------------
    # 1. Conversation Lifecycle & CRUD
    # --------------------------------------------------------------------------

    def create_conversation(
        self,
        db: Session,
        user_id: Optional[int] = None,
        title: Optional[str] = None,
    ) -> Conversation:
        """Creates a new active conversation associated with a specific user."""
        conv = Conversation(
            id=str(uuid.uuid4()),
            title=title.strip() if title and title.strip() else "New Conversation",
            user_id=user_id,
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
        logger.info(f"Created conversation {conv.id} for user_id={user_id}")
        return conv

    def list_conversations(
        self,
        db: Session,
        user_id: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Conversation]:
        """Lists active conversations, strictly filtering by user_id to enforce user isolation."""
        query = db.query(Conversation).filter(Conversation.is_active.is_(True))
        if user_id is not None:
            query = query.filter(Conversation.user_id == user_id)
        return query.order_by(Conversation.updated_at.desc()).offset(offset).limit(limit).all()

    def get_conversation(
        self,
        db: Session,
        conversation_id: str,
        user_id: Optional[int] = None,
    ) -> Conversation:
        """Retrieves a conversation by ID, enforcing user isolation."""
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conv or not conv.is_active:
            raise EntityNotFoundError(f"Conversation with ID '{conversation_id}' not found.")

        # Strict User Isolation check
        if user_id is not None and conv.user_id is not None and conv.user_id != user_id:
            logger.warning(
                f"Unauthorized conversation access attempt: user {user_id} tried to access conv {conversation_id} owned by user {conv.user_id}"
            )
            raise AuthorizationError("You do not have permission to access this conversation.")

        return conv

    def delete_conversation(
        self,
        db: Session,
        conversation_id: str,
        user_id: Optional[int] = None,
        hard_delete: bool = False,
    ) -> bool:
        """Deletes a conversation after verifying user ownership."""
        conv = self.get_conversation(db=db, conversation_id=conversation_id, user_id=user_id)
        if hard_delete:
            db.delete(conv)
        else:
            conv.is_active = False
            conv.updated_at = datetime.now(timezone.utc)
        db.commit()
        logger.info(f"Deleted conversation {conversation_id} (hard_delete={hard_delete})")
        return True

    def update_title(
        self,
        db: Session,
        conversation_id: str,
        new_title: str,
        user_id: Optional[int] = None,
    ) -> Conversation:
        """Updates the title of a conversation."""
        conv = self.get_conversation(db=db, conversation_id=conversation_id, user_id=user_id)
        conv.title = new_title.strip()
        conv.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(conv)
        return conv

    # --------------------------------------------------------------------------
    # 2. Message Persistence & Turn Tracking
    # --------------------------------------------------------------------------

    def add_message(
        self,
        db: Session,
        conversation_id: str,
        role: str,
        content: str,
        msg_metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
    ) -> Message:
        """Persists a message to the conversation, updating conversation timestamp and default title."""
        conv = self.get_conversation(db=db, conversation_id=conversation_id, user_id=user_id)

        msg = Message(
            id=str(uuid.uuid4()),
            conversation_id=conv.id,
            role=role,
            content=content,
            msg_metadata=msg_metadata or {},
            created_at=datetime.now(timezone.utc),
        )
        db.add(msg)

        # Update conversation timestamp
        conv.updated_at = datetime.now(timezone.utc)

        # Smart title update from first user turn if title is still default
        if conv.title == "New Conversation" and role == "user":
            clean_first = content.strip().split("\n")[0]
            conv.title = clean_first[:45] + ("..." if len(clean_first) > 45 else "")

        db.commit()
        db.refresh(msg)
        return msg

    def get_messages(
        self,
        db: Session,
        conversation_id: str,
        user_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Message]:
        """Retrieves messages ordered chronologically."""
        self.get_conversation(db=db, conversation_id=conversation_id, user_id=user_id)
        return (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .limit(limit)
            .all()
        )

    # --------------------------------------------------------------------------
    # 3. Context Retrieval & Sliding Window
    # --------------------------------------------------------------------------

    def get_relevant_context(
        self,
        db: Session,
        conversation_id: str,
        user_id: Optional[int] = None,
        max_recent_turns: int = 6,
    ) -> ConversationalContext:
        """Assembles conversational memory context using a sliding window and consolidated summary."""
        self.get_conversation(db=db, conversation_id=conversation_id, user_id=user_id)

        all_messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .all()
        )

        if not all_messages:
            return ConversationalContext(
                conversation_id=conversation_id,
                turns=[],
                formatted_history="",
                summary=None,
                token_count_estimate=0,
                total_messages_count=0,
                is_summarized=False,
            )

        # Locate latest system summary message if one exists
        latest_summary: Optional[str] = None
        dialogue_messages: List[Message] = []
        for m in all_messages:
            if m.role == "system" and m.msg_metadata.get("is_summary") is True:
                latest_summary = m.content
            else:
                dialogue_messages.append(m)

        # Select recent sliding window of dialogue turns
        recent_messages = dialogue_messages[-max_recent_turns:] if len(dialogue_messages) > max_recent_turns else dialogue_messages

        turns: List[ConversationTurn] = []
        formatted_lines: List[str] = []

        if latest_summary:
            formatted_lines.append(f"[CONVERSATION SUMMARY OF EARLIER TOPICS]:\n{latest_summary}\n")

        formatted_lines.append("[RECENT CONVERSATION TURNS]:")
        for m in recent_messages:
            role_label = "User" if m.role == "user" else ("Assistant" if m.role == "assistant" else m.role.capitalize())
            citations = m.msg_metadata.get("citations", [])
            formatted_lines.append(f"{role_label}: {m.content}")

            turns.append(
                ConversationTurn(
                    id=m.id,
                    role=m.role,
                    content=m.content,
                    created_at=m.created_at,
                    citations=citations,
                    metadata=m.msg_metadata,
                )
            )

        formatted_history = "\n".join(formatted_lines)
        token_estimate = max(1, len(formatted_history) // 4)

        return ConversationalContext(
            conversation_id=conversation_id,
            turns=turns,
            formatted_history=formatted_history,
            summary=latest_summary,
            token_count_estimate=token_estimate,
            total_messages_count=len(dialogue_messages),
            is_summarized=latest_summary is not None,
        )

    # --------------------------------------------------------------------------
    # 4. Context-Window Management & Auto-Summarization
    # --------------------------------------------------------------------------

    def summarize_and_compact(
        self,
        db: Session,
        conversation_id: str,
        user_id: Optional[int] = None,
        threshold_turns: int = 8,
        preserve_recent_turns: int = 4,
    ) -> Optional[str]:
        """Automatically compresses older conversation history into a consolidated memory summary.
        Preserves the most recent turns verbatim while summarizing earlier turns to keep
        context within LLM token budgets.
        """
        self.get_conversation(db=db, conversation_id=conversation_id, user_id=user_id)

        messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .all()
        )

        dialogue_msgs = [m for m in messages if not (m.role == "system" and m.msg_metadata.get("is_summary"))]

        # Only summarize if dialogue length exceeds threshold
        if len(dialogue_msgs) <= threshold_turns:
            return None

        # Turns to be summarized vs turns to preserve
        turns_to_summarize = dialogue_msgs[:-preserve_recent_turns]
        if not turns_to_summarize:
            return None

        transcript_lines = []
        for m in turns_to_summarize:
            role_name = "User" if m.role == "user" else "Assistant"
            transcript_lines.append(f"{role_name}: {m.content}")
        transcript_text = "\n".join(transcript_lines)

        user_prompt = (
            f"Please summarize the following conversation history into key factual bullet points "
            f"and decisions:\n\n{transcript_text}"
        )

        try:
            llm_response = self.router.generate(
                prompt=user_prompt,
                system_prompt=self.SUMMARIZE_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=600,
            )
            summary_content = llm_response.content.strip()
        except Exception as exc:
            logger.warning(f"Summarization LLM call failed ({exc}); skipping compaction for turn.")
            return None

        # Upsert summary message
        existing_summary_msg = (
            db.query(Message)
            .filter(
                Message.conversation_id == conversation_id,
                Message.role == "system",
            )
            .first()
        )

        if existing_summary_msg and existing_summary_msg.msg_metadata.get("is_summary"):
            existing_summary_msg.content = summary_content
            existing_summary_msg.msg_metadata = {
                "is_summary": True,
                "compacted_turns": len(turns_to_summarize),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        else:
            summary_msg = Message(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                role="system",
                content=summary_content,
                msg_metadata={
                    "is_summary": True,
                    "compacted_turns": len(turns_to_summarize),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                created_at=datetime.now(timezone.utc),
            )
            db.add(summary_msg)

        db.commit()
        logger.info(f"Summarized {len(turns_to_summarize)} turns for conversation {conversation_id}.")
        return summary_content


# Global default instance
conversation_memory = ConversationMemoryManager()
