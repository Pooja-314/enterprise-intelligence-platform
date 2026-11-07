"""Context data structures for Conversational Memory."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ConversationTurn:
    """Represents a single conversational turn (user or assistant)."""
    id: str
    role: str
    content: str
    created_at: datetime
    citations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationalContext:
    """Structured memory context delivered to reasoning engines."""
    conversation_id: str
    turns: List[ConversationTurn]
    formatted_history: str
    summary: Optional[str] = None
    token_count_estimate: int = 0
    total_messages_count: int = 0
    is_summarized: bool = False
