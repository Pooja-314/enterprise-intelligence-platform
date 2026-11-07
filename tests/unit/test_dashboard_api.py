"""Unit tests for backend /api/v1/dashboard/stats endpoint."""
import pytest
from sqlalchemy.orm import Session

from app.api.dashboard import get_dashboard_statistics
from app.db.models import Conversation, Document, DocumentChunk, Message, ProcessingStatus, User


def test_dashboard_stats_endpoint(db_session: Session):
    user = User(id=501, username="dash_user", email="dash@corp.com", hashed_password="pwd", role="user")
    db_session.add(user)

    doc = Document(
        id="doc_dash_1",
        title="Dashboard Test Doc",
        filename="dash.pdf",
        file_type="pdf",
        file_size_bytes=1000,
        content_hash="h_dash_1",
        processing_status=ProcessingStatus.COMPLETED.value,
        doc_metadata={},
    )
    db_session.add(doc)
    db_session.commit()

    chunk = DocumentChunk(
        id="chk_dash_1",
        document_id="doc_dash_1",
        chunk_index=0,
        content="Dashboard chunk text.",
        token_count=10,
        embedding=None,
        chunk_metadata={},
    )
    conv = Conversation(id="conv_dash_1", title="Dash Conv", user_id=501, is_active=True)
    db_session.add_all([chunk, conv])
    db_session.commit()

    msg = Message(id="msg_dash_1", conversation_id="conv_dash_1", role="user", content="Hello", msg_metadata={})
    db_session.add(msg)
    db_session.commit()

    stats = get_dashboard_statistics(current_user=user, db=db_session)

    assert stats["documents"]["total"] >= 1
    assert stats["documents"]["completed"] >= 1
    assert stats["documents"]["total_chunks"] >= 1
    assert stats["conversations"]["total_user_conversations"] == 1
    assert stats["conversations"]["total_user_messages"] == 1
    assert "system_status" in stats
