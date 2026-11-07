"""Dashboard analytics and telemetry aggregation endpoints."""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.telemetry import telemetry
from app.db.models import AuditLog, Conversation, Document, DocumentChunk, Message, User
from app.db.session import check_db_connection, get_db

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/stats")
def get_dashboard_statistics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Returns real operational analytics, document counts, and recent activity from the database."""
    # 1. Document metrics
    total_docs = db.query(Document).count()
    completed_docs = db.query(Document).filter(Document.processing_status == "COMPLETED").count()
    failed_docs = db.query(Document).filter(Document.processing_status == "FAILED").count()
    total_chunks = db.query(DocumentChunk).count()

    # 2. Document file type breakdown
    docs = db.query(Document.file_type).all()
    file_type_counts: Dict[str, int] = {}
    for d in docs:
        ft = d[0].lower() if d[0] else "unknown"
        file_type_counts[ft] = file_type_counts.get(ft, 0) + 1

    # 3. User Conversation metrics
    user_conversations_count = (
        db.query(Conversation)
        .filter(Conversation.user_id == current_user.id, Conversation.is_active.is_(True))
        .count()
    )

    user_messages_count = (
        db.query(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .filter(Conversation.user_id == current_user.id, Conversation.is_active.is_(True))
        .count()
    )

    # 4. Telemetry and Latency metrics
    telemetry_summary = telemetry.get_summary()

    # 5. Recent Activity from Audit Logs
    audit_records = (
        db.query(AuditLog)
        .order_by(AuditLog.created_at.desc())
        .limit(15)
        .all()
    )
    activity_feed = [
        {
            "id": a.id,
            "action": a.action,
            "resource_type": a.resource_type,
            "resource_id": a.resource_id,
            "user_id": a.user_id,
            "created_at": a.created_at.isoformat() if a.created_at else None,
            "status": a.details.get("status", "SUCCESS"),
            "latency_ms": a.details.get("latency_ms"),
        }
        for a in audit_records
    ]

    db_health = check_db_connection()

    return {
        "documents": {
            "total": total_docs,
            "completed": completed_docs,
            "failed": failed_docs,
            "total_chunks": total_chunks,
            "file_types": file_type_counts,
        },
        "conversations": {
            "total_user_conversations": user_conversations_count,
            "total_user_messages": user_messages_count,
        },
        "telemetry": telemetry_summary,
        "recent_activity": activity_feed,
        "system_status": {
            "database_connected": db_health.get("connected", False),
            "database_dialect": db_health.get("dialect", "unknown"),
        },
    }
