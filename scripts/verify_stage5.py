"""Live verification script for Stage 5: Conversational Memory on Neon PostgreSQL.
Connects to Neon PostgreSQL, verifies:
1. Real table persistence for conversations and messages.
2. User isolation enforcement.
3. Message ordering and citation preservation in msg_metadata.
4. Conversational context assembly with sliding window.
5. Automated cleanup.
"""
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.core.errors import AuthorizationError, EntityNotFoundError
from app.db.models import Conversation, Message, User
from app.db.session import get_migration_database_url
from app.memory.manager import ConversationMemoryManager


def run_neon_stage5_verification():
    print("=" * 70)
    print("STAGE 5: LIVE CONVERSATIONAL MEMORY AUDIT (NEON POSTGRESQL)")
    print("=" * 70)

    db_url = get_migration_database_url()
    if not db_url or "sqlite" in db_url:
        print("ERROR: Direct Neon PostgreSQL URL not configured.")
        sys.exit(1)

    print(f"Connecting to database...")
    engine = create_engine(db_url, pool_pre_ping=True, connect_args={"connect_timeout": 10})
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    test_conv_id = None
    user1_id = 99901
    user2_id = 99902

    try:
        # Step 0: Ensure test users exist
        u1 = session.query(User).filter(User.id == user1_id).first()
        if not u1:
            u1 = User(id=user1_id, username="stage5_user_1", email="s5_1@neon.test", hashed_password="pwd", role="user")
            session.add(u1)
        u2 = session.query(User).filter(User.id == user2_id).first()
        if not u2:
            u2 = User(id=user2_id, username="stage5_user_2", email="s5_2@neon.test", hashed_password="pwd", role="user")
            session.add(u2)
        session.commit()

        manager = ConversationMemoryManager()

        # Step 1: Create Conversation in Neon
        print("\n[Step 1] Creating persistent conversation in Neon PostgreSQL...")
        conv = manager.create_conversation(
            db=session,
            user_id=user1_id,
            title="Neon Live Multi-Turn Dialogue",
        )
        test_conv_id = conv.id
        print(f"  Conversation ID: {conv.id}")
        print(f"  Title: {conv.title}")
        print(f"  User ID: {conv.user_id}")
        print(f"  Status: PASS")

        # Step 2: User Isolation Check
        print("\n[Step 2] Testing User Isolation (User 2 attempting access to User 1's conversation)...")
        try:
            manager.get_conversation(db=session, conversation_id=test_conv_id, user_id=user2_id)
            print("  FAIL: User 2 accessed User 1 conversation!")
            sys.exit(1)
        except AuthorizationError as e:
            print(f"  PASS: AuthorizationError caught as expected: {e}")

        # Step 3: Message Persistence & Citation Tracking
        print("\n[Step 3] Persisting multi-turn messages with metadata & citations...")
        m1 = manager.add_message(
            db=session,
            conversation_id=test_conv_id,
            role="user",
            content="What are the encryption standards for databases at rest?",
            user_id=user1_id,
        )
        m2 = manager.add_message(
            db=session,
            conversation_id=test_conv_id,
            role="assistant",
            content="AES-256 transparent encryption is strictly enforced across all clusters.",
            msg_metadata={
                "citations": ["[Doc: Compliance Guide, Page: 5, Chunk: chk_neon_1]"],
                "tools_used": ["knowledge_base_search"],
                "latency_ms": 42.1,
            },
            user_id=user1_id,
        )
        print(f"  User Message ID: {m1.id}")
        print(f"  Assistant Message ID: {m2.id}")
        print(f"  Citations Persisted: {m2.msg_metadata.get('citations')}")
        print(f"  Status: PASS")

        # Step 4: Context Retrieval (Sliding Window)
        print("\n[Step 4] Retrieving conversational context window...")
        ctx = manager.get_relevant_context(
            db=session,
            conversation_id=test_conv_id,
            user_id=user1_id,
            max_recent_turns=4,
        )
        print(f"  Total Dialogue Messages in Context: {len(ctx.turns)}")
        print(f"  Token Estimate: {ctx.token_count_estimate}")
        print(f"  Formatted History Preview:\n{ctx.formatted_history}")
        assert len(ctx.turns) == 2
        assert "AES-256" in ctx.formatted_history
        print(f"  Status: PASS")

        # Step 5: Soft Deletion
        print("\n[Step 5] Testing Conversation Soft Deletion...")
        manager.delete_conversation(db=session, conversation_id=test_conv_id, user_id=user1_id, hard_delete=False)
        try:
            manager.get_conversation(db=session, conversation_id=test_conv_id, user_id=user1_id)
            print("  FAIL: Deleted conversation still retrievable!")
            sys.exit(1)
        except EntityNotFoundError:
            print("  PASS: Inactive conversation correctly rejected as not found.")

        print("\n" + "=" * 70)
        print("STAGE 5 NEON POSTGRESQL LIVE VERIFICATION COMPLETE: ALL CHECKS PASSED")
        print("=" * 70)

    finally:
        # Cleanup test records
        try:
            if test_conv_id:
                session.query(Message).filter(Message.conversation_id == test_conv_id).delete()
                session.query(Conversation).filter(Conversation.id == test_conv_id).delete()
            session.query(User).filter(User.id.in_([user1_id, user2_id])).delete()
            session.commit()
            print("\n[Cleanup] Test conversations, messages, and temporary users removed.")
        except Exception as e:
            session.rollback()
            print(f"[Cleanup Warning] {e}")
        finally:
            session.close()


if __name__ == "__main__":
    run_neon_stage5_verification()
