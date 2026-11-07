"""Sanitized Inspection of verification metadata in Neon PostgreSQL.
Queries specific metadata columns only (id, created_at, msg_metadata).
Never retrieves Message.content or Conversation records.
Never prints rejection_reason, raw citation strings, prompts, generated responses, document text, credentials, or secrets.
"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.db.session import SessionLocal
from app.db.models import Message


def inspect_metadata():
    print("=" * 80)
    print("SANITIZED VERIFICATION METADATA AUDIT (COLUMN-ONLY QUERY)")
    print("=" * 80)

    db = SessionLocal()
    try:
        # Requirement 1 & 2: Column-only query selecting strictly id, created_at, msg_metadata.
        # Excludes Message.content and Conversation records completely.
        records = (
            db.query(Message.id, Message.created_at, Message.msg_metadata)
            .filter(Message.role == "assistant")
            .order_by(Message.created_at.desc())
            .limit(3)
            .all()
        )

        print(f"Inspecting metadata for {len(records)} recent assistant record(s):\n")

        for idx, (msg_id, created_at, msg_metadata) in enumerate(records, 1):
            meta = msg_metadata or {}
            ver = meta.get("verification") or {}
            citations = meta.get("citations") or []
            tools_used = meta.get("tools_used") or []
            latency_ms = meta.get("latency_ms")

            # Extract sanitized document IDs, chunk IDs, and pages only (no raw text)
            sanitized_sources = []
            for c in citations[:8]:
                if isinstance(c, dict):
                    sanitized_sources.append({
                        "doc_id": str(c.get("document_id", "N/A"))[:8],
                        "page": c.get("page_number", "?"),
                        "chunk_id": str(c.get("chunk_id", "N/A"))[:8],
                    })
                elif isinstance(c, str):
                    doc_match = re.search(r"Doc:\s*([a-zA-Z0-9_.-]+)", c)
                    page_match = re.search(r"Page:\s*(\d+)", c)
                    chunk_match = re.search(r"Chunk:\s*([a-zA-Z0-9_-]+)", c)
                    sanitized_sources.append({
                        "doc_id": doc_match.group(1)[:8] if doc_match else "N/A",
                        "page": page_match.group(1) if page_match else "?",
                        "chunk_id": chunk_match.group(1)[:8] if chunk_match else "N/A",
                    })

            # Requirement 4: Print strictly the permitted telemetry points
            print(f"--- Assistant Message #{idx} ---")
            print(f"Timestamp: {created_at}")
            print(f"Verification Status: {ver.get('is_verified', 'N/A')}")
            print(f"Grounding Ratio: {ver.get('confidence_score', 'N/A')}")
            print(f"Total Claims Count: {ver.get('total_claims', 0)}")
            print(f"Supported Claims Count: {ver.get('supported_claims', 0)}")
            print(f"Unsupported Claims Count: {ver.get('unsupported_claims', 0)}")
            print(f"Citation Count: {len(citations)}")
            if sanitized_sources:
                print("Sanitized Sources:")
                for s in sanitized_sources:
                    print(f"  • Doc ID: {s['doc_id']} | Page: {s['page']} | Chunk ID: {s['chunk_id']}")
            print(f"Tools Used: {tools_used}")
            print(f"Latency: {latency_ms} ms")
            print()

    finally:
        # Requirement 5: Strictly read-only; close session cleanly
        db.close()


if __name__ == "__main__":
    inspect_metadata()
