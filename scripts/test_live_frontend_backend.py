"""Live integration test: Starts FastAPI backend and executes BackendApiClient operations over real HTTP."""
import os
import sys
import time
import threading
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from app.main import app
from app.config import settings
from frontend.api_client import BackendApiClient


class ServerThread(threading.Thread):
    def __init__(self, app, host="127.0.0.1", port=8001):
        super().__init__(daemon=True)
        self.server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))

    def run(self):
        self.server.run()

    def stop(self):
        self.server.should_exit = True


def test_live_backend_interaction():
    print("=" * 70)
    print("LIVE FRONTEND CLIENT <-> FASTAPI BACKEND INTEGRATION TEST")
    print("=" * 70)

    port = 8001
    server = ServerThread(app, host="127.0.0.1", port=port)
    server.start()

    # Wait for server ready
    time.sleep(1.5)

    client = BackendApiClient(base_url=f"http://127.0.0.1:{port}/api/v1")

    try:
        # 1. Health check
        print("\n[Check 1] Probing /health endpoint...")
        health = client.get_health()
        print(f"  App: {health.get('app_name')}")
        print(f"  Status: {health.get('status')}")
        assert health.get("app_name") == settings.APP_NAME
        print("  PASS")

        # 2. Login as admin
        print("\n[Check 2] Authenticating through /auth/login...")
        login_res = client.login(username=settings.ADMIN_USERNAME, password=settings.ADMIN_PASSWORD)
        print(f"  User: {login_res.get('username')}")
        print(f"  Role: {login_res.get('role')}")
        print(f"  Token Prefix: {login_res.get('access_token')[:20]}...")
        assert "access_token" in login_res
        print("  PASS")

        # 3. Fetch User Profile
        print("\n[Check 3] Verifying authenticated profile via /auth/me...")
        me = client.get_me()
        print(f"  Authenticated Username: {me.get('username')}")
        print(f"  Role: {me.get('role')}")
        assert me.get("username") == settings.ADMIN_USERNAME
        print("  PASS")

        # 4. Upload Document
        print("\n[Check 4] Uploading test document via /documents/upload...")
        doc_content = b"Zero trust cloud security architecture mandates mutual TLS and biometric MFA."
        upload_res = client.upload_document(filename="cloud_security.txt", file_bytes=doc_content)
        doc_id = upload_res.get("document_id")
        print(f"  Document ID: {doc_id}")
        print(f"  Total Chunks: {upload_res.get('total_chunks')}")
        print(f"  Status: {upload_res.get('status')}")
        assert upload_res.get("total_chunks") >= 1
        print("  PASS")

        # 5. Create Conversation
        print("\n[Check 5] Creating persistent conversation via /conversations...")
        conv_res = client.create_conversation(title="Live Client Test Conv")
        conv_id = conv_res.get("id")
        print(f"  Conversation ID: {conv_id}")
        print(f"  Title: {conv_res.get('title')}")
        assert conv_id is not None
        print("  PASS")

        # 6. Fetch Dashboard Stats
        print("\n[Check 6] Fetching dashboard operational statistics via /dashboard/stats...")
        stats = client.get_dashboard_stats()
        print(f"  Total Documents: {stats['documents']['total']}")
        print(f"  Total Chunks: {stats['documents']['total_chunks']}")
        print(f"  User Conversations: {stats['conversations']['total_user_conversations']}")
        assert stats["documents"]["total"] >= 1
        print("  PASS")

        # 7. Cleanup
        print("\n[Check 7] Deleting test document and conversation...")
        client.delete_document(doc_id)
        client.delete_conversation(conv_id, hard_delete=True)
        print("  Cleanup: PASS")

        print("\n" + "=" * 70)
        print("LIVE CLIENT INTEGRATION TEST COMPLETE: ALL CHECKS PASSED")
        print("=" * 70)

    finally:
        server.stop()


if __name__ == "__main__":
    test_live_backend_interaction()
