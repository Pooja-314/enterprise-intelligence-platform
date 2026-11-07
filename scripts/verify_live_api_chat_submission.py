import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import httpx
from app.config import settings

def run_test():
    base_url = "http://127.0.0.1:8000/api/v1"
    
    # 1. Login
    resp = httpx.post(
        f"{base_url}/auth/login",
        json={"username": settings.ADMIN_USERNAME, "password": settings.ADMIN_PASSWORD},
        timeout=15.0,
    )
    if resp.status_code != 200:
        print("Login failed with status:", resp.status_code)
        return
    token = resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 2. Create New Conversation
    conv_resp = httpx.post(f"{base_url}/conversations", json={"title": "Assessment Test"}, headers=headers, timeout=15.0)
    if conv_resp.status_code not in (200, 201):
        print("Create conversation failed:", conv_resp.status_code)
        return
    conv_id = conv_resp.json()["id"]
    print("New Conversation Created:", conv_id)
    
    # 3. Send the exact query
    query = (
        "Explain the five core modules required by the technical assessment. "
        "For each module, describe its key functional requirements and cite the relevant source document sections or pages."
    )
    print("Submitting query through live backend endpoint...")
    msg_resp = httpx.post(
        f"{base_url}/conversations/{conv_id}/messages",
        json={"content": query, "max_steps": 5},
        headers=headers,
        timeout=120.0,
    )
    print("Message Endpoint HTTP Status:", msg_resp.status_code)
    if msg_resp.status_code != 200:
        print("Error details:", msg_resp.status_code, msg_resp.text[:300])
        return
    
    data = msg_resp.json()
    assistant_msg = data.get("assistant_message", {})
    content = assistant_msg.get("content", "")
    meta = assistant_msg.get("msg_metadata", {})
    
    is_insufficient = "INSUFFICIENT_EVIDENCE" in content.upper()
    verif = meta.get("verification", {})
    citations = meta.get("citation_details") or meta.get("citations", [])
    tools_used = meta.get("tools_used", [])
    latency_ms = meta.get("latency_ms", 0)
    
    print("\n--- LIVE BACKEND API RESPONSE SUMMARY ---")
    print("Contains 'INSUFFICIENT_EVIDENCE':", is_insufficient)
    print("Verification Confidence Score:", verif.get("confidence_score"))
    print("Supported Claims:", verif.get("supported_claims"), "/", verif.get("total_claims"))
    print("Hallucination Detected:", verif.get("hallucination_detected"))
    print("Tools Used:", tools_used)
    print("Latency ms:", latency_ms)
    print("Citations Rendered Count:", len(citations))
    for idx, c in enumerate(citations[:5], 1):
        print(f"  [{idx}] Doc: {c.get('document_title')} | Page: {c.get('page_number')} | Chunk: {c.get('chunk_id')}")

if __name__ == "__main__":
    run_test()
