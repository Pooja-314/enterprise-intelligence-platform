"""HTTP Client for Enterprise Intelligence Platform Backend REST API."""
import os
from typing import Any, Dict, List, Optional
import httpx


class BackendApiError(Exception):
    """Custom exception raised when backend returns an error."""
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class BackendApiClient:
    """Synchronous HTTP client for Streamlit frontend calling FastAPI backend."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000/api/v1")
        # Ensure trailing slash stripped
        self.base_url = self.base_url.rstrip("/")
        self.token: Optional[str] = None
        self.timeout = httpx.Timeout(60.0, connect=10.0)

    def set_token(self, token: str) -> None:
        """Sets the Bearer JWT token for subsequent requests."""
        self.token = token

    def clear_token(self) -> None:
        """Clears the authentication token upon logout."""
        self.token = None

    def _get_headers(self, custom_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if custom_headers:
            headers.update(custom_headers)
        return headers

    def _handle_response(self, response: httpx.Response) -> Any:
        try:
            data = response.json()
        except Exception:
            data = {"message": response.text}

        if response.is_error:
            msg = data.get("message") or data.get("detail") or f"HTTP {response.status_code} Error"
            details = data.get("details", {})
            raise BackendApiError(message=str(msg), status_code=response.status_code, details=details)

        return data

    # --------------------------------------------------------------------------
    # 1. Authentication
    # --------------------------------------------------------------------------

    def login(self, username: str, password: str) -> Dict[str, Any]:
        url = f"{self.base_url}/auth/login"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json={"username": username, "password": password})
            data = self._handle_response(resp)
            if "access_token" in data:
                self.set_token(data["access_token"])
            return data

    def register(self, username: str, email: str, password: str) -> Dict[str, Any]:
        url = f"{self.base_url}/auth/register"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json={"username": username, "email": email, "password": password})
            return self._handle_response(resp)

    def get_me(self) -> Dict[str, Any]:
        url = f"{self.base_url}/auth/me"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, headers=self._get_headers())
            return self._handle_response(resp)

    # --------------------------------------------------------------------------
    # 2. System & Dashboard
    # --------------------------------------------------------------------------

    def get_health(self) -> Dict[str, Any]:
        # Health is mounted at root /health
        root_url = self.base_url.replace("/api/v1", "")
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{root_url}/health")
            return self._handle_response(resp)

    def get_dashboard_stats(self) -> Dict[str, Any]:
        url = f"{self.base_url}/dashboard/stats"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, headers=self._get_headers())
            return self._handle_response(resp)

    def get_telemetry(self) -> Dict[str, Any]:
        url = f"{self.base_url}/telemetry"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, headers=self._get_headers())
            return self._handle_response(resp)

    # --------------------------------------------------------------------------
    # 3. Document Management
    # --------------------------------------------------------------------------

    def upload_document(self, filename: str, file_bytes: bytes, allow_duplicate: bool = False) -> Dict[str, Any]:
        url = f"{self.base_url}/documents/upload"
        params = {"allow_duplicate": str(allow_duplicate).lower()}
        files = {"file": (filename, file_bytes, "application/octet-stream")}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, params=params, files=files, headers=self._get_headers())
            return self._handle_response(resp)

    def list_documents(self, limit: int = 50, offset: int = 0, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/documents"
        params = {"limit": limit, "offset": offset}
        if status_filter:
            params["status_filter"] = status_filter
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, params=params, headers=self._get_headers())
            return self._handle_response(resp)

    def get_document(self, document_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/documents/{document_id}"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, headers=self._get_headers())
            return self._handle_response(resp)

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/documents/{document_id}/chunks"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, headers=self._get_headers())
            return self._handle_response(resp)

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/documents/{document_id}"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.delete(url, headers=self._get_headers())
            return self._handle_response(resp)

    # --------------------------------------------------------------------------
    # 4. Conversations & Chat
    # --------------------------------------------------------------------------

    def create_conversation(self, title: Optional[str] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/conversations"
        payload = {"title": title} if title else {}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload, headers=self._get_headers())
            return self._handle_response(resp)

    def list_conversations(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/conversations"
        params = {"limit": limit, "offset": offset}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, params=params, headers=self._get_headers())
            return self._handle_response(resp)

    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/conversations/{conversation_id}"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, headers=self._get_headers())
            return self._handle_response(resp)

    def send_message(self, conversation_id: str, content: str, max_steps: int = 5) -> Dict[str, Any]:
        url = f"{self.base_url}/conversations/{conversation_id}/messages"
        payload = {"content": content, "max_steps": max_steps}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload, headers=self._get_headers())
            return self._handle_response(resp)

    def delete_conversation(self, conversation_id: str, hard_delete: bool = False) -> Dict[str, Any]:
        url = f"{self.base_url}/conversations/{conversation_id}"
        params = {"hard_delete": str(hard_delete).lower()}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.delete(url, params=params, headers=self._get_headers())
            return self._handle_response(resp)

    def summarize_conversation(self, conversation_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/conversations/{conversation_id}/summarize"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=self._get_headers())
            return self._handle_response(resp)

    # --------------------------------------------------------------------------
    # 5. AI Agent & Tools
    # --------------------------------------------------------------------------

    def list_tools(self) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/agent/tools"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, headers=self._get_headers())
            return self._handle_response(resp)

    def run_agent(self, query: str, max_steps: int = 5) -> Dict[str, Any]:
        url = f"{self.base_url}/agent/run"
        payload = {"query": query, "max_steps": max_steps}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload, headers=self._get_headers())
            return self._handle_response(resp)

    def run_report_generation_tool(self, topic: str, report_format: str = "executive_summary", max_chunks: int = 8) -> Dict[str, Any]:
        url = f"{self.base_url}/agent/tools/generate_report"
        payload = {"topic": topic, "report_format": report_format, "max_evidence_chunks": max_chunks}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload, headers=self._get_headers())
            return self._handle_response(resp)

    # --------------------------------------------------------------------------
    # 6. RAG Direct Query
    # --------------------------------------------------------------------------

    def query_rag(self, query: str, top_k: int = 5, document_id: Optional[str] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/rag/query"
        payload = {"query": query, "top_k": top_k, "document_id": document_id}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload, headers=self._get_headers())
            return self._handle_response(resp)
