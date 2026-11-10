"""Unit tests for Frontend BackendApiClient."""
import httpx
import pytest

from frontend.api_client import BackendApiClient, BackendApiError


def test_api_client_token_management():
    client = BackendApiClient(base_url="http://localhost:8000/api/v1")
    assert client.token is None
    assert "Authorization" not in client._get_headers()

    client.set_token("test_jwt_token_123")
    assert client.token == "test_jwt_token_123"
    assert client._get_headers()["Authorization"] == "Bearer test_jwt_token_123"

    client.clear_token()
    assert client.token is None
    assert "Authorization" not in client._get_headers()


def test_api_client_handle_response_success():
    client = BackendApiClient()
    req = httpx.Request("GET", "http://localhost:8000/api/v1/health")
    resp = httpx.Response(status_code=200, json={"status": "healthy"}, request=req)
    data = client._handle_response(resp)
    assert data["status"] == "healthy"


def test_api_client_handle_response_error():
    client = BackendApiClient()
    req = httpx.Request("POST", "http://localhost:8000/api/v1/auth/login")
    resp = httpx.Response(status_code=401, json={"message": "Invalid credentials", "error_code": "AUTH_FAIL"}, request=req)
    with pytest.raises(BackendApiError) as exc_info:
        client._handle_response(resp)

    assert exc_info.value.status_code == 401
    assert "Invalid credentials" in exc_info.value.message


def test_api_client_base_url_normalization():
    # URL with trailing slash and /api/v1
    client1 = BackendApiClient(base_url="http://localhost:8000/api/v1/")
    assert client1.base_url == "http://localhost:8000/api/v1"

    # URL with /api/v1
    client2 = BackendApiClient(base_url="http://localhost:8000/api/v1")
    assert client2.base_url == "http://localhost:8000/api/v1"

    # Host-only URL (Render service URL without /api/v1)
    client3 = BackendApiClient(base_url="https://enterprise-intelligence-backend-exl7.onrender.com")
    assert client3.base_url == "https://enterprise-intelligence-backend-exl7.onrender.com/api/v1"

    # Host-only URL with trailing slash
    client4 = BackendApiClient(base_url="https://enterprise-intelligence-backend-exl7.onrender.com/")
    assert client4.base_url == "https://enterprise-intelligence-backend-exl7.onrender.com/api/v1"


def test_login_endpoint_url_construction(monkeypatch):
    """Verifies login endpoint URL construction and POST request payload."""
    recorded_calls = []

    def mock_post(self, url, *args, **kwargs):
        recorded_calls.append({"method": "POST", "url": url, "json": kwargs.get("json")})
        req = httpx.Request("POST", url)
        return httpx.Response(
            status_code=200,
            json={"access_token": "mock_jwt_token", "user_id": 1, "username": "admin", "role": "admin"},
            request=req,
        )

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    # 1. When BACKEND_API_URL is configured without /api/v1 (e.g. Render default web service URL)
    client_bare_host = BackendApiClient(base_url="https://enterprise-intelligence-backend-exl7.onrender.com")
    res = client_bare_host.login("admin", "test_pass")
    assert res["access_token"] == "mock_jwt_token"
    assert recorded_calls[-1]["url"] == "https://enterprise-intelligence-backend-exl7.onrender.com/api/v1/auth/login"
    assert recorded_calls[-1]["json"] == {"username": "admin", "password": "test_pass"}

    # 2. When BACKEND_API_URL includes /api/v1
    client_full = BackendApiClient(base_url="https://enterprise-intelligence-backend-exl7.onrender.com/api/v1")
    client_full.login("admin", "test_pass")
    assert recorded_calls[-1]["url"] == "https://enterprise-intelligence-backend-exl7.onrender.com/api/v1/auth/login"
    # Ensure no path duplication like /api/v1/api/v1
    assert "/api/v1/api/v1" not in recorded_calls[-1]["url"]


def test_health_endpoint_url_construction(monkeypatch):
    """Verifies health check URL resolves to root /health regardless of base_url configuration."""
    recorded_calls = []

    def mock_get(self, url, *args, **kwargs):
        recorded_calls.append({"method": "GET", "url": url})
        req = httpx.Request("GET", url)
        return httpx.Response(status_code=200, json={"status": "healthy"}, request=req)

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    client = BackendApiClient(base_url="https://enterprise-intelligence-backend-exl7.onrender.com")
    res = client.get_health()
    assert res["status"] == "healthy"
    assert recorded_calls[-1]["url"] == "https://enterprise-intelligence-backend-exl7.onrender.com/health"


def test_run_agent_uses_dedicated_150s_timeout(monkeypatch):
    """Verifies run_agent uses a dedicated 150-second timeout with 10-second connect."""
    client_instance_timeouts = []

    original_client_init = httpx.Client.__init__

    def mock_client_init(self, *args, **kwargs):
        timeout = kwargs.get("timeout")
        client_instance_timeouts.append(timeout)
        original_client_init(self, *args, **kwargs)

    def mock_post(self, url, *args, **kwargs):
        req = httpx.Request("POST", url)
        return httpx.Response(
            status_code=200,
            json={
                "query": "test query",
                "final_answer": "test answer",
                "status": "COMPLETED",
                "tools_used": [],
                "citations": [],
                "steps": [],
                "total_latency_ms": 120.0,
                "tokens": {"total": 50},
            },
            request=req,
        )

    monkeypatch.setattr(httpx.Client, "__init__", mock_client_init)
    monkeypatch.setattr(httpx.Client, "post", mock_post)

    api_client = BackendApiClient(base_url="http://localhost:8000/api/v1")
    res = api_client.run_agent(query="Test agent query", max_steps=4)

    assert res["status"] == "COMPLETED"
    assert len(client_instance_timeouts) >= 1
    used_timeout = client_instance_timeouts[-1]
    assert isinstance(used_timeout, httpx.Timeout)
    assert used_timeout.read == 150.0
    assert used_timeout.connect == 10.0


def test_other_endpoints_retain_default_60s_timeout(monkeypatch):
    """Verifies other API methods retain the default 60s timeout."""
    client_instance_timeouts = []

    original_client_init = httpx.Client.__init__

    def mock_client_init(self, *args, **kwargs):
        timeout = kwargs.get("timeout")
        client_instance_timeouts.append(timeout)
        original_client_init(self, *args, **kwargs)

    def mock_get(self, url, *args, **kwargs):
        req = httpx.Request("GET", url)
        return httpx.Response(status_code=200, json=[], request=req)

    monkeypatch.setattr(httpx.Client, "__init__", mock_client_init)
    monkeypatch.setattr(httpx.Client, "get", mock_get)

    api_client = BackendApiClient(base_url="http://localhost:8000/api/v1")
    api_client.list_tools()

    assert len(client_instance_timeouts) >= 1
    used_timeout = client_instance_timeouts[-1]
    assert isinstance(used_timeout, httpx.Timeout)
    assert used_timeout.read == 60.0
    assert used_timeout.connect == 10.0
