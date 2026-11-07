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
    client1 = BackendApiClient(base_url="http://localhost:8000/api/v1/")
    assert client1.base_url == "http://localhost:8000/api/v1"

    client2 = BackendApiClient(base_url="http://localhost:8000/api/v1")
    assert client2.base_url == "http://localhost:8000/api/v1"
