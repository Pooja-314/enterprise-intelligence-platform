import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import httpx
from app.config import settings

def check_login():
    try:
        resp = httpx.post(
            "http://127.0.0.1:8000/api/v1/auth/login",
            json={"username": settings.ADMIN_USERNAME, "password": settings.ADMIN_PASSWORD},
            timeout=10.0,
        )
        has_token = False
        login_success = (resp.status_code == 200)
        if login_success:
            data = resp.json()
            has_token = bool(data.get("access_token"))

        print("HTTP Status Code:", resp.status_code)
        print("Login Success:", login_success)
        print("Access Token Returned:", has_token)
    except Exception as e:
        print("Connection failed:", type(e).__name__)

if __name__ == "__main__":
    check_login()
