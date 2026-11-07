"""Unit tests for backend /api/v1/auth/ endpoints."""
import pytest
from sqlalchemy.orm import Session

from app.api.auth import LoginRequest, RegisterRequest, get_current_user_profile, login, register
from app.config import settings
from app.core.errors import AuthenticationError
from app.db.models import User


def test_auth_admin_bootstrap_login(db_session: Session):
    # Logging in as configured admin before user row exists creates and authenticates admin
    req = LoginRequest(username=settings.ADMIN_USERNAME, password=settings.ADMIN_PASSWORD)
    res = login(request=req, db=db_session)

    assert res.access_token is not None
    assert res.username == settings.ADMIN_USERNAME
    assert res.role == "admin"
    assert res.token_type == "bearer"

    # User now exists in DB
    user = db_session.query(User).filter(User.username == settings.ADMIN_USERNAME).first()
    assert user is not None
    assert user.role == "admin"


def test_auth_invalid_password_rejected(db_session: Session):
    req = LoginRequest(username=settings.ADMIN_USERNAME, password="wrong_password_123")
    with pytest.raises(AuthenticationError):
        login(request=req, db=db_session)


def test_auth_register_and_login_new_user(db_session: Session):
    reg_req = RegisterRequest(username="sarah_connor", email="sarah@resistance.org", password="cyberdyne_pwd")
    profile = register(request=reg_req, db=db_session)
    assert profile.username == "sarah_connor"
    assert profile.email == "sarah@resistance.org"

    # Login with new credentials
    login_req = LoginRequest(username="sarah_connor", password="cyberdyne_pwd")
    token_res = login(request=login_req, db=db_session)
    assert token_res.username == "sarah_connor"
    assert token_res.role == "user"


def test_auth_get_current_user_profile(db_session: Session):
    user = User(id=401, username="test_profile", email="profile@corp.com", hashed_password="pwd", role="user")
    db_session.add(user)
    db_session.commit()

    profile = get_current_user_profile(current_user=user)
    assert profile.id == 401
    assert profile.username == "test_profile"
    assert profile.email == "profile@corp.com"
