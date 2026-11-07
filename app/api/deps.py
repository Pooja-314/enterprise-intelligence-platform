"""FastAPI authentication and session dependencies."""
from typing import Optional
from fastapi import Depends, Header
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.errors import AuthenticationError
from app.core.security import SecurityService
from app.db.models import User
from app.db.session import get_db

# OpenAPI HTTPBearer security scheme (auto_error=False allows fallback to custom headers / optional auth)
bearer_scheme = HTTPBearer(auto_error=False)


def get_or_create_user(
    db: Session,
    user_id: int,
    username: Optional[str] = None,
    email: Optional[str] = None,
) -> User:
    """Ensures a User record exists for valid foreign key reference."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        user = User(
            id=user_id,
            username=username or f"user_{user_id}",
            email=email or f"user_{user_id}@enterprise.local",
            hashed_password=SecurityService.hash_password("default_secure_pass_123!"),
            role="user",
            is_active=True,
        )
        db.add(user)
        try:
            db.commit()
            db.refresh(user)
        except Exception:
            db.rollback()
            user = db.query(User).filter(User.id == user_id).first()
    return user


def get_current_user(
    db: Session = Depends(get_db),
    token_auth: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    authorization: Optional[str] = Header(None, include_in_schema=False),
    x_user_id: Optional[str] = Header(None, include_in_schema=False),
) -> User:
    """Authenticates the request via Bearer JWT token or X-User-Id header.
    Enforces user identification for conversational isolation.
    """
    token = None
    if isinstance(token_auth, HTTPAuthorizationCredentials) and token_auth.credentials:
        token = token_auth.credentials
    elif authorization:
        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise AuthenticationError("Invalid Authorization header format. Expected 'Bearer <token>'.")
        token = parts[1]

    if token:
        payload = SecurityService.verify_token(token)
        raw_uid = payload.get("sub") or payload.get("user_id") or 1
        try:
            user_id = int(raw_uid)
        except ValueError:
            user_id = 1
        username = payload.get("username", f"user_{user_id}")
        email = payload.get("email", f"{username}@enterprise.local")
        return get_or_create_user(db=db, user_id=user_id, username=username, email=email)

    if x_user_id:
        try:
            uid = int(x_user_id)
            return get_or_create_user(db=db, user_id=uid)
        except ValueError:
            raise AuthenticationError("Invalid X-User-Id header value.")

    raise AuthenticationError("Missing authentication token. Bearer JWT required.")


def get_current_user_optional(
    db: Session = Depends(get_db),
    token_auth: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    authorization: Optional[str] = Header(None, include_in_schema=False),
    x_user_id: Optional[str] = Header(None, include_in_schema=False),
) -> Optional[User]:
    """Optional authentication: returns authenticated User if credentials present, otherwise None."""
    try:
        return get_current_user(
            db=db,
            token_auth=token_auth,
            authorization=authorization,
            x_user_id=x_user_id,
        )
    except AuthenticationError:
        return None
