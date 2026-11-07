"""Authentication REST API endpoints for user login, registration, and session verification."""
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.config import settings
from app.core.errors import AuthenticationError
from app.core.security import SecurityService
from app.db.models import AuditLog, User
from app.db.session import get_db

router = APIRouter(prefix="/auth", tags=["Authentication"])


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, description="Username or email.")
    password: str = Field(..., min_length=1, description="Account password.")


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=6)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str
    role: str
    expires_in_minutes: int


class UserProfileResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Authenticates user credentials and issues an HMAC-SHA256 JWT access token."""
    username_or_email = request.username.strip().lower()

    # 1. Lookup user in database
    user = (
        db.query(User)
        .filter(
            (User.username.ilike(username_or_email))
            | (User.email.ilike(username_or_email))
        )
        .first()
    )

    # 2. Bootstrap initial admin user if matching configured admin credentials
    if not user and (username_or_email == settings.ADMIN_USERNAME.lower() or username_or_email == "admin@enterprise.local"):
        if request.password == settings.ADMIN_PASSWORD:
            user = User(
                username=settings.ADMIN_USERNAME,
                email="admin@enterprise.local",
                hashed_password=SecurityService.hash_password(settings.ADMIN_PASSWORD),
                role="admin",
                is_active=True,
            )
            db.add(user)
            db.commit()
            db.refresh(user)

    if not user:
        raise AuthenticationError("Invalid username or password.")

    # 3. Verify password hash
    if not SecurityService.verify_password(request.password, user.hashed_password):
        # Fallback check for bootstrap admin password update
        if user.username == settings.ADMIN_USERNAME and request.password == settings.ADMIN_PASSWORD:
            user.hashed_password = SecurityService.hash_password(settings.ADMIN_PASSWORD)
            db.commit()
        else:
            raise AuthenticationError("Invalid username or password.")

    if not user.is_active:
        raise AuthenticationError("User account is deactivated.")

    # 4. Generate JWT access token
    token_payload = {
        "sub": user.id,
        "username": user.username,
        "email": user.email,
        "role": user.role,
    }
    access_token = SecurityService.create_access_token(
        data=token_payload,
        expires_delta_seconds=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )

    # Record login audit
    try:
        audit = AuditLog(
            user_id=user.id,
            action="user_login",
            resource_type="auth",
            resource_id=str(user.id),
            details={"username": user.username, "status": "SUCCESS"},
            created_at=datetime.now(timezone.utc),
        )
        db.add(audit)
        db.commit()
    except Exception:
        db.rollback()

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user_id=user.id,
        username=user.username,
        role=user.role,
        expires_in_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
    )


@router.post("/register", response_model=UserProfileResponse, status_code=status.HTTP_201_CREATED)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """Registers a new user account with hashed password."""
    # Check if username exists
    existing = db.query(User).filter(User.username.ilike(request.username)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username is already registered.")

    existing_email = db.query(User).filter(User.email.ilike(request.email)).first()
    if existing_email:
        raise HTTPException(status_code=400, detail="Email is already registered.")

    user = User(
        username=request.username.strip(),
        email=request.email.strip().lower(),
        hashed_password=SecurityService.hash_password(request.password),
        role="user",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return UserProfileResponse.model_validate(user)


@router.get("/me", response_model=UserProfileResponse)
def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Returns profile for currently authenticated user."""
    return UserProfileResponse.model_validate(current_user)
