import asyncio
import os
import socket
import sys
from typing import Generator
import pytest
import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

# Windows Sandbox loopback socket compatibility shim
if sys.platform == "win32":
    try:
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_sock.bind(("127.0.0.1", 0))
        test_sock.listen(1)
        port = test_sock.getsockname()[1]
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(0.1)
        probe.connect(("127.0.0.1", port))
        probe.close()
        test_sock.close()
    except (PermissionError, OSError):
        def _safe_socketpair(*args, **kwargs):
            s1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s1.setblocking(False)
            s2.setblocking(False)
            return s1, s2

        socket.socketpair = _safe_socketpair
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.config import settings
from app.core.telemetry import telemetry
from app.db.models import Base
from app.db.session import get_db, check_db_connection
from app.main import app

# Isolated In-Memory Test Database Engine
TEST_SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create all database schema tables for the test session."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Provides a fresh isolated database session per test."""
    with test_engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        with test_engine.begin() as conn:
            for table in reversed(Base.metadata.sorted_tables):
                conn.execute(table.delete())


@pytest.fixture(autouse=True)
def reset_telemetry_state():
    """Ensure telemetry records are cleanly isolated per test."""
    telemetry.reset()
    yield
    telemetry.reset()


@pytest.fixture
async def async_client(db_session: Session):
    """Asynchronous HTTP test client using ASGITransport with database session override."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
    app.dependency_overrides.clear()
