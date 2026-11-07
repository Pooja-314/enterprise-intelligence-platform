"""Unit tests for Frontend session state and components."""
import streamlit as st
import pytest

from frontend.state import clear_user_session, init_session_state, is_authenticated, set_user_session


def test_session_state_lifecycle():
    # 1. Initialize
    init_session_state()
    assert is_authenticated() is False
    assert st.session_state["auth_token"] is None
    assert st.session_state["current_user"] is None

    # 2. Login
    set_user_session("jwt_secret_token", {"id": 1, "username": "admin", "role": "admin"})
    assert is_authenticated() is True
    assert st.session_state["auth_token"] == "jwt_secret_token"
    assert st.session_state["current_user"]["username"] == "admin"

    # 3. Logout
    clear_user_session()
    assert is_authenticated() is False
    assert st.session_state["auth_token"] is None
    assert st.session_state["current_user"] is None
