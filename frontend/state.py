"""Streamlit session state management and authentication session guards."""
from typing import Any, Dict, Optional
import streamlit as st

from frontend.api_client import BackendApiClient


def get_api_client() -> BackendApiClient:
    """Returns or initializes the session-scoped BackendApiClient."""
    if "api_client" not in st.session_state:
        st.session_state["api_client"] = BackendApiClient()
    
    # Sync token if in session
    token = st.session_state.get("auth_token")
    if token:
        st.session_state["api_client"].set_token(token)
    return st.session_state["api_client"]


def init_session_state() -> None:
    """Initializes default Streamlit session state keys."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "auth_token" not in st.session_state:
        st.session_state["auth_token"] = None
    if "current_user" not in st.session_state:
        st.session_state["current_user"] = None
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "📊 Dashboard"
    if "active_conversation_id" not in st.session_state:
        st.session_state["active_conversation_id"] = None
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []


def is_authenticated() -> bool:
    """Checks whether a valid user session is active."""
    return bool(st.session_state.get("authenticated") and st.session_state.get("auth_token"))


def set_user_session(token: str, user_data: Dict[str, Any]) -> None:
    """Stores the authenticated session state."""
    st.session_state["authenticated"] = True
    st.session_state["auth_token"] = token
    st.session_state["current_user"] = user_data
    client = get_api_client()
    client.set_token(token)


def clear_user_session() -> None:
    """Clears all session state variables upon logout."""
    st.session_state["authenticated"] = False
    st.session_state["auth_token"] = None
    st.session_state["current_user"] = None
    st.session_state["active_conversation_id"] = None
    st.session_state["chat_messages"] = []
    if "api_client" in st.session_state:
        st.session_state["api_client"].clear_token()
