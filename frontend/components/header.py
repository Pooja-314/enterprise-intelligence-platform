"""Header and navigation status component."""
import streamlit as st
from frontend.state import clear_user_session


def render_header() -> None:
    """Renders the top enterprise application header."""
    user = st.session_state.get("current_user") or {}
    username = user.get("username", "User")
    role = user.get("role", "user").upper()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                <h2 style="margin: 0; padding: 0; color: #1E293B; font-weight: 700;">
                    🏢 Enterprise Intelligence Platform
                </h2>
                <span style="background: #E2E8F0; color: #475569; font-size: 11px; padding: 3px 8px; border-radius: 12px; font-weight: 600;">
                    v1.0 • Neon PostgreSQL • pgvector
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="text-align: right; margin-bottom: 8px;">
                <span style="font-size: 13px; font-weight: 600; color: #334155;">👤 {username}</span>
                <span style="background: #DBEAFE; color: #1D4ED8; font-size: 11px; padding: 2px 6px; border-radius: 4px; margin-left: 6px;">{role}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='margin-top: 0; margin-bottom: 16px; border: 0; border-top: 1px solid #E2E8F0;' />", unsafe_allow_html=True)
