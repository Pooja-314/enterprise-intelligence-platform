"""Enterprise Intelligence Platform - Streamlit Main Application Entry Point."""
import sys
from pathlib import Path

# Robust import resolution: ensure repository root is on sys.path
# This ensures 'frontend.*' and 'app.*' imports resolve cleanly regardless of execution context.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from frontend.components.auth_view import render_auth_view
from frontend.components.header import render_header
from frontend.pages_views.agent_view import render_agent_view
from frontend.pages_views.chat_view import render_chat_view
from frontend.pages_views.dashboard_view import render_dashboard_view
from frontend.pages_views.documents_view import render_documents_view
from frontend.pages_views.observability_view import render_observability_view
from frontend.state import clear_user_session, init_session_state, is_authenticated

# Configure Page
st.set_page_config(
    page_title="Enterprise Intelligence Platform",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session
init_session_state()


def main() -> None:
    """Main application orchestrator and view dispatcher."""
    # 1. Unauthenticated Gateway Guard
    if not is_authenticated():
        render_auth_view()
        return

    # 2. Render Authenticated Header
    render_header()

    # 3. Sidebar Navigation
    with st.sidebar:
        st.markdown("### 🏢 Navigation")
        nav_selection = st.radio(
            "Select Module",
            [
                "📊 Dashboard",
                "💬 AI Chat",
                "🤖 AI Agent",
                "📁 Documents",
                "📈 Observability",
            ],
            key="navigation_radio",
        )

        st.markdown("---")
        user = st.session_state.get("current_user") or {}
        st.markdown(f"**Connected User:** `{user.get('username', 'user')}`")
        st.caption(f"Role: `{user.get('role', 'user')}`")

        if st.button("🚪 Sign Out", use_container_width=True, type="secondary"):
            clear_user_session()
            st.rerun()

    # 4. Route to Selected Page View
    if nav_selection == "📊 Dashboard":
        render_dashboard_view()
    elif nav_selection == "💬 AI Chat":
        render_chat_view()
    elif nav_selection == "🤖 AI Agent":
        render_agent_view()
    elif nav_selection == "📁 Documents":
        render_documents_view()
    elif nav_selection == "📈 Observability":
        render_observability_view()


if __name__ == "__main__":
    main()
