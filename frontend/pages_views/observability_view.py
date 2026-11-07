"""Usage & Observability View: Audit Logs, Latency, and Telemetry."""
from typing import Any, Dict, List
import streamlit as st
import pandas as pd

from frontend.api_client import BackendApiError
from frontend.state import get_api_client


def render_observability_view() -> None:
    """Renders real telemetry metrics, audit logs, and performance percentiles."""
    st.markdown("### 📈 Enterprise Observability & Audit Telemetry")
    st.caption("Real-time telemetry records persisted in PostgreSQL and in-memory execution metrics.")

    client = get_api_client()

    try:
        stats = client.get_dashboard_stats()
        health = client.get_health()
    except Exception as e:
        st.error(f"Error fetching observability metrics: {e}")
        return

    telem = stats.get("telemetry", {})
    activity = stats.get("recent_activity", [])
    system_status = stats.get("system_status", {})
    llm_info = health.get("llm_providers", {})

    # Top System Status
    st.markdown("##### 🔌 Core Infrastructure Health")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Database Health", "Connected (Neon pgvector)" if system_status.get("database_connected") else "Degraded")
    with c2:
        groq_status = "Active & Configured" if llm_info.get("groq_configured") else "Not Configured"
        st.metric("Primary LLM (Groq)", groq_status)
    with c3:
        gemini_status = "Available (HTTP 503 Spikes)" if llm_info.get("gemini_configured") else "Not Configured"
        st.metric("Fallback LLM (Gemini)", gemini_status)

    st.markdown("---")

    # Tokens Breakdown
    st.markdown("##### 🪙 Token Consumption Breakdown")
    tokens = telem.get("tokens", {})
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.metric("Total Tokens", f"{tokens.get('total', 0):,}")
    with col_t2:
        st.metric("Prompt / Input Tokens", f"{tokens.get('prompt', 0):,}")
    with col_t3:
        st.metric("Completion / Output Tokens", f"{tokens.get('completion', 0):,}")

    st.markdown("---")

    # Audit Logs
    st.markdown("##### 📋 Audit Log Explorer")
    st.caption("Every tool invocation, user login, document parsing, and agent action is recorded.")

    if activity:
        df_activity = pd.DataFrame(activity)
        filter_action = st.selectbox(
            "Filter by Action Type",
            ["All"] + sorted(list(set(df_activity["action"].dropna()))),
        )
        if filter_action != "All":
            df_activity = df_activity[df_activity["action"] == filter_action]

        st.dataframe(
            df_activity[["id", "action", "resource_type", "status", "created_at", "latency_ms"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No audit records found in the database.")
