"""Main Executive Dashboard View."""
import streamlit as st
import pandas as pd

from frontend.api_client import BackendApiError
from frontend.state import get_api_client


def render_dashboard_view() -> None:
    """Renders the high-level executive intelligence dashboard."""
    st.markdown("### 📊 Operational Overview & System Telemetry")

    client = get_api_client()
    try:
        with st.spinner("Fetching operational statistics from backend..."):
            stats = client.get_dashboard_stats()
    except BackendApiError as e:
        st.error(f"Error loading dashboard statistics: {e.message}")
        return
    except Exception as exc:
        st.error(f"Failed to connect to backend: {str(exc)}")
        return

    docs = stats.get("documents", {})
    convs = stats.get("conversations", {})
    telem = stats.get("telemetry", {})
    activity = stats.get("recent_activity", [])
    status_info = stats.get("system_status", {})

    # Row 1: Key Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Total Documents",
            value=docs.get("total", 0),
            delta=f"{docs.get('completed', 0)} Completed",
            help="Total uploaded and parsed enterprise documents in database.",
        )
    with c2:
        st.metric(
            label="Indexed Chunks",
            value=docs.get("total_chunks", 0),
            help="Total vector embeddings stored in pgvector (384 dimensions).",
        )
    with c3:
        st.metric(
            label="Active Conversations",
            value=convs.get("total_user_conversations", 0),
            delta=f"{convs.get('total_user_messages', 0)} messages",
            help="Persistent multi-turn conversation sessions belonging to current user.",
        )
    with c4:
        avg_lat = telem.get("avg_latency_ms", 0.0)
        st.metric(
            label="Avg Query Latency",
            value=f"{avg_lat:.1f} ms" if avg_lat > 0 else "0.0 ms",
            help="Mean end-to-end execution latency for retrieval and generation.",
        )

    st.markdown("---")

    # Row 2: Ingestion & Telemetry Breakdown
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("##### 📁 Document Processing Status")
        file_types = docs.get("file_types", {})
        if file_types:
            df_types = pd.DataFrame(list(file_types.items()), columns=["Format", "Count"])
            st.bar_chart(df_types.set_index("Format"), height=220)
        else:
            st.info("No documents uploaded yet. Go to the 'Documents' tab to ingest files.")

        st.caption(
            f"Status: **{docs.get('completed', 0)}** Completed • "
            f"**{docs.get('failed', 0)}** Failed • "
            f"Database Connected: **{'Yes (PostgreSQL)' if status_info.get('database_connected') else 'Degraded'}**"
        )

    with col_right:
        st.markdown("##### ⚡ LLM Tokens & Usage Summary")
        tokens_info = telem.get("tokens", {})
        total_tok = tokens_info.get("total", 0)
        prompt_tok = tokens_info.get("prompt", 0)
        comp_tok = tokens_info.get("completion", 0)

        st.markdown(
            f"""
            - **Total Queries Executed:** `{telem.get('total_queries', 0)}`
            - **Total Tokens Consumed:** `{total_tok:,}`
            - **Prompt / Input Tokens:** `{prompt_tok:,}`
            - **Completion / Output Tokens:** `{comp_tok:,}`
            - **Primary Provider:** `Groq (openai/gpt-oss-20b)`
            - **Fallback Provider:** `Gemini (gemini-3.8-flash)`
            """
        )

    st.markdown("---")

    # Row 3: Persisted Activity Feed (Audit Logs)
    st.markdown("##### 🕒 Recent Activity (Audit Log)")
    if activity:
        df_activity = pd.DataFrame(activity)
        df_display = df_activity[["action", "resource_type", "status", "created_at", "latency_ms"]].copy()
        df_display.columns = ["Action", "Resource", "Status", "Timestamp", "Latency (ms)"]
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info("No recorded audit events yet.")
