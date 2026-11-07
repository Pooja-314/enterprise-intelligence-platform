"""AI Agent Multi-Step Reasoning & Tool Execution View."""
from typing import Any, Dict, List
import streamlit as st

from frontend.api_client import BackendApiError
from frontend.components.citations import render_citations
from frontend.state import get_api_client


def render_agent_view() -> None:
    """Renders the AI Agent orchestration interface with step traces and tool execution."""
    st.markdown("### 🤖 Autonomous Multi-Tool AI Agent Orchestrator")
    st.caption("Executes complex reasoning workflows using registered knowledge base search, document analysis, and report generation tools.")

    client = get_api_client()

    # 1. Inspect Available Tools
    with st.expander("🛠️ View Registered Agent Tools & Schemas", expanded=False):
        try:
            tools = client.list_tools()
            for t in tools:
                st.markdown(f"**Tool:** `{t.get('name')}`")
                st.markdown(f"*{t.get('description')}*")
                with st.expander("JSON Parameter Schema"):
                    st.json(t.get("parameters", {}))
                st.markdown("---")
        except Exception as ex:
            st.warning(f"Could not fetch tool schemas: {ex}")

    st.markdown("##### 🎯 Submit Agent Goal / Multi-Step Inquiry")

    with st.form("agent_query_form"):
        agent_goal = st.text_area(
            "Agent Instruction or Goal",
            placeholder="e.g. Generate a structured intelligence report on Q3 financial performance and verify all key claims.",
            height=90,
        )

        col_steps, col_submit = st.columns([1, 2])
        with col_steps:
            max_steps = st.slider("Max Reasoning Steps", min_value=1, max_value=8, value=5)
        with col_submit:
            st.write("")
            st.write("")
            execute_btn = st.form_submit_button("🚀 Run Autonomous Agent", use_container_width=True, type="primary")

    if execute_btn:
        if not agent_goal.strip():
            st.error("Please enter a goal or inquiry for the agent.")
            return

        with st.spinner("Agent is reasoning and executing tools..."):
            try:
                res = client.run_agent(query=agent_goal.strip(), max_steps=max_steps)

                status = res.get("status", "COMPLETED")
                final_answer = res.get("final_answer", "")
                tools_used = res.get("tools_used", [])
                citations = res.get("citations", [])
                steps = res.get("steps", [])
                latency = res.get("total_latency_ms", 0.0)

                # Header summary
                st.success(f"✅ Workflow Status: **{status}** • Latency: **{latency:.1f}ms** • Tools Called: **{len(tools_used)}**")

                # Step-by-Step Execution Trace Accordion
                st.markdown("##### 🔍 Execution Trace Scratchpad")
                for s in steps:
                    step_num = s.get("step_number")
                    action = s.get("action")
                    thought = s.get("thought")
                    obs = s.get("observation_summary")
                    step_lat = s.get("latency_ms", 0.0)
                    step_status = s.get("status")

                    icon = "🟢" if step_status == "SUCCESS" else ("🟡" if step_status == "RETRY" else "🔴")
                    with st.expander(f"{icon} Step {step_num}: Action `{action}` ({step_lat:.1f}ms)"):
                        st.markdown(f"**Thought:** {thought}")
                        st.markdown(f"**Action Input:**")
                        st.json(s.get("action_input", {}))
                        st.markdown(f"**Observation:**")
                        st.code(obs, language="text")

                # Final Grounded Output
                st.markdown("---")
                st.markdown("##### 📄 Final Synthesized Output")
                st.markdown(final_answer)

                render_citations(citations=citations)

            except BackendApiError as e:
                st.error(f"Agent Execution Error: {e.message}")
            except Exception as exc:
                st.error(f"Connection failure: {str(exc)}")
