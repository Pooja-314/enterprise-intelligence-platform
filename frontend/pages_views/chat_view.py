"""Conversational AI Chat View with persistent multi-turn memory."""
from typing import Any, Dict, List, Optional
import streamlit as st

from frontend.api_client import BackendApiError
from frontend.components.citations import render_citations
from frontend.state import get_api_client


def render_chat_view() -> None:
    """Renders conversational AI chat interface with persistent memory and session selection."""
    st.markdown("### 💬 Conversational Intelligence with Persistent Memory")

    client = get_api_client()

    # Layout: Sidebar for conversations, main pane for chat
    col_sidebar, col_chat = st.columns([1, 2.5])

    # --------------------------------------------------------------------------
    # 1. Conversation List & Creation
    # --------------------------------------------------------------------------
    with col_sidebar:
        st.markdown("##### 📁 Your Conversations")

        if st.button("➕ New Conversation", use_container_width=True, type="primary"):
            try:
                new_conv = client.create_conversation()
                st.session_state["active_conversation_id"] = new_conv["id"]
                st.session_state["chat_messages"] = []
                st.rerun()
            except Exception as e:
                st.error(f"Failed to create conversation: {e}")

        st.markdown("<hr style='margin: 8px 0;' />", unsafe_allow_html=True)

        try:
            conversations = client.list_conversations(limit=25)
        except Exception as e:
            st.error(f"Error fetching conversations: {e}")
            conversations = []

        if not conversations:
            st.info("No active conversations. Click 'New Conversation' to start.")
        else:
            for c in conversations:
                cid = c["id"]
                is_selected = (st.session_state.get("active_conversation_id") == cid)

                col_btn, col_del = st.columns([4, 1])
                with col_btn:
                    btn_label = f"{'▶ ' if is_selected else ''}{c.get('title', 'Conversation')[:30]}"
                    if st.button(btn_label, key=f"sel_conv_{cid}", use_container_width=True):
                        st.session_state["active_conversation_id"] = cid
                        st.session_state["chat_messages"] = []
                        st.rerun()

                with col_del:
                    if st.button("🗑️", key=f"del_conv_{cid}", help="Delete conversation"):
                        try:
                            client.delete_conversation(cid)
                            if st.session_state.get("active_conversation_id") == cid:
                                st.session_state["active_conversation_id"] = None
                                st.session_state["chat_messages"] = []
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Delete failed: {ex}")

    # --------------------------------------------------------------------------
    # 2. Main Chat Conversation History & Input
    # --------------------------------------------------------------------------
    with col_chat:
        active_id = st.session_state.get("active_conversation_id")

        if not active_id:
            if conversations:
                # Default to first conversation
                active_id = conversations[0]["id"]
                st.session_state["active_conversation_id"] = active_id
            else:
                st.info("👈 Create a new conversation on the left to begin querying.")
                return

        # Fetch conversation details and messages
        try:
            conv_detail = client.get_conversation(active_id)
        except BackendApiError as e:
            st.error(f"Error loading conversation: {e.message}")
            return
        except Exception as ex:
            st.error(f"Failed to load conversation: {ex}")
            return

        st.markdown(f"#### 🗨️ {conv_detail.get('title')}")

        if conv_detail.get("summary"):
            st.info(f"🧠 **Memory Summary (Compacted Context):**\n{conv_detail.get('summary')}")

        # Render message history
        messages = conv_detail.get("messages", [])
        if not messages:
            st.caption("No messages yet in this session. Ask any factual question grounded in your documents.")

        for msg in messages:
            role = msg.get("role")
            if role == "system" and msg.get("msg_metadata", {}).get("is_summary"):
                continue  # Displayed above in the summary box

            with st.chat_message("user" if role == "user" else "assistant"):
                st.markdown(msg.get("content", ""))

                # Display citations if assistant turn
                if role == "assistant":
                    meta = msg.get("msg_metadata", {})
                    citations = meta.get("citation_details") or meta.get("citations", [])
                    tools_used = meta.get("tools_used", [])
                    latency_ms = meta.get("latency_ms")
                    verification_data = meta.get("verification")

                    render_citations(citations=citations, verification_data=verification_data)

                    if tools_used or latency_ms:
                        tools_str = ", ".join(tools_used) if tools_used else "RAG Router"
                        st.caption(f"⚡ Tools: `{tools_str}` • Latency: `{latency_ms or 0:.1f}ms`")

        # Chat Input Bar
        user_prompt = st.chat_input("Ask a question (memory and follow-ups are preserved)...")
        if user_prompt:
            # Display user query immediately
            with st.chat_message("user"):
                st.markdown(user_prompt)

            # Send to backend
            with st.chat_message("assistant"):
                with st.spinner("Reasoning over document memory..."):
                    try:
                        res = client.send_message(
                            conversation_id=active_id,
                            content=user_prompt,
                            max_steps=5,
                        )
                        assistant_msg = res.get("assistant_message", {})
                        st.markdown(assistant_msg.get("content", ""))

                        meta = assistant_msg.get("msg_metadata", {})
                        citations = res.get("citation_details") or meta.get("citation_details") or res.get("citations", [])
                        verification_data = meta.get("verification")
                        render_citations(citations=citations, verification_data=verification_data)

                        if res.get("is_summarized"):
                            st.caption("⚡ Long dialogue context was automatically summarized to maintain token budget.")

                        st.rerun()
                    except BackendApiError as e:
                        st.error(f"Query Error: {e.message}")
                    except Exception as exc:
                        st.error(f"Error communicating with backend: {str(exc)}")
