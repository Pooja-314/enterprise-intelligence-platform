"""Authentication View: Login and Registration."""
import streamlit as st
from frontend.api_client import BackendApiError
from frontend.state import get_api_client, set_user_session


def render_auth_view() -> None:
    """Renders the login and registration screen for unauthenticated users."""
    st.markdown(
        """
        <div style="text-align: center; margin-top: 30px; margin-bottom: 25px;">
            <h1 style="color: #0F172A; font-weight: 800; font-size: 32px; margin-bottom: 4px;">
                🏢 Enterprise Intelligence Platform
            </h1>
            <p style="color: #64748B; font-size: 15px; margin: 0;">
                Hybrid RAG • Autonomous AI Agents • pgvector • Conversational Memory
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1.8, 1])
    with col2:
        tab_login, tab_register = st.tabs(["🔐 Sign In", "📝 Create Account"])

        with tab_login:
            st.markdown("##### Enterprise Sign In")
            with st.form("login_form"):
                username = st.text_input("Username or Email", placeholder="e.g. admin or alice").strip()
                password = st.text_input("Password", type="password", placeholder="Enter password")
                submit_btn = st.form_submit_button("Sign In", use_container_width=True, type="primary")

                if submit_btn:
                    if not username or not password:
                        st.error("Please provide both username and password.")
                    else:
                        client = get_api_client()
                        try:
                            with st.spinner("Authenticating with enterprise backend..."):
                                token_data = client.login(username=username, password=password)
                                user_info = {
                                    "id": token_data.get("user_id"),
                                    "username": token_data.get("username"),
                                    "role": token_data.get("role"),
                                }
                                set_user_session(token=token_data["access_token"], user_data=user_info)
                                st.success("Authentication successful! Redirecting...")
                                st.rerun()
                        except BackendApiError as e:
                            st.error(f"Authentication Failed: {e.message}")
                        except Exception as exc:
                            st.error(f"Connection error to backend: {str(exc)}")

            st.caption("Default admin credentials configured via environment settings.")

        with tab_register:
            st.markdown("##### New User Registration")
            with st.form("register_form"):
                new_username = st.text_input("Choose Username", placeholder="e.g. jane_doe").strip()
                new_email = st.text_input("Corporate Email", placeholder="jane@enterprise.com").strip()
                new_password = st.text_input("Choose Password (min 6 chars)", type="password")
                reg_btn = st.form_submit_button("Register Account", use_container_width=True)

                if reg_btn:
                    if not new_username or not new_email or not new_password:
                        st.error("All fields are required.")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters.")
                    else:
                        client = get_api_client()
                        try:
                            with st.spinner("Registering user account..."):
                                client.register(username=new_username, email=new_email, password=new_password)
                                st.success("Registration successful! You may now sign in above.")
                        except BackendApiError as e:
                            st.error(f"Registration Failed: {e.message}")
                        except Exception as exc:
                            st.error(f"Error connecting to backend: {str(exc)}")
