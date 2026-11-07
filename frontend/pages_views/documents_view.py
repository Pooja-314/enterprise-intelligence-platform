"""Document Ingestion & Management View."""
import streamlit as st
import pandas as pd

from frontend.api_client import BackendApiError
from frontend.state import get_api_client


def render_documents_view() -> None:
    """Renders document upload, listing, chunk inspection, and deletion."""
    st.markdown("### 📁 Enterprise Document Ingestion & Vector Storage")

    client = get_api_client()

    tab_list, tab_upload = st.tabs(["📚 Indexed Documents", "📤 Upload New Document"])

    # --------------------------------------------------------------------------
    # Tab 1: List, Inspect Chunks & Delete
    # --------------------------------------------------------------------------
    with tab_list:
        try:
            documents = client.list_documents(limit=100)
        except Exception as e:
            st.error(f"Error fetching document inventory: {str(e)}")
            documents = []

        if not documents:
            st.info("No documents currently indexed in the knowledge base. Use the 'Upload' tab to ingest files.")
        else:
            col_search, col_filter = st.columns([2, 1])
            with col_search:
                search_query = st.text_input("🔍 Filter by filename or title", placeholder="Type to filter...").strip().lower()
            with col_filter:
                status_filter = st.selectbox("Status Filter", ["All", "COMPLETED", "PENDING", "FAILED"])

            filtered_docs = documents
            if search_query:
                filtered_docs = [
                    d for d in filtered_docs
                    if search_query in d.get("title", "").lower() or search_query in d.get("filename", "").lower()
                ]
            if status_filter != "All":
                filtered_docs = [d for d in filtered_docs if d.get("processing_status") == status_filter]

            st.markdown(f"Showing **{len(filtered_docs)}** of **{len(documents)}** indexed documents")

            for doc in filtered_docs:
                with st.expander(f"📄 **{doc.get('title')}** ({doc.get('filename')}) • `{doc.get('file_type', '').upper()}` • {doc.get('processing_status')}"):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("File Size", f"{doc.get('file_size_bytes', 0):,} B")
                    c2.metric("Total Chunks", doc.get("total_chunks", 0))
                    c3.metric("Total Tokens", doc.get("total_tokens", 0))
                    c4.write(f"**Uploaded:** {doc.get('created_at', '')[:19]}")

                    if doc.get("error_message"):
                        st.error(f"Processing Error: {doc.get('error_message')}")

                    btn_col1, btn_col2 = st.columns([1, 4])
                    with btn_col1:
                        if st.button("🗑️ Delete", key=f"del_{doc['id']}", type="secondary"):
                            try:
                                client.delete_document(doc["id"])
                                st.success(f"Document '{doc['title']}' deleted successfully.")
                                st.rerun()
                            except Exception as ex:
                                st.error(f"Failed to delete document: {ex}")

                    with btn_col2:
                        if st.checkbox("🔍 Inspect Vector Chunks", key=f"chk_toggle_{doc['id']}"):
                            try:
                                chunks = client.get_document_chunks(doc["id"])
                                if chunks:
                                    st.markdown(f"**Stored Vector Chunks ({len(chunks)}):**")
                                    for idx, chunk in enumerate(chunks, 1):
                                        meta = chunk.get("chunk_metadata", {})
                                        loc = f"Page {meta.get('page_number')}" if meta.get("page_number") else f"Chunk #{idx}"
                                        st.caption(f"**[{loc}]** • Tokens: {chunk.get('token_count')} • ID: `{chunk.get('id')}`")
                                        st.code(chunk.get("content", ""), language="text")
                                else:
                                    st.info("No chunks found for this document.")
                            except Exception as ex:
                                st.error(f"Error fetching chunks: {ex}")

    # --------------------------------------------------------------------------
    # Tab 2: Upload Documents
    # --------------------------------------------------------------------------
    with tab_upload:
        st.markdown("##### Upload Document for Ingestion & Vectorization")
        st.caption("Supported formats: **PDF (.pdf)**, **Plain Text (.txt)**, **Markdown (.md)**, and **CSV Tables (.csv)**.")

        uploaded_file = st.file_uploader(
            "Select an enterprise file to upload",
            type=["pdf", "txt", "md", "csv"],
            help="Files will be parsed, cleaned, deduplicated via SHA-256 content hashing, chunked (500 tokens), and embedded into pgvector.",
        )

        allow_dup = st.checkbox(
            "Force duplicate re-ingestion",
            value=False,
            help="If unchecked, uploads matching an existing content hash are deduplicated and rejected to save storage.",
        )

        if st.button("🚀 Ingest & Index Document", type="primary", disabled=uploaded_file is None):
            if uploaded_file is not None:
                file_bytes = uploaded_file.getvalue()
                with st.spinner(f"Ingesting '{uploaded_file.name}' into vector database..."):
                    try:
                        res = client.upload_document(
                            filename=uploaded_file.name,
                            file_bytes=file_bytes,
                            allow_duplicate=allow_dup,
                        )
                        if res.get("is_duplicate"):
                            st.warning(f"⚠️ **Duplicate Document Detected!** A document with identical content is already indexed (Document ID: `{res.get('document_id')}`).")
                        else:
                            st.success(
                                f"✅ **Ingestion Complete!** Created **{res.get('total_chunks')}** vector chunks "
                                f"({res.get('total_tokens')} tokens) indexed in pgvector."
                            )
                        st.json(res)
                    except BackendApiError as e:
                        st.error(f"Ingestion Failed: {e.message}")
                    except Exception as exc:
                        st.error(f"Network error: {str(exc)}")
