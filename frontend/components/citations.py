"""Citation and Grounding UI Component."""
import re
from typing import Any, Dict, List, Optional
import streamlit as st


def render_citations(
    citations: List[Any],
    verification_data: Optional[Dict[str, Any]] = None,
    is_insufficient_evidence: bool = False,
) -> None:
    """Renders structured citations and claim verification badges for AI outputs.

    Each citation is rendered in its own separate Streamlit expander, displaying:
    - Document name
    - Page number (or row/section)
    - Chunk ID
    - Actual retrieved chunk text (with readable preview and expandable full text).
    """
    if is_insufficient_evidence:
        st.info("ℹ️ **Notice:** Insufficient factual evidence found in the enterprise knowledge base to ground a verified response.")
        return

    # 1. Claim Verification status banner if available (preserved from original)
    if verification_data:
        is_verified = verification_data.get("is_verified", False)
        confidence = verification_data.get("confidence_score", 0.0)
        supported = verification_data.get("supported_claims", 0)
        total = verification_data.get("total_claims", 0)

        badge_color = "#16A34A" if is_verified else "#D97706"
        badge_text = "GROUNDED & VERIFIED" if is_verified else "PARTIALLY SUPPORTED"

        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 8px; margin-top: 8px; margin-bottom: 6px;">
                <span style="background: {badge_color}15; color: {badge_color}; border: 1px solid {badge_color}; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px;">
                    🛡️ {badge_text}
                </span>
                <span style="font-size: 12px; color: #64748B;">
                    Grounding Confidence: <b>{confidence * 100:.1f}%</b> ({supported}/{total} claims grounded)
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not citations:
        return

    st.markdown(f"**📚 Verified Source Citations ({len(citations)} references):**")

    # 2. Render each citation in its own separate expander
    for i, c in enumerate(citations, 1):
        doc_title = "Document"
        chunk_id = "N/A"
        page = None
        row = None
        header = None
        content = ""

        if isinstance(c, dict):
            doc_title = c.get("document_title") or c.get("title") or "Document"
            chunk_id = c.get("chunk_id") or "N/A"
            page = c.get("page_number")
            row = c.get("row_number")
            header = c.get("section_header")
            content = c.get("content") or c.get("snippet") or c.get("text") or ""
        elif hasattr(c, "document_title"):
            doc_title = getattr(c, "document_title", "Document")
            chunk_id = getattr(c, "chunk_id", "N/A")
            page = getattr(c, "page_number", None)
            row = getattr(c, "row_number", None)
            header = getattr(c, "section_header", None)
            content = getattr(c, "content", "") or getattr(c, "snippet", "")
        elif isinstance(c, str):
            # Parse string format e.g.: [Doc: Title, Page: 2, Chunk: abc-123]
            doc_match = re.search(r"Doc:\s*([^,\]]+)", c)
            page_match = re.search(r"Page:\s*(\d+)", c)
            row_match = re.search(r"Row:\s*(\d+)", c)
            chunk_match = re.search(r"Chunk:\s*([a-zA-Z0-9_-]+)", c)

            doc_title = doc_match.group(1).strip() if doc_match else c
            page = int(page_match.group(1)) if page_match else None
            row = int(row_match.group(1)) if row_match else None
            chunk_id = chunk_match.group(1).strip() if chunk_match else "N/A"
            content = ""

        # Format location label
        loc_str = f"Page {page}" if page is not None else (f"Row {row}" if row is not None else (f"Section: {header}" if header else "General Section"))
        short_id = chunk_id[:8] if chunk_id != "N/A" else "N/A"

        expander_label = f"📄 [{i}] {doc_title} • {loc_str} • Chunk: {short_id}"
        with st.expander(expander_label, expanded=False):
            # Key Metadata
            st.markdown(f"- **Document Name:** `{doc_title}`")
            st.markdown(f"- **Page / Location:** {loc_str}")
            st.markdown(f"- **Chunk ID:** `{chunk_id}`")

            # Actual retrieved chunk text
            clean_text = content.strip() if isinstance(content, str) else ""
            if clean_text:
                preview_length = 200
                if len(clean_text) > preview_length:
                    preview = clean_text[:preview_length].rstrip() + "..."
                    st.markdown("**Retrieved Source Text (Preview):**")
                    st.info(preview)
                    with st.expander("📖 View Full Retrieved Text", expanded=False):
                        st.markdown(clean_text)
                else:
                    st.markdown("**Retrieved Source Text:**")
                    st.info(clean_text)
            else:
                st.caption("ℹ️ *(Actual chunk text preview unavailable for this citation record)*")
