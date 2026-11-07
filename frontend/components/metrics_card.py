"""KPI Metric Cards Component."""
from typing import Any
import streamlit as st


def render_metric_card(title: str, value: Any, delta: str = "", help_text: str = "") -> None:
    """Renders a styled enterprise KPI card."""
    st.metric(label=title, value=value, delta=delta if delta else None, help=help_text if help_text else None)
