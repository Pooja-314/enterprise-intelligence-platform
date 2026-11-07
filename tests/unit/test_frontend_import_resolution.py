"""Unit tests to verify frontend import path resolution under Streamlit execution conditions."""
import os
import subprocess
import sys
from pathlib import Path
import pytest


def test_frontend_app_direct_execution_path_resolution():
    """Tests executing frontend/app.py with sys.path[0] set to the frontend directory,
    reproducing the exact condition under `streamlit run frontend/app.py`.
    """
    python_exe = sys.executable
    project_root = Path(__file__).resolve().parent.parent.parent

    # Execute python frontend/app.py from project root
    result = subprocess.run(
        [python_exe, "frontend/app.py"],
        cwd=str(project_root),
        capture_output=True,
        text=True,
    )

    # Should exit cleanly (code 0) without ModuleNotFoundError: No module named 'frontend'
    assert result.returncode == 0, f"frontend/app.py failed with: {result.stderr}"
    assert "ModuleNotFoundError" not in result.stderr


def test_frontend_app_execution_from_frontend_directory():
    """Tests executing app.py from within the frontend/ directory."""
    python_exe = sys.executable
    frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"

    result = subprocess.run(
        [python_exe, "app.py"],
        cwd=str(frontend_dir),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"app.py from frontend/ failed with: {result.stderr}"
    assert "ModuleNotFoundError" not in result.stderr


def test_all_frontend_modules_importable():
    """Verifies that all frontend components and views can be imported without syntax or name errors."""
    import frontend.api_client
    import frontend.state
    import frontend.components.auth_view
    import frontend.components.header
    import frontend.components.citations
    import frontend.components.metrics_card
    import frontend.pages_views.dashboard_view
    import frontend.pages_views.documents_view
    import frontend.pages_views.chat_view
    import frontend.pages_views.agent_view
    import frontend.pages_views.observability_view
    import frontend.app

    assert hasattr(frontend.app, "main")
    assert hasattr(frontend.state, "init_session_state")
    assert hasattr(frontend.api_client, "BackendApiClient")
