"""Shared pytest fixtures for the prize-draw-orchestrator test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure the project root is importable so tests can `import mcp_client`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

MOCK_SERVER_PATH = Path(__file__).resolve().parent / "mock_mcp_server.py"


@pytest.fixture
def mock_mcp_server_path() -> str:
    """Absolute path to the mock MCP server script."""
    assert MOCK_SERVER_PATH.exists(), f"missing mock server: {MOCK_SERVER_PATH}"
    return str(MOCK_SERVER_PATH)


@pytest.fixture
def mock_mcp_env() -> dict:
    """Environment for the mock subprocess (inherit + force unbuffered)."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return env
