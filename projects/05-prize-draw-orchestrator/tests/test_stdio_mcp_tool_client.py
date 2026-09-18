"""Integration tests for StdioMCPToolClient against a mock MCP subprocess.

These spawn a real Python subprocess (tests/mock_mcp_server.py) and drive
StdioMCPToolClient end to end: process spawn, the MCP handshake, the tool
call, response unwrapping and error handling. Everything below goes through
`call_tool`, the client's only public entry point — an earlier version
probed for `_send_request`/`list_tools`/etc. and skipped when it found none,
which meant five of its six tests silently did nothing.
"""

from __future__ import annotations

import sys

import pytest

from mcp_client import MCPToolError, StdioMCPToolClient


def _client(mock_mcp_server_path: str, mode: str = "normal") -> StdioMCPToolClient:
    return StdioMCPToolClient(
        command=sys.executable,
        args=[mock_mcp_server_path, mode],
    )


def test_call_tool_returns_structured_content(mock_mcp_server_path: str):
    client = _client(mock_mcp_server_path)

    result = client.call_tool("echo", {"text": "hello"})

    assert result == {"echoed": "hello"}


def test_call_tool_round_trips_each_call_independently(mock_mcp_server_path: str):
    # Each call_tool spawns its own short-lived session, so a second call must
    # work without depending on state left by the first.
    client = _client(mock_mcp_server_path)

    assert client.call_tool("echo", {"text": "one"}) == {"echoed": "one"}
    assert client.call_tool("echo", {"text": "two"}) == {"echoed": "two"}


def test_tool_error_is_raised_as_mcp_tool_error(mock_mcp_server_path: str):
    client = _client(mock_mcp_server_path, mode="error")

    with pytest.raises(MCPToolError) as exc_info:
        client.call_tool("echo", {"text": "hello"})

    assert "echo" in str(exc_info.value)


def test_unknown_tool_is_raised_as_mcp_tool_error(mock_mcp_server_path: str):
    client = _client(mock_mcp_server_path)

    with pytest.raises(MCPToolError):
        client.call_tool("no_such_tool", {})


# A server that fails before or during the handshake surfaces as the
# ExceptionGroup the underlying anyio task group raises, NOT as the
# MCPToolError the client wraps tool-level failures in — so a caller currently
# has to catch both. Accept either rather than asserting the group
# specifically: the looser form keeps passing once the client is fixed to wrap
# these, instead of pinning the present gap in place.
@pytest.mark.parametrize("mode", ["crash", "malformed"])
def test_server_failing_before_handshake_raises(mock_mcp_server_path, mode):
    client = _client(mock_mcp_server_path, mode=mode)

    with pytest.raises((MCPToolError, ExceptionGroup)):
        client.call_tool("echo", {"text": "hello"})


def test_missing_server_command_raises(tmp_path):
    client = StdioMCPToolClient(
        command=sys.executable,
        args=[str(tmp_path / "does_not_exist.py")],
    )

    with pytest.raises((MCPToolError, ExceptionGroup)):
        client.call_tool("echo", {"text": "hello"})
