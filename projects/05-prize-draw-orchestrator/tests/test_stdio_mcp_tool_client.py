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


# A server that dies before the handshake surfaces as the ExceptionGroup the
# underlying anyio task group raises, NOT as the MCPToolError the client wraps
# tool-level failures in. These tests pin that as it stands rather than assert
# the nicer behaviour, so they document the gap instead of hiding it: a caller
# currently has to catch ExceptionGroup as well as MCPToolError to handle a
# server that fails to start. Worth a follow-up to wrap it.
@pytest.mark.parametrize(
    "mode, expected_in_message",
    [("crash", "TaskGroup"), ("normal", "TaskGroup")],
)
def test_server_that_never_completes_the_handshake_raises(
    mock_mcp_server_path, tmp_path, mode, expected_in_message
):
    if mode == "crash":
        client = _client(mock_mcp_server_path, mode="crash")
    else:
        # Same failure shape, reached a different way: the command itself is
        # missing, so the subprocess exits before speaking MCP at all.
        client = StdioMCPToolClient(
            command=sys.executable,
            args=[str(tmp_path / "does_not_exist.py")],
        )

    with pytest.raises(ExceptionGroup) as exc_info:
        client.call_tool("echo", {"text": "hello"})

    assert expected_in_message in str(exc_info.value)
