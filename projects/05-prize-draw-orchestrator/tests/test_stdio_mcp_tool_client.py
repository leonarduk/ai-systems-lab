"""Integration tests for StdioMCPToolClient against a mock MCP subprocess.

These spawn a real Python subprocess (tests/mock_mcp_server.py) and drive
StdioMCPToolClient end to end: process spawn, the MCP handshake, the tool
call, response unwrapping and error handling. Everything below goes through
`call_tool`, the client's only public entry point — an earlier version
probed for `_send_request`/`list_tools`/etc. and skipped when it found none,
which meant five of its six tests silently did nothing.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time

import pytest

from mcp_client import MCPToolError, StdioMCPToolClient


def _client(
    mock_mcp_server_path: str, mode: str = "normal", **kwargs
) -> StdioMCPToolClient:
    return StdioMCPToolClient(
        command=sys.executable,
        args=[mock_mcp_server_path, mode],
        **kwargs,
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


@pytest.mark.parametrize("mode", ["crash", "malformed"])
def test_server_failing_before_handshake_raises(mock_mcp_server_path, mode):
    client = _client(mock_mcp_server_path, mode=mode)

    with pytest.raises(MCPToolError):
        client.call_tool("echo", {"text": "hello"})


def test_missing_server_command_raises(tmp_path):
    client = StdioMCPToolClient(
        command=sys.executable,
        args=[str(tmp_path / "does_not_exist.py")],
    )

    with pytest.raises(MCPToolError):
        client.call_tool("echo", {"text": "hello"})


def test_unresponsive_server_times_out_instead_of_hanging(mock_mcp_server_path):
    # The "hang" server writes output the client cannot parse and then keeps
    # its pipe open, so the client waits on a handshake reply that never
    # arrives. Before the connect timeout existed this blocked forever — the
    # orchestrator polls on a schedule, so one such server stalled the whole
    # run silently.
    #
    # Run it on a daemon thread and bound the join rather than calling straight
    # through: if the timeout regresses, this fails in 20s with a message
    # naming the cause instead of hanging CI until the job is killed.
    client = _client(mock_mcp_server_path, mode="hang", connect_timeout=2.0)
    outcome = {}

    def call():
        try:
            outcome["returned"] = client.call_tool("echo", {"text": "hello"})
        except BaseException as exc:  # noqa: BLE001 - recorded and re-checked below
            outcome["raised"] = exc

    worker = threading.Thread(target=call, daemon=True)
    worker.start()
    worker.join(timeout=20)

    assert not worker.is_alive(), (
        "call_tool did not return within 20s despite a 2s connect timeout — "
        "the client is hanging on an unresponsive server again"
    )
    assert isinstance(outcome.get("raised"), MCPToolError), outcome
    assert "did not respond within the timeout" in str(outcome["raised"])


def test_timeout_does_not_leave_the_server_running(mock_mcp_server_path):
    # A timeout that abandoned the subprocess would leak one process per poll.
    def server_pids():
        listing = subprocess.run(
            ["ps", "-eo", "pid,args"], capture_output=True, text=True
        ).stdout
        return {
            line.split(None, 1)[0]
            for line in listing.splitlines()
            if f"{mock_mcp_server_path} hang" in line
        }

    before = server_pids()
    client = _client(mock_mcp_server_path, mode="hang", connect_timeout=2.0)
    with pytest.raises(MCPToolError):
        client.call_tool("echo", {"text": "hello"})

    # The client tears the process down as it unwinds; give the OS a moment to
    # reap it before looking.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and server_pids() - before:
        time.sleep(0.2)
    assert server_pids() - before == set()
