"""Integration tests for StdioMCPToolClient against a mock MCP subprocess.

These spawn a real Python subprocess (tests/mock_mcp_server.py) and drive
StdioMCPToolClient end to end: process spawn, the MCP handshake, the tool
call, response unwrapping and error handling. Everything below goes through
`call_tool`, the client's only public entry point — an earlier version
probed for `_send_request`/`list_tools`/etc. and skipped when it found none,
which meant five of its six tests silently did nothing.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import time

import pytest

from mcp_client import MCPToolError, StdioMCPToolClient, _leaves


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

    with pytest.raises(MCPToolError) as exc_info:
        client.call_tool("echo", {"text": "hello"})

    # Startup failures are wrapped, not swallowed.
    assert exc_info.value.__cause__ is not None


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
    raised = outcome.get("raised")
    assert isinstance(raised, MCPToolError), outcome
    assert "did not respond within the timeout" in str(raised)
    # The underlying failure is kept as __cause__ rather than discarded, so a
    # traceback still shows what actually went wrong inside the session.
    assert raised.__cause__ is not None


@pytest.mark.skipif(
    sys.platform == "win32", reason="process listing below is POSIX-only (ps)"
)
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


def test_call_finishing_inside_its_budget_is_not_cut_short(mock_mcp_server_path):
    # The other side of the timeout: a tool that takes real time must still
    # succeed. Without this, tightening the default call_timeout would look
    # fine — every other test here either fails fast or never returns.
    client = _client(mock_mcp_server_path, mode="slow", call_timeout=15.0)

    assert client.call_tool("echo", {"text": "hello"}) == {"echoed": "hello"}


def test_call_exceeding_its_budget_times_out(mock_mcp_server_path):
    # ...and the same call is cut off when the budget is below the work, so the
    # test above is passing because the budget is respected rather than because
    # the timeout never fires.
    client = _client(mock_mcp_server_path, mode="slow", call_timeout=0.2)

    with pytest.raises(MCPToolError) as exc_info:
        client.call_tool("echo", {"text": "hello"})

    assert "did not respond within the timeout" in str(exc_info.value)


def test_outer_backstop_bounds_a_stalled_session_setup(monkeypatch):
    # The inner timeouts only start once the session is up, so a setup that
    # never returns would slip past both. The backstop covers that. Run it on a
    # daemon thread with a bounded join for the same reason as the hang test
    # above: without the backstop there is nothing to end this, and the test
    # should fail rather than stall CI.
    import mcp.client.stdio as stdio_module

    class StallingSession:
        async def __aenter__(self):
            await asyncio.sleep(3600)

        async def __aexit__(self, *exc_info):
            return False

    monkeypatch.setattr(
        stdio_module, "stdio_client", lambda *args, **kwargs: StallingSession()
    )
    client = StdioMCPToolClient(
        command=sys.executable,
        args=["-c", "pass"],
        connect_timeout=0.2,
        call_timeout=0.2,
    )
    outcome = {}

    def call():
        try:
            outcome["returned"] = client.call_tool("echo", {"text": "hello"})
        except BaseException as exc:  # noqa: BLE001 - recorded and re-checked below
            outcome["raised"] = exc

    worker = threading.Thread(target=call, daemon=True)
    worker.start()
    worker.join(timeout=20)

    assert not worker.is_alive(), "session setup was never bounded by the backstop"
    assert isinstance(outcome.get("raised"), MCPToolError), outcome
    assert "did not respond within the timeout" in str(outcome["raised"])


def test_timeout_is_reported_even_when_it_is_not_the_first_group_member(monkeypatch):
    # The real hang produces a single-member group, so nothing above
    # distinguishes "scan every member" from "look at the first". Drive the
    # branch directly: anyio can collect a reader-task failure alongside the
    # timeout and promises no order, and reporting that as a generic failure
    # would hide the actual cause.
    import mcp.client.stdio as stdio_module

    def exploding_stdio_client(*args, **kwargs):
        raise ExceptionGroup(
            "session failed", [ValueError("reader died"), TimeoutError()]
        )

    monkeypatch.setattr(stdio_module, "stdio_client", exploding_stdio_client)
    client = StdioMCPToolClient(command=sys.executable, args=["-c", "pass"])

    with pytest.raises(MCPToolError) as exc_info:
        client.call_tool("echo", {"text": "hello"})

    assert "did not respond within the timeout" in str(exc_info.value)


def test_interrupt_is_not_converted_into_a_tool_error(monkeypatch):
    # A caller asking to stop must not have that turned into MCPToolError, or
    # they cannot interrupt the orchestrator's poll loop.
    import mcp.client.stdio as stdio_module

    def exploding_stdio_client(*args, **kwargs):
        raise BaseExceptionGroup("interrupted", [KeyboardInterrupt()])

    monkeypatch.setattr(stdio_module, "stdio_client", exploding_stdio_client)
    client = StdioMCPToolClient(command=sys.executable, args=["-c", "pass"])

    with pytest.raises(BaseExceptionGroup):
        client.call_tool("echo", {"text": "hello"})


class TestLeaves:
    """`_leaves` decides whether a failure is reported as a timeout.

    The timeout is raised inside anyio's task group, so it reaches the client
    wrapped. A task group can collect several exceptions — a timeout alongside
    a cancellation or a broken pipe from the reader task — and does not promise
    an order, so scanning only the first member would misreport a timeout as a
    generic failure depending on which arrived first.
    """

    def test_plain_exception_is_its_own_leaf(self):
        exc = ValueError("boom")
        assert _leaves(exc) == [exc]

    def test_group_members_are_flattened(self):
        first, second = ValueError("a"), TimeoutError()
        assert _leaves(ExceptionGroup("g", [first, second])) == [first, second]

    def test_nested_groups_are_flattened(self):
        deep = TimeoutError()
        group = ExceptionGroup(
            "outer", [ValueError("a"), ExceptionGroup("inner", [deep])]
        )
        assert deep in _leaves(group)

    def test_timeout_is_found_when_it_is_not_first(self):
        group = ExceptionGroup("g", [ValueError("reader died"), TimeoutError()])
        assert any(isinstance(leaf, TimeoutError) for leaf in _leaves(group))
