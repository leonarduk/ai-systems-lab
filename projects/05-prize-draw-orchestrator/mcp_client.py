"""Client interface for calling the prize-draw MCP server's tools.

This module defines a small, generic protocol (`MCPToolClient`) for calling
named MCP tools with a dict of arguments and getting a dict back — the same
shape as the real Model Context Protocol's `ClientSession.call_tool`. The
orchestrator (`orchestrator.py`) is written against this protocol only, so it
does not care whether it's talking to a stub, a fake used in tests, or a real
MCP server subprocess.

As of this writing, issue #22 (the MCP server exposing `search_draws`,
`parse_entry_page`, `submit_entry`, `check_log`) has not been merged into
`main`, so `StdioMCPToolClient` below is a genuine MCP stdio client
implementation (using the official `mcp` Python SDK, the same package the
other servers in `projects/01-mcp-server-suite` are built on) but has not
been exercised against a live server. Once issue #22 lands, point
`MCP_SERVER_COMMAND` / `MCP_SERVER_ARGS` at its entry point and this client
should work unmodified, since it only depends on the tool-call contract
documented in the issue.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Protocol

# The handshake is local and should be near-instant: a server that has not
# answered `initialize` within this long is not going to. Tool calls get a
# much larger budget because they do real work (fetching and parsing pages),
# so a single shared timeout would either cut those short or leave a dead
# handshake hanging for a minute.
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_CALL_TIMEOUT_SECONDS = 60.0


def _leaves(exc: BaseException) -> list[BaseException]:
    """Flatten a (possibly nested) exception group to its non-group members.

    ExceptionGroup's own str() is just "unhandled errors in a TaskGroup", which
    says nothing about what failed, so callers need the members. All of them:
    a task group can collect several, and the one that matters — a timeout —
    is not reliably first.
    """
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for sub in exc.exceptions for leaf in _leaves(sub)]
    return [exc]


class MCPToolError(RuntimeError):
    """Raised when an MCP tool call fails or returns an unusable result."""


class MCPToolClient(Protocol):
    """Minimal interface for calling MCP tools by name.

    Any implementation — a stub, a test fake, or a real MCP client session —
    only needs to satisfy this one method.
    """

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call the named tool with `arguments` and return its result as a dict."""
        ...


class StdioMCPToolClient:
    """Calls tools on a real MCP server process over stdio.

    Connects lazily on first use and reuses the connection for subsequent
    calls. Requires the `mcp` package (see requirements.txt).
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        call_timeout: float = DEFAULT_CALL_TIMEOUT_SECONDS,
    ):
        """Store the server subprocess command/args/env for later lazy connection."""
        self.command = command
        self.args = args or []
        self.env = env
        self.connect_timeout = connect_timeout
        self.call_timeout = call_timeout

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run one MCP tool call in a short-lived stdio session.

        A fresh session per call keeps this client simple and safe to use
        from a synchronous orchestrator loop; it costs a process spawn per
        call, which is acceptable for the polling cadence this tool runs at
        (see README for scheduling guidance).
        """
        return asyncio.run(self._call_tool_async(name, arguments))

    def _timeout_message(self, name: str) -> str:
        return (
            f"MCP server {self.command!r} did not respond within the timeout "
            f"while calling tool {name!r} (connect {self.connect_timeout}s, "
            f"call {self.call_timeout}s). The server may be unresponsive, or may "
            "be writing output the client cannot parse as JSON-RPC — without this "
            "timeout that case blocks forever, because the client keeps waiting "
            "on a pipe the server never closes."
        )

    async def _call_tool_async(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except (
            ImportError
        ) as exc:  # pragma: no cover - exercised only without the mcp extra installed
            raise MCPToolError(
                "The 'mcp' package is required for StdioMCPToolClient. Install it with "
                "`pip install mcp` (see requirements.txt)."
            ) from exc

        server_params = StdioServerParameters(
            command=self.command, args=self.args, env=self.env
        )
        # Set when a timeout fires inside the session. The exception itself
        # does not survive the trip out: cancelling the call tears the stream
        # down, that teardown fails with BrokenResourceError, and the enclosing
        # task group raises a fresh group containing only *that* — the
        # TimeoutError is replaced, not collected. Without this flag a call
        # timeout is indistinguishable from a transport failure.
        timed_out = False
        try:
            # The inner timeouts bound the two waits that can actually stall.
            # The outer one is a backstop for everything they do not cover —
            # spawning the subprocess, and the context managers' own setup and
            # teardown — so no path through here is unbounded.
            async with asyncio.timeout(self.connect_timeout + self.call_timeout):
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        try:
                            async with asyncio.timeout(self.connect_timeout):
                                await session.initialize()
                            async with asyncio.timeout(self.call_timeout):
                                result = await session.call_tool(
                                    name, arguments=arguments
                                )
                        except TimeoutError:
                            timed_out = True
                            raise
        except TimeoutError as exc:
            raise MCPToolError(self._timeout_message(name)) from exc
        except BaseExceptionGroup as group:
            # stdio_client runs its reader and writer in an anyio task group, so
            # anything that goes wrong before or during the session arrives
            # wrapped. Unwrap so callers only ever have to handle MCPToolError,
            # and keep the original as __cause__.
            leaves = _leaves(group)
            # Checked before the timeout below, deliberately: a group can carry
            # both, and a caller asking to stop outranks reporting why the call
            # was slow. Never convert an interrupt into a tool error — swallowing
            # it here would leave them unable to stop the poll loop.
            if any(
                isinstance(leaf, (KeyboardInterrupt, SystemExit)) for leaf in leaves
            ):
                raise
            # Scan every member rather than taking the first: a timeout can be
            # collected alongside a cancellation or a broken-pipe error from the
            # reader task, and anyio does not promise which comes first.
            if timed_out or any(isinstance(leaf, TimeoutError) for leaf in leaves):
                raise MCPToolError(self._timeout_message(name)) from group
            raise MCPToolError(
                f"MCP server {self.command!r} failed while calling tool {name!r}: "
                f"{leaves[0]!r}"
            ) from group

        if getattr(result, "isError", False):
            raise MCPToolError(f"MCP tool '{name}' returned an error: {result}")

        structured = getattr(result, "structuredContent", None)
        if isinstance(structured, dict):
            return structured

        for block in getattr(result, "content", []):
            text = getattr(block, "text", None)
            if text:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"raw": text}
        return {}
