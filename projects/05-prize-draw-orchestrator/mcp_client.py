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

import json
from typing import Any, Protocol

DEFAULT_TIMEOUT_SECONDS = 30.0
"""Seconds to wait for the handshake, and again for the tool call, by default.

Generous enough for a server that has to start an interpreter and import its
dependencies, short enough that a misbehaving server cannot stall a polling
run indefinitely.
"""


class MCPToolError(RuntimeError):
    """Raised when an MCP tool call fails or returns an unusable result."""


def _find_tool_error(exc: BaseException) -> MCPToolError | None:
    """Return the first `MCPToolError` in `exc`, descending into exception groups.

    The `mcp` stdio client runs its plumbing in an anyio task group, which
    re-raises whatever escapes its body wrapped in an `ExceptionGroup`. When
    that body is one of our own `MCPToolError`s we want the original back, not
    a group around it.
    """
    if isinstance(exc, MCPToolError):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        for sub_exc in exc.exceptions:
            found = _find_tool_error(sub_exc)
            if found is not None:
                return found
    return None


def _first_leaf(exc: BaseException) -> BaseException:
    """Return the first non-group exception inside `exc` (or `exc` itself)."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


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

    Every failure this client can run into — a server that won't spawn, one
    that never completes the handshake, one that never answers the call, or a
    tool that returns an error — surfaces as `MCPToolError`, so callers have a
    single exception type to catch.
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = DEFAULT_TIMEOUT_SECONDS,
    ):
        """Store the server subprocess command/args/env for later lazy connection.

        `timeout` bounds the handshake and the tool call separately, in
        seconds; `None` disables it (and with it the only protection against a
        server that goes quiet mid-conversation).
        """
        self.command = command
        self.args = args or []
        self.env = env
        self.timeout = timeout

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run one MCP tool call in a short-lived stdio session.

        A fresh session per call keeps this client simple and safe to use
        from a synchronous orchestrator loop; it costs a process spawn per
        call, which is acceptable for the polling cadence this tool runs at
        (see README for scheduling guidance).
        """
        import asyncio

        return asyncio.run(self._call_tool_async(name, arguments))

    async def _call_tool_async(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            import anyio

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
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Without these deadlines a server that writes something
                    # unparseable and then holds its stdout pipe open leaves us
                    # waiting forever: the library reports the parse failure to
                    # the session, which ignores it, and no reply ever arrives.
                    # Only a server that *closes* the pipe fails promptly.
                    try:
                        with anyio.fail_after(self.timeout):
                            await session.initialize()
                    except TimeoutError as exc:
                        raise MCPToolError(
                            f"MCP server {self.command!r} did not complete the "
                            f"initialize handshake within {self.timeout} seconds."
                        ) from exc

                    try:
                        with anyio.fail_after(self.timeout):
                            result = await session.call_tool(name, arguments=arguments)
                    except TimeoutError as exc:
                        raise MCPToolError(
                            f"MCP tool '{name}' did not return within "
                            f"{self.timeout} seconds."
                        ) from exc
        except MCPToolError:
            raise
        except Exception as exc:
            # Anything that goes wrong around the session — a command that does
            # not exist, a server that exits before the handshake — reaches us
            # as the ExceptionGroup anyio's task group raises. Unwrap it so
            # callers only ever have to catch MCPToolError.
            tool_error = _find_tool_error(exc)
            if tool_error is not None:
                raise tool_error
            cause = _first_leaf(exc)
            raise MCPToolError(
                f"MCP server {self.command!r} failed before tool '{name}' "
                f"returned a result: {cause!r}"
            ) from cause

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
