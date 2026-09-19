"""Mock MCP server subprocess, used to integration-test StdioMCPToolClient.

StdioMCPToolClient talks to its server through the `mcp` library's stdio
client, which performs a real MCP handshake (`initialize`, then
`tools/call`) before any tool runs. A hand-rolled line-based JSON-RPC
responder therefore cannot serve it — the client rejects it with
"Method not found: 'initialize'" before the first tool call. This builds
on the library's own server instead, so the protocol is genuine and only
the tool behaviour is faked.

The first CLI argument selects the behaviour under test:

- "normal":    `echo` returns its argument as structured content.
- "error":     `echo` raises, so the client sees a tool error.
- "crash":     exit non-zero before serving, to model a server that dies on spawn.
- "malformed": write a non-JSON line to stdout and exit, to model a server that
               answers with something the client cannot parse.
- "hang":      write a non-JSON line but keep the pipe open, to model a server
               that neither answers nor dies.
- "slow":      handshake normally, then never return from `echo`, to model a
               server that goes quiet only once the tool is called.

"malformed" and "hang" differ only in whether the pipe closes, and that is the
whole point of the pair: closing it turns the client's parse failure into a
prompt error, while holding it open leaves the client waiting on a reply that
never comes. "hang" is therefore only safe to test against a client that
imposes its own handshake timeout.
"""

from __future__ import annotations

import asyncio
import sys

from mcp.server.fastmcp import FastMCP

# Long enough that only the client's own timeout can end a "slow" call, short
# enough that a stray process cannot outlive the test run by much.
SLOW_TOOL_SECONDS = 300


def build_server(mode: str) -> FastMCP:
    server = FastMCP("mock-mcp-server")

    @server.tool()
    async def echo(text: str) -> dict:
        """Echo back the provided text."""
        if mode == "error":
            raise RuntimeError("mock server was asked to fail")
        if mode == "slow":
            await asyncio.sleep(SLOW_TOOL_SECONDS)
        return {"echoed": text}

    return server


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
    if mode == "crash":
        return 1
    if mode == "malformed":
        # Not JSON, so the client's JSONRPCMessage parse fails. Exiting closes
        # the pipe, which turns that parse failure into a prompt error rather
        # than a hang.
        sys.stdout.write("this is not json-rpc\n")
        sys.stdout.flush()
        return 0
    if mode == "hang":
        # Same unparseable output, but the pipe stays open: the client's parse
        # failure is reported to a session that ignores it, and the handshake
        # reply never arrives. Reading stdin until the client closes it keeps
        # this process alive for exactly as long as the client holds on.
        sys.stdout.write("this is not json-rpc\n")
        sys.stdout.flush()
        for _ in sys.stdin:
            sys.stdout.write("still not json\n")
            sys.stdout.flush()
        return 0
    build_server(mode).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
