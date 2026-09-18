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

Note "malformed" exits after writing. A server that writes garbage and then
holds the pipe open hangs the client forever instead of failing — there is no
handshake timeout — so that variant cannot be asserted on without a test that
never returns. Raised as a follow-up rather than pinned here.
"""

from __future__ import annotations

import sys

from mcp.server.fastmcp import FastMCP


def build_server(mode: str) -> FastMCP:
    server = FastMCP("mock-mcp-server")

    @server.tool()
    def echo(text: str) -> dict:
        """Echo back the provided text."""
        if mode == "error":
            raise RuntimeError("mock server was asked to fail")
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
    build_server(mode).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
