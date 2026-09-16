"""Mock MCP server subprocess for integration testing.

Reads JSON-RPC requests from stdin (one per line) and writes canned
JSON-RPC responses to stdout. Behavior is controlled by the first CLI
argument (mode):

- "normal": respond to known methods with fixed results.
- "malformed": respond with non-JSON garbage.
- "crash": exit immediately with a non-zero code.
"""

from __future__ import annotations

import json
import sys


def _write(obj) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _write_raw(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _handle_normal(request: dict) -> None:
    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    if method == "list_tools":
        _write(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo back the provided text.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"text": {"type": "string"}},
                                "required": ["text"],
                            },
                        }
                    ]
                },
            }
        )
        return

    if method == "call_tool":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name == "echo":
            text = arguments.get("text", "")
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"echo: {text}"}],
                        "isError": False,
                    },
                }
            )
            return
        _write(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown tool: {name!r}",
                },
            }
        )
        return

    _write(
        {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method!r}",
            },
        }
    )


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "normal"

    if mode == "crash":
        sys.stderr.write("mock_mcp_server: simulated crash\n")
        sys.stderr.flush()
        return 3

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        if mode == "malformed":
            _write_raw("this is not valid json {{{")
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }
            )
            continue

        _handle_normal(request)

    return 0


if __name__ == "__main__":
    sys.exit(main())
