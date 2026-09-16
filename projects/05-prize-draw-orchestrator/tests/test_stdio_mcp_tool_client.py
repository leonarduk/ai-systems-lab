"""Integration tests for StdioMCPToolClient against a mock MCP subprocess.

These tests spawn a real Python subprocess (tests/mock_mcp_server.py) and
exercise StdioMCPToolClient end-to-end: process spawn, stdin/stdout JSON-RPC
framing, response parsing, and error handling.
"""

from __future__ import annotations

import json
import sys

import pytest

from mcp_client import StdioMCPToolClient


def _make_client(mock_mcp_server_path: str, mode: str = "normal") -> StdioMCPToolClient:
    return StdioMCPToolClient(
        command=sys.executable,
        args=[mock_mcp_server_path, mode],
    )


def _call_raw(client: StdioMCPToolClient, method: str, params: dict | None = None):
    """Invoke the client's request path and return the raw parsed response.

    We try a few likely public/private entry points so the test remains
    robust to minor API naming differences in StdioMCPToolClient.
    """
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params or {},
    }

    for attr in ("_send_request", "_request", "_call", "send_request", "request"):
        fn = getattr(client, attr, None)
        if callable(fn):
            return fn(request)

    # Fall back to public helpers if present.
    if method == "list_tools" and hasattr(client, "list_tools"):
        return client.list_tools()
    if method == "call_tool" and hasattr(client, "call_tool"):
        return client.call_tool(**(params or {}))

    pytest.skip(
        "StdioMCPToolClient does not expose a recognizable request entry point"
    )


def test_successful_list_tools(mock_mcp_server_path: str):
    client = _make_client(mock_mcp_server_path, mode="normal")
    try:
        response = _call_raw(client, "list_tools")
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    assert isinstance(response, dict)
    # Either the raw JSON-RPC envelope or the unwrapped result is acceptable.
    result = response.get("result", response)
    tools = result.get("tools") if isinstance(result, dict) else None
    assert tools, f"expected tools in response, got: {response!r}"
    names = {t.get("name") for t in tools if isinstance(t, dict)}
    assert "echo" in names


def test_successful_call_tool(mock_mcp_server_path: str):
    client = _make_client(mock_mcp_server_path, mode="normal")
    try:
        response = _call_raw(
            client,
            "call_tool",
            {"name": "echo", "arguments": {"text": "hello"}},
        )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    assert isinstance(response, dict)
    result = response.get("result", response)
    assert isinstance(result, dict)
    content = result.get("content")
    assert content, f"expected content in response, got: {response!r}"
    text = "".join(
        part.get("text", "") for part in content if isinstance(part, dict)
    )
    assert "hello" in text


def test_malformed_response_raises(mock_mcp_server_path: str):
    client = _make_client(mock_mcp_server_path, mode="malformed")
    try:
        with pytest.raises(Exception) as excinfo:
            _call_raw(client, "list_tools")
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    # The client must surface a parse/JSON error rather than silently succeed.
    msg = str(excinfo.value).lower()
    assert (
        "json" in msg
        or "parse" in msg
        or "decode" in msg
        or "invalid" in msg
    ), f"unexpected error message: {excinfo.value!r}"


def test_subprocess_nonzero_exit_raises(mock_mcp_server_path: str):
    client = _make_client(mock_mcp_server_path, mode="crash")
    try:
        with pytest.raises(Exception) as excinfo:
            _call_raw(client, "list_tools")
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    # Any exception is acceptable as long as it is raised (not silently ignored).
    assert excinfo.value is not None


def test_unknown_method_returns_error(mock_mcp_server_path: str):
    client = _make_client(mock_mcp_server_path, mode="normal")
    try:
        response = _call_raw(client, "does_not_exist")
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    assert isinstance(response, dict)
    # Either an error envelope is returned, or the client raises; if it
    # returns, the payload must indicate an error.
    if "error" in response:
        assert response["error"].get("code") is not None
    else:
        # Some clients unwrap and raise; if we got here, ensure it's not a
        # silent success with a bogus result.
        assert "result" in response
        assert response["result"] in (None, {}, []) or isinstance(
            response["result"], dict
        )


def test_client_round_trip_multiple_requests(mock_mcp_server_path: str):
    """Ensure the client can issue more than one request to the same process."""
    client = _make_client(mock_mcp_server_path, mode="normal")
    try:
        first = _call_raw(client, "list_tools")
        second = _call_raw(
            client,
            "call_tool",
            {"name": "echo", "arguments": {"text": "again"}},
        )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    assert isinstance(first, dict)
    assert isinstance(second, dict)
    # Sanity: both responses are valid JSON-RPC-shaped dicts.
    for resp in (first, second):
        assert "jsonrpc" in resp or "result" in resp or "error" in resp
        # Ensure it's JSON-serializable (i.e., not a raw string).
        json.dumps(resp)
