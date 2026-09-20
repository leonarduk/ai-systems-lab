"""Tests for app.py's chat orchestration logic. No live Gradio server, no
live DeepSeek calls — guardrails and llm are monkeypatched."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

import app  # noqa: E402
import gradio as gr  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def make_request(ip="1.2.3.4", session_hash="session-abc"):
    return SimpleNamespace(client=SimpleNamespace(host=ip), session_hash=session_hash)


class TestClientIpAndSessionId:
    def test_client_ip_from_request(self):
        assert app._client_ip(make_request(ip="9.9.9.9")) == "9.9.9.9"

    def test_client_ip_missing_request_is_unknown(self):
        assert app._client_ip(None) == "unknown"

    def test_client_ip_missing_client_is_unknown(self):
        assert app._client_ip(SimpleNamespace(client=None)) == "unknown"

    def test_session_id_from_request(self):
        assert app._session_id(make_request(session_hash="abc123")) == "abc123"

    def test_session_id_missing_request_is_unknown(self):
        assert app._session_id(None) == "unknown"

    def test_session_id_empty_hash_is_unknown(self):
        assert app._session_id(SimpleNamespace(session_hash="")) == "unknown"


class TestToConversation:
    def test_appends_message_to_history(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        conversation = app._to_conversation(history, "what's next?")
        assert conversation == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "what's next?"},
        ]

    def test_empty_history(self):
        assert app._to_conversation([], "hi") == [{"role": "user", "content": "hi"}]


class TestChat:
    def test_refusal_short_circuits_before_calling_llm(self, monkeypatch):
        monkeypatch.setattr(
            app.guardrails, "check_request", lambda *a: (False, "rate limited")
        )
        called = {"llm": False}
        monkeypatch.setattr(
            app.llm,
            "send_message",
            lambda *a, **k: called.update(llm=True) or ("x", {}),
        )

        reply = app.chat("hi", [], make_request())

        assert reply == "rate limited"
        assert called["llm"] is False

    def test_allowed_request_calls_llm_and_records_usage(self, monkeypatch):
        monkeypatch.setattr(app.guardrails, "check_request", lambda *a: (True, None))
        monkeypatch.setattr(
            app.llm,
            "send_message",
            lambda conversation, system_prompt: ("a reply", {"output_tokens": 10}),
        )
        recorded = {}
        monkeypatch.setattr(
            app.guardrails, "record_usage", lambda usage: recorded.update(usage)
        )

        reply = app.chat("hi", [], make_request())

        assert reply == "a reply"
        assert recorded == {"output_tokens": 10}


class TestBuildDemo:
    def test_builds_without_error(self):
        demo = app.build_demo()
        assert isinstance(demo, gr.Blocks)


class TestHealthEndpoint:
    def test_health_returns_200_with_ok_json(self):
        client = TestClient(app.build_health_app())
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_has_no_side_effects(self, monkeypatch):
        # Guard against the health endpoint accidentally calling into
        # guardrails/llm/tools — it must be a pure liveness probe.
        def boom(*a, **k):
            raise AssertionError("health check must not call external code")

        monkeypatch.setattr(app.guardrails, "check_request", boom)
        monkeypatch.setattr(app.guardrails, "record_usage", boom)
        monkeypatch.setattr(app.llm, "send_message", boom)

        client = TestClient(app.build_health_app())
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_is_still_reachable_when_mounted_with_the_chat_ui(self):
        # The realistic failure mode: build_health_app() passes in isolation
        # but the route silently disappears once actually combined with the
        # Gradio demo for a real run. demo.launch(app_kwargs={"app": ...})
        # looked plausible but silently drops the custom app entirely — this
        # was only caught by launching it and hitting both routes for real.
        # gr.mount_gradio_app is what actually merges the two correctly.
        client = TestClient(app.build_app())
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_chat_ui_still_serves_at_root_when_health_app_is_mounted(self):
        client = TestClient(app.build_app())
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
