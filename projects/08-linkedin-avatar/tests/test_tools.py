"""Tests for avatar/tools.py. No test performs a real network call."""

import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import avatar.tool_definitions as tool_definitions  # noqa: E402
import avatar.tools as tools  # noqa: E402


class FakeResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise tools.requests.HTTPError(f"status {self.status_code}")


@pytest.fixture(autouse=True)
def _clear_pushover_env(monkeypatch):
    monkeypatch.delenv("PUSHOVER_USER", raising=False)
    monkeypatch.delenv("PUSHOVER_TOKEN", raising=False)


@pytest.fixture(autouse=True)
def _clear_telegram_env(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)


class TestPushoverNotify:
    def test_logs_when_credentials_missing(self):
        result = tools._pushover_notify("title", "message")
        assert result["status"] == "logged"

    def test_posts_when_credentials_present(self, monkeypatch):
        monkeypatch.setenv("PUSHOVER_USER", "u")
        monkeypatch.setenv("PUSHOVER_TOKEN", "t")

        captured = {}

        def fake_post(url, data, timeout):
            captured["url"] = url
            captured["data"] = data
            return FakeResponse(200)

        monkeypatch.setattr(tools.requests, "post", fake_post)

        result = tools._pushover_notify("title", "message")

        assert result["status"] == "sent"
        assert captured["url"] == tools.PUSHOVER_URL
        assert captured["data"]["title"] == "title"
        assert captured["data"]["message"] == "message"
        assert captured["data"]["token"] == "t"
        assert captured["data"]["user"] == "u"

    def test_http_failure_does_not_raise(self, monkeypatch):
        monkeypatch.setenv("PUSHOVER_USER", "u")
        monkeypatch.setenv("PUSHOVER_TOKEN", "t")

        def fake_post(url, data, timeout):
            return FakeResponse(500)

        monkeypatch.setattr(tools.requests, "post", fake_post)

        result = tools._pushover_notify("title", "message")

        assert result["status"] == "failed"

    def test_connection_error_does_not_raise(self, monkeypatch):
        monkeypatch.setenv("PUSHOVER_USER", "u")
        monkeypatch.setenv("PUSHOVER_TOKEN", "t")

        def fake_post(url, data, timeout):
            raise tools.requests.ConnectionError("no network")

        monkeypatch.setattr(tools.requests, "post", fake_post)

        result = tools._pushover_notify("title", "message")

        assert result["status"] == "failed"


class TestTelegramNotify:
    def test_logs_when_credentials_missing(self):
        result = tools._telegram_notify("title", "message")
        assert result["status"] == "logged"

    def test_posts_when_credentials_present(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "cid")

        captured = {}

        def fake_post(url, data, timeout):
            captured["url"] = url
            captured["data"] = data
            return FakeResponse(200)

        monkeypatch.setattr(tools.requests, "post", fake_post)

        result = tools._telegram_notify("title", "message")

        assert result["status"] == "sent"
        assert captured["url"] == tools.TELEGRAM_API_URL.format(token="tok")
        assert captured["data"]["chat_id"] == "cid"
        assert "title" in captured["data"]["text"]
        assert "message" in captured["data"]["text"]

    def test_http_failure_does_not_raise(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "cid")

        def fake_post(url, data, timeout):
            return FakeResponse(500)

        monkeypatch.setattr(tools.requests, "post", fake_post)

        result = tools._telegram_notify("title", "message")

        assert result["status"] == "failed"

    def test_connection_error_does_not_raise(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "cid")

        def fake_post(url, data, timeout):
            raise tools.requests.ConnectionError("no network")

        monkeypatch.setattr(tools.requests, "post", fake_post)

        result = tools._telegram_notify("title", "message")

        assert result["status"] == "failed"

    def test_token_never_appears_in_logs_on_http_failure(self, monkeypatch, caplog):
        # Telegram's URL embeds the token (.../bot<TOKEN>/sendMessage), unlike
        # Pushover's fixed URL — a naive logger.exception would leak it into
        # Render's logs, which is exactly what happened before this was fixed.
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "super-secret-token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "cid")
        monkeypatch.setattr(tools.requests, "post", lambda *a, **k: FakeResponse(400))

        with caplog.at_level(logging.ERROR):
            result = tools._telegram_notify("title", "message")

        assert result["status"] == "failed"
        assert "super-secret-token" not in caplog.text

    def test_token_never_appears_in_logs_on_connection_error(self, monkeypatch, caplog):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "super-secret-token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "cid")
        monkeypatch.setattr(
            tools.requests,
            "post",
            lambda *a, **k: (_ for _ in ()).throw(tools.requests.ConnectionError("no network")),
        )

        with caplog.at_level(logging.ERROR):
            result = tools._telegram_notify("title", "message")

        assert result["status"] == "failed"
        assert "super-secret-token" not in caplog.text


class TestNotifyFanOut:
    def test_logged_when_nothing_configured(self):
        result = tools._notify("title", "message")
        assert result["status"] == "logged"
        assert result["channels"]["pushover"]["status"] == "logged"
        assert result["channels"]["telegram"]["status"] == "logged"

    def test_sent_when_any_channel_sends(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "cid")
        monkeypatch.setattr(tools.requests, "post", lambda *a, **k: FakeResponse(200))

        result = tools._notify("title", "message")

        assert result["status"] == "sent"

    def test_failed_when_configured_channel_errors_and_none_send(self, monkeypatch):
        monkeypatch.setenv("PUSHOVER_USER", "u")
        monkeypatch.setenv("PUSHOVER_TOKEN", "t")
        monkeypatch.setattr(
            tools.requests,
            "post",
            lambda *a, **k: (_ for _ in ()).throw(tools.requests.ConnectionError()),
        )

        result = tools._notify("title", "message")

        assert result["status"] == "failed"


class TestRecordContact:
    def test_happy_path_logs_without_credentials(self):
        result = tools.record_contact(
            email="a@example.com", name="Alice", notes="wants a chat"
        )
        assert result["recorded"] is True
        assert result["status"] == "logged"

    def test_optional_fields_default_to_none(self):
        result = tools.record_contact(email="a@example.com")
        assert result["recorded"] is True

    def test_no_api_key_or_env_var_in_notification(self, monkeypatch):
        monkeypatch.setenv("PUSHOVER_USER", "u")
        monkeypatch.setenv("PUSHOVER_TOKEN", "super-secret-token")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-should-never-appear")

        captured = {}

        def fake_post(url, data, timeout):
            captured["message"] = data["message"]
            return FakeResponse(200)

        monkeypatch.setattr(tools.requests, "post", fake_post)

        tools.record_contact(email="a@example.com", name="Alice", notes="hello")

        assert "super-secret-token" not in captured["message"]
        assert "sk-should-never-appear" not in captured["message"]


class TestRecordUnknownQuestion:
    def test_happy_path(self):
        result = tools.record_unknown_question(question="What did he do at Acme?")
        assert result["recorded"] is True

    def test_pushover_failure_is_non_fatal(self, monkeypatch):
        monkeypatch.setenv("PUSHOVER_USER", "u")
        monkeypatch.setenv("PUSHOVER_TOKEN", "t")
        monkeypatch.setattr(
            tools.requests,
            "post",
            lambda *a, **k: (_ for _ in ()).throw(tools.requests.ConnectionError()),
        )

        result = tools.record_unknown_question(question="What did he do at Acme?")

        assert result["recorded"] is False
        assert result["status"] == "failed"


class TestLookupProject:
    @pytest.fixture
    def snapshot(self, tmp_path, monkeypatch):
        records = [
            {
                "name": "issue-worm",
                "description": "Multi-agent coder",
                "url": "https://github.com/leonarduk/issue-worm",
                "topics": ["agents"],
                "languages": ["Python"],
                "stars": 3,
                "pushed_at": "2026-08-20",
                "readme_excerpt": "...",
                "curated_note": None,
            },
        ]
        path = tmp_path / "github.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        monkeypatch.setattr(tools, "GITHUB_SNAPSHOT_PATH", path)
        return path

    def test_exact_match(self, snapshot):
        result = tools.lookup_project(name="issue-worm")
        assert result["found"] is True
        assert result["project"]["name"] == "issue-worm"

    def test_case_insensitive_match(self, snapshot):
        result = tools.lookup_project(name="Issue-Worm")
        assert result["found"] is True

    def test_fuzzy_match(self, snapshot):
        result = tools.lookup_project(name="issueworm")
        assert result["found"] is True
        assert result["project"]["name"] == "issue-worm"

    def test_no_match_does_not_raise(self, snapshot):
        result = tools.lookup_project(name="totally-unrelated-repo-xyz")
        assert result["found"] is False
        assert "message" in result

    def test_missing_snapshot_file_does_not_raise(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            tools, "GITHUB_SNAPSHOT_PATH", tmp_path / "does-not-exist.json"
        )
        result = tools.lookup_project(name="issue-worm")
        assert result["found"] is False


class TestDispatch:
    def test_dispatches_known_tool(self):
        result = tools.dispatch(tools.RECORD_UNKNOWN_QUESTION, {"question": "hi"})
        assert result["recorded"] is True

    def test_unknown_tool_name_returns_error_without_raising(self):
        result = tools.dispatch("delete_everything", {})
        assert "error" in result

    def test_bad_arguments_return_error_without_raising(self):
        result = tools.dispatch(tools.RECORD_CONTACT, {"unexpected_field": "x"})
        assert "error" in result


class TestToolDefinitions:
    def test_reexported_from_tools_module(self):
        assert tools.TOOL_DEFINITIONS is tool_definitions.TOOL_DEFINITIONS

    def test_every_definition_is_strict_and_closed(self):
        for tool in tool_definitions.TOOL_DEFINITIONS:
            function = tool["function"]
            assert function["strict"] is True
            params = function["parameters"]
            assert params["additionalProperties"] is False
            assert set(params["required"]) == set(params["properties"].keys())
