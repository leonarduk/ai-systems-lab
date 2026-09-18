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
            lambda *a, **k: (_ for _ in ()).throw(
                tools.requests.ConnectionError("no network")
            ),
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

    def _configure_both_channels(self, monkeypatch):
        monkeypatch.setenv("PUSHOVER_USER", "u")
        monkeypatch.setenv("PUSHOVER_TOKEN", "t")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "cid")

    def test_partial_failure_still_counts_as_sent(self, monkeypatch):
        # Both channels configured; Pushover succeeds, Telegram returns 500.
        # _notify is documented as "sent if any channel sent": the notification
        # did reach me, so a transient outage on one channel must not report the
        # whole fan-out as failed. The per-channel statuses still record what
        # happened. Issue #191 originally specified "failed" here; its AC was
        # amended to match this contract rather than change _notify, which the
        # issue itself puts out of scope.
        self._configure_both_channels(monkeypatch)

        def fake_post(url, data, timeout):
            if url == tools.PUSHOVER_URL:
                return FakeResponse(200)
            return FakeResponse(500)

        monkeypatch.setattr(tools.requests, "post", fake_post)

        result = tools._notify("title", "message")

        assert result["status"] == "sent"
        assert result["channels"]["pushover"]["status"] == "sent"
        assert result["channels"]["telegram"]["status"] == "failed"
        assert result["channels"]["telegram"]["detail"]

    def test_partial_failure_by_exception_still_counts_as_sent(self, monkeypatch):
        # Same, but the failing channel raises rather than returning an error
        # status — _notify is documented never to raise.
        self._configure_both_channels(monkeypatch)

        def fake_post(url, data, timeout):
            if url == tools.PUSHOVER_URL:
                return FakeResponse(200)
            raise tools.requests.ConnectionError("no network")

        monkeypatch.setattr(tools.requests, "post", fake_post)

        result = tools._notify("title", "message")

        assert result["status"] == "sent"
        assert result["channels"]["pushover"]["status"] == "sent"
        assert result["channels"]["telegram"]["status"] == "failed"

    def test_partial_failure_is_reported_as_recorded(self, monkeypatch):
        # This is why "sent" is the right overall status: the caller must not
        # tell a visitor their contact request was lost when one of two
        # channels delivered it.
        self._configure_both_channels(monkeypatch)

        def fake_post(url, data, timeout):
            if url == tools.PUSHOVER_URL:
                return FakeResponse(200)
            return FakeResponse(500)

        monkeypatch.setattr(tools.requests, "post", fake_post)

        assert tools.record_contact("visitor@example.com")["recorded"] is True

    def test_every_configured_channel_failing_is_a_failure(self, monkeypatch):
        # The other side of the same rule: with nothing getting through, the
        # fan-out is a failure rather than a partial success.
        self._configure_both_channels(monkeypatch)
        monkeypatch.setattr(
            tools.requests, "post", lambda url, data, timeout: FakeResponse(500)
        )

        result = tools._notify("title", "message")

        assert result["status"] == "failed"
        assert tools.record_contact("visitor@example.com")["recorded"] is False

    def test_telegram_failure_does_not_leak_the_bot_token(self, monkeypatch, caplog):
        # Telegram's URL embeds the bot token, so the failure path logs only a
        # status code and never the exception or URL. A regression to
        # logger.exception here would write the token into the logs.
        self._configure_both_channels(monkeypatch)

        def fake_post(url, data, timeout):
            if url == tools.PUSHOVER_URL:
                return FakeResponse(200)
            # requests puts the failing URL in the exception message, which is
            # how the token would reach the log. Reproduce that faithfully —
            # a bare exception would make this test pass either way.
            exc = tools.requests.HTTPError(
                f"500 Server Error for url: "
                f"{tools.TELEGRAM_API_URL.format(token='tok')}"
            )
            exc.response = FakeResponse(500)
            raise exc

        monkeypatch.setattr(tools.requests, "post", fake_post)

        with caplog.at_level(logging.DEBUG):
            tools._notify("title", "message")

        # The failure path definitely ran (and logged the status code)...
        assert "Telegram notification failed" in caplog.text
        assert "500" in caplog.text
        # ...without the token reaching the log.
        assert "tok" not in caplog.text


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

    def test_malformed_snapshot_file_does_not_raise(self, tmp_path, monkeypatch):
        path = tmp_path / "github.json"
        path.write_text("{invalid json", encoding="utf-8")
        monkeypatch.setattr(tools, "GITHUB_SNAPSHOT_PATH", path)
        result = tools.lookup_project(name="issue-worm")
        assert result["found"] is False
        assert "message" in result

    def test_record_missing_name_is_skipped(self, tmp_path, monkeypatch):
        records = [
            {"description": "no name here", "url": "https://example.com/broken"},
            {
                "name": "issue-worm",
                "description": "Multi-agent coder",
                "url": "https://github.com/leonarduk/issue-worm",
            },
        ]
        path = tmp_path / "github.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        monkeypatch.setattr(tools, "GITHUB_SNAPSHOT_PATH", path)

        result = tools.lookup_project(name="issue-worm")

        assert result["found"] is True
        assert result["project"]["name"] == "issue-worm"

    def test_all_records_malformed_does_not_raise(self, tmp_path, monkeypatch):
        records = [
            {"description": "no name"},
            "not-a-dict",
            {"name": ""},
            {"name": None},
        ]
        path = tmp_path / "github.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        monkeypatch.setattr(tools, "GITHUB_SNAPSHOT_PATH", path)

        result = tools.lookup_project(name="issue-worm")

        assert result["found"] is False
        assert "message" in result

    def test_snapshot_not_a_list_does_not_raise(self, tmp_path, monkeypatch):
        path = tmp_path / "github.json"
        path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
        monkeypatch.setattr(tools, "GITHUB_SNAPSHOT_PATH", path)

        result = tools.lookup_project(name="issue-worm")

        assert result["found"] is False
        assert "message" in result


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
