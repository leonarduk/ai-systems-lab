"""Tests for avatar/llm.py. No live DeepSeek/OpenAI API calls."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, RateLimitError

sys.path.insert(0, str(Path(__file__).parent.parent))

from avatar import llm  # noqa: E402


def make_usage(prompt_tokens=100, completion_tokens=20, cache_hit=0, cache_miss=None):
    if cache_miss is None:
        cache_miss = prompt_tokens - cache_hit
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_cache_hit_tokens=cache_hit,
        prompt_cache_miss_tokens=cache_miss,
    )


def make_tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def make_response(content=None, tool_calls=None, usage=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)], usage=usage or make_usage()
    )


class FakeClient:
    """Records every create() call and returns queued responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

        class _Completions:
            def create(inner_self, **kwargs):
                self.calls.append(kwargs)
                if not self._responses:
                    raise AssertionError("FakeClient ran out of queued responses")
                response = self._responses.pop(0)
                if isinstance(response, Exception):
                    raise response
                return response

        self.chat = SimpleNamespace(completions=_Completions())


class TestSendMessageHappyPath:
    def test_single_response_with_no_tool_calls(self):
        client = FakeClient([make_response(content="Hello there.")])

        reply, usage = llm.send_message(
            [{"role": "user", "content": "hi"}], "system", client=client
        )

        assert reply == "Hello there."
        assert len(client.calls) == 1

    def test_two_round_tool_conversation_completes(self):
        tool_call = make_tool_call("call_1", "lookup_project", {"name": "issue-worm"})
        client = FakeClient(
            [
                make_response(tool_calls=[tool_call]),
                make_response(content="issue-worm is a multi-agent coder."),
            ]
        )

        reply, usage = llm.send_message(
            [{"role": "user", "content": "tell me about issue-worm"}],
            "system",
            client=client,
        )

        assert reply == "issue-worm is a multi-agent coder."
        assert len(client.calls) == 2
        # The second call's messages must include the assistant tool_calls
        # message and a matching tool-result message.
        second_call_messages = client.calls[1]["messages"]
        roles = [m["role"] for m in second_call_messages]
        assert "tool" in roles
        tool_message = next(m for m in second_call_messages if m["role"] == "tool")
        assert tool_message["tool_call_id"] == "call_1"

    def test_multiple_tool_calls_in_one_response_all_get_matching_results(self):
        call_a = make_tool_call("call_a", "record_unknown_question", {"question": "q1"})
        call_b = make_tool_call("call_b", "record_unknown_question", {"question": "q2"})
        client = FakeClient(
            [
                make_response(tool_calls=[call_a, call_b]),
                make_response(content="done"),
            ]
        )

        reply, usage = llm.send_message(
            [{"role": "user", "content": "hi"}], "system", client=client
        )

        assert reply == "done"
        second_call_messages = client.calls[1]["messages"]
        tool_messages = [m for m in second_call_messages if m["role"] == "tool"]
        assert {m["tool_call_id"] for m in tool_messages} == {"call_a", "call_b"}

    def test_bad_json_tool_arguments_does_not_crash(self):
        bad_call = SimpleNamespace(
            id="call_bad",
            function=SimpleNamespace(name="lookup_project", arguments="{not json"),
        )
        client = FakeClient(
            [
                make_response(tool_calls=[bad_call]),
                make_response(content="handled"),
            ]
        )

        reply, usage = llm.send_message(
            [{"role": "user", "content": "hi"}], "system", client=client
        )

        assert reply == "handled"
        tool_message = next(
            m for m in client.calls[1]["messages"] if m["role"] == "tool"
        )
        assert "error" in json.loads(tool_message["content"])


class TestUsageAccounting:
    def test_usage_accumulated_across_rounds(self):
        tool_call = make_tool_call(
            "call_1", "record_unknown_question", {"question": "q"}
        )
        client = FakeClient(
            [
                make_response(
                    tool_calls=[tool_call],
                    usage=make_usage(
                        prompt_tokens=100, completion_tokens=10, cache_hit=20
                    ),
                ),
                make_response(
                    content="done",
                    usage=make_usage(
                        prompt_tokens=150, completion_tokens=15, cache_hit=140
                    ),
                ),
            ]
        )

        _, usage = llm.send_message(
            [{"role": "user", "content": "hi"}], "system", client=client
        )

        assert usage["input_tokens"] == 250
        assert usage["output_tokens"] == 25
        assert usage["cache_hit_tokens"] == 160

    def test_usage_returned_even_on_single_call(self):
        client = FakeClient(
            [
                make_response(
                    content="hi", usage=make_usage(prompt_tokens=50, cache_hit=10)
                )
            ]
        )

        _, usage = llm.send_message([], "system", client=client)

        assert usage["input_tokens"] == 50
        assert usage["cache_hit_tokens"] == 10


class TestErrorHandling:
    def _rate_limit_error(self):
        response = httpx.Response(
            429, request=httpx.Request("POST", "https://api.deepseek.com")
        )
        return RateLimitError("rate limited", response=response, body=None)

    def _status_error(self):
        response = httpx.Response(
            500, request=httpx.Request("POST", "https://api.deepseek.com")
        )
        return APIStatusError("server error", response=response, body=None)

    def _connection_error(self):
        return APIConnectionError(
            request=httpx.Request("POST", "https://api.deepseek.com")
        )

    def test_rate_limit_returns_friendly_message(self):
        client = FakeClient([self._rate_limit_error()])
        reply, usage = llm.send_message([], "system", client=client)
        assert reply == llm.FRIENDLY_RATE_LIMIT_MESSAGE

    def test_status_error_returns_friendly_message(self):
        client = FakeClient([self._status_error()])
        reply, usage = llm.send_message([], "system", client=client)
        assert reply == llm.FRIENDLY_ERROR_MESSAGE

    def test_connection_error_returns_friendly_message(self):
        client = FakeClient([self._connection_error()])
        reply, usage = llm.send_message([], "system", client=client)
        assert reply == llm.FRIENDLY_ERROR_MESSAGE

    def test_errors_never_raise_out_of_send_message(self):
        client = FakeClient([self._rate_limit_error()])
        # Should not raise.
        llm.send_message([], "system", client=client)


class TestToolLoopTermination:
    def test_hard_cap_stops_infinite_tool_loop(self):
        endless_tool_call = make_tool_call(
            "call_x", "record_unknown_question", {"question": "q"}
        )
        responses = [
            make_response(tool_calls=[endless_tool_call])
            for _ in range(llm.MAX_TOOL_LOOP_ITERATIONS)
        ]
        client = FakeClient(responses)

        reply, usage = llm.send_message(
            [{"role": "user", "content": "hi"}], "system", client=client
        )

        assert reply == llm.FRIENDLY_ERROR_MESSAGE
        assert len(client.calls) == llm.MAX_TOOL_LOOP_ITERATIONS


class TestEmptyContentWarning:
    def test_warns_when_content_none_and_no_tool_calls(self, caplog):
        client = FakeClient([make_response(content=None)])

        with caplog.at_level("WARNING", logger="avatar.llm"):
            reply, usage = llm.send_message(
                [{"role": "user", "content": "hi"}], "system", client=client
            )

        assert reply == ""
        assert any(
            "empty content with no tool calls" in record.message
            for record in caplog.records
        )

    def test_warns_when_content_empty_string_and_no_tool_calls(self, caplog):
        client = FakeClient([make_response(content="")])

        with caplog.at_level("WARNING", logger="avatar.llm"):
            reply, usage = llm.send_message(
                [{"role": "user", "content": "hi"}], "system", client=client
            )

        assert reply == ""
        assert any(
            "empty content with no tool calls" in record.message
            for record in caplog.records
        )

    def test_no_warning_when_tool_calls_present(self, caplog):
        tool_call = make_tool_call(
            "call_1", "record_unknown_question", {"question": "q"}
        )
        client = FakeClient(
            [
                make_response(content=None, tool_calls=[tool_call]),
                make_response(content="done"),
            ]
        )

        with caplog.at_level("WARNING", logger="avatar.llm"):
            reply, usage = llm.send_message(
                [{"role": "user", "content": "hi"}], "system", client=client
            )

        assert reply == "done"
        assert not any(
            "empty content with no tool calls" in record.message
            for record in caplog.records
        )

    def test_no_warning_when_content_present(self, caplog):
        client = FakeClient([make_response(content="Hello there.")])

        with caplog.at_level("WARNING", logger="avatar.llm"):
            reply, usage = llm.send_message(
                [{"role": "user", "content": "hi"}], "system", client=client
            )

        assert reply == "Hello there."
        assert not any(
            "empty content with no tool calls" in record.message
            for record in caplog.records
        )


class TestBuildClient:
    def test_reads_api_key_from_env_only(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-123")
        captured = {}

        class FakeOpenAI:
            def __init__(self, base_url, api_key):
                captured["base_url"] = base_url
                captured["api_key"] = api_key

        monkeypatch.setattr(llm, "OpenAI", FakeOpenAI)

        llm._build_client()

        assert captured["api_key"] == "test-key-123"
        assert captured["base_url"] == llm.DEEPSEEK_BASE_URL

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(KeyError):
            llm._build_client()


class TestModelSelection:
    def test_defaults_to_deepseek_v4_flash(self, monkeypatch):
        monkeypatch.delenv("AVATAR_MODEL", raising=False)
        assert llm._model() == llm.DEFAULT_MODEL

    def test_reads_avatar_model_env_var(self, monkeypatch):
        monkeypatch.setenv("AVATAR_MODEL", "deepseek-v4-pro")
        assert llm._model() == "deepseek-v4-pro"
