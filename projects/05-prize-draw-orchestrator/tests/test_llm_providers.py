import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_providers import (
    ClaudeProvider,
    DeepSeekProvider,
    LLMProviderError,
    OllamaProvider,
    _parse_json_object,
    build_llm_provider,
)


class FakeConfig:
    def __init__(self, **kwargs):
        self.llm_provider = kwargs.get("llm_provider", "ollama")
        self.ollama_host = kwargs.get("ollama_host", "http://localhost:11434")
        self.ollama_model = kwargs.get("ollama_model", "llama3")
        self.deepseek_api_key = kwargs.get("deepseek_api_key", "")
        self.deepseek_model = kwargs.get("deepseek_model", "deepseek-v4-flash")
        self.anthropic_api_key = kwargs.get("anthropic_api_key", "")
        self.claude_model = kwargs.get("claude_model", "claude-sonnet-4-6")


class TestParseJsonObject:
    def test_plain_json_object(self):
        assert _parse_json_object('{"eligible": true}', "Test") == {"eligible": True}

    def test_fenced_json_with_language_tag(self):
        raw = '```json\n{"eligible": true}\n```'
        assert _parse_json_object(raw, "Test") == {"eligible": True}

    def test_fenced_json_without_language_tag(self):
        raw = '```\n{"eligible": false}\n```'
        assert _parse_json_object(raw, "Test") == {"eligible": False}

    def test_json_wrapped_in_prose(self):
        raw = 'Sure, here is the result: {"eligible": true} — hope that helps!'
        assert _parse_json_object(raw, "Test") == {"eligible": True}

    def test_fenced_json_with_surrounding_prose(self):
        raw = 'Here you go:\n```json\n{"eligible": true}\n```\nDone.'
        assert _parse_json_object(raw, "Test") == {"eligible": True}

    def test_invalid_json_raises(self):
        with pytest.raises(LLMProviderError, match="did not return valid JSON"):
            _parse_json_object("not json at all", "Test")

    def test_non_object_json_raises(self):
        with pytest.raises(LLMProviderError, match="isn't an object"):
            _parse_json_object("[1, 2, 3]", "Test")


class TestOllamaProvider:
    def test_generate_json_parses_response(self):
        mock_response = Mock()
        mock_response.json.return_value = {"response": '{"eligible": true}'}
        mock_response.raise_for_status = Mock()

        with patch(
            "llm_providers.requests.post", return_value=mock_response
        ) as mock_post:
            result = OllamaProvider().generate_json("prompt", schema={"type": "object"})

        assert result == {"eligible": True}
        called_payload = mock_post.call_args.kwargs["json"]
        assert called_payload["format"] == {"type": "object"}

    def test_generate_json_handles_fenced_response(self):
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": '```json\n{"eligible": true}\n```'
        }
        mock_response.raise_for_status = Mock()

        with patch("llm_providers.requests.post", return_value=mock_response):
            result = OllamaProvider().generate_json("prompt")

        assert result == {"eligible": True}

    def test_connection_error_raises_llm_provider_error(self):
        with patch(
            "llm_providers.requests.post",
            side_effect=requests.exceptions.ConnectionError(),
        ):
            with pytest.raises(LLMProviderError, match="Could not reach local Ollama"):
                OllamaProvider().generate_json("prompt")

    def test_non_json_response_raises(self):
        mock_response = Mock()
        mock_response.json.return_value = {"response": "not json"}
        mock_response.raise_for_status = Mock()
        with patch("llm_providers.requests.post", return_value=mock_response):
            with pytest.raises(LLMProviderError, match="did not return valid JSON"):
                OllamaProvider().generate_json("prompt")


class TestDeepSeekProvider:
    def test_requires_api_key(self):
        with pytest.raises(LLMProviderError, match="DEEPSEEK_API_KEY"):
            DeepSeekProvider(api_key="")

    def test_generate_json_parses_chat_completion(self):
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"eligible": false}'}}]
        }
        mock_response.raise_for_status = Mock()

        with patch("llm_providers.requests.post", return_value=mock_response):
            result = DeepSeekProvider(api_key="secret").generate_json("prompt")

        assert result == {"eligible": False}

    def test_generate_json_handles_fenced_content(self):
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '```json\n{"eligible": true}\n```'}}]
        }
        mock_response.raise_for_status = Mock()

        with patch("llm_providers.requests.post", return_value=mock_response):
            result = DeepSeekProvider(api_key="secret").generate_json("prompt")

        assert result == {"eligible": True}

    def test_no_choices_raises(self):
        mock_response = Mock()
        mock_response.json.return_value = {"choices": []}
        mock_response.raise_for_status = Mock()
        with patch("llm_providers.requests.post", return_value=mock_response):
            with pytest.raises(LLMProviderError, match="no choices"):
                DeepSeekProvider(api_key="secret").generate_json("prompt")


class TestClaudeProvider:
    def test_requires_api_key(self):
        with pytest.raises(LLMProviderError, match="ANTHROPIC_API_KEY"):
            ClaudeProvider(api_key="")

    def test_generate_json_parses_content_blocks(self):
        mock_response = Mock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": '{"eligible": true}'}]
        }
        mock_response.raise_for_status = Mock()

        with patch(
            "llm_providers.requests.post", return_value=mock_response
        ) as mock_post:
            result = ClaudeProvider(api_key="secret").generate_json("prompt")

        assert result == {"eligible": True}
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["x-api-key"] == "secret"


class TestBuildLLMProvider:
    def test_defaults_to_ollama(self):
        provider = build_llm_provider(FakeConfig(llm_provider="ollama"))
        assert isinstance(provider, OllamaProvider)

    def test_selects_deepseek(self):
        provider = build_llm_provider(
            FakeConfig(llm_provider="deepseek", deepseek_api_key="k")
        )
        assert isinstance(provider, DeepSeekProvider)

    def test_selects_claude(self):
        provider = build_llm_provider(
            FakeConfig(llm_provider="claude", anthropic_api_key="k")
        )
        assert isinstance(provider, ClaudeProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(LLMProviderError, match="Unknown LLM_PROVIDER"):
            build_llm_provider(FakeConfig(llm_provider="bogus"))

    def test_deepseek_missing_api_key_raises(self, monkeypatch):
        """Selecting deepseek with no config key and no env var must raise a
        clear LLMProviderError naming DEEPSEEK_API_KEY (not a KeyError)."""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(LLMProviderError, match="DEEPSEEK_API_KEY"):
            build_llm_provider(FakeConfig(llm_provider="deepseek", deepseek_api_key=""))

    def test_claude_missing_api_key_raises(self, monkeypatch):
        """Selecting claude with no config key and no env var must raise a
        clear LLMProviderError naming ANTHROPIC_API_KEY (not a KeyError)."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(LLMProviderError, match="ANTHROPIC_API_KEY"):
            build_llm_provider(FakeConfig(llm_provider="claude", anthropic_api_key=""))

    def test_deepseek_falls_back_to_env_var(self, monkeypatch):
        """When config has no key but DEEPSEEK_API_KEY is set, the env var is used."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "env-secret")
        provider = build_llm_provider(
            FakeConfig(llm_provider="deepseek", deepseek_api_key="")
        )
        assert isinstance(provider, DeepSeekProvider)
        assert provider.api_key == "env-secret"

    def test_claude_falls_back_to_env_var(self, monkeypatch):
        """When config has no key but ANTHROPIC_API_KEY is set, the env var is used."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-secret")
        provider = build_llm_provider(
            FakeConfig(llm_provider="claude", anthropic_api_key="")
        )
        assert isinstance(provider, ClaudeProvider)
        assert provider.api_key == "env-secret"


class TestParseJsonObjectEdgeCases:
    """Edge cases for the JSON extraction helpers introduced in PR #237.

    These tests pin the *current* behavior of the fenced/prose/brace-slicing
    heuristics. They intentionally assert that ambiguous inputs raise
    ``LLMProviderError`` rather than silently mis-parsing, so any future
    change to the extraction logic must update these tests explicitly.
    """

    def _provider(self):
        # OllamaProvider is the simplest concrete provider; its generate_json
        # delegates to the shared _parse_json_object helper.
        return OllamaProvider()

    def _mock_response(self, content):
        mock_response = Mock()
        mock_response.json.return_value = {"response": content}
        mock_response.raise_for_status = Mock()
        return mock_response

    def test_empty_json_fenced_block_raises(self):
        """A ```json fence with empty inner content must raise, not return {}."""
        mock_response = self._mock_response("```json\n```")
        with patch("llm_providers.requests.post", return_value=mock_response):
            with pytest.raises(LLMProviderError):
                self._provider().generate_json("prompt")

    def test_empty_bare_fenced_block_raises(self):
        """A bare ``` fence with empty inner content must raise, not return {}."""
        mock_response = self._mock_response("```\n```")
        with patch("llm_providers.requests.post", return_value=mock_response):
            with pytest.raises(LLMProviderError):
                self._provider().generate_json("prompt")

    def test_multiple_json_objects_raises(self):
        """Two JSON objects in one response must raise rather than pick one."""
        mock_response = self._mock_response(
            '{"eligible": true} but note {"x": 1} is unrelated'
        )
        with patch("llm_providers.requests.post", return_value=mock_response):
            with pytest.raises(LLMProviderError):
                self._provider().generate_json("prompt")

    def test_trailing_brace_in_prose_raises(self):
        """A valid object followed by prose containing a `}` must raise."""
        mock_response = self._mock_response(
            '{"eligible": true} trailing text with a } brace'
        )
        with patch("llm_providers.requests.post", return_value=mock_response):
            with pytest.raises(LLMProviderError):
                self._provider().generate_json("prompt")
