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


class TestParseJsonObject:
    def test_bare_json_object(self):
        assert _parse_json_object('{"eligible": true}', "Test") == {"eligible": True}

    def test_bare_json_with_surrounding_whitespace(self):
        assert _parse_json_object('  \n {"eligible": true} \n ', "Test") == {
            "eligible": True
        }

    def test_fenced_json_with_language_tag(self):
        raw = '```json\n{"eligible": true}\n```'
        assert _parse_json_object(raw, "Test") == {"eligible": True}

    def test_fenced_json_without_language_tag(self):
        raw = '```\n{"eligible": false}\n```'
        assert _parse_json_object(raw, "Test") == {"eligible": False}

    def test_fenced_json_with_prose_around_fence(self):
        raw = 'Here is the JSON:\n```json\n{"eligible": true}\n```\nHope that helps!'
        assert _parse_json_object(raw, "Test") == {"eligible": True}

    def test_prose_prefixed_json(self):
        raw = 'Here is the JSON: {"eligible": true}'
        assert _parse_json_object(raw, "Test") == {"eligible": True}

    def test_prose_prefixed_and_suffixed_json(self):
        raw = 'Sure! {"eligible": true} Let me know if you need more.'
        assert _parse_json_object(raw, "Test") == {"eligible": True}

    def test_nested_braces_preserved(self):
        raw = '{"outer": {"inner": {"deep": 1}}}'
        assert _parse_json_object(raw, "Test") == {
            "outer": {"inner": {"deep": 1}}
        }

    def test_prose_prefixed_nested_braces(self):
        raw = 'Result: {"outer": {"inner": 1}} done.'
        assert _parse_json_object(raw, "Test") == {"outer": {"inner": 1}}

    def test_invalid_input_still_raises(self):
        with pytest.raises(LLMProviderError, match="did not return valid JSON"):
            _parse_json_object("not json at all", "Test")

    def test_non_object_json_raises(self):
        with pytest.raises(LLMProviderError, match="isn't an object"):
            _parse_json_object("[1, 2, 3]", "Test")


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
            build_llm_provider(
                FakeConfig(llm_provider="deepseek", deepseek_api_key="")
            )

    def test_claude_missing_api_key_raises(self, monkeypatch):
        """Selecting claude with no config key and no env var must raise a
        clear LLMProviderError naming ANTHROPIC_API_KEY (not a KeyError)."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(LLMProviderError, match="ANTHROPIC_API_KEY"):
            build_llm_provider(
                FakeConfig(llm_provider="claude", anthropic_api_key="")
            )

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
