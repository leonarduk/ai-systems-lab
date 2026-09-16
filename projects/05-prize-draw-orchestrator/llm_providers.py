"""Configurable LLM-provider abstraction: local Ollama (default), DeepSeek, or
Claude, selected via `LLM_PROVIDER` (see `config.py` / README).

Every provider exposes the same `generate_json(prompt, schema)` method so the
orchestrator's reasoning code (parsing, filtering, eligibility, tie-breakers)
never needs to know which backend answered it.

Privacy note: `OllamaProvider` never sends prompt content outside the local
machine. `DeepSeekProvider` and `ClaudeProvider` send the full prompt —
including any competition content and any personal data embedded in it — to
that provider's hosted API. Both must be explicitly configured (see
`config.py`); neither is ever selected by default.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

import requests

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_TIMEOUT = 60


class LLMProviderError(RuntimeError):
    """Raised when an LLM backend can't be reached or returns unusable output."""


class LLMProvider(Protocol):
    """Common interface every LLM backend implements."""

    def generate_json(
        self, prompt: str, schema: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send `prompt` to the backend and return its response parsed as a JSON object.

        `schema` is a best-effort hint: providers that support constrained/
        structured output (currently Ollama) use it to guarantee the shape;
        providers that don't (DeepSeek, Claude) rely on prompt instructions
        and best-effort JSON parsing instead. Raises `LLMProviderError` if
        the backend can't be reached or the response can't be parsed as a
        JSON object.
        """
        ...


def _strip_code_fences_and_prose(raw_text: str) -> str:
    """Normalize an LLM reply so it can be handed to the JSON parser.

    Handles the two common ways a model wraps otherwise-valid JSON:

    * A markdown code fence — either ```` ```json ... ``` ```` or a bare
      ```` ``` ... ``` ````. The contents of the first fence are returned.
    * Leading prose such as ``Here is the JSON:`` before the payload. The
      substring from the first ``{`` to the last ``}`` is returned.

    Bare JSON objects are returned unchanged (aside from surrounding
    whitespace). If no JSON-looking substring is found, the trimmed input is
    returned so the caller's parser can raise its usual error.
    """
    text = raw_text.strip()

    fence_start = text.find("```")
    if fence_start != -1:
        after_fence = text[fence_start + 3 :]
        # Drop an optional language tag (e.g. "json") on the opening fence line.
        newline_index = after_fence.find("\n")
        if newline_index != -1:
            first_line = after_fence[:newline_index].strip().lower()
            if first_line in ("", "json"):
                after_fence = after_fence[newline_index + 1 :]
        fence_end = after_fence.find("```")
        if fence_end != -1:
            text = after_fence[:fence_end].strip()

    # Trim leading prose by slicing from the first `{` to the last `}`.
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace : last_brace + 1]

    return text.strip()


def _parse_json_object(raw_text: str, provider_name: str) -> dict[str, Any]:
    """Parse `raw_text` as a JSON object, raising `LLMProviderError` if it isn't one.

    Strips markdown code fences and leading prose before parsing so that
    otherwise-valid JSON wrapped by the model is still accepted.
    """
    cleaned = _strip_code_fences_and_prose(raw_text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMProviderError(
            f"{provider_name} did not return valid JSON: {raw_text[:200]!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise LLMProviderError(
            f"{provider_name} returned JSON that isn't an object: {raw_text[:200]!r}"
        )
    return parsed


def _resolve_api_key(config_value: Any, env_var_name: str) -> str:
    """Return the API key from config if set, otherwise fall back to the env var.

    Returns an empty string if neither source provides a value. Callers are
    responsible for raising a clear `LLMProviderError` when the result is empty.
    """
    if config_value:
        return str(config_value)
    return os.environ.get(env_var_name, "") or ""


class OllamaProvider:
    """Local Ollama backend. Default provider; no data leaves the machine."""

    name = "ollama"

    def __init__(
        self,
        host: str = DEFAULT_OLLAMA_HOST,
        model: str = DEFAULT_OLLAMA_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """Configure the Ollama endpoint, model, and per-request timeout."""
        self.host = host
        self.model = model
        self.timeout = timeout

    def generate_json(
        self, prompt: str, schema: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Call Ollama's `/api/generate` with structured-output `format`, parse the JSON reply."""
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": schema if schema is not None else "json",
        }
        try:
            response = requests.post(
                f"{self.host.rstrip('/')}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            raw_text = response.json().get("response", "")
        except requests.exceptions.ConnectionError as exc:
            raise LLMProviderError(
                f"Could not reach local Ollama server at {self.host}. Is `ollama serve` running?"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMProviderError(f"Ollama request failed: {exc}") from exc

        return _parse_json_object(raw_text, "Ollama")


class DeepSeekProvider:
    """DeepSeek backend (OpenAI-compatible chat completions API).

    Opt-in: configuring this provider sends prompt content — including
    competition text and any personal data embedded in it — to DeepSeek's
    hosted API. See README for the required informed-consent configuration.
    """

    name = "deepseek"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_DEEPSEEK_MODEL,
        base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """Configure the DeepSeek API key, model, base URL, and timeout."""
        if not api_key:
            raise LLMProviderError(
                "DeepSeek provider requires DEEPSEEK_API_KEY to be set."
            )
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def generate_json(
        self, prompt: str, schema: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Call DeepSeek's chat completions API and parse the reply as a JSON object."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as exc:
            raise LLMProviderError(f"DeepSeek request failed: {exc}") from exc

        choices = data.get("choices", [])
        if not choices:
            raise LLMProviderError(f"DeepSeek returned no choices: {data!r}")
        content = choices[0].get("message", {}).get("content", "")
        return _parse_json_object(content, "DeepSeek")


class ClaudeProvider:
    """Anthropic Claude backend (Messages API).

    Opt-in: configuring this provider sends prompt content — including
    competition text and any personal data embedded in it — to Anthropic's
    hosted API. See README for the required informed-consent configuration.
    """

    name = "claude"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_CLAUDE_MODEL,
        max_tokens: int = 2000,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """Configure the Anthropic API key, model, max tokens, and timeout."""
        if not api_key:
            raise LLMProviderError(
                "Claude provider requires ANTHROPIC_API_KEY to be set."
            )
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate_json(
        self, prompt: str, schema: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Call Anthropic's Messages API and parse the reply's text content as JSON."""
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as exc:
            raise LLMProviderError(f"Claude request failed: {exc}") from exc

        content = data.get("content", [])
        text = "\n".join(
            block.get("text", "") for block in content if block.get("type") == "text"
        ).strip()
        return _parse_json_object(text, "Claude")


def build_llm_provider(config: Any) -> LLMProvider:
    """Construct the configured `LLMProvider` from a `Config` object.

    `config.llm_provider` selects the backend: 'ollama' (default), 'deepseek',
    or 'claude'. API keys are read from the config object first, then fall
    back to the corresponding environment variable (`DEEPSEEK_API_KEY` /
    `ANTHROPIC_API_KEY`). Raises `LLMProviderError` for an unknown provider
    name or when a required API key is missing from both sources.
    """
    provider = (config.llm_provider or "ollama").strip().lower()

    if provider == "ollama":
        return OllamaProvider(host=config.ollama_host, model=config.ollama_model)

    if provider == "deepseek":
        api_key = _resolve_api_key(
            getattr(config, "deepseek_api_key", None), "DEEPSEEK_API_KEY"
        )
        if not api_key:
            raise LLMProviderError(
                "DeepSeek provider selected but no API key found. "
                "Set DEEPSEEK_API_KEY or config.deepseek_api_key."
            )
        return DeepSeekProvider(api_key=api_key, model=config.deepseek_model)

    if provider == "claude":
        api_key = _resolve_api_key(
            getattr(config, "anthropic_api_key", None), "ANTHROPIC_API_KEY"
        )
        if not api_key:
            raise LLMProviderError(
                "Claude provider selected but no API key found. "
                "Set ANTHROPIC_API_KEY or config.anthropic_api_key."
            )
        return ClaudeProvider(api_key=api_key, model=config.claude_model)

    raise LLMProviderError(
        f"Unknown LLM_PROVIDER '{provider}'. Expected 'ollama', 'deepseek', or 'claude'."
    )
