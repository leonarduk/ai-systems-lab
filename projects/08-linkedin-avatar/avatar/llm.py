"""DeepSeek client and tool-use loop.

DeepSeek's API is OpenAI-compatible, so this uses the `openai` SDK pointed at
DeepSeek's base URL rather than the Anthropic SDK. Caching is automatic on
DeepSeek's side (design §4) — there's nothing to configure here, only usage
to log so real cache behaviour is visible. See docs/design.md §7.
"""

import json
import logging
import os

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from avatar import tools
from avatar.tool_definitions import TOOL_DEFINITIONS

logger = logging.getLogger(__name__)

# Strict-mode tool schemas (design §5 / issue #123) are a DeepSeek beta
# feature gated behind the /beta base URL.
DEEPSEEK_BASE_URL = "https://api.deepseek.com/beta"
DEFAULT_MODEL = "deepseek-v4-flash"

MAX_REPLY_TOKENS = 1024
MAX_TOOL_LOOP_ITERATIONS = 5

FRIENDLY_RATE_LIMIT_MESSAGE = (
    "My twin is getting a lot of questions right now — please try again in a moment."
)
FRIENDLY_ERROR_MESSAGE = "Something went wrong on my end. Please try again, or reach Steve directly via LinkedIn."


def _build_client(api_key=None):
    api_key = api_key or os.environ["DEEPSEEK_API_KEY"]
    return OpenAI(base_url=DEEPSEEK_BASE_URL, api_key=api_key)


def _model():
    return os.environ.get("AVATAR_MODEL", DEFAULT_MODEL)


def _empty_usage():
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_hit_tokens": 0,
        "cache_miss_tokens": 0,
    }


def _accumulate_usage(totals, usage):
    """Fold one response's usage into the running totals and log it, so real
    cache behaviour (hit vs. miss) is visible rather than assumed."""
    if usage is None:
        return

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    cache_hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
    cache_miss = getattr(usage, "prompt_cache_miss_tokens", None)
    if cache_miss is None:
        cache_miss = max(prompt_tokens - cache_hit, 0)

    totals["input_tokens"] += prompt_tokens
    totals["output_tokens"] += completion_tokens
    totals["cache_hit_tokens"] += cache_hit
    totals["cache_miss_tokens"] += cache_miss

    logger.info(
        "DeepSeek usage: prompt=%s completion=%s cache_hit=%s cache_miss=%s",
        prompt_tokens,
        completion_tokens,
        cache_hit,
        cache_miss,
    )


def _tool_call_message(message, tool_calls):
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ],
    }


def _dispatch_tool_call(call):
    tool_name = call.function.name
    try:
        arguments = json.loads(call.function.arguments)
    except json.JSONDecodeError:
        logger.exception("Bad JSON arguments for tool call %s", tool_name)
        result = {"error": "invalid tool arguments"}
    else:
        try:
            result = tools.dispatch(tool_name, arguments)
        except Exception:
            # Tool implementations are documented to never raise (tools.py's
            # own module docstring) — reaching here means one broke that
            # contract, so the exception is unvetted. Some of them embed
            # secrets in their message (e.g. Telegram's API URL carries the
            # bot token — see tools._telegram_notify), so only the tool name
            # goes into the result that flows back into the LLM's context;
            # the exception itself is logged server-side only.
            logger.exception("Tool %s raised an exception during dispatch", tool_name)
            result = {"error": f"tool '{tool_name}' failed"}

    return {
        "role": "tool",
        "tool_call_id": call.id,
        "content": json.dumps(result),
    }


def send_message(conversation, system_prompt, api_key=None, client=None):
    """Send a conversation to DeepSeek, running the tool-use loop to
    completion. `conversation` is a list of {"role", "content"} messages,
    excluding the system prompt. Returns (reply_text, usage_totals)."""
    client = client or _build_client(api_key)
    model = _model()
    messages = [{"role": "system", "content": system_prompt}] + list(conversation)
    usage_totals = _empty_usage()

    for _ in range(MAX_TOOL_LOOP_ITERATIONS):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                max_tokens=MAX_REPLY_TOKENS,
            )
        except RateLimitError:
            logger.exception("DeepSeek rate limit hit")
            return FRIENDLY_RATE_LIMIT_MESSAGE, usage_totals
        except APIStatusError:
            logger.exception("DeepSeek API returned an error status")
            return FRIENDLY_ERROR_MESSAGE, usage_totals
        except APIConnectionError:
            logger.exception("Could not connect to DeepSeek")
            return FRIENDLY_ERROR_MESSAGE, usage_totals

        _accumulate_usage(usage_totals, response.usage)

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)

        if not tool_calls:
            return message.content or "", usage_totals

        messages.append(_tool_call_message(message, tool_calls))
        for call in tool_calls:
            messages.append(_dispatch_tool_call(call))

    logger.warning(
        "Tool loop exceeded %s iterations without a final answer",
        MAX_TOOL_LOOP_ITERATIONS,
    )
    return FRIENDLY_ERROR_MESSAGE, usage_totals
