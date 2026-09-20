"""Rate limits, input cap and daily spend kill-switch.

The one control that must be correct: an unmetered public LLM endpoint is an
open invitation, and the daily budget kill-switch is the only thing standing
between a bored visitor with a script and an unbounded bill. See
docs/design.md §6. Every check here fails closed — an error inside the
budget tracker blocks the call, it never waves it through.
"""

import logging
import os
import threading
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DEFAULT_MAX_INPUT_CHARS = 1500
DEFAULT_SESSION_RATE_LIMIT = "20/hour"
DEFAULT_IP_RATE_LIMIT = "40/day"
DEFAULT_DAILY_BUDGET_USD = 2.00

# Matches avatar.llm.DEFAULT_MODEL (#124) — duplicated rather than imported
# so this module has no dependency on avatar.llm's tool-use loop.
DEFAULT_MODEL_FOR_PRICING = "deepseek-v4-flash"

_UNIT_SECONDS = {"hour": 3600, "day": 86400}

# Cleanup tuning for _SlidingWindowLimiter. When the number of tracked keys
# exceeds _CLEANUP_KEY_THRESHOLD, we sweep out keys whose most recent event is
# older than the window (they can no longer affect any decision). The sweep is
# amortised: it only runs when the map has grown past the threshold, so the
# hot path stays cheap for normal traffic.
_CLEANUP_KEY_THRESHOLD = 1000

# Verified against api-docs.deepseek.com/quick_start/pricing on 2026-08-30.
# USD per 1M tokens, using the more expensive **peak** rate as the
# conservative default (off-peak is half these figures) — prices move, so
# re-check this page before trusting these numbers again.
PRICE_TABLE_USD_PER_MTOK = {
    "deepseek-v4-flash": {"cache_hit": 0.014, "cache_miss": 0.44, "output": 1.32},
    "deepseek-v4-pro": {"cache_hit": 0.044, "cache_miss": 1.32, "output": 3.96},
}

INPUT_TOO_LONG_MESSAGE = (
    "That's a bit long for this chat — could you shorten it and try again?"
)
RATE_LIMITED_MESSAGE = (
    "You've reached the limit of questions for now — please try again a bit later, "
    "or reach Steve directly via LinkedIn."
)
BUDGET_EXHAUSTED_MESSAGE = (
    "My twin is resting for today — you can reach Steve directly via LinkedIn or email."
)


def _env_int(name, default):
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def _env_float(name, default):
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


def _parse_rate_limit(spec):
    """Parse "20/hour" or "40/day" into (max_events, window_seconds)."""
    count_str, _, unit = spec.partition("/")
    return int(count_str), _UNIT_SECONDS[unit]


def estimate_cost_usd(usage, model=None):
    """Estimate the USD cost of one response's usage figures against the
    price table. `usage` is the dict avatar.llm.send_message returns."""
    model = model or os.environ.get("AVATAR_MODEL", DEFAULT_MODEL_FOR_PRICING)
    prices = PRICE_TABLE_USD_PER_MTOK.get(model)
    if prices is None:
        logger.warning(
            "No price table entry for model %s; using %s",
            model,
            DEFAULT_MODEL_FOR_PRICING,
        )
        prices = PRICE_TABLE_USD_PER_MTOK[DEFAULT_MODEL_FOR_PRICING]

    cache_hit_tokens = usage.get("cache_hit_tokens", 0)
    cache_miss_tokens = usage.get("cache_miss_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    cost = (
        cache_hit_tokens * prices["cache_hit"]
        + cache_miss_tokens * prices["cache_miss"]
        + output_tokens * prices["output"]
    ) / 1_000_000
    return cost


class _SlidingWindowLimiter:
    """Thread-safe sliding-window rate limiter, held in process memory.

    Stale keys (sessions/IPs that have not been seen within the window) are
    swept out opportunistically once the tracked-key count crosses
    `_CLEANUP_KEY_THRESHOLD`, so a long-running process does not accumulate
    one permanent entry per unique visitor.
    """

    def __init__(
        self,
        max_events,
        window_seconds,
        clock=time.time,
        cleanup_threshold=_CLEANUP_KEY_THRESHOLD,
    ):
        self.max_events = max_events
        self.window_seconds = window_seconds
        self._clock = clock
        self._cleanup_threshold = cleanup_threshold
        self._events = {}
        self._lock = threading.Lock()

    def _prune_stale_keys(self, cutoff):
        """Drop keys whose most recent event is older than `cutoff`.

        Caller must hold `self._lock`. A key is only removed when *all* of
        its recorded events are outside the active window, so a key that is
        still rate-limiting cannot be reset by cleanup.
        """
        stale = [key for key, events in self._events.items() if not events or events[-1] < cutoff]
        for key in stale:
            del self._events[key]

    def allow(self, key):
        now = self._clock()
        cutoff = now - self.window_seconds
        with self._lock:
            events = [t for t in self._events.get(key, []) if t >= cutoff]
            if len(events) >= self.max_events:
                self._events[key] = events
                return False
            events.append(now)
            self._events[key] = events

            if len(self._events) > self._cleanup_threshold:
                self._prune_stale_keys(cutoff)

            return True


class GuardrailState:
    """Holds the rate limiters and spend tracker for one running app.
    Accepts explicit config and a clock so tests can drive both directly,
    instead of going through environment variables and wall-clock time."""

    def __init__(
        self,
        max_input_chars=None,
        session_rate_limit=None,
        ip_rate_limit=None,
        daily_budget_usd=None,
        clock=time.time,
    ):
        self.max_input_chars = (
            _env_int("AVATAR_MAX_INPUT_CHARS", DEFAULT_MAX_INPUT_CHARS)
            if max_input_chars is None
            else max_input_chars
        )
        self.daily_budget_usd = (
            _env_float("AVATAR_DAILY_BUDGET_USD", DEFAULT_DAILY_BUDGET_USD)
            if daily_budget_usd is None
            else daily_budget_usd
        )

        session_count, session_window = _parse_rate_limit(
            session_rate_limit
            or os.environ.get("AVATAR_SESSION_RATE_LIMIT", DEFAULT_SESSION_RATE_LIMIT)
        )
        ip_count, ip_window = _parse_rate_limit(
            ip_rate_limit
            or os.environ.get("AVATAR_IP_RATE_LIMIT", DEFAULT_IP_RATE_LIMIT)
        )
        self._session_limiter = _SlidingWindowLimiter(
            session_count, session_window, clock=clock
        )
        self._ip_limiter = _SlidingWindowLimiter(ip_count, ip_window, clock=clock)

        self._clock = clock
        self._budget_lock = threading.Lock()
        self._budget_day = None
        self._spent_usd = 0.0

    def _current_utc_day(self):
        return datetime.fromtimestamp(self._clock(), tz=timezone.utc).date()

    def _reset_budget_if_new_day(self):
        today = self._current_utc_day()
        if self._budget_day != today:
            self._budget_day = today
            self._spent_usd = 0.0

    def is_budget_exhausted(self):
        with self._budget_lock:
            self._reset_budget_if_new_day()
            return self._spent_usd >= self.daily_budget_usd

    def record_usage(self, usage, model=None):
        cost = estimate_cost_usd(usage, model)
        with self._budget_lock:
            self._reset_budget_if_new_day()
            self._spent_usd += cost
        return cost

    def check_request(self, session_id, ip, message):
        """Returns (allowed, refusal_message). refusal_message is None when allowed."""
        if len(message) > self.max_input_chars:
            return False, INPUT_TOO_LONG_MESSAGE

        try:
            budget_exhausted = self.is_budget_exhausted()
        except Exception:
            logger.exception("Budget tracker failed; failing closed")
            budget_exhausted = True

        if budget_exhausted:
            return False, BUDGET_EXHAUSTED_MESSAGE

        if not self._session_limiter.allow(session_id):
            return False, RATE_LIMITED_MESSAGE
        if not self._ip_limiter.allow(ip):
            return False, RATE_LIMITED_MESSAGE

        return True, None


_default_state = None
_default_state_lock = threading.Lock()


def _get_default_state():
    global _default_state
    if _default_state is None:
        with _default_state_lock:
            if _default_state is None:
                _default_state = GuardrailState()
    return _default_state


def check_request(session_id, ip, message):
    return _get_default_state().check_request(session_id, ip, message)


def record_usage(usage, model=None):
    return _get_default_state().record_usage(usage, model)
