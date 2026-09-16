"""Tests for avatar/guardrails.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from avatar import guardrails  # noqa: E402


class FakeClock:
    def __init__(self, start=1_000_000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_state(**overrides):
    defaults = {
        "max_input_chars": 1500,
        "session_rate_limit": "20/hour",
        "ip_rate_limit": "40/day",
        "daily_budget_usd": 2.00,
        "clock": FakeClock(),
    }
    defaults.update(overrides)
    return guardrails.GuardrailState(**defaults), defaults["clock"]


class TestInputLengthCap:
    def test_oversized_message_rejected(self):
        state, _ = make_state(max_input_chars=10)
        allowed, reason = state.check_request("session-1", "1.2.3.4", "x" * 11)
        assert allowed is False
        assert reason == guardrails.INPUT_TOO_LONG_MESSAGE

    def test_message_at_limit_is_allowed(self):
        state, _ = make_state(max_input_chars=10)
        allowed, reason = state.check_request("session-1", "1.2.3.4", "x" * 10)
        assert allowed is True
        assert reason is None


class TestSlidingWindowRateLimits:
    def test_session_limit_blocks_after_max_events(self):
        state, clock = make_state(session_rate_limit="2/hour", ip_rate_limit="1000/day")
        assert state.check_request("s1", "1.1.1.1", "hi")[0] is True
        assert state.check_request("s1", "1.1.1.1", "hi")[0] is True
        allowed, reason = state.check_request("s1", "1.1.1.1", "hi")
        assert allowed is False
        assert reason == guardrails.RATE_LIMITED_MESSAGE

    def test_session_limit_releases_after_window_passes(self):
        state, clock = make_state(session_rate_limit="1/hour", ip_rate_limit="1000/day")
        assert state.check_request("s1", "1.1.1.1", "hi")[0] is True
        assert state.check_request("s1", "1.1.1.1", "hi")[0] is False

        clock.advance(3601)

        assert state.check_request("s1", "1.1.1.1", "hi")[0] is True

    def test_rate_limit_releases_at_exact_window_boundary(self):
        # Boundary test: the sliding window uses `cutoff = now - window_seconds`
        # and evicts events with `timestamp <= cutoff`. At exactly
        # `window_seconds` elapsed, the original event must be evicted and a
        # new request allowed. This catches off-by-one regressions (e.g. using
        # `<` instead of `<=`, or `>` instead of `>=`).
        clock = FakeClock()
        limiter = guardrails._SlidingWindowLimiter(
            max_events=1, window_seconds=3600, clock=clock
        )

        # Fill the window.
        assert limiter.allow() is True
        # Immediately after, the limit is enforced.
        assert limiter.allow() is False

        # Advance to exactly window_seconds - 1: still within the window.
        clock.advance(3599)
        assert limiter.allow() is False

        # Advance one more second to land exactly on window_seconds elapsed.
        clock.advance(1)
        assert limiter.allow() is True

    def test_rate_limit_enforced_just_before_window_boundary(self):
        # Inverse boundary test: at `window_seconds - 1` elapsed, the original
        # event is still inside the window and the limit must remain enforced.
        clock = FakeClock()
        limiter = guardrails._SlidingWindowLimiter(
            max_events=1, window_seconds=3600, clock=clock
        )

        assert limiter.allow() is True
        assert limiter.allow() is False

        clock.advance(3599)
        assert limiter.allow() is False

    def test_ip_limit_is_independent_of_session_limit(self):
        state, clock = make_state(session_rate_limit="1000/hour", ip_rate_limit="1/day")
        assert state.check_request("session-a", "9.9.9.9", "hi")[0] is True
        # Different session, same IP: IP limit still blocks.
        allowed, reason = state.check_request("session-b", "9.9.9.9", "hi")
        assert allowed is False
        assert reason == guardrails.RATE_LIMITED_MESSAGE

    def test_different_sessions_have_independent_limits(self):
        state, clock = make_state(session_rate_limit="1/hour", ip_rate_limit="1000/day")
        assert state.check_request("session-a", "1.1.1.1", "hi")[0] is True
        assert state.check_request("session-b", "2.2.2.2", "hi")[0] is True


class TestEstimateCostUsd:
    def test_known_usage_figures(self):
        usage = {
            "cache_hit_tokens": 1_000_000,
            "cache_miss_tokens": 0,
            "output_tokens": 0,
        }
        cost = guardrails.estimate_cost_usd(usage, model="deepseek-v4-flash")
        assert cost == pytest.approx(0.014)

    def test_cache_miss_and_output_priced_separately(self):
        usage = {
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 1_000_000,
            "output_tokens": 1_000_000,
        }
        cost = guardrails.estimate_cost_usd(usage, model="deepseek-v4-flash")
        assert cost == pytest.approx(0.44 + 1.32)

    def test_unknown_model_falls_back_to_default(self):
        usage = {
            "cache_hit_tokens": 1_000_000,
            "cache_miss_tokens": 0,
            "output_tokens": 0,
        }
        cost = guardrails.estimate_cost_usd(usage, model="not-a-real-model")
        default_cost = guardrails.estimate_cost_usd(
            usage, model=guardrails.DEFAULT_MODEL_FOR_PRICING
        )
        assert cost == pytest.approx(default_cost)


class TestBudgetKillSwitch:
    def test_crossing_budget_switches_to_fixed_message(self):
        state, clock = make_state(daily_budget_usd=0.01)
        big_usage = {
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 0,
            "output_tokens": 1_000_000,
        }

        assert state.check_request("s1", "1.1.1.1", "hi")[0] is True
        state.record_usage(big_usage, model="deepseek-v4-flash")

        allowed, reason = state.check_request("s2", "2.2.2.2", "hi")
        assert allowed is False
        assert reason == guardrails.BUDGET_EXHAUSTED_MESSAGE

    def test_budget_resets_on_utc_day_boundary(self):
        state, clock = make_state(daily_budget_usd=0.01)
        big_usage = {
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 0,
            "output_tokens": 1_000_000,
        }
        state.record_usage(big_usage, model="deepseek-v4-flash")
        assert state.is_budget_exhausted() is True

        clock.advance(86400)

        assert state.is_budget_exhausted() is False

    def test_budget_tracker_failure_fails_closed(self, monkeypatch):
        state, clock = make_state()

        def boom():
            raise RuntimeError("budget tracker exploded")

        monkeypatch.setattr(state, "is_budget_exhausted", boom)

        allowed, reason = state.check_request("s1", "1.1.1.1", "hi")

        assert allowed is False
        assert reason == guardrails.BUDGET_EXHAUSTED_MESSAGE

    def test_oversized_message_rejected_without_touching_budget(self):
        state, clock = make_state(max_input_chars=5, daily_budget_usd=1000)
        allowed, reason = state.check_request("s1", "1.1.1.1", "way too long")
        assert allowed is False
        assert reason == guardrails.INPUT_TOO_LONG_MESSAGE
        # Budget untouched: still not exhausted at a huge budget.
        assert state.is_budget_exhausted() is False


class TestParseRateLimit:
    def test_parses_hour_and_day(self):
        assert guardrails._parse_rate_limit("20/hour") == (20, 3600)
        assert guardrails._parse_rate_limit("40/day") == (40, 86400)


class TestDefaultModuleFunctions:
    def test_check_request_and_record_usage_use_a_shared_default_state(
        self, monkeypatch
    ):
        monkeypatch.setattr(guardrails, "_default_state", None)
        monkeypatch.setenv("AVATAR_DAILY_BUDGET_USD", "1000")
        monkeypatch.setenv("AVATAR_SESSION_RATE_LIMIT", "1000/hour")
        monkeypatch.setenv("AVATAR_IP_RATE_LIMIT", "1000/day")

        allowed, reason = guardrails.check_request("s1", "1.1.1.1", "hi")

        assert allowed is True
        assert reason is None
