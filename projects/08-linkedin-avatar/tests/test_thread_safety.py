"""Thread-safety stress tests for guardrails.

These tests exercise concurrent access to the rate limiter and budget
tracker. They are deliberately written to be non-flaky: they assert on
invariants that must hold regardless of scheduling (e.g. "the number of
allowed requests never exceeds the configured limit", "the recorded spend
equals the sum of the individual costs"), rather than on timing.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from avatar.guardrails import (
    GuardrailState,
    _SlidingWindowLimiter,
    estimate_cost_usd,
)


class _FakeClock:
    """Monotonic fake clock so tests never depend on wall time."""

    def __init__(self, start=1_000_000.0):
        self._now = start
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            return self._now

    def advance(self, seconds):
        with self._lock:
            self._now += seconds


def test_sliding_window_limiter_is_thread_safe():
    """Concurrent `allow` calls must never exceed `max_events` in a window."""
    max_events = 50
    limiter = _SlidingWindowLimiter(
        max_events=max_events, window_seconds=3600, clock=_FakeClock()
    )

    num_threads = 16
    calls_per_thread = 200
    barrier = threading.Barrier(num_threads)
    results = []
    results_lock = threading.Lock()

    def worker():
        barrier.wait()
        local = []
        for _ in range(calls_per_thread):
            local.append(limiter.allow("shared-key"))
        with results_lock:
            results.extend(local)

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(worker) for _ in range(num_threads)]
        for f in futures:
            f.result()  # propagate any exception

    allowed = sum(1 for r in results if r)
    assert allowed == max_events
    assert len(results) == num_threads * calls_per_thread


def test_sliding_window_limiter_cleanup_bounds_memory():
    """Many unique keys over time must not grow `_events` without bound."""
    clock = _FakeClock()
    limiter = _SlidingWindowLimiter(
        max_events=5,
        window_seconds=60,
        clock=clock,
        cleanup_threshold=100,
    )

    # Simulate 10k unique sessions spread across time. Each session is only
    # seen once, so after its window elapses it becomes stale and eligible
    # for cleanup.
    for i in range(10_000):
        limiter.allow(f"session-{i}")
        if i % 50 == 0:
            clock.advance(120)  # push earlier keys out of the window

    # The map must be bounded by the cleanup threshold plus the keys added
    # since the last sweep — never the full 10k.
    assert len(limiter._events) < 10_000
    assert len(limiter._events) <= 100 + 50


def test_sliding_window_limiter_cleanup_preserves_active_keys():
    """Cleanup must not evict keys that are still within the active window."""
    clock = _FakeClock()
    limiter = _SlidingWindowLimiter(
        max_events=3,
        window_seconds=3600,
        clock=clock,
        cleanup_threshold=5,
    )

    # Fill the limiter with active keys, then add enough new keys to trigger
    # a sweep. The original keys must survive because they are still inside
    # the window.
    for i in range(5):
        assert limiter.allow(f"active-{i}")

    for i in range(20):
        limiter.allow(f"new-{i}")

    for i in range(5):
        assert f"active-{i}" in limiter._events


def test_guardrail_state_concurrent_check_and_record():
    """Concurrent `check_request` + `record_usage` must stay consistent."""
    state = GuardrailState(
        max_input_chars=1000,
        session_rate_limit="1000/hour",
        ip_rate_limit="1000/hour",
        daily_budget_usd=1_000_000.0,  # effectively unlimited for this test
    )

    num_threads = 16
    calls_per_thread = 100
    barrier = threading.Barrier(num_threads)

    usage = {
        "cache_hit_tokens": 100,
        "cache_miss_tokens": 200,
        "output_tokens": 300,
    }
    per_call_cost = estimate_cost_usd(usage, model="deepseek-v4-flash")

    def worker(thread_id):
        barrier.wait()
        for i in range(calls_per_thread):
            allowed, refusal = state.check_request(
                session_id=f"session-{thread_id}",
                ip=f"10.0.0.{thread_id}",
                message="hello",
            )
            assert allowed is True
            assert refusal is None
            state.record_usage(usage, model="deepseek-v4-flash")

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(worker, tid) for tid in range(num_threads)]
        for f in futures:
            f.result()

    expected_total = per_call_cost * num_threads * calls_per_thread
    # Allow a tiny epsilon for float accumulation order.
    assert state._spent_usd == pytest.approx(expected_total, rel=1e-9)


def test_guardrail_state_concurrent_budget_enforcement():
    """Concurrent requests must not overshoot the daily budget kill-switch."""
    # Budget allows exactly 10 calls at the per-call cost.
    usage = {
        "cache_hit_tokens": 0,
        "cache_miss_tokens": 1_000_000,  # $0.44 per call
        "output_tokens": 0,
    }
    per_call_cost = estimate_cost_usd(usage, model="deepseek-v4-flash")
    budget = per_call_cost * 10

    state = GuardrailState(
        max_input_chars=1000,
        session_rate_limit="10000/hour",
        ip_rate_limit="10000/hour",
        daily_budget_usd=budget,
    )

    num_threads = 16
    calls_per_thread = 50
    barrier = threading.Barrier(num_threads)
    allowed_count = 0
    count_lock = threading.Lock()

    def worker(thread_id):
        nonlocal allowed_count
        barrier.wait()
        local_allowed = 0
        for _ in range(calls_per_thread):
            allowed, _ = state.check_request(
                session_id=f"s-{thread_id}",
                ip=f"ip-{thread_id}",
                message="hi",
            )
            if allowed:
                local_allowed += 1
                state.record_usage(usage, model="deepseek-v4-flash")
        with count_lock:
            allowed_count += local_allowed

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(worker, tid) for tid in range(num_threads)]
        for f in futures:
            f.result()

    # The budget kill-switch is checked before the call, so at most one
    # extra call can slip through per thread between the check and the
    # record. The invariant we care about: we never allow *all* requests
    # through, and the recorded spend never exceeds the budget by more than
    # the in-flight allowance.
    assert allowed_count < num_threads * calls_per_thread
    assert state._spent_usd <= budget + per_call_cost * num_threads
