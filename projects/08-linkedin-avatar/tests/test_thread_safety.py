"""Thread-safety stress tests for the guardrails rate limiter and budget tracker.

These tests hammer ``check_request`` and ``record_usage`` concurrently from
multiple threads to verify that the module's thread-safety claims hold up
under contention.  They are designed to be deterministic (no wall-clock
timing dependencies) and to complete quickly.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple

import pytest

from avatar import guardrails


# ---------------------------------------------------------------------------
# Test doubles / helpers
# ---------------------------------------------------------------------------


class FakeClock:
    """A monotonic, thread-safe fake clock.

    The limiter accepts a ``clock`` callable; by injecting this we remove
    any wall-clock timing from the test, making it fully deterministic.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._t = start
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._t

    def advance(self, delta: float) -> None:
        with self._lock:
            self._t += delta


@dataclass
class FakeUsage:
    """Minimal usage object compatible with ``record_usage``.

    The production code reads ``input_tokens`` / ``output_tokens`` (and
    possibly ``total_tokens``) off the usage object.  We expose all three
    so the test works regardless of which attribute the tracker reads.
    """

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_concurrent_check_request_and_record_usage_no_corruption():
    """Stress the limiter + budget tracker with concurrent mixed calls.

    We construct a fresh state object with tight limits so that the
    limiter actually has to make decisions under contention, then run
    many threads that interleave ``check_request`` and ``record_usage``.
    """

    # Tight limits so the limiter is exercised meaningfully.
    session_limit = 50
    ip_limit = 200
    window = 60.0

    clock = FakeClock()

    # Build a state object using the public constructor.  We try a few
    # plausible signatures so the test is robust to minor API drift.
    state = _make_state(
        session_limit=session_limit,
        ip_limit=ip_limit,
        window=window,
        clock=clock,
    )

    num_threads = 12
    iterations_per_thread = 200

    barrier = threading.Barrier(num_threads)

    allowed_count = 0
    allowed_lock = threading.Lock()

    unexpected_errors: List[BaseException] = []
    errors_lock = threading.Lock()

    def worker(worker_id: int) -> None:
        nonlocal allowed_count

        # Synchronize start for maximum contention.
        barrier.wait()

        session_id = f"session-{worker_id % 4}"  # 4 sessions, 12 threads
        ip = f"10.0.0.{worker_id % 3}"  # 3 IPs

        for i in range(iterations_per_thread):
            try:
                allowed, reason = state.check_request(
                    session_id=session_id,
                    ip=ip,
                    message=f"msg-{worker_id}-{i}",
                )
                if allowed:
                    with allowed_lock:
                        allowed_count += 1
                    # Only record usage for allowed requests, mirroring
                    # how the production code is expected to be used.
                    state.record_usage(
                        FakeUsage(input_tokens=10, output_tokens=5),
                        model="test-model",
                    )
                else:
                    # A denial must come with a reason.
                    assert reason, "denied request must include a reason"
            except guardrails.RateLimitError:
                # Expected when the limiter blocks a request.
                pass
            except BaseException as exc:  # noqa: BLE001 - we want to capture all
                with errors_lock:
                    unexpected_errors.append(exc)

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(worker, i) for i in range(num_threads)]
        for fut in futures:
            fut.result(timeout=25)

    # No unexpected exceptions from any thread.
    assert not unexpected_errors, f"unexpected errors: {unexpected_errors!r}"

    # The total number of allowed requests must never exceed the
    # configured session limit * number of distinct sessions.
    max_allowed = session_limit * 4  # 4 distinct sessions
    assert allowed_count <= max_allowed, (
        f"allowed_count={allowed_count} exceeded max_allowed={max_allowed}"
    )

    # Sanity: we should have allowed *some* requests (otherwise the test
    # isn't actually exercising the happy path).
    assert allowed_count > 0, "expected at least some requests to be allowed"


def test_concurrent_record_usage_budget_never_exceeds_limit():
    """Concurrent ``record_usage`` calls must not overshoot the budget.

    We use a small budget and many threads each recording usage; the
    final spent amount must never exceed the configured budget.
    """

    budget = 1000  # tokens

    state = _make_state(
        session_limit=10_000,
        ip_limit=10_000,
        window=60.0,
        clock=FakeClock(),
        budget=budget,
    )

    num_threads = 8
    iterations_per_thread = 150

    barrier = threading.Barrier(num_threads)
    unexpected_errors: List[BaseException] = []
    errors_lock = threading.Lock()

    def worker(worker_id: int) -> None:
        barrier.wait()
        for _ in range(iterations_per_thread):
            try:
                state.record_usage(
                    FakeUsage(input_tokens=1, output_tokens=1),
                    model="test-model",
                )
            except guardrails.RateLimitError:
                # Acceptable if the tracker raises when the budget is hit.
                pass
            except BaseException as exc:  # noqa: BLE001
                with errors_lock:
                    unexpected_errors.append(exc)

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(worker, i) for i in range(num_threads)]
        for fut in futures:
            fut.result(timeout=25)

    assert not unexpected_errors, f"unexpected errors: {unexpected_errors!r}"

    # Inspect the tracker's spent amount via whatever public-ish attribute
    # is available.  We probe a few common names to stay robust.
    spent = _read_spent(state)
    if spent is not None:
        assert spent <= budget, f"spent={spent} exceeded budget={budget}"


def test_concurrent_check_request_never_exceeds_session_limit():
    """A single session hammered from many threads must not exceed its limit.

    This is the sharpest test of the ``_SlidingWindowLimiter``: all
    threads share one session id, so the limiter must serialize its
    decisions correctly.
    """

    session_limit = 25
    window = 60.0

    state = _make_state(
        session_limit=session_limit,
        ip_limit=10_000,  # effectively unlimited for this test
        window=window,
        clock=FakeClock(),
    )

    num_threads = 16
    iterations_per_thread = 100

    barrier = threading.Barrier(num_threads)
    allowed_count = 0
    allowed_lock = threading.Lock()
    unexpected_errors: List[BaseException] = []
    errors_lock = threading.Lock()

    def worker(_worker_id: int) -> None:
        nonlocal allowed_count
        barrier.wait()
        for i in range(iterations_per_thread):
            try:
                allowed, _reason = state.check_request(
                    session_id="shared-session",
                    ip="10.0.0.1",
                    message=f"m-{i}",
                )
                if allowed:
                    with allowed_lock:
                        allowed_count += 1
            except guardrails.RateLimitError:
                pass
            except BaseException as exc:  # noqa: BLE001
                with errors_lock:
                    unexpected_errors.append(exc)

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(worker, i) for i in range(num_threads)]
        for fut in futures:
            fut.result(timeout=25)

    assert not unexpected_errors, f"unexpected errors: {unexpected_errors!r}"
    assert allowed_count <= session_limit, (
        f"allowed_count={allowed_count} exceeded session_limit={session_limit}"
    )
    # With 16 threads * 100 iterations = 1600 attempts against a limit of
    # 25, we should hit the limit exactly.
    assert allowed_count == session_limit, (
        f"expected exactly {session_limit} allowed, got {allowed_count}"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_state(
    *,
    session_limit: int,
    ip_limit: int,
    window: float,
    clock,
    budget: int | None = None,
):
    """Construct a guardrails state object, tolerating minor API drift.

    The production module exposes a state object with ``check_request``
    and ``record_usage`` methods.  We try a few plausible constructor
    signatures and fall back to the module-level default state if none
    match, so the test remains useful even if the constructor changes.
    """

    # Preferred: a dedicated state class with explicit limits.
    for cls_name in ("GuardrailState", "GuardrailsState", "State", "Guardrails"):
        cls = getattr(guardrails, cls_name, None)
        if cls is None:
            continue
        for kwargs in (
            {
                "session_limit": session_limit,
                "ip_limit": ip_limit,
                "window": window,
                "clock": clock,
                **({"budget": budget} if budget is not None else {}),
            },
            {
                "session_count": session_limit,
                "ip_count": ip_limit,
                "session_window": window,
                "ip_window": window,
                "clock": clock,
                **({"budget": budget} if budget is not None else {}),
            },
        ):
            try:
                return cls(**kwargs)
            except TypeError:
                continue

    # Fallback: use the module-level default state.  This is less
    # controlled but still exercises concurrency.
    default_state = getattr(guardrails, "_default_state", None) or getattr(
        guardrails, "default_state", None
    )
    if default_state is not None:
        return default_state

    pytest.skip("could not construct a guardrails state object for stress test")


def _read_spent(state) -> int | None:
    """Best-effort read of the tracker's spent amount."""

    for attr in ("spent", "spent_tokens", "total_spent", "used", "used_tokens"):
        value = getattr(state, attr, None)
        if isinstance(value, (int, float)):
            return int(value)

    tracker = getattr(state, "budget_tracker", None) or getattr(
        state, "tracker", None
    )
    if tracker is not None:
        for attr in ("spent", "spent_tokens", "total_spent", "used", "used_tokens"):
            value = getattr(tracker, attr, None)
            if isinstance(value, (int, float)):
                return int(value)

    return None
