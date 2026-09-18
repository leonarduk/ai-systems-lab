#!/usr/bin/env python3
"""Check whether the `as_of` date in pricing.json is stale.

Reads the `as_of` field from projects/05-llm-cost-comparison/pricing.json,
parses it strictly as an ISO 8601 calendar date (YYYY-MM-DD), and compares
its age in whole months against a configurable threshold.

Modes
-----
The check supports two modes, selected via the ``STALENESS_MODE`` environment
variable (or the ``--mode`` CLI flag):

* ``warn`` (default): print a warning and exit 0 when stale. This matches the
  AI review's suggestion that the check be non-blocking by default.
* ``fail``: exit non-zero when stale, so CI blocks the PR.

Configuration
-------------
* ``STALENESS_THRESHOLD_MONTHS`` (env var) or ``--threshold-months`` (CLI):
  maximum allowed age in whole months. Defaults to ``DEFAULT_THRESHOLD_MONTHS``
  below. The threshold is never hardcoded into the comparison logic.
* ``PRICING_JSON_PATH`` (env var) or ``--pricing-json`` (CLI): path to the
  pricing file. Defaults to the sibling ``pricing.json`` in this project.

Scope
-----
This script is intentionally scoped to *staleness* only. It does not validate
the schema of ``as_of`` beyond the strict ``YYYY-MM-DD`` parse required to
compute an age, and it does not reject future dates. Those are tracked as a
separate follow-up.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

# --- Configuration (tunable without editing comparison logic) ---------------

#: Default maximum allowed age, in whole months, before the check reports
#: staleness. Override with the STALENESS_THRESHOLD_MONTHS env var or the
#: --threshold-months CLI flag.
DEFAULT_THRESHOLD_MONTHS = 6

#: Default mode: "warn" (non-blocking) or "fail" (blocking).
DEFAULT_MODE = "warn"

#: Default location of the pricing file, relative to this script.
DEFAULT_PRICING_JSON = Path(__file__).resolve().parent.parent / "pricing.json"

VALID_MODES = ("warn", "fail")


def _months_between(earlier: date, later: date) -> int:
    """Return the number of whole calendar months between two dates.

    Uses calendar-month arithmetic (not a fixed 30-day approximation) so the
    result is robust to varying month lengths. A partial month does not count.
    """
    months = (later.year - earlier.year) * 12 + (later.month - earlier.month)
    if later.day < earlier.day:
        months -= 1
    return months


def _parse_as_of(raw: object) -> date:
    """Strictly parse the ``as_of`` value as an ISO 8601 ``YYYY-MM-DD`` date.

    Raises ``ValueError`` for anything that is not a string in exactly that
    format (e.g. ``"2026-7-1"``, ``"2026-07-01T00:00:00Z"``, or a non-string).
    """
    if not isinstance(raw, str):
        raise ValueError(f"as_of must be a string, got {type(raw).__name__}")
    try:
        parsed = datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            f"as_of {raw!r} is not a valid ISO 8601 date (expected YYYY-MM-DD)"
        ) from exc
    # strptime is lenient about zero-padding in some locales; re-format and
    # compare to enforce strict YYYY-MM-DD.
    if parsed.isoformat() != raw:
        raise ValueError(
            f"as_of {raw!r} is not strictly formatted as YYYY-MM-DD"
        )
    return parsed


def _load_as_of(pricing_json: Path) -> date:
    if not pricing_json.is_file():
        raise FileNotFoundError(f"pricing file not found: {pricing_json}")
    with pricing_json.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "as_of" not in data:
        raise ValueError(f"{pricing_json} is missing the 'as_of' field")
    return _parse_as_of(data["as_of"])


def _resolve_threshold(cli_value: int | None) -> int:
    if cli_value is not None:
        return cli_value
    env_value = os.environ.get("STALENESS_THRESHOLD_MONTHS")
    if env_value is not None:
        try:
            return int(env_value)
        except ValueError as exc:
            raise ValueError(
                f"STALENESS_THRESHOLD_MONTHS must be an integer, got {env_value!r}"
            ) from exc
    return DEFAULT_THRESHOLD_MONTHS


def _resolve_mode(cli_value: str | None) -> str:
    mode = cli_value or os.environ.get("STALENESS_MODE") or DEFAULT_MODE
    mode = mode.lower()
    if mode not in VALID_MODES:
        raise ValueError(
            f"mode must be one of {VALID_MODES}, got {mode!r}"
        )
    return mode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check whether pricing.json's as_of date is stale.",
    )
    parser.add_argument(
        "--pricing-json",
        type=Path,
        default=Path(os.environ.get("PRICING_JSON_PATH", DEFAULT_PRICING_JSON)),
        help="Path to pricing.json (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold-months",
        type=int,
        default=None,
        help=(
            "Maximum allowed age in whole months "
            f"(default: ${DEFAULT_THRESHOLD_MONTHS} or STALENESS_THRESHOLD_MONTHS)"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default=None,
        help=(
            "warn = non-blocking (exit 0), fail = blocking (exit 1). "
            f"Default: {DEFAULT_MODE} or STALENESS_MODE"
        ),
    )
    parser.add_argument(
        "--today",
        type=str,
        default=None,
        help="Override today's date (YYYY-MM-DD) for testing.",
    )
    args = parser.parse_args(argv)

    try:
        threshold = _resolve_threshold(args.threshold_months)
        mode = _resolve_mode(args.mode)
    except ValueError as exc:
        print(f"::error::configuration error: {exc}", file=sys.stderr)
        return 2

    if threshold < 0:
        print(
            f"::error::threshold must be non-negative, got {threshold}",
            file=sys.stderr,
        )
        return 2

    try:
        as_of = _load_as_of(args.pricing_json)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"::error::failed to read as_of: {exc}", file=sys.stderr)
        return 2

    if args.today is not None:
        try:
            today = _parse_as_of(args.today)
        except ValueError as exc:
            print(f"::error::invalid --today value: {exc}", file=sys.stderr)
            return 2
    else:
        today = date.today()

    age_months = _months_between(as_of, today)

    if age_months > threshold:
        message = (
            f"pricing.json as_of={as_of.isoformat()} is {age_months} month(s) old "
            f"(threshold: {threshold} month(s), today: {today.isoformat()}). "
            "Please refresh the pricing data and update as_of."
        )
        if mode == "fail":
            print(f"::error::{message}")
            return 1
        print(f"::warning::{message}")
        return 0

    print(
        f"pricing.json as_of={as_of.isoformat()} is {age_months} month(s) old "
        f"(threshold: {threshold} month(s)) — OK."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
