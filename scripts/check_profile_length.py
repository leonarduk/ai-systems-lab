#!/usr/bin/env python3
"""Enforce a maximum length on the LinkedIn avatar profile document.

The profile at projects/08-linkedin-avatar/knowledge/profile.md is injected
into LLM context windows by the avatar MCP server. If it grows unboundedly it
risks exceeding context windows, diluting signal, and becoming hard to review.

This script fails (exit code 1) when the profile exceeds the configured word
or byte limit. Limits are read from [tool.profile-length] in pyproject.toml
when available, otherwise sensible defaults are used.

Run locally:
    python scripts/check_profile_length.py
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE = REPO_ROOT / "projects" / "08-linkedin-avatar" / "knowledge" / "profile.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"

DEFAULT_MAX_WORDS = 4000
DEFAULT_MAX_BYTES = 25000


def load_limits() -> tuple[int, int]:
    """Return (max_words, max_bytes) from pyproject.toml, falling back to defaults."""
    max_words, max_bytes = DEFAULT_MAX_WORDS, DEFAULT_MAX_BYTES
    if not PYPROJECT.is_file():
        return max_words, max_bytes
    try:
        with PYPROJECT.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"warning: could not read {PYPROJECT}: {exc}", file=sys.stderr)
        return max_words, max_bytes

    section = data.get("tool", {}).get("profile-length", {})
    if isinstance(section, dict):
        if isinstance(section.get("max_words"), int):
            max_words = section["max_words"]
        if isinstance(section.get("max_bytes"), int):
            max_bytes = section["max_bytes"]
    return max_words, max_bytes


def check(profile: Path, max_words: int, max_bytes: int) -> int:
    if not profile.is_file():
        print(f"error: profile not found: {profile}", file=sys.stderr)
        return 1

    raw = profile.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    byte_count = len(raw)
    word_count = len(text.split())

    print(f"profile: {profile.relative_to(REPO_ROOT) if profile.is_relative_to(REPO_ROOT) else profile}")
    print(f"  words: {word_count} / {max_words}")
    print(f"  bytes: {byte_count} / {max_bytes}")

    failures = []
    if word_count > max_words:
        failures.append(f"word count {word_count} exceeds limit {max_words}")
    if byte_count > max_bytes:
        failures.append(f"byte count {byte_count} exceeds limit {max_bytes}")

    if failures:
        print("", file=sys.stderr)
        for msg in failures:
            print(f"error: {msg}", file=sys.stderr)
        print(
            "Trim the profile or raise [tool.profile-length] limits in pyproject.toml.",
            file=sys.stderr,
        )
        return 1

    print("OK: profile is within configured limits.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help=f"Path to profile.md (default: {DEFAULT_PROFILE})",
    )
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--max-bytes", type=int, default=None)
    args = parser.parse_args(argv)

    cfg_words, cfg_bytes = load_limits()
    max_words = args.max_words if args.max_words is not None else cfg_words
    max_bytes = args.max_bytes if args.max_bytes is not None else cfg_bytes

    return check(args.profile, max_words, max_bytes)


if __name__ == "__main__":
    sys.exit(main())
