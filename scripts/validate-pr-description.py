#!/usr/bin/env python3
"""Validate that changed workflow files are listed in the PR description.

Reads the PR description from the PR_DESCRIPTION environment variable (or a
file path given via PR_DESCRIPTION_FILE), parses the "Files Affected" section,
and compares it against the list of changed workflow files provided via the
CHANGED_WORKFLOW_FILES environment variable (newline-separated) or as
positional arguments.

Exits non-zero if any changed workflow file is missing from the description.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


HEADING_RE = re.compile(
    r"^\s{0,3}#{1,6}\s*files?\s+affected\s*:?\s*$",
    re.IGNORECASE,
)
ANY_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*\S)\s*$")
BACKTICK_RE = re.compile(r"`([^`]+)`")


def read_description() -> str:
    desc_file = os.environ.get("PR_DESCRIPTION_FILE")
    if desc_file:
        try:
            return Path(desc_file).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"::error::Failed to read PR description file {desc_file}: {exc}")
            sys.exit(2)

    return os.environ.get("PR_DESCRIPTION", "")


def extract_files_affected(description: str) -> list[str]:
    """Return the list of file paths listed under the Files Affected heading."""
    lines = description.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if HEADING_RE.match(line):
            start = idx + 1
            break

    if start is None:
        return []

    collected: list[str] = []
    for line in lines[start:]:
        if ANY_HEADING_RE.match(line):
            break
        bullet = BULLET_RE.match(line)
        if not bullet:
            continue
        content = bullet.group(1).strip()
        # Prefer backticked paths if present, otherwise take the first token.
        backticks = BACKTICK_RE.findall(content)
        if backticks:
            collected.extend(bt.strip() for bt in backticks if bt.strip())
        else:
            # Strip trailing parenthetical annotations like "(modify — ...)".
            token = re.split(r"\s+[—\-–(]", content, maxsplit=1)[0].strip()
            token = token.rstrip(",;:")
            if token:
                collected.append(token)

    return collected


def normalize(path: str) -> str:
    p = path.strip().strip("`").strip()
    p = p.lstrip("./")
    return p


def get_changed_files() -> list[str]:
    env_val = os.environ.get("CHANGED_WORKFLOW_FILES", "")
    files = [line.strip() for line in env_val.splitlines() if line.strip()]
    files.extend(arg for arg in sys.argv[1:] if arg.strip())
    return files


def main() -> int:
    description = read_description()
    changed = get_changed_files()

    if not changed:
        print("No changed workflow files detected; nothing to validate.")
        return 0

    if not description.strip():
        print("::warning::PR description is empty; cannot validate Files Affected section.")
        print("Missing from Files Affected section:")
        for f in changed:
            print(f"  - {f}")
        return 1

    listed_raw = extract_files_affected(description)
    listed = {normalize(p) for p in listed_raw}

    missing = [f for f in changed if normalize(f) not in listed]

    if not missing:
        print("All changed workflow files are listed in the Files Affected section.")
        return 0

    print("::warning::The following changed workflow files are not listed in the PR description's 'Files Affected' section:")
    for f in missing:
        print(f"  - {f}")
    print()
    print("Please update the PR description to include these files under '## Files Affected'.")
    print("(This check is informational and does not block merge.)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
