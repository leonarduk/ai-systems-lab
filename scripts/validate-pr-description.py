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


def extract_files_affected(description: str) -> list[str] | None:
    """Return the file paths listed under the Files Affected heading.

    Returns None (not an empty list) when no such heading exists at all, so
    callers can tell "the section is there but empty" apart from "there is
    no section" — the two mean very different things (see main()).
    """
    lines = description.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if HEADING_RE.match(line):
            start = idx + 1
            break

    if start is None:
        return None

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
    # A literal "./" prefix, not a leading run of '.' and '/' characters:
    # str.lstrip("./") strips the character *set* {'.', '/'}, so a real repo
    # path like ".github/workflows/foo.yml" loses its leading dot too,
    # becoming "github/workflows/foo.yml". That happens to cancel out here
    # because both the listed path and the actual changed-file path go
    # through the same normalize(), but it's a landmine for any caller that
    # compares a normalized path against something that didn't go through
    # this function — strip only an actual "./" prefix instead.
    while p.startswith("./"):
        p = p[2:]
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

    listed_raw = extract_files_affected(description)

    if listed_raw is None:
        # No "Files Affected" heading at all — which, checked against every
        # real PR in this repo's history (including #89, the incident this
        # check exists for), is not an exception: no PR here has ever used
        # this heading. issue-worm's own generated PRs write "## What" as
        # prose describing the change, not a parseable file list. Treating
        # "doesn't use an optional heading nobody uses" as a documentation
        # violation would fire ::warning:: on every single workflow-touching
        # PR from here on, forever, with no way to distinguish a genuinely
        # undocumented change from an ordinarily-written one — the exact
        # kind of unfixable, un-actionable noise that trains reviewers to
        # ignore every warning this check ever produces. So this is
        # informational only, and does not fail the check.
        print(
            "PR description has no 'Files Affected' section — nothing to "
            "cross-check. (Add one to get this check's real value: it "
            "verifies the section against the diff once you opt in.)"
        )
        return 0

    listed = {normalize(p) for p in listed_raw}
    missing = [f for f in changed if normalize(f) not in listed]

    if not missing:
        print("All changed workflow files are listed in the Files Affected section.")
        return 0

    # Here, unlike the no-heading case above, there IS a real drift: the
    # author opted into a Files Affected section and then missed a file —
    # the same species of mistake as PR #89 (which had no Files Affected
    # section; the omission there was from its "## What" bullet list
    # instead, so this check wouldn't actually have caught #89 itself, but
    # it catches the identical failure mode for anyone who does use this
    # heading). This is worth a real warning.
    print(
        "::warning::The following changed workflow files are not listed in the PR description's 'Files Affected' section:"
    )
    for f in missing:
        print(f"  - {f}")
    print()
    print(
        "Please update the PR description to include these files under '## Files Affected'."
    )
    print("(This check is informational and does not block merge.)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
