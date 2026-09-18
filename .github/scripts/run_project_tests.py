#!/usr/bin/env python
"""Run each project's pytest suite in isolation, mirroring the "test" job in
.github/workflows/python-ci.yml (per-project pytest, not a single repo-wide
run — a flat `pytest projects/` hits name collisions between projects that
lack package structure, e.g. several unrelated test_server.py modules, and
some tests only import cleanly with their own project directory as rootdir).

Written in Python (not the workflow's bash loop) so it also works as a local
/ CI-mirroring check on Windows via cicaid's `.cicaid-checks.toml`.

Invariant: ``projects_root`` (i.e. ``projects/``) must NOT contain a
``requirements.txt``. If one existed, ``find_project_dirs`` would resolve
every test file's project dir to ``projects_root`` itself, collapsing the
per-project runs into the single flat ``pytest projects/`` run this script
exists to avoid (name collisions, wrong rootdirs). ``main()`` enforces this
with a loud guard rather than silently degrading.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECTS_ROOT = REPO_ROOT / "projects"

# Excluded for the same reasons as the "test" job in python-ci.yml:
# hardcodes a specific developer's local path / shells out to a hardcoded
# `python3` not guaranteed to be on PATH (e.g. on Windows).
EXCLUDED = {
    PROJECTS_ROOT / "01-mcp-server-suite" / "servers" / "git-mcp-server" / "test_git_server_integration.py",
    REPO_ROOT / ".github" / "scripts" / "test_extract_verdict.py",
    # projects/08-linkedin-avatar has its own build.py, imported as
    # `from build import ...`. In CI's fresh venv this shadows the pip
    # `build` package fine, but on a local machine that already has `build`
    # installed (a common packaging tool) the installed package can win,
    # causing a false-positive local-only collection error unrelated to any
    # real regression.
    PROJECTS_ROOT / "08-linkedin-avatar" / "tests" / "test_build_github_snapshot.py",
    PROJECTS_ROOT / "08-linkedin-avatar" / "tests" / "test_build_profile.py",
}

# GitHub Actions ::group::/::error:: annotations are only meaningful in a
# workflow log; printed locally via cicaid they're just noise.
_IN_CI = os.environ.get("GITHUB_ACTIONS") == "true"


def _group(title: str) -> None:
    print(f"::group::{title}" if _IN_CI else f"=== {title} ===")


def _end_group() -> None:
    if _IN_CI:
        print("::endgroup::")


def _error(message: str) -> None:
    print(f"::error::{message}" if _IN_CI else f"ERROR: {message}")


def find_project_dirs(
    projects_root: Path = PROJECTS_ROOT, excluded: set[Path] | None = None
) -> list[Path]:
    """Discover one pytest rootdir per project under ``projects_root``.

    Mirrors python-ci.yml's own walk: starting from each test file's
    directory, walk *upward toward and including* ``projects_root`` looking
    for the nearest ``requirements.txt``; a project directory itself (with
    no further ancestor to check) is still a valid stopping point. If no
    ``requirements.txt`` is found anywhere in that ancestry, the project is
    still tested from the test file's own directory, matching python-ci.yml
    rather than being silently skipped.

    Note: a ``requirements.txt`` placed directly at ``projects_root`` is
    unsupported. It would make every test file resolve to ``projects_root``,
    collapsing the per-project runs into the flat ``pytest projects/`` run
    this module exists to avoid. ``main()`` guards against this; callers
    passing a synthetic ``projects_root`` should not rely on that layout.
    """
    if excluded is None:
        excluded = EXCLUDED
    test_files = sorted(
        p
        for p in projects_root.rglob("*.py")
        if (p.name.startswith("test_") or p.name.endswith("_test.py"))
        and ".venv" not in p.parts
        and p not in excluded
    )
    seen: dict[Path, None] = {}
    for test_file in test_files:
        proj_dir = test_file.parent
        walk = test_file.parent
        while True:
            if (walk / "requirements.txt").exists():
                proj_dir = walk
                break
            if walk == projects_root or projects_root not in walk.parents:
                break
            walk = walk.parent
        seen.setdefault(proj_dir, None)
    return list(seen)


def main() -> int:
    if (PROJECTS_ROOT / "requirements.txt").exists():
        _error(
            f"{PROJECTS_ROOT.relative_to(REPO_ROOT)}/requirements.txt must not "
            "exist: it would collapse the per-project pytest runs into a "
            "single flat `pytest projects/` run (name collisions, wrong "
            "rootdirs). Move it into the individual project directory that "
            "needs it."
        )
        return 2

    if importlib.util.find_spec("pytest") is None:
        _error(
            "pytest is not installed in this environment "
            f"({sys.executable}) -- install requirements-dev.txt first. "
            "Treating this as an environment problem, not a test failure."
        )
        return 2

    overall_status = 0
    for proj_dir in find_project_dirs():
        rel = proj_dir.relative_to(REPO_ROOT)
        _group(f"Testing {rel}")
        ignore_args = [
            f"--ignore={excluded}"
            for excluded in EXCLUDED
            if proj_dir in excluded.parents
        ]
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(proj_dir), "-q", *ignore_args],
            cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            _error(f"Tests failed in {rel}")
            overall_status = 1
        _end_group()
    return overall_status


if __name__ == "__main__":
    raise SystemExit(main())
