#!/usr/bin/env python
"""Run each project's pytest suite in isolation, mirroring the "test" job in
.github/workflows/python-ci.yml (per-project pytest, not a single repo-wide
run — a flat `pytest projects/` hits name collisions between projects that
lack package structure, e.g. several unrelated test_server.py modules, and
some tests only import cleanly with their own project directory as rootdir).

Written in Python (not the workflow's bash loop) so it also works as a local
/ CI-mirroring check on Windows via cicaid's `.cicaid-checks.toml`.
"""
from __future__ import annotations

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


def find_project_dirs() -> list[Path]:
    test_files = sorted(
        p
        for p in PROJECTS_ROOT.rglob("*.py")
        if (p.name.startswith("test_") or p.name.endswith("_test.py"))
        and ".venv" not in p.parts
        and p not in EXCLUDED
    )
    seen: dict[Path, None] = {}
    for test_file in test_files:
        proj_dir = test_file.parent
        walk = test_file.parent
        while walk != PROJECTS_ROOT and PROJECTS_ROOT in walk.parents:
            if (walk / "requirements.txt").exists():
                proj_dir = walk
                break
            walk = walk.parent
        seen.setdefault(proj_dir, None)
    return list(seen)


def main() -> int:
    overall_status = 0
    for proj_dir in find_project_dirs():
        rel = proj_dir.relative_to(REPO_ROOT)
        print(f"::group::Testing {rel}")
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
            print(f"::error::Tests failed in {rel}")
            overall_status = 1
        print("::endgroup::")
    return overall_status


if __name__ == "__main__":
    raise SystemExit(main())
