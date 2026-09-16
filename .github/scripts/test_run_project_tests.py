"""Unit tests for run_project_tests.find_project_dirs (leonarduk/ai-systems-lab#255).

The script it tests exists specifically to stop issue-worm's local verifier
from narrating untested "tests pass" claims -- so the discovery logic it
relies on needs its own regression guard, not just a manual smoke run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_project_tests  # noqa: E402
from run_project_tests import find_project_dirs  # noqa: E402


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_uses_nearest_ancestor_requirements_txt(tmp_path: Path):
    """A tests/ subfolder's requirements.txt lives in its parent -- the
    project dir should be that parent, not the tests/ folder itself."""
    projects_root = tmp_path / "projects"
    _touch(projects_root / "foo" / "requirements.txt")
    _touch(projects_root / "foo" / "tests" / "test_thing.py")

    assert find_project_dirs(projects_root, excluded=set()) == [
        projects_root / "foo"
    ]


def test_falls_back_to_test_files_own_directory_when_no_requirements_txt(
    tmp_path: Path,
):
    """No requirements.txt anywhere in the ancestry -> still tested from the
    test file's own directory, matching python-ci.yml, rather than silently
    dropped."""
    projects_root = tmp_path / "projects"
    _touch(projects_root / "bar" / "test_thing.py")

    assert find_project_dirs(projects_root, excluded=set()) == [
        projects_root / "bar"
    ]


def test_requirements_txt_directly_in_projects_root_is_found(tmp_path: Path):
    """A project whose requirements.txt sits directly at projects_root
    (rather than in some deeper ancestor) must still be found -- the walk
    must check projects_root itself, not stop just before it.

    NOTE: this layout is unsupported by ``main()`` (see the guard test
    below); this test only pins the low-level walk behavior of
    ``find_project_dirs`` for callers that pass a synthetic root.
    """
    projects_root = tmp_path / "projects"
    _touch(projects_root / "requirements.txt")
    _touch(projects_root / "baz" / "test_thing.py")

    assert find_project_dirs(projects_root, excluded=set()) == [projects_root]


def test_excluded_test_files_do_not_contribute_a_project_dir(tmp_path: Path):
    """A project whose only test file is excluded shouldn't be discovered
    at all -- excluding a file is meant to skip it entirely, not just skip
    it from an otherwise-still-run project."""
    projects_root = tmp_path / "projects"
    only_test = projects_root / "quux" / "test_thing.py"
    _touch(only_test)

    assert find_project_dirs(projects_root, excluded={only_test}) == []


def test_multiple_test_files_in_the_same_project_yield_one_entry(tmp_path: Path):
    projects_root = tmp_path / "projects"
    _touch(projects_root / "multi" / "requirements.txt")
    _touch(projects_root / "multi" / "test_a.py")
    _touch(projects_root / "multi" / "test_b.py")

    assert find_project_dirs(projects_root, excluded=set()) == [
        projects_root / "multi"
    ]


def test_main_rejects_requirements_txt_at_projects_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """``main()`` must fail loudly (exit 2, environment/config problem, not
    a test failure) if ``projects_root`` contains a ``requirements.txt`` --
    otherwise the runner silently degrades into the flat ``pytest projects/``
    run it exists to avoid."""
    projects_root = tmp_path / "projects"
    _touch(projects_root / "requirements.txt")
    _touch(projects_root / "foo" / "test_thing.py")

    monkeypatch.setattr(run_project_tests, "PROJECTS_ROOT", projects_root)
    monkeypatch.setattr(run_project_tests, "REPO_ROOT", tmp_path)

    # Ensure the guard fires before any pytest discovery/exec.
    def _fail_run(*_args, **_kwargs):  # pragma: no cover - must not be called
        raise AssertionError("subprocess.run should not be reached")

    monkeypatch.setattr(run_project_tests.subprocess, "run", _fail_run)

    assert run_project_tests.main() == 2
