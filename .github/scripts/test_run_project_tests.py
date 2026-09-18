"""Unit tests for run_project_tests.find_project_dirs (leonarduk/ai-systems-lab#255).

The script it tests exists specifically to stop issue-worm's local verifier
from narrating untested "tests pass" claims -- so the discovery logic it
relies on needs its own regression guard, not just a manual smoke run.
"""
from __future__ import annotations

import importlib.util
import subprocess
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
# ---------------------------------------------------------------------------
# Tests for main()'s --ignore wiring.
#
# main() iterates over find_project_dirs() and, for each project dir, builds
# `--ignore=<abs path>` args for every EXCLUDED entry whose parent is that
# project dir. That filter is the part most likely to silently regress: if
# it's removed or inverted, the runner would start collecting files that were
# meant to be excluded (or stop ignoring them) with no test failing.
#
# These tests are hermetic: they monkeypatch find_project_dirs (the actual
# source of main()'s project list), REPO_ROOT, EXCLUDED, subprocess.run, and
# importlib.util.find_spec so nothing touches the real repo layout.
# ---------------------------------------------------------------------------


def _build_synthetic_repo(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Create a synthetic repo layout mirroring the real one.

    Returns (repo_root, projects_root, project_with_excluded, project_clean).
    """
    repo_root = tmp_path / "repo"
    projects_root = repo_root / "projects"

    project_with_excluded = projects_root / "08-linkedin-avatar"
    _touch(project_with_excluded / "requirements.txt")
    _touch(project_with_excluded / "tests" / "test_thing.py")
    excluded_file = project_with_excluded / "tests" / "test_build_profile.py"
    _touch(excluded_file)

    project_clean = projects_root / "01-clean-project"
    _touch(project_clean / "requirements.txt")
    _touch(project_clean / "test_thing.py")

    return repo_root, projects_root, project_with_excluded, project_clean


def _run_main_capturing_argv(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
    projects_root: Path,
    project_dirs: list[Path],
    excluded: set[Path],
) -> list[list[str]]:
    """Invoke main() with subprocess.run stubbed out; return captured argvs."""
    captured: list[list[str]] = []

    def fake_run(argv, *args, **kwargs):
        captured.append(list(argv))

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(run_project_tests, "REPO_ROOT", repo_root)
    monkeypatch.setattr(run_project_tests, "PROJECTS_ROOT", projects_root)
    monkeypatch.setattr(run_project_tests, "EXCLUDED", excluded)
    monkeypatch.setattr(
        run_project_tests, "find_project_dirs", lambda *a, **k: list(project_dirs)
    )
    monkeypatch.setattr(run_project_tests.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_project_tests.importlib.util,
        "find_spec",
        lambda name: object() if name == "pytest" else None,
    )

    rc = run_project_tests.main()
    assert rc == 0
    return captured


def _argv_for_project(captured: list[list[str]], proj_dir: Path) -> list[str]:
    """Find the captured pytest argv whose project-dir argument matches."""
    for argv in captured:
        if str(proj_dir) in argv:
            return argv
    raise AssertionError(
        f"no captured argv referenced project dir {proj_dir}; got {captured!r}"
    )


def test_main_passes_ignore_args_for_excluded_files_in_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A project dir containing an excluded file must get --ignore=<abs path>
    for that file; a project dir with no excluded files must get none; and
    the .github/scripts-level EXCLUDED entry must never appear in any argv."""
    repo_root, projects_root, project_with_excluded, project_clean = (
        _build_synthetic_repo(tmp_path)
    )

    excluded_file = (
        project_with_excluded / "tests" / "test_build_profile.py"
    )
    scripts_excluded = (
        repo_root / ".github" / "scripts" / "test_extract_verdict.py"
    )
    _touch(scripts_excluded)

    excluded = {excluded_file, scripts_excluded}

    captured = _run_main_capturing_argv(
        monkeypatch,
        repo_root,
        projects_root,
        [project_with_excluded, project_clean],
        excluded,
    )

    # Positive case: the project containing the excluded file gets an
    # --ignore arg pointing at the absolute path of that excluded file.
    argv_with = _argv_for_project(captured, project_with_excluded)
    assert f"--ignore={excluded_file}" in argv_with

    # Negative case: the clean project gets no --ignore arg at all.
    argv_clean = _argv_for_project(captured, project_clean)
    assert not any(a.startswith("--ignore=") for a in argv_clean)

    # The .github/scripts-level EXCLUDED entry must never appear in any
    # project's --ignore args (its parent is not under projects/).
    for argv in captured:
        for arg in argv:
            assert arg != f"--ignore={scripts_excluded}"


def test_main_ignore_filter_fails_if_inverted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Guard against the filter being inverted: a project dir that is *not*
    the parent of an excluded file must not receive an --ignore for it, and
    the project that *is* the parent must receive it."""
    repo_root, projects_root, project_with_excluded, project_clean = (
        _build_synthetic_repo(tmp_path)
    )

    excluded_file = (
        project_with_excluded / "tests" / "test_build_profile.py"
    )
    excluded = {excluded_file}

    captured = _run_main_capturing_argv(
        monkeypatch,
        repo_root,
        projects_root,
        [project_with_excluded, project_clean],
        excluded,
    )

    argv_with = _argv_for_project(captured, project_with_excluded)
    argv_clean = _argv_for_project(captured, project_clean)

    assert f"--ignore={excluded_file}" in argv_with
    assert f"--ignore={excluded_file}" not in argv_clean


def test_main_ignore_args_use_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """--ignore args must be absolute paths (matching the real EXCLUDED set),
    so pytest resolves them regardless of cwd."""
    repo_root, projects_root, project_with_excluded, project_clean = (
        _build_synthetic_repo(tmp_path)
    )

    excluded_file = (
        project_with_excluded / "tests" / "test_build_profile.py"
    )
    excluded = {excluded_file}

    captured = _run_main_capturing_argv(
        monkeypatch,
        repo_root,
        projects_root,
        [project_with_excluded],
        excluded,
    )

    argv_with = _argv_for_project(captured, project_with_excluded)
    ignore_args = [a for a in argv_with if a.startswith("--ignore=")]
    assert ignore_args == [f"--ignore={excluded_file}"]
    assert Path(ignore_args[0].split("=", 1)[1]).is_absolute()
