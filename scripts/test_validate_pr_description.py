"""Unit tests for validate-pr-description.py.

The module under test is named with a hyphen (validate-pr-description.py),
which isn't a valid Python identifier, so it can't be `import`ed normally.
Loaded via importlib.util from its file path instead.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parent / "validate-pr-description.py"
_spec = importlib.util.spec_from_file_location("validate_pr_description", _MODULE_PATH)
vpd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vpd)


class TestExtractFilesAffected:
    """Direct tests of the parser, independent of main()'s exit-code logic."""

    def test_no_heading_returns_none_not_empty_list(self) -> None:
        # None and [] mean different things to main(): None means "this PR
        # doesn't use the convention at all" (not a violation); [] would mean
        # "opted in, but listed nothing" (a real, if odd, violation). Mixing
        # them up is exactly the bug that made every ordinary PR trip the
        # check — see test_false_positive_on_a_realistic_pr_body below.
        result = vpd.extract_files_affected("## What\nSome prose.\n")
        assert result is None

    def test_heading_with_backticked_bullets(self) -> None:
        desc = "## Files Affected\n- `a/b.yml`\n- `c/d.yml`\n"
        assert vpd.extract_files_affected(desc) == ["a/b.yml", "c/d.yml"]

    def test_heading_with_plain_bullets_and_annotations(self) -> None:
        desc = "## Files Affected\n- a/b.yml (modify — add a step)\n"
        assert vpd.extract_files_affected(desc) == ["a/b.yml"]

    def test_stops_at_the_next_heading(self) -> None:
        desc = "## Files Affected\n- `a.yml`\n## Testing\n- `b.yml`\n"
        assert vpd.extract_files_affected(desc) == ["a.yml"]

    def test_case_insensitive_and_any_heading_level(self) -> None:
        for text in ("# files affected", "### FILES AFFECTED", "## Files Affected:"):
            assert vpd.extract_files_affected(f"{text}\n- `a.yml`\n") == ["a.yml"]

    def test_singular_file_affected_also_matches(self) -> None:
        assert vpd.extract_files_affected("## File Affected\n- `a.yml`\n") == ["a.yml"]

    def test_empty_section_returns_empty_list_not_none(self) -> None:
        # Opted in (heading present) but listed nothing — a real, if
        # unusual, case that must be distinguishable from "no heading".
        assert vpd.extract_files_affected("## Files Affected\n\n## Why\n") == []


class TestNormalize:
    def test_strips_backticks_and_whitespace(self) -> None:
        assert vpd.normalize(" `a/b.yml` ") == "a/b.yml"

    def test_strips_a_leading_dot_slash_prefix(self) -> None:
        assert vpd.normalize("./a/b.yml") == "a/b.yml"

    def test_does_not_over_strip_a_path_starting_with_a_dot(self) -> None:
        # str.lstrip("./") strips the character *set*, not the literal
        # prefix, so ".github/workflows/x.yml" would lose its leading dot
        # too under the old implementation. Pinned directly against the
        # real path shape every workflow file in this repo has.
        assert vpd.normalize(".github/workflows/x.yml") == ".github/workflows/x.yml"

    def test_does_not_over_strip_a_dotfile_style_name(self) -> None:
        assert vpd.normalize(".foo/bar.yml") == ".foo/bar.yml"


class TestMain:
    """Exercises main() through subprocess, the same way CI invokes it."""

    def _run(self, description: str, changed: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(_MODULE_PATH)],
            env={
                "PR_DESCRIPTION": description,
                "CHANGED_WORKFLOW_FILES": changed,
                "PATH": "/usr/bin:/bin",
            },
            capture_output=True,
            text=True,
        )

    def test_no_changed_files_is_a_silent_success(self) -> None:
        result = self._run("anything", "")
        assert result.returncode == 0

    def test_false_positive_on_a_realistic_pr_body(self) -> None:
        # The actual bug this test suite exists to pin. Every PR reviewed
        # in this repo's history writes "## What" as prose, never a
        # "Files Affected" heading — including PR #89, the incident this
        # check is supposedly built from. Before the fix, this exact input
        # produced ::warning:: and exit 1 on every such PR, forever.
        body = (
            "## What\n"
            "Added a `User-Agent` header to the Octopus Agile API request.\n\n"
            "## Why\nMany APIs reject requests without one.\n\n"
            "## Testing\n- Ran the existing suite.\n"
        )
        result = self._run(body, ".github/workflows/python-ci.yml")
        assert result.returncode == 0
        assert "::warning::" not in result.stdout
        assert "no 'Files Affected' section" in result.stdout

    def test_empty_description_is_also_not_a_violation(self) -> None:
        # Same reasoning as the no-heading case: an empty description
        # doesn't use the convention either, so it isn't held to it.
        result = self._run("", ".github/workflows/python-ci.yml")
        assert result.returncode == 0
        assert "::warning::" not in result.stdout

    def test_incomplete_files_affected_section_is_flagged(self) -> None:
        # The actual failure mode issue #114 is about: a file changed in
        # the diff is missing from a Files Affected section the author DID
        # write. This must still warn and exit non-zero.
        body = "## Files Affected\n- `.github/workflows/a.yml`\n"
        changed = ".github/workflows/a.yml\n.github/workflows/b.yml"
        result = self._run(body, changed)
        assert result.returncode == 1
        assert "::warning::" in result.stdout
        assert ".github/workflows/b.yml" in result.stdout
        # The file that *was* listed must not also be reported as missing.
        assert "  - .github/workflows/a.yml" not in result.stdout

    def test_complete_files_affected_section_passes(self) -> None:
        body = "## Files Affected\n- `.github/workflows/a.yml`\n"
        result = self._run(body, ".github/workflows/a.yml")
        assert result.returncode == 0
        assert "::warning::" not in result.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
