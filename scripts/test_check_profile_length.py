"""Tests for scripts/check_profile_length.py.

Loaded via importlib since the module lives in a hyphen-free but
non-package top-level scripts/ directory (matches the pattern used for
scripts/test_validate_pr_description.py).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parent / "check_profile_length.py"
spec = importlib.util.spec_from_file_location("check_profile_length", MODULE_PATH)
check_profile_length = importlib.util.module_from_spec(spec)
sys.modules["check_profile_length"] = check_profile_length
spec.loader.exec_module(check_profile_length)


@pytest.fixture
def isolated_repo(tmp_path, monkeypatch):
    """Point the module's REPO_ROOT/PYPROJECT/DEFAULT_PROFILE at a scratch
    directory so tests never touch the real repo's pyproject.toml or
    profile.md."""
    monkeypatch.setattr(check_profile_length, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(check_profile_length, "PYPROJECT", tmp_path / "pyproject.toml")
    profile = tmp_path / "profile.md"
    monkeypatch.setattr(check_profile_length, "DEFAULT_PROFILE", profile)
    return tmp_path, profile


class TestLoadLimits:
    def test_missing_pyproject_returns_defaults(self, isolated_repo):
        assert check_profile_length.load_limits() == (
            check_profile_length.DEFAULT_MAX_WORDS,
            check_profile_length.DEFAULT_MAX_BYTES,
        )

    def test_reads_configured_limits(self, isolated_repo):
        tmp_path, _ = isolated_repo
        (tmp_path / "pyproject.toml").write_text(
            "[tool.profile-length]\nmax_words = 10\nmax_bytes = 20\n",
            encoding="utf-8",
        )
        assert check_profile_length.load_limits() == (10, 20)

    def test_malformed_pyproject_falls_back_to_defaults(self, isolated_repo, capsys):
        tmp_path, _ = isolated_repo
        (tmp_path / "pyproject.toml").write_text("not valid toml [[[", encoding="utf-8")

        result = check_profile_length.load_limits()

        assert result == (
            check_profile_length.DEFAULT_MAX_WORDS,
            check_profile_length.DEFAULT_MAX_BYTES,
        )
        assert "warning" in capsys.readouterr().err

    def test_missing_section_returns_defaults(self, isolated_repo):
        tmp_path, _ = isolated_repo
        (tmp_path / "pyproject.toml").write_text(
            "[tool.other]\nfoo = 1\n", encoding="utf-8"
        )
        assert check_profile_length.load_limits() == (
            check_profile_length.DEFAULT_MAX_WORDS,
            check_profile_length.DEFAULT_MAX_BYTES,
        )

    def test_non_integer_values_are_ignored(self, isolated_repo):
        tmp_path, _ = isolated_repo
        (tmp_path / "pyproject.toml").write_text(
            '[tool.profile-length]\nmax_words = "lots"\nmax_bytes = 20\n',
            encoding="utf-8",
        )
        assert check_profile_length.load_limits() == (
            check_profile_length.DEFAULT_MAX_WORDS,
            20,
        )


class TestCheck:
    def test_passes_when_under_limits(self, tmp_path, capsys):
        profile = tmp_path / "profile.md"
        profile.write_text("a b c", encoding="utf-8")

        exit_code = check_profile_length.check(profile, max_words=10, max_bytes=100)

        assert exit_code == 0
        assert "OK" in capsys.readouterr().out

    def test_exact_boundary_passes(self, tmp_path):
        profile = tmp_path / "profile.md"
        content = "a " * 9 + "a"  # exactly 10 words
        profile.write_bytes(content.encode("utf-8"))

        exit_code = check_profile_length.check(
            profile, max_words=10, max_bytes=len(content.encode("utf-8"))
        )

        assert exit_code == 0

    def test_fails_when_over_word_limit(self, tmp_path, capsys):
        profile = tmp_path / "profile.md"
        profile.write_text("a b c d e", encoding="utf-8")

        exit_code = check_profile_length.check(profile, max_words=3, max_bytes=1000)

        assert exit_code == 1
        assert "word count" in capsys.readouterr().err

    def test_fails_when_over_byte_limit(self, tmp_path, capsys):
        profile = tmp_path / "profile.md"
        profile.write_text("a" * 100, encoding="utf-8")

        exit_code = check_profile_length.check(profile, max_words=1000, max_bytes=10)

        assert exit_code == 1
        assert "byte count" in capsys.readouterr().err

    def test_missing_file_fails(self, tmp_path, capsys):
        profile = tmp_path / "does-not-exist.md"

        exit_code = check_profile_length.check(profile, max_words=10, max_bytes=100)

        assert exit_code == 1
        assert "not found" in capsys.readouterr().err


class TestMain:
    def test_uses_pyproject_limits_by_default(self, isolated_repo):
        tmp_path, profile = isolated_repo
        (tmp_path / "pyproject.toml").write_text(
            "[tool.profile-length]\nmax_words = 3\nmax_bytes = 1000\n",
            encoding="utf-8",
        )
        profile.write_text("a b c d e", encoding="utf-8")

        assert check_profile_length.main([]) == 1

    def test_cli_flags_override_pyproject_limits(self, isolated_repo):
        tmp_path, profile = isolated_repo
        (tmp_path / "pyproject.toml").write_text(
            "[tool.profile-length]\nmax_words = 3\nmax_bytes = 1000\n",
            encoding="utf-8",
        )
        profile.write_text("a b c d e", encoding="utf-8")

        assert check_profile_length.main(["--max-words", "100"]) == 0

    def test_explicit_profile_path_overrides_default(self, isolated_repo, tmp_path):
        other_profile = tmp_path / "other.md"
        other_profile.write_text("a b c", encoding="utf-8")

        exit_code = check_profile_length.main(
            ["--profile", str(other_profile), "--max-words", "10", "--max-bytes", "100"]
        )

        assert exit_code == 0
