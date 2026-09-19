"""Unit tests for extract_verdict.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Add the scripts directory to the Python path so we can import extract_verdict
sys.path.insert(0, str(Path(__file__).parent))

from extract_verdict import extract_verdict


class TestExtractVerdict:
    """Test the extract_verdict function with various verdict formats."""

    def test_plain_approve(self) -> None:
        """Test plain **APPROVE** format."""
        review_text = """
## Review

Some analysis here.

**APPROVE** — no blocking concerns
"""
        assert extract_verdict(review_text) == "APPROVE"

    def test_plain_request_changes(self) -> None:
        """Test plain **REQUEST CHANGES** format."""
        review_text = """
## Review

Found an issue.

**REQUEST CHANGES** — blocking bug on line 42
"""
        assert extract_verdict(review_text) == "REQUEST CHANGES"

    def test_backtick_wrapped_approve(self) -> None:
        """Test backtick-wrapped **`APPROVE`** format (DeepSeek variant)."""
        review_text = """
## Review

Everything looks good.

**`APPROVE`** — no blocking concerns
"""
        assert extract_verdict(review_text) == "APPROVE"

    def test_backtick_wrapped_request_changes(self) -> None:
        """Test backtick-wrapped **`REQUEST CHANGES`** format."""
        review_text = """
## Review

Found blocking issues.

**`REQUEST CHANGES`** — unhandled edge case
"""
        assert extract_verdict(review_text) == "REQUEST CHANGES"

    def test_verdict_in_middle_of_text(self) -> None:
        """Test that verdict is found even when surrounded by other text."""
        review_text = """
### 1. Acceptance criteria
Looks good.

### 2. Bugs
None found.

**APPROVE** — ready to merge

Some closing thoughts.
"""
        assert extract_verdict(review_text) == "APPROVE"

    def test_backtick_wrapped_in_middle_of_text(self) -> None:
        """Test backtick-wrapped verdict in middle of review."""
        review_text = """
### Analysis

All sections reviewed.

**`APPROVE`** — no blockers

Final note: great work!
"""
        assert extract_verdict(review_text) == "APPROVE"

    def test_no_verdict(self) -> None:
        """Test that None is returned when no valid verdict is found."""
        review_text = """
## Review

Some analysis here with no verdict line.

Looks good overall.
"""
        assert extract_verdict(review_text) is None

    def test_empty_string(self) -> None:
        """Test that None is returned for empty input."""
        assert extract_verdict("") is None

    def test_whitespace_only(self) -> None:
        """Test that None is returned for whitespace-only input."""
        assert extract_verdict("   \n\n   ") is None

    def test_verdict_with_extra_context(self) -> None:
        """Test verdict extraction with full review context."""
        review_text = """
## Acceptance criteria
All criteria met.

## Bugs and logic errors
None found.

## API, data, and workflow safety
No issues.

## Test coverage
Tests cover the new functionality.

**APPROVE** — no blocking concerns (non-blocking items go in section 5 above)
"""
        assert extract_verdict(review_text) == "APPROVE"

    def test_backtick_variant_with_full_context(self) -> None:
        """Test backtick variant with full review context."""
        review_text = """
## Acceptance criteria
AC met.

## Bugs
Found one.

**`REQUEST CHANGES`** — unhandled exception at line 123
"""
        assert extract_verdict(review_text) == "REQUEST CHANGES"

    def test_case_sensitivity(self) -> None:
        """Test that verdict matching is case-sensitive (reject lowercase)."""
        review_text = "**approve** — no issues"
        assert extract_verdict(review_text) is None

    def test_verdict_embedded_in_sentence_no_match(self) -> None:
        """Test that a verdict word embedded in a sentence (not alone on its line) doesn't match."""
        review_text = "I think we should APPROVE — no issues"
        assert extract_verdict(review_text) is None

    def test_unbolded_verdict_alone_on_line_matches(self) -> None:
        """Test that an unbolded verdict alone on its line matches via the fallback."""
        review_text = "APPROVE — no issues"  # missing bold markers, but alone on the line
        assert extract_verdict(review_text) == "APPROVE"

    def test_single_backtick_malformed_format_matches(self) -> None:
        """Test that the regex matches a single-backtick malformed format (e.g., **`APPROVE**)."""
        review_text = "**`APPROVE** — missing closing backtick"
        verdict = extract_verdict(review_text)
        assert verdict == "APPROVE"

    def test_multiple_verdicts_returns_first(self) -> None:
        """Test that when multiple verdicts appear, the first one is returned."""
        review_text = """
Initial analysis: **APPROVE** — looks good

Wait, I found an issue: **REQUEST CHANGES** — blocking bug
"""
        assert extract_verdict(review_text) == "APPROVE"

    def test_bulleted_unbolded_approve(self) -> None:
        """Test fallback for a bulleted, unbolded verdict (observed from DeepSeek on PR #4801)."""
        review_text = """
## Acceptance criteria
All criteria met.

## Bugs
None found.

- APPROVE
"""
        assert extract_verdict(review_text) == "APPROVE"

    def test_bulleted_unbolded_request_changes(self) -> None:
        """Test fallback for a bulleted, unbolded REQUEST CHANGES verdict."""
        review_text = """
## Bugs
Found a blocking issue at line 42.

- REQUEST CHANGES — unhandled null case
"""
        assert extract_verdict(review_text) == "REQUEST CHANGES"

    def test_plain_unbolded_line_approve(self) -> None:
        """Test fallback for a plain (non-bulleted) unbolded verdict line."""
        review_text = """
## Review

Everything checks out.

APPROVE
"""
        assert extract_verdict(review_text) == "APPROVE"

    def test_unbolded_verdict_prefers_last_occurrence(self) -> None:
        """When multiple unbolded verdict-only lines appear, the last one wins."""
        review_text = """
- APPROVE

Actually, on reflection:

- REQUEST CHANGES — found a blocking bug
"""
        assert extract_verdict(review_text) == "REQUEST CHANGES"

    def test_unbolded_word_in_prose_does_not_match(self) -> None:
        """A verdict word embedded in a sentence must not trigger the fallback."""
        review_text = """
## Acceptance criteria
The reviewer should APPROVE this once the tests pass.

## Bugs
None found.
"""
        assert extract_verdict(review_text) is None

    def test_unbolded_word_in_checklist_item_does_not_match(self) -> None:
        """A checklist item that merely mentions the verdict word must not trigger the fallback."""
        review_text = """
## Acceptance criteria
- [ ] Verify APPROVE flow renders the confirmation banner
- [ ] REQUEST CHANGES button disabled while loading

## Bugs
None found.
"""
        assert extract_verdict(review_text) is None

    def test_bold_verdict_preferred_over_unbolded_mention(self) -> None:
        """When both a bold verdict and a stray unbolded mention exist, the bold one wins."""
        review_text = """
## Acceptance criteria
- [ ] Verify APPROVE flow renders correctly

## Verdict
**REQUEST CHANGES** — missing test coverage
"""
        assert extract_verdict(review_text) == "REQUEST CHANGES"


class TestExtractVerdictScriptMain:
    """Test the main() function when called via the script."""

    def test_main_with_approve_verdict(self, tmp_path) -> None:
        """Test main() returns 0 for APPROVE verdict."""
        temp_path = tmp_path / "review.md"
        temp_path.write_text("**APPROVE** — no issues")

        result = subprocess.run(
            [sys.executable or "python3", ".github/scripts/extract_verdict.py", str(temp_path), "TestProvider"],
            cwd=str(Path(__file__).parent.parent.parent),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "APPROVED" in result.stdout

    def test_main_with_request_changes_verdict(self, tmp_path) -> None:
        """Test main() returns 1 for REQUEST CHANGES verdict."""
        temp_path = tmp_path / "review.md"
        temp_path.write_text("**REQUEST CHANGES** — blocking issue")

        result = subprocess.run(
            [sys.executable or "python3", ".github/scripts/extract_verdict.py", str(temp_path), "TestProvider"],
            cwd=str(Path(__file__).parent.parent.parent),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "CHANGES REQUESTED" in result.stdout

    def test_main_with_backtick_approve(self, tmp_path) -> None:
        """Test main() returns 0 for backtick-wrapped APPROVE."""
        temp_path = tmp_path / "review.md"
        temp_path.write_text("**`APPROVE`** — no issues")

        result = subprocess.run(
            [sys.executable or "python3", ".github/scripts/extract_verdict.py", str(temp_path), "DeepSeek"],
            cwd=str(Path(__file__).parent.parent.parent),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "APPROVED" in result.stdout

    def test_main_with_no_verdict(self, tmp_path) -> None:
        """Test main() returns 1 when no valid verdict found."""
        temp_path = tmp_path / "review.md"
        temp_path.write_text("Some review text without a verdict line.")

        result = subprocess.run(
            [sys.executable or "python3", ".github/scripts/extract_verdict.py", str(temp_path), "TestProvider"],
            cwd=str(Path(__file__).parent.parent.parent),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "did not include a valid verdict" in result.stderr


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
