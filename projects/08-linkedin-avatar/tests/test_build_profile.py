"""Tests for build/build_profile.py. No real personal data — all fixtures are synthetic."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from build import build_profile  # noqa: E402


class FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class FakePdfReader:
    def __init__(self, path):
        self.path = path
        self.pages = [FakePage("Page one text"), FakePage("Page two text")]


class TestExtractText:
    def test_joins_pages(self, monkeypatch):
        monkeypatch.setattr(build_profile, "PdfReader", FakePdfReader)
        text = build_profile.extract_text("fake.pdf")
        assert "Page one text" in text
        assert "Page two text" in text

    def test_handles_none_from_extract_text(self, monkeypatch):
        class NonePage:
            def extract_text(self):
                return None

        class ReaderWithNonePage:
            def __init__(self, path):
                self.pages = [NonePage()]

        monkeypatch.setattr(build_profile, "PdfReader", ReaderWithNonePage)
        text = build_profile.extract_text("fake.pdf")
        assert text == ""


class TestRedactEmail:
    def test_redacts_email(self):
        result = build_profile.redact("Contact me at jane.doe@example.com please")
        assert "jane.doe@example.com" not in result
        assert build_profile.REDACTED in result


class TestRedactLinkedInUrl(object):
    def test_redacts_profile_url(self):
        result = build_profile.redact("linkedin.com/in/janedoe123")
        assert "linkedin.com/in/janedoe123" not in result
        assert build_profile.REDACTED in result

    def test_redacts_full_https_url(self):
        result = build_profile.redact("https://www.linkedin.com/in/janedoe123/")
        assert "linkedin.com" not in result


class TestRedactPhone:
    @pytest.mark.parametrize(
        "phone",
        [
            "+44 7911 123456",
            "07911 123456",
            "(020) 7946 0958",
            "+1 415 555 2671",
        ],
    )
    def test_redacts_real_looking_phone_numbers(self, phone):
        result = build_profile.redact(f"Call me on {phone} any time")
        assert phone not in result
        assert build_profile.REDACTED in result

    @pytest.mark.parametrize(
        "near_miss",
        [
            "Python 3.11 and 3.12",
            "v2.0.1",
            "released in 2026",
            "Q3 2026",
            "01.2020 - 12.2023",
            "Jan 2020 - Dec 2023",
            "07/2020 - 08/2023",
            "2020 - 2023",
        ],
    )
    def test_does_not_redact_near_misses(self, near_miss):
        result = build_profile.redact(near_miss)
        assert result == near_miss


class TestRedactPostcode:
    def test_redacts_uk_postcode(self):
        result = build_profile.redact("Based near SW1A 1AA in London")
        assert "SW1A 1AA" not in result
        assert build_profile.REDACTED in result


class TestRedactStreetAddress:
    def test_redacts_full_street_line(self):
        result = build_profile.redact("123 Example Street, London")
        assert result.strip() == build_profile.REDACTED

    def test_does_not_redact_job_title_or_employer_lines(self):
        for line in [
            "Senior Software Engineer",
            "Acme Corporation",
            "Built a multi-agent coder for GitHub issues",
            "10 years of experience in distributed systems",
        ]:
            assert build_profile.redact(line) == line


class TestNormalize:
    def test_converts_known_headings_to_markdown(self):
        text = "Experience\nSome role\nEducation\nSome degree"
        result = build_profile.normalize(text)
        assert "## Experience" in result
        assert "## Education" in result

    def test_drops_page_furniture(self):
        text = "Some content\nPage 1 of 4\nMore content"
        result = build_profile.normalize(text)
        assert "Page 1 of 4" not in result

    def test_collapses_repeated_blank_lines(self):
        text = "A\n\n\n\n\nB"
        result = build_profile.normalize(text)
        assert "\n\n\n" not in result

    def test_heading_match_is_case_insensitive(self):
        result = build_profile.normalize("experience")
        assert result.strip() == "## experience"


class TestBuildProfile:
    def test_writes_redacted_normalized_markdown(self, tmp_path, monkeypatch):
        monkeypatch.setattr(build_profile, "PdfReader", FakePdfReader)
        out_path = tmp_path / "profile.md"

        build_profile.build_profile(Path("fake.pdf"), out_path)

        content = out_path.read_text(encoding="utf-8")
        assert "Page one text" in content

    def test_deterministic_across_two_runs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(build_profile, "PdfReader", FakePdfReader)
        out_path = tmp_path / "profile.md"

        build_profile.build_profile(Path("fake.pdf"), out_path)
        first = out_path.read_text(encoding="utf-8")

        build_profile.build_profile(Path("fake.pdf"), out_path)
        second = out_path.read_text(encoding="utf-8")

        assert first == second


class TestFindContactLeaks:
    def test_finds_planted_email(self):
        leaks = build_profile.find_contact_leaks("## Contact\njane.doe@example.com\n")
        assert any(pattern == "email" for _, pattern, _ in leaks)

    def test_finds_planted_phone(self):
        leaks = build_profile.find_contact_leaks("## Contact\n+44 7911 123456\n")
        assert any(pattern == "phone" for _, pattern, _ in leaks)

    def test_finds_planted_postcode(self):
        leaks = build_profile.find_contact_leaks("## Location\nSW1A 1AA\n")
        assert any(pattern == "postcode" for _, pattern, _ in leaks)

    def test_finds_planted_street_address(self):
        leaks = build_profile.find_contact_leaks("## Address\n123 Example Street\n")
        assert any(pattern == "street_address" for _, pattern, _ in leaks)

    def test_clean_file_has_no_leaks(self):
        text = "## Experience\nSenior Software Engineer\nAcme Corporation\n"
        assert build_profile.find_contact_leaks(text) == []


class TestOutPathIsSafe:
    def test_parent_of_knowledge_dir_is_rejected(self, tmp_path, monkeypatch):
        knowledge = tmp_path / "knowledge"
        knowledge.mkdir()
        monkeypatch.setattr(build_profile, "KNOWLEDGE_DIR", knowledge)

        # A parent of knowledge/ (e.g. the repo root) must not be considered safe.
        assert build_profile._out_path_is_safe(tmp_path / "profile.md") is False

    def test_path_inside_knowledge_dir_is_accepted(self, tmp_path, monkeypatch):
        knowledge = tmp_path / "knowledge"
        knowledge.mkdir()
        monkeypatch.setattr(build_profile, "KNOWLEDGE_DIR", knowledge)

        assert build_profile._out_path_is_safe(knowledge / "profile.md") is True


class TestCli:
    def test_check_exits_zero_on_clean_file(self, tmp_path, capsys):
        clean = tmp_path / "clean.md"
        clean.write_text("## Experience\nSenior Software Engineer\n", encoding="utf-8")

        exit_code = build_profile.main(["--check", str(clean)])

        assert exit_code == 0

    def test_check_exits_one_on_planted_email(self, tmp_path, capsys):
        dirty = tmp_path / "dirty.md"
        dirty.write_text("## Contact\njane.doe@example.com\n", encoding="utf-8")

        exit_code = build_profile.main(["--check", str(dirty)])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "email" in captured.err

    def test_check_exits_one_on_planted_phone(self, tmp_path, capsys):
        dirty = tmp_path / "dirty.md"
        dirty.write_text("## Contact\n+44 7911 123456\n", encoding="utf-8")

        exit_code = build_profile.main(["--check", str(dirty)])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "phone" in captured.err

    def test_check_exits_one_on_planted_postcode(self, tmp_path, capsys):
        dirty = tmp_path / "dirty.md"
        dirty.write_text("## Location\nSW1A 1AA\n", encoding="utf-8")

        exit_code = build_profile.main(["--check", str(dirty)])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "postcode" in captured.err

    def test_check_exits_one_on_planted_street_address(self, tmp_path, capsys):
        dirty = tmp_path / "dirty.md"
        dirty.write_text("## Address\n123 Example Street\n", encoding="utf-8")

        exit_code = build_profile.main(["--check", str(dirty)])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "street_address" in captured.err

    def test_pdf_required_without_check(self):
        with pytest.raises(SystemExit):
            build_profile.main([])

    def test_out_path_outside_knowledge_dir_is_rejected(self, monkeypatch):
        monkeypatch.setattr(build_profile, "PdfReader", FakePdfReader)
        with pytest.raises(SystemExit):
            build_profile.main(
                ["--pdf", "fake.pdf", "--out", "/tmp/outside/profile.md"]
            )

    def test_default_out_path_is_accepted(self, monkeypatch, tmp_path):
        monkeypatch.setattr(build_profile, "PdfReader", FakePdfReader)
        monkeypatch.setattr(build_profile, "KNOWLEDGE_DIR", tmp_path)

        exit_code = build_profile.main(
            ["--pdf", "fake.pdf", "--out", str(tmp_path / "profile.md")]
        )

        assert exit_code == 0
        assert (tmp_path / "profile.md").exists()
