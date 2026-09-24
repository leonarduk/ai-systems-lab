"""Tests for scripts/validate_profile.py.

Loaded via importlib since the module lives in a non-package top-level
scripts/ directory (matches the pattern used for
scripts/test_check_profile_length.py and
scripts/test_validate_pr_description.py).
"""

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).parent / "validate_profile.py"
spec = importlib.util.spec_from_file_location("validate_profile", MODULE_PATH)
validate_profile = importlib.util.module_from_spec(spec)
sys.modules["validate_profile"] = validate_profile
spec.loader.exec_module(validate_profile)


class TestSplitSections:
    def test_splits_top_level_headings(self):
        text = "preamble\n## Summary\nsummary text\n## Experience\nexperience text\n"
        sections = validate_profile.split_sections(text)
        assert sections[""] == "preamble\n"
        assert sections["summary"] == "\nsummary text\n"
        assert sections["experience"] == "\nexperience text\n"

    def test_heading_names_are_lowercased(self):
        text = "## SUMMARY\ntext\n"
        sections = validate_profile.split_sections(text)
        assert "summary" in sections
        assert "SUMMARY" not in sections

    def test_no_headings_returns_whole_text_under_empty_key(self):
        text = "just plain text, no headings"
        sections = validate_profile.split_sections(text)
        assert sections == {"": text}


class TestFindTechnologies:
    def test_finds_simple_technology(self):
        assert "Python" in validate_profile.find_technologies("I write Python code.")

    def test_case_insensitive(self):
        assert "AWS" in validate_profile.find_technologies("Deployed on aws.")

    def test_spring_boot_not_double_counted_as_bare_spring(self):
        found = validate_profile.find_technologies("Built services with Spring Boot.")
        assert "Spring Boot" in found
        assert "Spring" not in found

    def test_bare_spring_is_found_when_not_followed_by_boot(self):
        found = validate_profile.find_technologies("Used the Spring framework.")
        assert "Spring" in found

    def test_no_false_positive_on_unrelated_text(self):
        assert (
            validate_profile.find_technologies("A regular sentence about cats.")
            == set()
        )


class TestHasProductionClaim:
    def test_detects_in_production(self):
        assert validate_profile.has_production_claim(
            "It has been running in production for years."
        )

    def test_detects_production_ready(self):
        assert validate_profile.has_production_claim(
            "Delivered a production-ready system."
        )

    def test_generic_verbs_alone_do_not_count(self):
        # Regression: the original pattern list included bare "built",
        # "delivered", "deployed" — so common in resume prose that the
        # production-claim check could never fire. Narrowed to
        # production-specific phrasing only.
        text = "Built and delivered several internal tools and deployed them."
        assert not validate_profile.has_production_claim(text)

    def test_no_match_on_unrelated_text(self):
        assert not validate_profile.has_production_claim("A regular sentence.")


class TestCheckTechnologiesInSummaryBackedByExperience:
    def test_no_mismatch_when_all_summary_tech_appears_in_experience(self):
        summary = "Experienced with Python and AWS."
        experience = "Built services in Python, deployed on AWS."
        assert (
            validate_profile.check_technologies_in_summary_backed_by_experience(
                summary, experience
            )
            == []
        )

    def test_flags_technology_only_in_summary(self):
        summary = "Experienced with Kubernetes."
        experience = "Built services in Python."
        mismatches = (
            validate_profile.check_technologies_in_summary_backed_by_experience(
                summary, experience
            )
        )
        assert len(mismatches) == 1
        assert mismatches[0].kind == "summary-not-in-experience"
        assert "Kubernetes" in mismatches[0].message


class TestCheckTechnologiesInExperienceReflectedInSummary:
    def test_warns_on_missing_headline_technology(self):
        summary = "A generalist engineer."
        experience = "Built systems in Java."
        warnings = (
            validate_profile.check_technologies_in_experience_reflected_in_summary(
                summary, experience
            )
        )
        assert len(warnings) == 1
        assert warnings[0].kind == "experience-not-in-summary"

    def test_no_warning_when_summary_already_mentions_it(self):
        summary = "Java engineer."
        experience = "Built systems in Java."
        assert (
            validate_profile.check_technologies_in_experience_reflected_in_summary(
                summary, experience
            )
            == []
        )

    def test_non_headline_technology_never_warns(self):
        summary = "A generalist engineer."
        experience = "Used Perl for scripting."
        assert (
            validate_profile.check_technologies_in_experience_reflected_in_summary(
                summary, experience
            )
            == []
        )


class TestCheckContradictions:
    def test_flags_exclusivity_claim_contradicted_by_experience(self):
        summary = "Java only developer."
        experience = "Also wrote Python tooling."
        mismatches = validate_profile.check_contradictions(summary, experience)
        assert len(mismatches) == 1
        assert mismatches[0].kind == "contradiction"

    def test_no_contradiction_when_experience_matches_exclusivity_claim(self):
        summary = "Java only developer."
        experience = "Wrote Java services."
        assert validate_profile.check_contradictions(summary, experience) == []

    def test_no_exclusivity_claim_no_contradiction(self):
        summary = "Polyglot developer."
        experience = "Wrote Java and Python."
        assert validate_profile.check_contradictions(summary, experience) == []


class TestCheckProductionClaims:
    def test_no_mismatch_when_experience_has_specific_production_language(self):
        summary = "I have production experience."
        experience = "Built a system that has been running in production for 3 years."
        assert validate_profile.check_production_claims(summary, experience) == []

    def test_flags_when_experience_only_has_generic_verbs(self):
        # The regression case: generic accomplishment verbs are not
        # sufficient evidence of production experience.
        summary = "I have deep production experience building systems."
        experience = "Built and delivered several internal tools and deployed them."
        mismatches = validate_profile.check_production_claims(summary, experience)
        assert len(mismatches) == 1
        assert mismatches[0].kind == "production-claim-unsupported"

    def test_no_claim_in_summary_means_no_check(self):
        summary = "A generalist engineer."
        experience = "Built and delivered several internal tools."
        assert validate_profile.check_production_claims(summary, experience) == []


class TestValidate:
    def test_clean_profile_passes_with_no_errors(self):
        text = (
            "## Summary\n"
            "Java engineer with production experience.\n"
            "## Experience\n"
            "Built and shipped Java services running in production.\n"
        )
        errors, warnings = validate_profile.validate(text)
        assert errors == []

    def test_missing_summary_section_is_an_error(self):
        text = "## Experience\nBuilt things.\n"
        errors, warnings = validate_profile.validate(text)
        assert any(e.kind == "missing-section" for e in errors)

    def test_missing_experience_section_is_an_error(self):
        text = "## Summary\nA developer.\n"
        errors, warnings = validate_profile.validate(text)
        assert any(e.kind == "missing-section" for e in errors)

    def test_unsupported_summary_technology_is_an_error(self):
        text = (
            "## Summary\nExperienced with Kubernetes.\n"
            "## Experience\nBuilt services in Python.\n"
        )
        errors, warnings = validate_profile.validate(text)
        assert any(e.kind == "summary-not-in-experience" for e in errors)


class TestMain:
    def test_exits_zero_on_clean_profile(self, tmp_path):
        profile = tmp_path / "profile.md"
        profile.write_text(
            "## Summary\nJava engineer with production experience.\n"
            "## Experience\nBuilt and shipped Java services running in production.\n",
            encoding="utf-8",
        )
        assert validate_profile.main([str(profile)]) == 0

    def test_exits_one_on_mismatch(self, tmp_path):
        profile = tmp_path / "profile.md"
        profile.write_text(
            "## Summary\nExperienced with Kubernetes.\n"
            "## Experience\nBuilt services in Python.\n",
            encoding="utf-8",
        )
        assert validate_profile.main([str(profile)]) == 1

    def test_missing_file_exits_two(self, tmp_path):
        assert validate_profile.main([str(tmp_path / "nope.md")]) == 2

    def test_strict_flag_fails_on_warnings_alone(self, tmp_path):
        profile = tmp_path / "profile.md"
        profile.write_text(
            "## Summary\nA generalist engineer.\n"
            "## Experience\nBuilt systems in Java.\n",
            encoding="utf-8",
        )
        # Non-strict: a warning-only run still exits 0.
        assert validate_profile.main([str(profile)]) == 0
        # Strict: the same warning-only run now fails.
        assert validate_profile.main([str(profile), "--strict"]) == 1

    def test_current_repo_profile_passes(self):
        # Guards against a future edit reintroducing the truncation this
        # test file was added alongside fixing — the real profile.md must
        # always pass its own validator.
        default_profile = (
            Path(__file__).resolve().parent.parent / validate_profile.DEFAULT_PROFILE
        )
        assert validate_profile.main([str(default_profile)]) == 0
