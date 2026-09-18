import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fakes import FakeLLMProvider, FakeMCPToolClient

import pytest

import orchestrator

from orchestrator import (
    _validate_extraction,
    check_duplicate,
    extract_and_classify,
    process_candidate,
    run_once,
)

CRITERIA = {
    "prize_types": ["cash"],
    "min_prize_value": 50,
    "regions": ["UK"],
    "max_days_to_closing": 30,
}


def make_candidate(draw_id="draw-1", url="https://example.com/draw-1"):
    return {"draw_id": draw_id, "url": url, "title": "Win cash!"}


class TestCheckDuplicate:
    def test_returns_false_for_unseen_draw(self):
        client = FakeMCPToolClient(already_logged=set())
        assert check_duplicate(client, "draw-1") is False

    def test_returns_true_for_seen_draw(self):
        client = FakeMCPToolClient(already_logged={"draw-1"})
        assert check_duplicate(client, "draw-1") is True


_ELIGIBLE_RESPONSE_WITH_NULL_ENTRY_URL = {
    "prize": "GBP 100 cash",
    "closing_date": "2026-08-15",
    "entry_requirements": "Fill in the web form",
    "entry_url": None,
    "requires_purchase": False,
    "has_complex_tie_breaker": False,
    "tie_breaker_answer": None,
    "eligible": True,
    "reason": "Matches all criteria",
}


class TestProcessCandidate:
    def test_duplicate_is_skipped_before_parsing(self):
        client = FakeMCPToolClient(already_logged={"draw-1"})
        llm = FakeLLMProvider()
        outcome, details = process_candidate(
            client,
            llm,
            CRITERIA,
            make_candidate(),
            dry_run=True,
            confirm_personal_data=False,
        )
        assert outcome == "duplicate"
        assert (
            "parse_entry_page",
            {"draw_id": "draw-1", "url": "https://example.com/draw-1"},
        ) not in client.calls

    def test_eligible_draw_is_entered_and_logged(self):
        client = FakeMCPToolClient(
            pages={"draw-1": {"content": "Win 100 pounds cash, no purchase necessary"}}
        )
        llm = FakeLLMProvider(
            fixed_response={
                "prize": "GBP 100 cash",
                "closing_date": "2026-08-15",
                "entry_requirements": "Fill in the web form",
                "entry_url": "https://example.com/enter",
                "requires_purchase": False,
                "has_complex_tie_breaker": False,
                "tie_breaker_answer": None,
                "eligible": True,
                "reason": "Matches all criteria",
            }
        )
        outcome, details = process_candidate(
            client,
            llm,
            CRITERIA,
            make_candidate(),
            dry_run=True,
            confirm_personal_data=False,
        )
        assert outcome == "entered"
        assert client.submitted == [
            {
                "draw_id": "draw-1",
                "fields": {
                    "entry_url": "https://example.com/enter",
                    "tie_breaker_answer": None,
                },
                "confirm_personal_data": False,
                "dry_run": True,
            }
        ]
        # Dry runs must not write a check_log record: doing so would mark the
        # draw "seen" and cause check_duplicate to skip it once the operator
        # goes live, defeating the dry-run-then-go-live workflow.
        assert client.records == []

    def test_requires_purchase_is_flagged_for_review_not_entered(self):
        client = FakeMCPToolClient(
            pages={"draw-1": {"content": "Buy a ticket to enter"}}
        )
        llm = FakeLLMProvider(
            fixed_response={
                "prize": "Car",
                "eligible": True,
                "requires_purchase": True,
                "has_complex_tie_breaker": False,
                "entry_requirements": "",
                "reason": "",
            }
        )
        outcome, details = process_candidate(
            client,
            llm,
            CRITERIA,
            make_candidate(),
            dry_run=True,
            confirm_personal_data=False,
        )
        assert outcome == "needs_review"
        assert client.submitted == []

    def test_complex_tie_breaker_is_flagged_for_review(self):
        client = FakeMCPToolClient(
            pages={"draw-1": {"content": "Tell us in 50 words why you deserve to win"}}
        )
        llm = FakeLLMProvider(
            fixed_response={
                "prize": "Holiday",
                "eligible": True,
                "requires_purchase": False,
                "has_complex_tie_breaker": True,
                "entry_requirements": "",
                "reason": "",
            }
        )
        outcome, _ = process_candidate(
            client,
            llm,
            CRITERIA,
            make_candidate(),
            dry_run=True,
            confirm_personal_data=False,
        )
        assert outcome == "needs_review"
        assert client.submitted == []

    def test_ineligible_draw_is_flagged_not_entered(self):
        client = FakeMCPToolClient(pages={"draw-1": {"content": "US residents only"}})
        llm = FakeLLMProvider(
            fixed_response={
                "prize": "Gadget",
                "eligible": False,
                "requires_purchase": False,
                "has_complex_tie_breaker": False,
                "entry_requirements": "",
                "reason": "US residents only, caller requires UK",
            }
        )
        outcome, details = process_candidate(
            client,
            llm,
            CRITERIA,
            make_candidate(),
            dry_run=True,
            confirm_personal_data=False,
        )
        assert outcome == "needs_review"
        assert details["reason"] == "US residents only, caller requires UK"

    def test_personal_data_requirement_blocks_entry_without_confirmation(self):
        client = FakeMCPToolClient(
            pages={"draw-1": {"content": "Enter your bank details to claim"}}
        )
        llm = FakeLLMProvider(
            fixed_response={
                "prize": "Cash",
                "eligible": True,
                "requires_purchase": False,
                "has_complex_tie_breaker": False,
                "entry_requirements": "Provide your bank account and postcode",
                "reason": "",
            }
        )
        outcome, details = process_candidate(
            client,
            llm,
            CRITERIA,
            make_candidate(),
            dry_run=True,
            confirm_personal_data=False,
        )
        assert outcome == "needs_review"
        assert client.submitted == []
        assert "personal" in details["reason"].lower()

    def test_null_entry_url_falls_back_to_candidate_url(self):
        # Regression test: `entry_url` is nullable in `_NULLABLE_EXTRACTION_KEYS`,
        # so a `null` value from the LLM is a valid, expected input. The
        # `parsed.get("entry_url") or candidate.get("url")` fallback in
        # `process_candidate` must populate `entry_url` from the candidate's
        # `url` rather than letting `None` propagate downstream.
        client = FakeMCPToolClient(
            pages={"draw-1": {"content": "Win 100 pounds cash, no purchase necessary"}}
        )
        llm = FakeLLMProvider(fixed_response=_ELIGIBLE_RESPONSE_WITH_NULL_ENTRY_URL)
        outcome, details = process_candidate(
            client,
            llm,
            CRITERIA,
            make_candidate(url="https://example.com/draw-1"),
            dry_run=True,
            confirm_personal_data=False,
        )
        assert outcome == "entered"
        # process_candidate applies the fallback where it builds submit_fields
        # (orchestrator.py: `parsed.get("entry_url") or candidate.get("url")`),
        # not by writing back into the response it returns — so the submission
        # is where a null entry_url has to be resolved, and details["entry_url"]
        # legitimately stays None. That is the only read of entry_url in the
        # project, so there is no second consumer to guard. Issue #233's
        # acceptance criteria name the returned candidate instead; the
        # discrepancy is recorded on the issue.
        #
        # Assert only that field: pinning the whole payload would make this
        # break on unrelated changes to the submission shape.
        assert len(client.submitted) == 1
        assert (
            client.submitted[0]["fields"]["entry_url"] == "https://example.com/draw-1"
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Issue #233's acceptance criteria assume process_candidate writes the "
            "resolved entry_url back into the response it returns. It does not — "
            "the fallback is applied only where submit_fields is built "
            "(orchestrator.py:276). Recorded as issue #543; this xfail is the "
            "executable form of that discrepancy and will start failing loudly "
            "if write-back is ever added."
        ),
    )
    def test_returned_candidate_carries_resolved_entry_url(self):
        client = FakeMCPToolClient(
            pages={"draw-1": {"content": "Win 100 pounds cash, no purchase necessary"}}
        )
        llm = FakeLLMProvider(fixed_response=_ELIGIBLE_RESPONSE_WITH_NULL_ENTRY_URL)

        _, details = process_candidate(
            client,
            llm,
            CRITERIA,
            make_candidate(url="https://example.com/draw-1"),
            dry_run=True,
            confirm_personal_data=False,
        )

        assert details["entry_url"] == "https://example.com/draw-1"

    def test_null_entry_url_with_no_candidate_url_stays_null(self):
        # The fallback is a plain `or`, so with nothing to fall back to the
        # submission carries None rather than inventing a URL.
        client = FakeMCPToolClient(
            pages={"draw-1": {"content": "Win 100 pounds cash, no purchase necessary"}}
        )
        llm = FakeLLMProvider(fixed_response=_ELIGIBLE_RESPONSE_WITH_NULL_ENTRY_URL)

        process_candidate(
            client,
            llm,
            CRITERIA,
            make_candidate(url=None),
            dry_run=True,
            confirm_personal_data=False,
        )

        assert client.submitted[0]["fields"]["entry_url"] is None

    def test_personal_data_requirement_allows_entry_with_explicit_confirmation(self):
        client = FakeMCPToolClient(
            pages={"draw-1": {"content": "Enter your bank details to claim"}}
        )
        llm = FakeLLMProvider(
            fixed_response={
                "prize": "Cash",
                "eligible": True,
                "requires_purchase": False,
                "has_complex_tie_breaker": False,
                "entry_requirements": "Provide your bank account and postcode",
                "entry_url": "https://example.com/enter",
                "reason": "",
            }
        )
        outcome, _ = process_candidate(
            client,
            llm,
            CRITERIA,
            make_candidate(),
            dry_run=True,
            confirm_personal_data=True,
        )
        assert outcome == "entered"
        assert client.submitted[0]["confirm_personal_data"] is True


class TestValidateExtraction:
    def _valid_payload(self, **overrides):
        payload = {
            "prize": "GBP 100 cash",
            "closing_date": "2026-08-15",
            "entry_requirements": "Fill in the web form",
            "entry_url": "https://example.com/enter",
            "requires_purchase": False,
            "has_complex_tie_breaker": False,
            "tie_breaker_answer": None,
            "eligible": True,
            "reason": "Matches all criteria",
        }
        payload.update(overrides)
        return payload

    def test_accepts_valid_payload(self):
        payload = self._valid_payload()
        assert _validate_extraction(payload) is payload

    def test_accepts_null_for_nullable_keys(self):
        payload = self._valid_payload(
            closing_date=None, entry_url=None, tie_breaker_answer=None
        )
        assert _validate_extraction(payload) is payload

    def test_rejects_non_object(self):
        with pytest.raises(ValueError, match="must be a JSON object"):
            _validate_extraction(["not", "an", "object"])

    def test_rejects_missing_required_key(self):
        payload = self._valid_payload()
        del payload["entry_requirements"]
        with pytest.raises(
            ValueError, match="missing required key 'entry_requirements'"
        ):
            _validate_extraction(payload)

    def test_rejects_wrong_type_for_string_field(self):
        payload = self._valid_payload(entry_requirements=123)
        with pytest.raises(ValueError, match="key 'entry_requirements'.*expected str"):
            _validate_extraction(payload)

    def test_rejects_wrong_type_for_boolean_field(self):
        payload = self._valid_payload(eligible="yes")
        with pytest.raises(ValueError, match="key 'eligible'.*expected bool"):
            _validate_extraction(payload)

    def test_rejects_null_for_non_nullable_key(self):
        payload = self._valid_payload(entry_requirements=None)
        with pytest.raises(
            ValueError, match="key 'entry_requirements' must not be null"
        ):
            _validate_extraction(payload)

    def test_rejects_bool_in_string_field(self):
        # bool is a subclass of int; make sure we don't accidentally accept it.
        payload = self._valid_payload(prize=True)
        with pytest.raises(ValueError, match="key 'prize'.*expected str"):
            _validate_extraction(payload)


class TestValidationContractMatchesSchema:
    """The Python validator is derived from `_EXTRACTION_SCHEMA`, not restated.

    A hand-maintained copy drifted from the schema once already: it required
    `closing_date` and `entry_url`, which the schema treats as optional and
    which no caller reads, so valid responses were rejected.
    """

    def test_required_keys_match_schema(self):
        assert orchestrator._REQUIRED_EXTRACTION_KEYS == frozenset(
            orchestrator._EXTRACTION_SCHEMA["required"]
        )

    def test_every_schema_property_is_validated(self):
        assert set(orchestrator._EXTRACTION_FIELD_TYPES) == set(
            orchestrator._EXTRACTION_SCHEMA["properties"]
        )

    def test_nullable_keys_are_the_ones_the_schema_allows_null(self):
        expected = {
            key
            for key, prop in orchestrator._EXTRACTION_SCHEMA["properties"].items()
            if "null"
            in (prop["type"] if isinstance(prop["type"], list) else [prop["type"]])
        }
        assert orchestrator._NULLABLE_EXTRACTION_KEYS == expected

    def test_optional_key_may_be_omitted(self):
        # closing_date is a schema property but not required, so a response
        # without it must validate.
        payload = {
            "prize": "Cash",
            "entry_requirements": "",
            "eligible": True,
            "requires_purchase": False,
            "has_complex_tie_breaker": False,
            "reason": "",
        }
        assert _validate_extraction(payload) is payload


class TestExtractAndClassifyValidation:
    def test_invalid_llm_response_raises_value_error(self):
        llm = FakeLLMProvider(
            fixed_response={
                "prize": "Cash",
                "eligible": True,
                "requires_purchase": False,
                "has_complex_tie_breaker": False,
                "reason": "",
                # entry_requirements and entry_url deliberately omitted
            }
        )
        with pytest.raises(ValueError, match="missing required key"):
            extract_and_classify(llm, CRITERIA, "some page content")

    def test_invalid_llm_response_surfaces_as_error_in_run_once(self):
        client = FakeMCPToolClient(
            draws=[make_candidate("draw-1")],
            pages={"draw-1": {"content": "x"}},
        )
        llm = FakeLLMProvider(
            fixed_response={
                "prize": "Cash",
                "eligible": True,
                "requires_purchase": False,
                "has_complex_tie_breaker": False,
                "reason": "",
                # entry_requirements and entry_url deliberately omitted
            }
        )
        summary = run_once(
            client, llm, CRITERIA, dry_run=True, confirm_personal_data=False
        )
        assert len(summary.errors) == 1
        assert summary.errors[0]["draw_id"] == "draw-1"
        assert "missing required key" in summary.errors[0]["error"]
        assert summary.entered == []


class TestRunOnce:
    def test_summarizes_multiple_candidates(self):
        client = FakeMCPToolClient(
            draws=[
                make_candidate("draw-1"),
                make_candidate("draw-2"),
                make_candidate("draw-3"),
            ],
            pages={
                "draw-1": {"content": "eligible cash draw"},
                "draw-2": {"content": "requires purchase"},
            },
            already_logged={"draw-3"},
        )
        llm = FakeLLMProvider(
            responses=[
                {
                    "prize": "Cash",
                    "eligible": True,
                    "requires_purchase": False,
                    "has_complex_tie_breaker": False,
                    "entry_requirements": "",
                    "entry_url": "https://example.com/1",
                    "reason": "",
                },
                {
                    "prize": "TV",
                    "eligible": True,
                    "requires_purchase": True,
                    "has_complex_tie_breaker": False,
                    "entry_requirements": "",
                    "reason": "",
                },
            ]
        )

        summary = run_once(
            client, llm, CRITERIA, dry_run=True, confirm_personal_data=False
        )

        assert len(summary.found) == 3
        assert len(summary.entered) == 1
        assert len(summary.needs_review) == 1
        assert len(summary.skipped_duplicates) == 1
        assert "3 candidate(s) found" in summary.as_text()

    def test_llm_error_on_one_candidate_is_recorded_and_others_still_processed(self):
        from llm_providers import LLMProviderError

        class RaisingThenWorkingLLM:
            def __init__(self):
                self.calls = 0

            def generate_json(self, prompt, schema=None):
                self.calls += 1
                if self.calls == 1:
                    raise LLMProviderError("boom")
                return {
                    "prize": "Cash",
                    "eligible": True,
                    "requires_purchase": False,
                    "has_complex_tie_breaker": False,
                    "entry_requirements": "",
                    "entry_url": "https://example.com/2",
                    "reason": "",
                }

        client = FakeMCPToolClient(
            draws=[make_candidate("draw-1"), make_candidate("draw-2")],
            pages={"draw-1": {"content": "x"}, "draw-2": {"content": "y"}},
        )
        summary = run_once(
            client,
            RaisingThenWorkingLLM(),
            CRITERIA,
            dry_run=True,
            confirm_personal_data=False,
        )

        assert len(summary.errors) == 1
        assert summary.errors[0]["draw_id"] == "draw-1"
        assert len(summary.entered) == 1
        assert summary.entered[0]["draw_id"] == "draw-2"
