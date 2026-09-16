import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fakes import FakeLLMProvider, FakeMCPToolClient

from orchestrator import (
    _EXTRACTION_SCHEMA,
    _NULLABLE_EXTRACTION_KEYS,
    check_duplicate,
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


class TestExtractionSchemaNullability:
    def test_nullable_keys_are_nullable_in_schema(self):
        for key in _NULLABLE_EXTRACTION_KEYS:
            prop = _EXTRACTION_SCHEMA["properties"][key]
            assert "null" in prop["type"], (
                f"{key} must be nullable in _EXTRACTION_SCHEMA to match "
                f"_NULLABLE_EXTRACTION_KEYS"
            )

    def test_nullable_keys_remain_required(self):
        for key in _NULLABLE_EXTRACTION_KEYS:
            assert key in _EXTRACTION_SCHEMA["required"]

    def test_non_nullable_fields_keep_single_string_type(self):
        for key, prop in _EXTRACTION_SCHEMA["properties"].items():
            if key in _NULLABLE_EXTRACTION_KEYS:
                continue
            assert prop["type"] != ["string", "null"]


class TestCheckDuplicate:
    def test_returns_false_for_unseen_draw(self):
        client = FakeMCPToolClient(already_logged=set())
        assert check_duplicate(client, "draw-1") is False

    def test_returns_true_for_seen_draw(self):
        client = FakeMCPToolClient(already_logged={"draw-1"})
        assert check_duplicate(client, "draw-1") is True


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
