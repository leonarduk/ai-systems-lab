"""Orchestrator: schedules runs, drives the MCP tool contract from issue #22
(`search_draws`, `parse_entry_page`, `submit_entry`, `check_log`), and uses
the configured `LLMProvider` (see `llm_providers.py`) for parsing, filtering,
eligibility reasoning, and simple tie-breaker answers.

The MCP tool calls and the LLM calls are both injected (`MCPToolClient`,
`LLMProvider`), so `run_once` has no network or subprocess dependency of its
own and is fully unit-testable with fakes (see tests/).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from llm_providers import LLMProvider, LLMProviderError
from mcp_client import MCPToolClient, MCPToolError

logger = logging.getLogger("prize_draw_orchestrator")

_EXTRACTION_PROMPT_TEMPLATE = """You are helping a person decide whether to enter a prize draw
competition, based on their configured criteria. Read the competition page content below and
respond with ONLY a JSON object (no prose, no markdown fences) with exactly these keys:

- "prize": short description of the prize
- "closing_date": ISO 8601 date the competition closes, or null if not stated
- "entry_requirements": short description of what entering requires
- "entry_url": the URL to enter, or null if not found
- "requires_purchase": true if entry requires a purchase ("no purchase necessary" -> false)
- "has_complex_tie_breaker": true only if there's a free-text tie-breaker question that needs
  creative/subjective writing (a simple factual or yes/no question is NOT complex)
- "tie_breaker_answer": if there's a simple, objectively-answerable tie-breaker question, your
  best short answer to it; otherwise null
- "eligible": true if the competition satisfies ALL of the caller's criteria below and does not
  require a purchase; false otherwise
- "reason": one short sentence explaining the eligible/false decision

Caller's criteria:
{criteria}

Competition page content:
{content}
"""

_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "prize": {"type": "string"},
        "closing_date": {"type": ["string", "null"]},
        "entry_requirements": {"type": "string"},
        "entry_url": {"type": ["string", "null"]},
        "requires_purchase": {"type": "boolean"},
        "has_complex_tie_breaker": {"type": "boolean"},
        "tie_breaker_answer": {"type": ["string", "null"]},
        "eligible": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": [
        "prize",
        "eligible",
        "requires_purchase",
        "has_complex_tie_breaker",
        "reason",
    ],
}


@dataclass
class RunSummary:
    """Outcome of one orchestrator pass, used to build the end-of-run notification."""

    found: list[dict[str, Any]] = field(default_factory=list)
    entered: list[dict[str, Any]] = field(default_factory=list)
    needs_review: list[dict[str, Any]] = field(default_factory=list)
    skipped_duplicates: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def as_text(self) -> str:
        """Render a human-readable run summary suitable for a log line or chat message."""
        lines = [
            f"Prize draw run summary: {len(self.found)} candidate(s) found, "
            f"{len(self.entered)} entered, {len(self.needs_review)} need review, "
            f"{len(self.skipped_duplicates)} duplicate(s) skipped, {len(self.errors)} error(s).",
        ]
        for draw in self.entered:
            lines.append(
                f"  ENTERED   {draw['draw_id']}: {draw.get('prize', '(unknown prize)')}"
            )
        for draw in self.needs_review:
            lines.append(
                f"  REVIEW    {draw['draw_id']}: {draw.get('reason', '(needs manual follow-up)')}"
            )
        for err in self.errors:
            lines.append(f"  ERROR     {err['draw_id']}: {err['error']}")
        return "\n".join(lines)


def _requires_personal_data(entry_requirements: str) -> bool:
    """Heuristic: does entering plausibly require personal/financial data?

    Errs on the side of caution — the caller must always pass an explicit
    `confirm_personal_data` opt-in before such a draw can be entered
    (see Config.confirm_personal_data), so a false positive here only ever
    costs a flag-for-review, never an unwanted submission.
    """
    keywords = (
        "address",
        "phone",
        "card",
        "payment",
        "bank",
        "national insurance",
        "date of birth",
        "postcode",
    )
    text = entry_requirements.lower()
    return any(keyword in text for keyword in keywords)


def check_duplicate(mcp_client: MCPToolClient, draw_id: str) -> bool:
    """Return True if `draw_id` has already been entered/recorded (per `check_log`)."""
    result = mcp_client.call_tool("check_log", {"query": {"draw_id": draw_id}})
    return bool(result.get("seen") or result.get("entries"))


def extract_and_classify(
    llm: LLMProvider, criteria: dict[str, Any], page_content: str
) -> dict[str, Any]:
    """Ask the configured LLM to normalize competition details and classify eligibility."""
    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(criteria=criteria, content=page_content)
    return llm.generate_json(prompt, schema=_EXTRACTION_SCHEMA)


def process_candidate(
    mcp_client: MCPToolClient,
    llm: LLMProvider,
    criteria: dict[str, Any],
    candidate: dict[str, Any],
    dry_run: bool,
    confirm_personal_data: bool,
) -> tuple[str, dict[str, Any]]:
    """Process one candidate draw end-to-end.

    Returns (`outcome`, `details`) where `outcome` is one of "duplicate",
    "entered", "needs_review", or "error".
    """
    draw_id = candidate["draw_id"]

    if check_duplicate(mcp_client, draw_id):
        return "duplicate", {"draw_id": draw_id}

    page = mcp_client.call_tool(
        "parse_entry_page", {"draw_id": draw_id, "url": candidate.get("url")}
    )
    parsed = extract_and_classify(llm, criteria, page.get("content", ""))
    parsed["draw_id"] = draw_id

    if parsed.get("requires_purchase"):
        parsed["reason"] = (
            parsed.get("reason") or "Requires a purchase; skipped for human review."
        )
        return "needs_review", parsed

    if parsed.get("has_complex_tie_breaker"):
        parsed["reason"] = (
            parsed.get("reason") or "Complex/creative tie-breaker needs a human answer."
        )
        return "needs_review", parsed

    if not parsed.get("eligible"):
        parsed["reason"] = parsed.get("reason") or "Does not meet configured criteria."
        return "needs_review", parsed

    entry_requirements = parsed["entry_requirements"]
    if _requires_personal_data(entry_requirements) and not confirm_personal_data:
        parsed["reason"] = (
            "Entry requires personal/financial data; set CONFIRM_PERSONAL_DATA=true "
            "after reviewing this draw to allow automatic entry."
        )
        return "needs_review", parsed

    submit_fields = {
        "entry_url": parsed.get("entry_url") or candidate.get("url"),
        "tie_breaker_answer": parsed.get("tie_breaker_answer"),
    }
    result = mcp_client.call_tool(
        "submit_entry",
        {
            "draw_id": draw_id,
            "fields": submit_fields,
            "confirm_personal_data": confirm_personal_data,
            "dry_run": dry_run,
        },
    )
    if not dry_run:
        mcp_client.call_tool(
            "check_log",
            {
                "record": {
                    "draw_id": draw_id,
                    "status": result.get("status", "entered"),
                    "prize": parsed.get("prize"),
                }
            },
        )
    parsed["submit_result"] = result
    return "entered", parsed


def run_once(
    mcp_client: MCPToolClient,
    llm: LLMProvider,
    criteria: dict[str, Any],
    dry_run: bool = True,
    confirm_personal_data: bool = False,
) -> RunSummary:
    """Run one full pass: search, evaluate, and (dry-run or real) enter eligible draws."""
    summary = RunSummary()

    search_result = mcp_client.call_tool("search_draws", {"criteria": criteria})
    candidates = search_result.get("draws", [])
    summary.found = candidates
    logger.info(
        "Found %d candidate draw(s) matching configured criteria", len(candidates)
    )

    for candidate in candidates:
        draw_id = candidate.get("draw_id", "(unknown)")
        try:
            outcome, details = process_candidate(
                mcp_client,
                llm,
                criteria,
                candidate,
                dry_run=dry_run,
                confirm_personal_data=confirm_personal_data,
            )
        except (MCPToolError, LLMProviderError) as exc:
            logger.error("Error processing draw %s: %s", draw_id, exc)
            summary.errors.append({"draw_id": draw_id, "error": str(exc)})
            continue
        except (
            Exception
        ) as exc:  # noqa: BLE001 - one bad candidate must not abort the run
            logger.exception("Unexpected error processing draw %s", draw_id)
            summary.errors.append({"draw_id": draw_id, "error": str(exc)})
            continue

        if outcome == "duplicate":
            summary.skipped_duplicates.append(details)
        elif outcome == "entered":
            summary.entered.append(details)
        elif outcome == "needs_review":
            summary.needs_review.append(details)

    logger.info("%s", summary.as_text())
    return summary


def run_forever(
    mcp_client: MCPToolClient,
    llm: LLMProvider,
    criteria: dict[str, Any],
    interval_minutes: int,
    dry_run: bool = True,
    confirm_personal_data: bool = False,
    sleep_fn=time.sleep,
    max_iterations: int | None = None,
) -> None:
    """Run `run_once` on a fixed interval, logging (and swallowing) per-iteration errors
    so a single bad run never kills the scheduling loop. `max_iterations` is exposed only
    for tests; production callers omit it and run indefinitely."""
    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        try:
            run_once(
                mcp_client,
                llm,
                criteria,
                dry_run=dry_run,
                confirm_personal_data=confirm_personal_data,
            )
        except (
            Exception
        ):  # noqa: BLE001 - a scheduling loop must never die on one bad run
            logger.exception(
                "Unhandled error during scheduled run; will retry next interval"
            )
        iterations += 1
        if max_iterations is None or iterations < max_iterations:
            sleep_fn(interval_minutes * 60)
