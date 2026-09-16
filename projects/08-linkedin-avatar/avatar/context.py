"""Assemble the system prompt from the three knowledge files.

Fixed order — role/identity, summary.txt, profile.md, GitHub index, rules —
so the prompt's stable prefix never moves and DeepSeek's automatic caching
has the best chance of a hit. Nothing here may vary between calls: no
timestamps, no request IDs, no randomised ordering. See docs/design.md §4.
"""

import json
import logging
import math
import os
from pathlib import Path

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"

DEFAULT_MAX_CONTEXT_TOKENS = 40000

# DeepSeek has no count-tokens endpoint and its offline tokenizer is a
# downloadable demo zip, not a pip package — not worth a project dependency
# for a build-time estimate. ~3 chars/token is a deliberately conservative
# read of DeepSeek's own ~0.3 tokens/English-character guidance: it
# overestimates, so the budget check errs toward trimming rather than
# quietly shipping an oversized prompt.
CHARS_PER_TOKEN_ESTIMATE = 3.0

ROLE_BLOCK = (
    "You are Steve Leonard's AI twin. A recruiter, hiring manager or fellow "
    "engineer is here to ask about his career and the projects in his GitHub — "
    "you read both. Say you don't know rather than guessing."
)

# The rules are prompt content, not code: they live in rules.md alongside this
# module so wording changes show up as pure content diffs. Loaded once at
# import time — the text is static, so there is nothing to gain from re-reading
# it per call, and a missing file should fail loudly at startup rather than
# silently ship a prompt with no rules in it.
RULES_PATH = Path(__file__).resolve().parent / "rules.md"
RULES_BLOCK = RULES_PATH.read_text(encoding="utf-8").strip()


class PromptTooLargeError(Exception):
    """The assembled prompt cannot fit AVATAR_MAX_CONTEXT_TOKENS even after
    demoting every repo record to an index line."""


def estimate_tokens(text):
    return math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE)


def _read_text_file(path):
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def _load_github_records(path):
    if not path.exists():
        logger.warning(
            "GitHub snapshot missing at %s; using a profile-only prompt", path
        )
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning(
            "GitHub snapshot at %s is not valid JSON; using a profile-only prompt", path
        )
        return []

    if not isinstance(data, list):
        logger.warning(
            "GitHub snapshot at %s is not a JSON list (got %s); using a profile-only prompt",
            path,
            type(data).__name__,
        )
        return []

    return data


def _format_index_line(record):
    return f"- {record['name']}: {record.get('description') or 'no description'}"


def _format_full_record(record):
    lines = [f"### {record['name']}", record.get("description") or ""]
    if record.get("topics"):
        lines.append("Topics: " + ", ".join(record["topics"]))
    if record.get("languages"):
        lines.append("Languages: " + ", ".join(record["languages"]))
    lines.append(f"Stars: {record.get('stars', 0)}")
    if record.get("curated_note"):
        lines.append(record["curated_note"])
    if record.get("readme_excerpt"):
        lines.append(record["readme_excerpt"])
    return "\n".join(line for line in lines if line)


def _github_section(records, section_budget):
    """Render the GitHub index, demoting oldest-pushed records to index
    lines until the section fits section_budget — or, if it still doesn't
    fit with every record demoted, doing the best it can. The overall
    budget is enforced once, on the whole assembled prompt, in
    build_system_prompt — so the error message it raises always reports
    the real configured budget rather than this section's slice of it."""
    if not records:
        return ""

    full_blocks = {r["name"]: _format_full_record(r) for r in records}
    index_lines = {r["name"]: _format_index_line(r) for r in records}
    full_tokens = {name: estimate_tokens(text) for name, text in full_blocks.items()}
    index_tokens = {name: estimate_tokens(text) for name, text in index_lines.items()}

    demoted = set()
    total = sum(full_tokens.values())

    oldest_first = sorted(records, key=lambda r: r.get("pushed_at") or "")
    for record in oldest_first:
        if total <= section_budget:
            break
        name = record["name"]
        total += index_tokens[name] - full_tokens[name]
        demoted.add(name)

    lines = ["## GitHub projects"]
    for record in records:  # keep the snapshot's own (name-sorted) order
        name = record["name"]
        lines.append(index_lines[name] if name in demoted else full_blocks[name])
    return "\n\n".join(lines)


def _default_max_tokens():
    raw = os.environ.get("AVATAR_MAX_CONTEXT_TOKENS")
    if raw is None:
        return DEFAULT_MAX_CONTEXT_TOKENS
    return int(raw)


def build_system_prompt(max_tokens=None, knowledge_dir=None):
    """Assemble the system prompt. Pure function of the files on disk
    and of AVATAR_MAX_CONTEXT_TOKENS (unless max_tokens is given explicitly)."""
    knowledge_dir = knowledge_dir or KNOWLEDGE_DIR
    max_tokens = _default_max_tokens() if max_tokens is None else max_tokens

    summary = _read_text_file(knowledge_dir / "summary.txt") or ""
    profile = _read_text_file(knowledge_dir / "profile.md") or ""
    github_records = _load_github_records(knowledge_dir / "github.json")

    static_sections = [ROLE_BLOCK, summary, profile]
    static_tokens = sum(estimate_tokens(section) for section in static_sections)
    rules_tokens = estimate_tokens(RULES_BLOCK)
    budget_for_github = max_tokens - static_tokens - rules_tokens

    github_section = _github_section(github_records, max(budget_for_github, 0))

    sections = [ROLE_BLOCK]
    if summary:
        sections.append(summary)
    if profile:
        sections.append(profile)
    if github_section:
        sections.append(github_section)
    sections.append(RULES_BLOCK)

    prompt = "\n\n".join(sections)

    total_tokens = estimate_tokens(prompt)
    if total_tokens > max_tokens:
        raise PromptTooLargeError(
            f"Assembled prompt needs ~{total_tokens} tokens, over the {max_tokens}-token budget"
        )

    return prompt
