#!/usr/bin/env python3
"""
Validate consistency between the Summary and Experience sections of a
LinkedIn profile Markdown file.

This script is READ-ONLY: it never modifies the profile document.

It performs heuristic checks to catch cases where the Summary section
claims experience (technologies, platforms, projects) that is not
supported by the Experience section, or where the two sections
contradict each other.

Usage:
    python3 scripts/validate_profile.py [path/to/profile.md]

Default path: projects/08-linkedin-avatar/knowledge/profile.md

Exit codes:
    0 - No mismatches found (or only warnings in non-strict mode)
    1 - Mismatches found
    2 - Usage / file error

The script has no external dependencies beyond the Python standard
library. It is intended to be advisory: in CI it runs in warn-only mode
by default (see .github/workflows/validate-profile.yml).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

DEFAULT_PROFILE = "projects/08-linkedin-avatar/knowledge/profile.md"

# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------

# Top-level (##) headings we care about. We match case-insensitively and
# tolerate trailing whitespace.
SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def split_sections(text: str) -> dict[str, str]:
    """Split a Markdown document into a mapping of top-level (##) section
    name (lower-cased) to its body text.

    Content before the first ## heading is stored under the key "".
    """
    sections: dict[str, str] = {}
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        sections[""] = text
        return sections

    # Preamble before the first heading.
    sections[""] = text[: matches[0].start()]

    for i, m in enumerate(matches):
        name = m.group(1).strip().lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[start:end]
    return sections


# ---------------------------------------------------------------------------
# Technology / keyword extraction
# ---------------------------------------------------------------------------

# Canonical technology tokens we look for. Each entry maps a canonical name
# to a list of regex patterns that match it in free text. Patterns are
# matched case-insensitively with word boundaries where appropriate.
#
# This list is deliberately conservative: it only includes technologies
# that are meaningful to cross-check between Summary and Experience.
TECHNOLOGIES: dict[str, list[str]] = {
    "Java": [r"\bJava\b"],
    "Spring Boot": [r"\bSpring\s+Boot\b", r"\bSpringBoot\b"],
    "Spring": [r"\bSpring\b(?!\s+Boot)"],
    "Python": [r"\bPython\b"],
    "AWS": [r"\bAWS\b", r"\bAmazon Web Services\b"],
    "Azure": [r"\bAzure\b"],
    "Cloud Foundry": [r"\bCloud\s+Foundry\b"],
    "Kafka": [r"\bKafka\b"],
    "Sybase IQ": [r"\bSybase\s+IQ\b", r"\bSybase\b"],
    "MCP": [r"\bMCP\b", r"\bModel\s+Context\s+Protocol\b"],
    "LLM": [r"\bLLM\b", r"\bLLMs\b", r"\bLarge\s+Language\s+Model"],
    "Parquet": [r"\bParquet\b"],
    "Kubernetes": [r"\bKubernetes\b", r"\bK8s\b"],
    "Docker": [r"\bDocker\b"],
    "Perl": [r"\bPerl\b"],
    "C++": [r"\bC\+\+"],
    "Karate": [r"\bKarate\b"],
    "Copilot": [r"\bCopilot\b"],
    "Murex": [r"\bMurex\b"],
    "Fidessa": [r"\bFidessa\b"],
    "REST": [r"\bREST\b", r"\bRESTful\b"],
    "SQL": [r"\bSQL\b"],
    "Gen AI": [r"\bGen\s*AI\b", r"\bGenerative\s+AI\b"],
}

# Phrases that indicate a claim of production / hands-on experience.
PRODUCTION_CLAIM_PATTERNS = [
    r"\bin\s+production\b",
    r"\bproduction[- ]ready\b",
    r"\bproduction\s+experience\b",
    r"\bshipped\b",
    r"\bdelivered\b",
    r"\bbuilt\b",
    r"\bdesigned\s+and\s+built\b",
    r"\bdeployed\b",
]


def find_technologies(text: str) -> set[str]:
    """Return the set of canonical technology names mentioned in `text`."""
    found: set[str] = set()
    for canonical, patterns in TECHNOLOGIES.items():
        for pat in patterns:
            if re.search(pat, text, flags=re.IGNORECASE):
                found.add(canonical)
                break
    return found


def has_production_claim(text: str) -> bool:
    for pat in PRODUCTION_CLAIM_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False


# ---------------------------------------------------------------------------
# Consistency rules
# ---------------------------------------------------------------------------

class Mismatch:
    def __init__(self, kind: str, message: str) -> None:
        self.kind = kind
        self.message = message

    def __str__(self) -> str:
        return f"[{self.kind}] {self.message}"


def check_technologies_in_summary_backed_by_experience(
    summary: str, experience: str
) -> list[Mismatch]:
    """Any technology mentioned in the Summary should also appear in the
    Experience section (as evidence of hands-on use)."""
    mismatches: list[Mismatch] = []
    summary_techs = find_technologies(summary)
    experience_techs = find_technologies(experience)

    for tech in sorted(summary_techs - experience_techs):
        mismatches.append(
            Mismatch(
                "summary-not-in-experience",
                f"Summary mentions '{tech}' but it does not appear in the "
                f"Experience section. Either add an Experience bullet "
                f"evidencing it, or remove the claim from the Summary.",
            )
        )
    return mismatches


def check_technologies_in_experience_reflected_in_summary(
    summary: str, experience: str
) -> list[Mismatch]:
    """Technologies that appear prominently in Experience should generally
    be reflected in the Summary. This is a softer check and is reported as
    a warning rather than a hard mismatch."""
    warnings: list[Mismatch] = []
    summary_techs = find_technologies(summary)
    experience_techs = find_technologies(experience)

    # Only warn for a small set of "headline" technologies that a Summary
    # would be expected to mention if they appear in Experience.
    headline = {"Java", "Python", "AWS", "LLM", "MCP", "Kafka"}
    for tech in sorted((experience_techs & headline) - summary_techs):
        warnings.append(
            Mismatch(
                "experience-not-in-summary",
                f"Experience mentions '{tech}' but the Summary does not. "
                f"Consider whether the Summary should mention it.",
            )
        )
    return warnings


def check_contradictions(summary: str, experience: str) -> list[Mismatch]:
    """Detect obvious contradictions between the two sections.

    Currently checks for exclusivity claims in the Summary (e.g. "Java
    only") that are contradicted by other technologies in Experience.
    """
    mismatches: list[Mismatch] = []

    exclusivity_patterns = [
        (r"\bJava\s+only\b", "Java"),
        (r"\bPython\s+only\b", "Python"),
        (r"\bonly\s+Java\b", "Java"),
        (r"\bonly\s+Python\b", "Python"),
    ]
    experience_techs = find_technologies(experience)

    for pat, claimed in exclusivity_patterns:
        if re.search(pat, summary, flags=re.IGNORECASE):
            others = experience_techs - {claimed}
            if others:
                mismatches.append(
                    Mismatch(
                        "contradiction",
                        f"Summary claims '{claimed} only' but Experience "
                        f"also mentions: {', '.join(sorted(others))}.",
                    )
                )
    return mismatches


def check_production_claims(summary: str, experience: str) -> list[Mismatch]:
    """If the Summary claims production experience, the Experience section
    should contain at least one production-related phrase."""
    mismatches: list[Mismatch] = []
    if has_production_claim(summary) and not has_production_claim(experience):
        mismatches.append(
            Mismatch(
                "production-claim-unsupported",
                "Summary claims production/hands-on experience but the "
                "Experience section contains no matching production "
                "language (e.g. 'in production', 'delivered', 'built').",
            )
        )
    return mismatches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate(profile_text: str) -> tuple[list[Mismatch], list[Mismatch]]:
    """Return (errors, warnings) for the given profile text."""
    sections = split_sections(profile_text)

    summary = sections.get("summary", "")
    experience = sections.get("experience", "")

    errors: list[Mismatch] = []
    warnings: list[Mismatch] = []

    if not summary:
        errors.append(
            Mismatch("missing-section", "No '## Summary' section found.")
        )
    if not experience:
        errors.append(
            Mismatch("missing-section", "No '## Experience' section found.")
        )
    if errors:
        return errors, warnings

    errors.extend(
        check_technologies_in_summary_backed_by_experience(summary, experience)
    )
    errors.extend(check_contradictions(summary, experience))
    errors.extend(check_production_claims(summary, experience))

    warnings.extend(
        check_technologies_in_experience_reflected_in_summary(
            summary, experience
        )
    )

    return errors, warnings


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate consistency between the Summary and Experience "
            "sections of a LinkedIn profile Markdown file. Read-only."
        )
    )
    parser.add_argument(
        "profile",
        nargs="?",
        default=DEFAULT_PROFILE,
        help=f"Path to profile.md (default: {DEFAULT_PROFILE})",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (exit non-zero on warnings too).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    path = Path(args.profile)
    if not path.is_file():
        print(f"error: profile file not found: {path}", file=sys.stderr)
        return 2

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"error: could not read {path}: {exc}", file=sys.stderr)
        return 2

    errors, warnings = validate(text)

    if errors:
        print(f"FAIL: {len(errors)} mismatch(es) found in {path}\n")
        for m in errors:
            print(f"  {m}")
    else:
        print(f"OK: no mismatches found in {path}")

    if warnings:
        print(f"\nWARN: {len(warnings)} advisory note(s):")
        for w in warnings:
            print(f"  {w}")

    if errors:
        return 1
    if warnings and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
