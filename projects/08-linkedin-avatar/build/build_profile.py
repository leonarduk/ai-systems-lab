"""Turn a local linkedin.pdf export into a redacted knowledge/profile.md.

Usage:
    python build/build_profile.py --pdf linkedin.pdf --out knowledge/profile.md
    python build/build_profile.py --check knowledge/profile.md

The PDF export carries an email address, a phone number and often a location —
none of which belong in a public repo. This script strips them by pattern,
normalises LinkedIn's section headings into markdown, and writes the reviewed
result. See docs/design.md §3.2: when a pattern is ambiguous, redact — a
missing detail costs a slightly worse answer, a leaked one is permanent.
"""

import argparse
import re
import sys
from pathlib import Path

from pypdf import PdfReader

REDACTED = "[redacted]"

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"

KNOWN_HEADINGS = {
    "summary",
    "experience",
    "education",
    "skills",
    "certifications",
    "licenses & certifications",
    "licenses and certifications",
    "volunteering",
    "volunteer experience",
    "recommendations",
    "accomplishments",
    "honors & awards",
    "honors and awards",
    "courses",
    "publications",
    "languages",
    "interests",
    "contact",
}

PAGE_FURNITURE_RE = re.compile(r"^\s*page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE)

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

LINKEDIN_URL_RE = re.compile(
    r"(https?://)?(www\.)?linkedin\.com/in/[A-Za-z0-9\-_%]+/?", re.IGNORECASE
)

# Broad candidate for anything digit-and-separator shaped; filtered below by
# digit count so short things like version numbers or dates aren't caught.
PHONE_CANDIDATE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")

# A "-" or en-dash with whitespace on both sides is a range separator
# ("Jan 2020 - Dec 2023", "01.2020 - 12.2023"), not a phone number — real
# phone formatting keeps dashes tight against the digits either side.
RANGE_SEPARATOR_RE = re.compile(r"\s[-–]\s")

UK_POSTCODE_RE = re.compile(
    r"\b[A-Za-z]{1,2}\d[A-Za-z\d]?\s*\d[A-Za-z]{2}\b",
)

STREET_ADDRESS_RE = re.compile(
    r"^\s*\d{1,4}[a-zA-Z]?\s+\S.*\b"
    r"(Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Close|Court|Ct|Way|Place|Pl)\b\.?",
    re.IGNORECASE,
)


def extract_text(pdf_path):
    """Extract text from every page of the PDF, joined with blank lines."""
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _looks_like_phone(candidate):
    if RANGE_SEPARATOR_RE.search(candidate):
        return False
    digits = re.sub(r"\D", "", candidate)
    return 9 <= len(digits) <= 15


def _redact_phones(text):
    def replace_if_phone(match):
        if _looks_like_phone(match.group(0)):
            return REDACTED
        return match.group(0)

    return PHONE_CANDIDATE_RE.sub(replace_if_phone, text)


def redact(text):
    """Replace contact-shaped content with [redacted]. Returns the redacted text."""
    text = EMAIL_RE.sub(REDACTED, text)
    text = LINKEDIN_URL_RE.sub(REDACTED, text)
    text = UK_POSTCODE_RE.sub(REDACTED, text)
    text = _redact_phones(text)

    lines = [
        REDACTED if STREET_ADDRESS_RE.match(line) else line for line in text.splitlines()
    ]
    return "\n".join(lines)


def normalize(text):
    """Turn known section labels into markdown headings and tidy whitespace."""
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if PAGE_FURNITURE_RE.match(line):
            continue

        if line.strip().lower() in KNOWN_HEADINGS:
            lines.append(f"## {line.strip()}")
            continue

        lines.append(line)

    normalized = "\n".join(lines)
    # Collapse runs of 3+ blank lines down to a single blank line.
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip() + "\n"


def build_profile(pdf_path, out_path):
    """Extract, redact and normalize a LinkedIn PDF export into markdown."""
    text = extract_text(pdf_path)
    redacted_text = redact(text)
    profile_markdown = normalize(redacted_text)
    out_path.write_text(profile_markdown, encoding="utf-8")
    return profile_markdown


def find_contact_leaks(text):
    """Return a list of (line_number, pattern_name, snippet) for anything
    contact-shaped still present in already-redacted text."""
    leaks = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if EMAIL_RE.search(line):
            leaks.append((line_no, "email", line.strip()))
        if LINKEDIN_URL_RE.search(line):
            leaks.append((line_no, "linkedin URL", line.strip()))
        if UK_POSTCODE_RE.search(line):
            leaks.append((line_no, "postcode", line.strip()))
        if STREET_ADDRESS_RE.match(line):
            leaks.append((line_no, "street address", line.strip()))
        for match in PHONE_CANDIDATE_RE.finditer(line):
            if _looks_like_phone(match.group(0)):
                leaks.append((line_no, "phone number", line.strip()))
                break
    return leaks


def _out_path_is_safe(out_path):
    """Return True only if out_path resolves inside the real knowledge/ dir.

    Guards against two escape routes:
      1. out_path itself being a symlink or containing ``..`` that escapes
         the knowledge directory.
      2. ``knowledge/`` itself being a symlink to somewhere outside the
         project — in that case we refuse to write at all, because the
         resolved containment check would otherwise happily accept a path
         that lands outside the intended sandbox.
    """
    knowledge_dir = KNOWLEDGE_DIR.resolve()

    # If knowledge/ is a symlink, refuse outright. Even if it currently
    # points somewhere benign, a symlinked knowledge/ is not the layout
    # this tool expects and could be swapped at any time.
    if KNOWLEDGE_DIR.is_symlink():
        return False

    resolved = out_path.resolve()
    return resolved == knowledge_dir or knowledge_dir in resolved.parents


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", help="Path to the local linkedin.pdf export")
    parser.add_argument(
        "--out",
        default=str(KNOWLEDGE_DIR / "profile.md"),
        help="Where to write the redacted markdown (must be inside knowledge/)",
    )
    parser.add_argument(
        "--check",
        metavar="PROFILE_MD",
        help="Re-run redaction detection over an existing profile.md and exit 1 on any leak",
    )
    args = parser.parse_args(argv)

    if args.check:
        check_path = Path(args.check)
        text = check_path.read_text(encoding="utf-8")
        leaks = find_contact_leaks(text)
        if leaks:
            for line_no, pattern_name, snippet in leaks:
                print(
                    f"{check_path}:{line_no}: possible {pattern_name} — {snippet}",
                    file=sys.stderr,
                )
            return 1
        print(f"{check_path}: no contact-shaped content found")
        return 0

    if not args.pdf:
        parser.error("--pdf is required unless --check is given")

    out_path = Path(args.out)
    if not _out_path_is_safe(out_path):
        parser.error(f"--out must be inside {KNOWLEDGE_DIR}")

    build_profile(Path(args.pdf), out_path)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
