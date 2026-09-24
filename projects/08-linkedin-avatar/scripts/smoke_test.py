#!/usr/bin/env python3
"""Post-deploy smoke test for the LinkedIn Avatar app.

Hits the deployed public URL and verifies:
  1. Required environment variables are present (checked against a hardcoded
     list kept in sync with what ``app.py`` and its dependencies read).
  2. The root URL responds with HTTP 200 and the page is actually the avatar
     app, not just any 200 (e.g. a Render "waking up" placeholder).
  3. The chat endpoint returns a non-empty, plausible answer to a test question.
  4. The contact-capture flow accepts a request and, unless ``--dry-run``,
     PUSHOVER_USER/PUSHOVER_TOKEN are confirmed able to deliver a
     notification. This does NOT prove the deployed app's own
     contact-capture code called Pushover for this conversation — there is
     no way to observe that from outside the app without adding
     instrumentation to it (see issue #182's review discussion). It only
     confirms two necessary preconditions: the chat flow accepts and
     replies to a contact-shaped message, and the configured credentials
     work.

Exits non-zero on the first failure with a clear, actionable message.

Usage:
    python scripts/smoke_test.py --base-url https://example.com
    python scripts/smoke_test.py --base-url https://example.com --dry-run

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from avatar.styles import TITLE as EXPECTED_ROOT_MARKER  # noqa: E402

DEFAULT_BASE_URL = "https://ai-systems-lab-s8gy.onrender.com"

# Env vars that app.py (and the modules it imports) actually read.
# Keep this list in sync with app.py / avatar/*.py — it is the source of
# truth for the "env vars present" check.
REQUIRED_ENV_VARS = [
    "DEEPSEEK_API_KEY",
]
OPTIONAL_ENV_VARS = [
    "AVATAR_PROVIDER",
    "AVATAR_MODEL",
    "AVATAR_DAILY_BUDGET_USD",
    "AVATAR_MAX_CONTEXT_TOKENS",
    "AVATAR_MAX_INPUT_CHARS",
    "AVATAR_SESSION_RATE_LIMIT",
    "AVATAR_IP_RATE_LIMIT",
    "PUSHOVER_USER",
    "PUSHOVER_TOKEN",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "GRADIO_SERVER_NAME",
    "GRADIO_SERVER_PORT",
]

TEST_QUESTION = "Tell me about issue-worm."
CONTACT_MESSAGE = (
    "I'd like to talk to him about a role, here's my email: smoke-test@example.com"
)

PUSHOVER_API = "https://api.pushover.net/1/messages.json"


class SmokeTestError(RuntimeError):
    """Raised when a smoke-test check fails."""


def _log(msg: str) -> None:
    print(msg, flush=True)


def check_env_vars() -> None:
    """Fail fast if a required env var is missing."""
    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise SmokeTestError(
            "Env var(s) missing: "
            + ", ".join(missing)
            + ". Set them before running the smoke test (see docs/deployment.md)."
        )
    _log(f"[ok] required env vars present: {', '.join(REQUIRED_ENV_VARS)}")


def _http_get(url: str, timeout: float = 60.0) -> tuple[int, bytes]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        raise SmokeTestError(f"GET {url} failed: {exc.reason}") from exc


def _http_post_json(
    url: str, payload: dict, timeout: float = 120.0
) -> tuple[int, bytes]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        raise SmokeTestError(f"POST {url} failed: {exc.reason}") from exc


def check_root(base_url: str) -> None:
    status, body = _http_get(base_url)
    if status != 200:
        raise SmokeTestError(
            f"Root URL {base_url} returned HTTP {status} (expected 200). "
            "The app may be asleep, crashed, or misconfigured — check Render logs."
        )
    if not body:
        raise SmokeTestError(f"Root URL {base_url} returned an empty body.")
    # A bare 200 isn't enough — a Render "service unavailable" placeholder,
    # a proxy error page, or an unrelated app on the same host would also
    # return 200. Require the app's own page title (set via
    # gr.Blocks(title=styles.TITLE) in app.py) to actually be present.
    text = body.decode("utf-8", errors="replace")
    if EXPECTED_ROOT_MARKER not in text:
        raise SmokeTestError(
            f"Root URL {base_url} returned HTTP 200 but the page doesn't "
            f"look like the avatar app — expected to find {EXPECTED_ROOT_MARKER!r} "
            "in the response body."
        )
    _log(
        f"[ok] root URL responded 200 with the expected app content ({len(body)} bytes)"
    )


def _gradio_chat(base_url: str, message: str) -> str:
    """Call Gradio's /api/predict endpoint for the ChatInterface.

    Gradio's exact payload shape varies by version, so we try the modern
    /gradio_api/call/chat endpoint first and fall back to /api/predict.
    """
    # Modern Gradio (4.x+): POST to /gradio_api/call/<fn>, then GET the event stream.
    call_url = base_url.rstrip("/") + "/gradio_api/call/chat"
    payload = {"data": [message, []]}
    status, body = _http_post_json(call_url, payload)
    if status == 200:
        try:
            event_id = json.loads(body).get("event_id")
        except json.JSONDecodeError:
            event_id = None
        if event_id:
            stream_url = f"{call_url}/{event_id}"
            s_status, s_body = _http_get(stream_url)
            if s_status == 200:
                text = s_body.decode("utf-8", errors="replace")
                # SSE stream: find the last "data: [...]" line.
                for line in reversed(text.splitlines()):
                    if line.startswith("data: "):
                        try:
                            parsed = json.loads(line[len("data: "):])
                        except json.JSONDecodeError:
                            continue
                        if isinstance(parsed, list) and parsed:
                            return str(parsed[0])
                return text

    # Legacy fallback.
    legacy_url = base_url.rstrip("/") + "/api/predict"
    status, body = _http_post_json(
        legacy_url, {"data": [message, []], "fn_index": 0}
    )
    if status != 200:
        raise SmokeTestError(
            f"Chat endpoint returned HTTP {status}. Body: "
            f"{body[:300].decode('utf-8', errors='replace')}"
        )
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SmokeTestError(f"Chat endpoint returned non-JSON: {exc}") from exc
    data = parsed.get("data") or []
    if not data:
        raise SmokeTestError(f"Chat endpoint returned no data: {parsed}")
    return str(data[0])


def check_chat(base_url: str) -> None:
    reply = _gradio_chat(base_url, TEST_QUESTION)
    if not reply or not reply.strip():
        raise SmokeTestError("Chat endpoint returned an empty reply.")
    lowered = reply.lower()
    # A refusal or an error string is a failure, not a valid answer.
    for bad in ("error", "traceback", "rate limit", "budget"):
        if bad in lowered and len(reply) < 200:
            raise SmokeTestError(
                f"Chat endpoint returned a suspicious reply: {reply[:200]!r}"
            )
    _log(f"[ok] chat endpoint replied ({len(reply)} chars): {reply[:80]!r}...")


def check_contact_capture(base_url: str, dry_run: bool) -> None:
    """Drive the contact-capture conversation flow and, unless dry-run,
    confirm PUSHOVER_USER/PUSHOVER_TOKEN can deliver a notification.

    This does not prove the deployed app's own contact-capture code called
    Pushover for this conversation — there's no way to observe that from
    outside the app without adding instrumentation to it. It only confirms
    two necessary preconditions: the chat flow accepts and replies to a
    contact-shaped message, and the configured credentials actually work.
    A broken server-side notification call (e.g. a stale token set only on
    the deployed instance) would not be caught by this check.
    """
    reply = _gradio_chat(base_url, CONTACT_MESSAGE)
    if not reply or not reply.strip():
        raise SmokeTestError("Contact-capture flow returned an empty reply.")
    _log(f"[ok] contact-capture flow replied: {reply[:80]!r}...")

    if dry_run:
        _log("[ok] --dry-run: skipping real Pushover send")
        return

    user = os.environ.get("PUSHOVER_USER")
    token = os.environ.get("PUSHOVER_TOKEN")
    if not user or not token:
        _log(
            "[skip] PUSHOVER_USER/PUSHOVER_TOKEN not set — cannot verify "
            "the credentials work. Re-run with them set to check."
        )
        return

    # NOTE: this calls Pushover directly with these credentials — it proves
    # they can deliver a notification, not that the deployed app's own
    # contact-capture path fired one for this conversation (see the
    # docstring above).
    # Send a clearly-labelled test notification so it's obvious in the app.
    payload = urllib.parse.urlencode(
        {
            "user": user,
            "token": token,
            "title": "Smoke test",
            "message": "LinkedIn Avatar smoke test — contact capture verified.",
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        PUSHOVER_API,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = resp.status
            body = resp.read()
    except urllib.error.HTTPError as exc:
        raise SmokeTestError(
            f"Pushover API returned HTTP {exc.code}: {exc.read()[:200]!r}"
        ) from exc
    except urllib.error.URLError as exc:
        raise SmokeTestError(f"Pushover API unreachable: {exc.reason}") from exc

    if status != 200:
        raise SmokeTestError(
            f"Pushover API returned HTTP {status}: {body[:200]!r}"
        )
    _log(
        "[ok] Pushover credentials work (a real notification was sent — "
        "this confirms the credentials, not that the app's own "
        "contact-capture path triggered a send)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SMOKE_TEST_BASE_URL", DEFAULT_BASE_URL),
        help=f"Base URL of the deployed app (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate request payloads without sending a real Pushover notification.",
    )
    args = parser.parse_args(argv)

    base_url = args.base_url.rstrip("/")
    _log(f"Smoke testing {base_url} (dry_run={args.dry_run})")

    checks = [
        ("env vars", lambda: check_env_vars()),
        ("root URL", lambda: check_root(base_url)),
        ("chat endpoint", lambda: check_chat(base_url)),
        (
            "contact capture",
            lambda: check_contact_capture(base_url, args.dry_run),
        ),
    ]

    for name, fn in checks:
        try:
            fn()
        except SmokeTestError as exc:
            _log(f"[FAIL] {name}: {exc}")
            return 1
        except Exception as exc:  # noqa: BLE001 — surface anything unexpected
            _log(f"[FAIL] {name}: unexpected error: {exc!r}")
            return 1

    _log("All smoke-test checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
