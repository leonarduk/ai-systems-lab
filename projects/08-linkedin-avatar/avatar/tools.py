"""record_contact, record_unknown_question, lookup_project.

Three tools for the DeepSeek tool-use loop (avatar/llm.py). Two fan a notification
out to every configured channel (Pushover, Telegram), one reads github.json
locally. None of them may raise — a failed notification or an unknown project
name must degrade to an error result, never take down the chat turn. See
docs/design.md §5.
"""

import difflib
import json
import logging
import os
from pathlib import Path

import requests

from .tool_definitions import (
    LOOKUP_PROJECT,
    RECORD_CONTACT,
    RECORD_UNKNOWN_QUESTION,
    TOOL_DEFINITIONS,
)

logger = logging.getLogger(__name__)

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"
TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
GITHUB_SNAPSHOT_PATH = (
    Path(__file__).resolve().parent.parent / "knowledge" / "github.json"
)

__all__ = [
    "LOOKUP_PROJECT",
    "RECORD_CONTACT",
    "RECORD_UNKNOWN_QUESTION",
    "TOOL_DEFINITIONS",
    "dispatch",
    "lookup_project",
    "record_contact",
    "record_unknown_question",
]


def _pushover_notify(title, message):
    """POST a Pushover notification. Never raises — logs and returns a status dict."""
    user = os.environ.get("PUSHOVER_USER")
    token = os.environ.get("PUSHOVER_TOKEN")

    if not user or not token:
        logger.info("Pushover not configured; logging instead. %s: %s", title, message)
        return {"status": "logged", "detail": "PUSHOVER_USER/PUSHOVER_TOKEN not set"}

    try:
        response = requests.post(
            PUSHOVER_URL,
            data={"token": token, "user": user, "title": title, "message": message},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException:
        logger.exception("Pushover notification failed: %s", title)
        return {"status": "failed", "detail": "notification could not be sent"}

    return {"status": "sent"}


def _telegram_notify(title, message):
    """POST a Telegram notification. Never raises — logs and returns a status dict."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        logger.info("Telegram not configured; logging instead. %s: %s", title, message)
        return {"status": "logged", "detail": "TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set"}

    try:
        response = requests.post(
            TELEGRAM_API_URL.format(token=token),
            data={"chat_id": chat_id, "text": f"{title}\n\n{message}"},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        # Telegram's URL embeds the bot token (unlike Pushover's, which keeps
        # it in the POST body) — logger.exception would print that URL, so
        # log only the status code, never the exception object itself.
        status = getattr(exc.response, "status_code", "no response")
        logger.error("Telegram notification failed (status=%s): %s", status, title)
        return {"status": "failed", "detail": "notification could not be sent"}

    return {"status": "sent"}


def _notify(title, message):
    """Fan a notification out to every configured channel. Never raises.

    Overall status is "sent" if any channel sent, "logged" if every channel
    merely logged (none configured), else "failed" — e.g. a configured
    channel errored and no other channel picked up the slack.
    """
    channels = {
        "pushover": _pushover_notify(title, message),
        "telegram": _telegram_notify(title, message),
    }
    statuses = {result["status"] for result in channels.values()}
    if "sent" in statuses:
        status = "sent"
    elif statuses == {"logged"}:
        status = "logged"
    else:
        status = "failed"
    return {"status": status, "channels": channels}


def record_contact(email, name=None, notes=None):
    """Notify me that a visitor wants to be contacted."""
    lines = [f"Email: {email}"]
    if name:
        lines.append(f"Name: {name}")
    if notes:
        lines.append(f"Notes: {notes}")
    result = _notify("LinkedIn Avatar: contact request", "\n".join(lines))
    return {"recorded": result["status"] in ("sent", "logged"), **result}


def record_unknown_question(question):
    """Notify me that the avatar didn't know the answer to something."""
    result = _notify("LinkedIn Avatar: unknown question", question)
    return {"recorded": result["status"] in ("sent", "logged"), **result}


def _validated_records(records):
    """Return only well-formed github.json records.

    A record is well-formed if it is a dict with a non-empty string "name".
    Malformed records are skipped with a warning rather than raising, so a
    single bad entry (from the separate snapshot generator) can't take down
    the lookup. Extra fields are preserved untouched.
    """
    valid = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            logger.warning(
                "Skipping malformed github.json record at index %d: not an object", index
            )
            continue
        name = record.get("name")
        if not isinstance(name, str) or not name.strip():
            logger.warning(
                "Skipping malformed github.json record at index %d: missing 'name'", index
            )
            continue
        valid.append(record)
    return valid


def lookup_project(name):
    """Fetch the full github.json record for one repo, fuzzy-matching the name."""
    try:
        raw = GITHUB_SNAPSHOT_PATH.read_text(encoding="utf-8")
        records = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read GitHub snapshot at %s", GITHUB_SNAPSHOT_PATH)
        return {
            "found": False,
            "message": "the GitHub project index is unavailable right now",
        }

    if not isinstance(records, list):
        logger.error(
            "GitHub snapshot at %s is not a list of records", GITHUB_SNAPSHOT_PATH
        )
        return {
            "found": False,
            "message": "the GitHub project index is unavailable right now",
        }

    by_name = {record["name"].lower(): record for record in _validated_records(records)}
    query = name.strip().lower()

    if query in by_name:
        return {"found": True, "project": by_name[query]}

    matches = difflib.get_close_matches(query, by_name.keys(), n=1, cutoff=0.6)
    if matches:
        return {"found": True, "project": by_name[matches[0]]}

    return {"found": False, "message": f"no project matching '{name}' was found"}


_DISPATCH_TABLE = {
    RECORD_CONTACT: record_contact,
    RECORD_UNKNOWN_QUESTION: record_unknown_question,
    LOOKUP_PROJECT: lookup_project,
}


def dispatch(name, arguments):
    """Call a tool by name with keyword arguments. Never raises."""
    handler = _DISPATCH_TABLE.get(name)
    if handler is None:
        return {"error": f"unknown tool: {name}"}

    try:
        return handler(**arguments)
    except TypeError:
        logger.exception("Bad arguments for tool %s: %r", name, arguments)
        return {"error": f"invalid arguments for tool: {name}"}
