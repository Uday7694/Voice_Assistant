"""Every call, written down as it happens.

One line of JSON per event, appended to a file named for the day: ``2026-08-25.jsonl``.
A day per file because that is how calls arrive and how anyone ever asks for them
("what happened on Tuesday?"), and JSON Lines because it is the one format that is both
tail-able while a call is running and loadable straight into a training pipeline
afterwards.

Written per turn rather than per call. A call that is still open has already produced
useful rows, and a process that dies mid-call — which is what a dropped line looks like
from here — loses nothing that already happened.

Every row carries ``session`` and ``caller``, so a day's file groups two ways: by call,
and by the person who made it. That is the whole reason to log the caller's number
rather than only a session id, and it is also the reason this file is sensitive. See
the note on redaction below.

Failure here is never allowed to end a call. A disk that is full, a permission that is
wrong, a value that will not serialise — all of it is logged and swallowed. A caller
mid-booking does not care that the analytics pipeline is unhappy.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DIRECTORY = Path(os.getenv("TRANSCRIPT_DIR", "logs/conversations"))

ENABLED = os.getenv("TRANSCRIPT_LOGGING", "1").strip().lower() not in {"0", "false", "no"}

# Whether to write phone numbers as themselves or as a stable hash.
#
# Off by default because the point of the log is training data, and a corpus of real
# calls with the identifiers stripped out cannot answer "did this caller ring twice".
# Turn it on for any deployment where the log leaves the building: these are hospital
# calls, the number is the patient, and a JSONL file is trivially copyable. Hashing
# keeps "same caller" answerable while making "which caller" unanswerable.
REDACT = os.getenv("TRANSCRIPT_REDACT", "").strip().lower() in {"1", "true", "yes", "on"}

# Appends from several calls land in one file, and a torn line is a corrupt record for
# whoever loads it later. The lock is per process; a multi-process deployment should
# give each worker its own directory rather than trusting O_APPEND atomicity.
_LOCK = threading.Lock()

_DIGITS = re.compile(r"\D")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _caller_id(session) -> str:
    """Who this call is from: their number when known, otherwise the session.

    A phone number is the only identifier that survives a caller ringing back, which is
    what makes a per-caller history possible at all.
    """
    phone = str(session.slots.get("phone", "") or "")
    if not phone:
        return session.session_id
    digits = _DIGITS.sub("", phone)
    if REDACT:
        import hashlib

        return "sha256:" + hashlib.sha256(digits.encode()).hexdigest()[:16]
    return digits


def path_for(day: str | None = None) -> Path:
    return DIRECTORY / f"{day or datetime.now().strftime('%Y-%m-%d')}.jsonl"


def write(record: dict[str, Any]) -> None:
    """Append one record. Never raises."""
    if not ENABLED:
        return
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        log.warning("Unserialisable transcript record dropped: %r", sorted(record))
        return

    try:
        with _LOCK:
            DIRECTORY.mkdir(parents=True, exist_ok=True)
            with open(path_for(), "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except OSError as exc:
        log.warning("Could not write transcript: %s", exc)


# --- the records ------------------------------------------------------------


def _envelope(session, kind: str) -> dict[str, Any]:
    return {
        "type": kind,
        "at": _now(),
        "session": session.session_id,
        "caller": _caller_id(session),
        "agent": session.agent_name,
        "channel": str(session.metadata.get("channel", "")),
        "language": session.language,
    }


def call_started(session) -> None:
    write(_envelope(session, "call_start"))


def turn(
    session,
    *,
    user_text: str,
    agent_text: str,
    intent: Any = None,
    tools: list[dict[str, Any]] | None = None,
    stage_ms: dict[str, float] | None = None,
) -> None:
    """One exchange: what the caller said, what the agent said back, and why.

    The intent, the node and the slots are logged alongside the words because a
    transcript on its own trains what to say and not when to say it — and the decision
    of *when* is the part of this system that is not a model.
    """
    record = _envelope(session, "turn")
    record.update(
        {
            "turn": len(session.history),
            "node": session.node_id,
            "user": user_text,
            "agent_said": agent_text,
            "slots": dict(session.slots),
        }
    )
    if intent is not None:
        record["intent"] = {
            "name": getattr(intent, "name", ""),
            "confidence": round(float(getattr(intent, "confidence", 0.0)), 3),
        }
    if tools:
        record["tools"] = tools
    if stage_ms:
        record["latency_ms"] = {k: round(v) for k, v in stage_ms.items()}
    write(record)


def call_ended(session, *, reason: str, outcome: str) -> None:
    """The last word on a call: how it finished and what it produced."""
    record = _envelope(session, "call_end")
    record.update(
        {
            "reason": reason,
            "outcome": outcome,
            "turns": len(session.history),
            "slots": dict(session.slots),
            "escalated": bool(session.escalated),
            "flagged": bool(session.flagged),
        }
    )
    if session.flag_reason:
        record["flag_reason"] = session.flag_reason
    write(record)


# --- reading them back ------------------------------------------------------


def read(day: str | None = None) -> list[dict[str, Any]]:
    """Every record for a day, in the order it happened.

    Lenient on purpose: a half-written final line — a process killed mid-append — must
    not make the rest of the day unreadable.
    """
    try:
        raw = path_for(day).read_text(encoding="utf-8")
    except OSError:
        return []

    records = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("Skipping malformed transcript line in %s", path_for(day))
    return records


def calls(day: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """A day's records grouped into calls, keyed by session id."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in read(day):
        grouped.setdefault(str(record.get("session", "")), []).append(record)
    return grouped
