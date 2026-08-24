"""Session storage.

An in-memory implementation behind a narrow interface, so swapping in Redis (working
state) and Postgres (transcripts) later touches one file.
"""

from __future__ import annotations

from typing import Protocol

from .models import Session


class SessionStore(Protocol):
    def get(self, session_id: str) -> Session | None: ...
    def put(self, session: Session) -> None: ...
    def drop(self, session_id: str) -> None: ...


class InMemorySessionStore:
    """Process-local store. Fine for the CLI and tests; not for multiple workers."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def put(self, session: Session) -> None:
        self._sessions[session.session_id] = session

    def drop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def all(self) -> tuple[Session, ...]:
        return tuple(self._sessions.values())


def transcript(session: Session) -> str:
    """Flat transcript for logging and, later, the Postgres episodic record."""
    return "\n".join(f"{turn.role.value}: {turn.text}" for turn in session.history)
