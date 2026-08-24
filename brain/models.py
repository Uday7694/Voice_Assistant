"""Core domain types.

Everything here is immutable. State transitions return new objects so a turn can be
replayed, logged, or rolled back without hidden side effects.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Role(str, Enum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class Turn(BaseModel):
    model_config = {"frozen": True}

    role: Role
    text: str
    at: float = Field(default_factory=time.time)


# --- Intent ---------------------------------------------------------------

# Global intents exist in every flow node and short-circuit the planner.
GLOBAL_INTENTS = (
    "escalate_to_human",
    "end_conversation",
    "repeat_that",
    "out_of_scope",
    # A caller swearing is not a caller asking for a human. Without a label of its own,
    # the classifier reached for escalate_to_human and every insult transferred the call
    # instantly — which is both wrong and an easy way to skip the queue.
    "abusive",
)


class IntentResult(BaseModel):
    model_config = {"frozen": True}

    name: str
    confidence: float = 0.0
    slots: dict[str, str] = Field(default_factory=dict)
    language: str = "en-IN"

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.55


# --- Session --------------------------------------------------------------


class Session(BaseModel):
    """One conversation. Immutable: every mutation returns a new Session."""

    model_config = {"frozen": True}

    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    agent_name: str = ""
    node_id: str = ""
    slots: dict[str, str] = Field(default_factory=dict)
    history: tuple[Turn, ...] = ()
    language: str = ""
    """Sticky. A caller who switches mid-call switches it, a single 'yes' does not."""

    turns_in_node: int = 0
    no_match_streak: int = 0
    abuse_streak: int = 0
    """Consecutive abusive turns. Reset by one ordinary turn: people cool down."""
    ended: bool = False
    escalated: bool = False
    flagged: bool = False
    """Marked for human review. Separate from `escalated`: a call can be handed to a
    person for ordinary reasons, and those should not appear in an abuse report."""

    flag_reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def with_turn(self, role: Role, text: str) -> "Session":
        return self.model_copy(update={"history": self.history + (Turn(role=role, text=text),)})

    def with_slots(self, new_slots: dict[str, str]) -> "Session":
        merged = {**self.slots, **{k: v for k, v in new_slots.items() if v}}
        return self.model_copy(update={"slots": merged})

    def at_node(self, node_id: str) -> "Session":
        if node_id == self.node_id:
            return self.model_copy(update={"turns_in_node": self.turns_in_node + 1})
        return self.model_copy(update={"node_id": node_id, "turns_in_node": 0})

    def recent(self, limit: int) -> tuple[Turn, ...]:
        return self.history[-limit:] if limit else self.history


# --- Events emitted to whatever transport is attached ---------------------


class Event(BaseModel):
    model_config = {"frozen": True}
    kind: str


class SayEvent(Event):
    """A speakable chunk. Emitted at sentence boundaries so TTS can start early."""

    kind: Literal["say"] = "say"
    text: str
    is_final: bool = False


class ToolEvent(Event):
    kind: Literal["tool"] = "tool"
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    ok: bool = True
    ms: float = 0.0


class TransitionEvent(Event):
    kind: Literal["transition"] = "transition"
    from_node: str
    to_node: str
    reason: str = ""


class IntentEvent(Event):
    kind: Literal["intent"] = "intent"
    intent: IntentResult


class EndEvent(Event):
    kind: Literal["end"] = "end"
    reason: str = "completed"


class EscalateEvent(Event):
    kind: Literal["escalate"] = "escalate"
    reason: str = ""


class ErrorEvent(Event):
    kind: Literal["error"] = "error"
    message: str
    recoverable: bool = True


class TimingEvent(Event):
    """Per-stage latency, so regressions are visible from day one."""

    kind: Literal["timing"] = "timing"
    stage_ms: dict[str, float] = Field(default_factory=dict)
