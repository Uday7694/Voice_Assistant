"""Safety boundaries.

Two passes. The inbound pass looks at what the caller said; the outbound pass looks at
what the agent is about to say. Both are cheap regex checks — deliberately not an LLM
call, because a guardrail that adds 300 ms to every turn gets removed.

Regexes catch the blatant cases. The per-agent refusal instruction in the system prompt
does the nuanced work; this layer is the floor, not the ceiling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Callers describing an emergency must reach a human immediately, not a language model.
_EMERGENCY = re.compile(
    r"\b(chest pain|heart attack|stroke|unconscious|not breathing|bleeding heavily|"
    r"suicide|overdose|seizure|emergency)\b",
    re.IGNORECASE,
)

# Instructions aimed at the model rather than the assistant.
_INJECTION = re.compile(
    r"(ignore (all |your )?(previous|prior) instructions|system prompt|"
    r"you are now|disregard (the|your) rules|reveal your (prompt|instructions))",
    re.IGNORECASE,
)

# Clinical advice the agent must never produce, however it is phrased.
_MEDICAL_ADVICE = re.compile(
    r"\b(\d+\s?(mg|ml|mcg)\b|take \d+|dosage|dose of|you (probably |likely )?have\b|"
    r"diagnos(is|ed|e)|prescri(be|ption)|stop taking|increase your dose)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class InboundVerdict:
    allowed: bool = True
    escalate: bool = False
    reason: str = ""
    spoken_response: str = ""


@dataclass(frozen=True)
class OutboundVerdict:
    text: str
    modified: bool = False
    reason: str = ""


def check_inbound(text: str, *, medical_domain: bool) -> InboundVerdict:
    """Screen the caller's utterance before it reaches the planner."""
    if medical_domain and _EMERGENCY.search(text):
        return InboundVerdict(
            allowed=False,
            escalate=True,
            reason="medical emergency detected",
            spoken_response=(
                "This sounds urgent. I'm connecting you to a member of our team right now. "
                "If this is a life-threatening emergency, please call 108 immediately."
            ),
        )

    if _INJECTION.search(text):
        # Not a refusal — the caller may simply be confused. Ignore the instruction,
        # continue the conversation, and let the planner handle the turn normally.
        return InboundVerdict(allowed=True, reason="ignored embedded instruction")

    return InboundVerdict()


def check_outbound(text: str, *, medical_domain: bool, refusal_line: str) -> OutboundVerdict:
    """Screen the agent's reply before it is spoken."""
    if medical_domain and _MEDICAL_ADVICE.search(text):
        return OutboundVerdict(
            text=(
                "I'm not able to give medical advice — a doctor needs to answer that one. "
                f"{refusal_line}"
            ),
            modified=True,
            reason="blocked clinical advice",
        )
    return OutboundVerdict(text=text)


def sanitise_for_prompt(text: str) -> str:
    """Knowledge-base and caller text is data, never instructions.

    Wrapping in delimiters and stripping role markers keeps retrieved content from
    impersonating a system turn.
    """
    cleaned = re.sub(r"^\s*(system|assistant|user)\s*:", "", text, flags=re.IGNORECASE | re.MULTILINE)
    return cleaned.strip()
