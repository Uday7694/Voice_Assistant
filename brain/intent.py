"""LLM intent detection and slot extraction, on the fast model.

This runs on every user turn, so it is deliberately one small JSON call. It answers
three questions at once: what does the user want, what values did they just give, and
what language are they speaking.
"""

from __future__ import annotations

import logging

from .config import FAST_TIMEOUT
from .flow import Agent, Node
from .llm import LLMClient, LLMTimeout
from .models import GLOBAL_INTENTS, IntentResult, Session

log = logging.getLogger(__name__)

_SYSTEM = """You classify a single user utterance in a voice conversation.

Return ONLY a JSON object:
{
  "intent": "<one of the allowed intents>",
  "confidence": <0.0-1.0>,
  "slots": {"<slot>": "<value in plain text>"},
  "language": "<BCP-47 tag, e.g. en-IN, hi-IN, te-IN>"
}

Rules:
- Pick "unknown" if nothing fits. Never invent an intent outside the allowed list.
- Fill a slot from anything the USER said in this conversation, including earlier turns.
  A caller who said "heart doctor" three turns ago has given you the department.
- Never invent a value out of nothing. But if the assistant just offered a list of
  options and the caller picks one — including loosely ("tomorrow morning works",
  "the first one", "Friday is fine") — record the assistant's EXACT option text as the
  slot value. Choosing from what was offered is the caller's decision, not a guess.
- If the caller's wording matches none of the offered options, leave the slot empty.
- Map what the caller said onto the expected value (e.g. "heart doctor" -> "cardiology",
  "bone doctor" -> "orthopaedics").
- Normalise dates and times to plain readable text (e.g. "tomorrow 4 pm").
- The user may mix English with an Indian language. Classify on meaning, not language.
- Report the language the caller is SPEAKING, not the script they typed it in. Indian
  callers routinely write their own language in Latin letters: "mera naam amit hai" and
  "aray madam appointment book kr do" are Hindi (hi-IN), not English, even though every
  character is ASCII. Report English only when the words themselves are English.
- Speech input is imperfect. Tolerate transcription noise and partial words.
- The current step describes what the ASSISTANT is doing, not what the caller is
  allowed to say. A caller can ask for anything at any point. Classify the utterance on
  its own meaning and never answer "unknown" merely because it does not fit the step.
- Reserve "unknown" for utterances you genuinely cannot interpret: silence, noise, or
  something unrelated to the assistant's purpose.
- Use "abusive" for insults, swearing or threats aimed at the assistant or anyone else,
  in any language or script. Do NOT use "escalate_to_human" for these: that intent means
  the caller explicitly asked to speak to a person. Frustration on its own ("this is
  useless", "you are not helping") is not abuse."""


def _prompt_for(node: Node, agent: Agent, session: Session, text: str) -> list[dict[str, str]]:
    allowed = list(dict.fromkeys(node.expected_intents + GLOBAL_INTENTS + ("unknown",)))
    slots = list(node.required_slots) or ["(none for this step)"]
    recent = "\n".join(f"{t.role.value}: {t.text}" for t in session.recent(8))

    context = (
        f"Assistant purpose: {agent.persona}\n"
        f"Current step: {node.id} — {node.goal}\n"
        f"Allowed intents: {', '.join(allowed)}\n"
        f"Slots to look for: {', '.join(slots)}\n"
        f"Slots already known: {session.slots or '(none)'}\n"
        f"Recent conversation:\n{recent or '(this is the first turn)'}"
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"{context}\n\nUser just said: {text!r}"},
    ]


async def classify(
    llm: LLMClient, agent: Agent, session: Session, text: str
) -> IntentResult:
    """Classify one utterance. Never raises — a failure degrades to 'unknown'."""
    node = agent.node(session.node_id or agent.entry_node)
    try:
        data = await llm.json_call(
            _prompt_for(node, agent, session, text), timeout=FAST_TIMEOUT
        )
    except LLMTimeout:
        log.warning("Intent classification timed out; falling back to unknown")
        return IntentResult(name="unknown", confidence=0.0)
    except Exception:  # noqa: BLE001 - classification must never break the turn
        log.exception("Intent classification failed")
        return IntentResult(name="unknown", confidence=0.0)

    return _coerce(data, node)


def _coerce(data: dict, node: Node) -> IntentResult:
    """Trust nothing the model returns; clamp it back into the allowed space."""
    name = str(data.get("intent", "unknown")).strip() or "unknown"
    allowed = set(node.expected_intents) | set(GLOBAL_INTENTS) | {"unknown"}
    if name not in allowed:
        log.info("Model produced out-of-vocabulary intent %r; treating as unknown", name)
        name = "unknown"

    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0

    raw_slots = data.get("slots") or {}
    slots = {
        str(k): str(v).strip()
        for k, v in raw_slots.items()
        if isinstance(raw_slots, dict) and v not in (None, "", "null")
    }

    language = str(data.get("language") or "en-IN")
    return IntentResult(name=name, confidence=confidence, slots=slots, language=language)
