"""LLM intent detection and slot extraction, on the fast model.

This runs on every user turn, so it is deliberately one small JSON call. It answers
three questions at once: what does the user want, what values did they just give, and
what language are they speaking.
"""

from __future__ import annotations

import logging

from . import slots as slot_mod
from .config import FAST_TIMEOUT
from .flow import Agent, Node
from .llm import LLMClient, LLMTimeout
from .models import GLOBAL_INTENTS, IntentResult, Session

log = logging.getLogger(__name__)

_SYSTEM = """You classify one utterance from a voice call.

Return ONLY this JSON object:
{"intent": "<allowed intent>", "confidence": <0.0-1.0>,
 "slots": {"<slot>": "<value>"}, "language": "<BCP-47 tag>"}

SLOTS ARE THE POINT. Every value the caller gives - a name, a department, a doctor, a
time, a phone number - goes in "slots", on the turn they say it, whatever the intent
turns out to be. A turn that carried a value and returned "slots": {} is wrong. Take
values from earlier turns too: a caller who said "heart doctor" three turns ago has
given you the department.

- Intent: one from the allowed list, or "unknown". Never invent one.
- Never invent a value out of nothing. But when the assistant has just offered options
  and the caller picks one, however loosely ("the first one", "Friday is fine"), record
  the assistant's exact option text.
- Map onto an expected value only when the meaning is plain: "heart doctor" ->
  "cardiology". When a slot lists allowed values and the caller matches none, record
  their own words unchanged - never the nearest-sounding listed value. A caller asking
  for a department that does not exist has to be told so.
- Normalise dates and times to plain text ("tomorrow 4 pm").
- Report the language SPOKEN, not the script it is written in. Indian callers routinely
  write their own language in Latin letters: "mera naam amit hai" is Hindi (hi-IN), not
  English, though every character is Latin. Answer English only for English words.
- This is speech, imperfectly transcribed. Tolerate noise and near-misses in values, and
  in the request itself: a sentence that otherwise reads as booking a doctor is one even
  if a word came through wrong. Indic transcription turns "appointment" into "apartment"
  routinely. Answer "out_of_scope" only when the caller genuinely wants something else
  entirely, never because one word sounds wrong.
- The current step describes what the ASSISTANT is doing, not what the caller is allowed
  to say. A caller can ask for anything at any point. Classify on meaning alone, and
  never answer "unknown" merely because the utterance does not fit the step.
- Reserve "unknown" for silence, noise, or the genuinely uninterpretable.
- "asks_identity": the caller asks what you are - a person, a machine, a recording. A
  caller who has explicitly asked to speak to a person is "escalate_to_human".
- "abusive": insults, swearing or threats, in any language or script. Not
  "escalate_to_human", and not mere frustration ("this is useless")."""


def _vocabularies(agent: Agent, session: Session) -> str:
    """List the closed value sets for this flow's slots, one line each.

    Every rule-bound slot, not only the current step's. A caller names a department in
    their opening sentence, long before the step that asks for one, and the value has to
    be recognised — or rejected — at the moment it is spoken.
    """
    lines = []
    for rule in agent.slot_rules:
        known = slot_mod.options(rule.validator, session.slots)
        if known:
            # Terse on purpose. Every character here is spent on every turn of every
            # call, and the classifier runs against a token-per-minute ceiling; the rule
            # about keeping unmatched wording is stated once in the system prompt rather
            # than repeated per slot.
            lines.append(f"Allowed {rule.slot}: {', '.join(known)} (or the caller's own words)")
    return "".join(f"{line}\n" for line in lines)


def _prompt_for(node: Node, agent: Agent, session: Session, text: str) -> list[dict[str, str]]:
    allowed = list(dict.fromkeys(node.expected_intents + GLOBAL_INTENTS + ("unknown",)))
    slots = list(node.required_slots) or ["(none for this step)"]
    # Callers do not answer in the order the flow asks. "I need a cardiology
    # appointment" carries the department three steps before the flow gets to it, and
    # dropping it means asking for something the caller already said — which is the
    # exact moment a call starts to feel like a form.
    elsewhere = [s for s in agent.all_slots if s not in node.required_slots]
    recent = "\n".join(f"{t.role.value}: {t.text}" for t in session.recent(8))

    context = (
        # The purpose line, not the persona. The classifier needs to know what this desk
        # does; the paragraph about being warm and brisk is for the half that speaks, and
        # every character here is spent on every turn against a token-per-minute ceiling.
        f"Assistant purpose: {agent.purpose}\n"
        f"Current step: {node.id} — {node.goal}\n"
        f"Allowed intents: {', '.join(allowed)}\n"
        f"Slots to look for: {', '.join(slots)}\n"
        + (f"Also record if the caller mentions: {', '.join(elsewhere)}\n" if elsewhere else "")
        + f"{_vocabularies(agent, session)}"
        + (
            f"The assistant's last question asked for: {session.asked_slot}. A bare "
            "answer - a name, a time, a number - fills that slot and no other.\n"
            if session.asked_slot
            else ""
        )
        + f"Slots already known: {session.slots or '(none)'}\n"
        + f"Recent conversation:\n{recent or '(this is the first turn)'}"
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
            _prompt_for(node, agent, session, text),
            timeout=FAST_TIMEOUT,
            # An empty object is not an answer. Measured: two calls in five came back
            # as "{}" - no error, no timeout, nothing to parse - and every one of those
            # turns degraded to "unknown", left the call at greet and shifted the whole
            # conversation one step out of line. Rejecting it here sends the turn to the
            # next provider instead, which is the difference between a demo that books
            # an appointment and one that does not.
            accept=lambda payload: bool(payload.get("intent")),
        )
    except LLMTimeout:
        log.warning("Intent classification timed out; falling back to unknown")
        return IntentResult(name="unknown", confidence=0.0)
    except Exception:  # noqa: BLE001 - classification must never break the turn
        log.exception("Intent classification failed")
        return IntentResult(name="unknown", confidence=0.0)

    return _coerce(data, node, agent)


# A label the model chose is an opinion; a value it pulled out of the utterance is an
# observation, and the two fail independently. Measured on the live call that prompted
# this: "amit ji" came back as "unknown" with patient_name="amit ji" — the model heard
# the name perfectly and still could not name what the caller was doing. Treated as
# unknown, that turn grew the no-match streak, triggered the out-of-scope line, and the
# call escalated three turns later having understood every word of it.
#
# So when the utterance carried a value this flow actually wants, the turn is the caller
# providing details, whatever the model called it. Confidence is lifted with it for the
# same reason: it is the *label* that was uncertain, and leaving a 0.2 on it means the
# turn still counts as a no-match downstream.
MIN_CONFIDENCE_FROM_SLOTS = 0.6


def _coerce(data: dict, node: Node, agent: Agent | None = None) -> IntentResult:
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

    if (
        name == "unknown"
        and agent is not None
        and any(slot in agent.all_slots for slot in slots)
        and "provide_details" in set(node.expected_intents) | set(GLOBAL_INTENTS)
    ):
        log.info("Unknown intent carried flow slots %s; reading it as provide_details", sorted(slots))
        name = "provide_details"
        confidence = max(confidence, MIN_CONFIDENCE_FROM_SLOTS)

    language = str(data.get("language") or "en-IN")
    return IntentResult(name=name, confidence=confidence, slots=slots, language=language)
