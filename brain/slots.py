"""Slot validation at capture time.

A caller said "architecture" when asked for a department. It was stored as the
department, survived two more turns while the agent collected a name, and was only
rejected when `check_availability` finally ran — by which point the caller had answered
three questions on the strength of a value the system was always going to refuse.

The rule this module exists to enforce: a slot with a known vocabulary is checked in
the turn it is spoken, not in the turn it is used. A tool result is the last line of
defence, not the first.

Validators are registered by name and referenced from agent config by that name, so a
flow stays a data structure — the vocabulary of a domain lives with that domain's
tools, not in the orchestrator.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SlotVerdict:
    """The outcome of checking one value.

    ``value`` is the canonical form, not the caller's wording: "synus" is accepted and
    stored as "ent", so every later prompt and tool call agrees on one spelling.
    """

    ok: bool
    value: str = ""
    options: tuple[str, ...] = ()
    """What the caller may choose instead. Spoken back to them on a rejection, which is
    the whole point of catching it early — a rejection with no alternatives just makes
    the caller guess again."""

    reason: str = ""


Validator = Callable[[str], SlotVerdict]
ContextualValidator = Callable[[str, dict], SlotVerdict]
"""A validator that also sees what the call already knows.

Most values judge themselves: a department is in the sheet or it is not. Some cannot.
"The first one" is a perfectly good answer to "ten, or half past twelve?" and means
nothing without the list it is choosing from — and the list depends on the doctor,
which is a slot filled two turns earlier. Registered with ``contextual=True`` and
handed the session's slots alongside the value."""

_VALIDATORS: dict[str, Validator | ContextualValidator] = {}
_CONTEXTUAL: set[str] = set()
_VOCABULARY: dict[str, tuple[str, ...]] = {}
_RECOVERY: dict[str, str] = {}
_OPTIONS: dict[str, Callable[[dict[str, str]], Sequence[str]]] = {}

# How a slot may be read out of an utterance the classifier fumbled.
#
#   "anywhere" - run the validator over the whole sentence. Safe only where the value
#                comes from a closed set: "heart" is in the department sheet or it is
#                not, so a match is a fact.
#   "bare"     - only when the utterance is the value and nothing else. For a phone
#                number: "9949210999" is unmistakable, while fishing ten digits out of
#                a sentence is how a booking ends up under the wrong number.
#   ""         - never. A name has neither a vocabulary nor a shape, and "Ramesh" is as
#                likely to be the patient as the doctor.
RECOVER_ANYWHERE = "anywhere"
RECOVER_BARE = "bare"


def register(
    name: str,
    validator: Validator,
    vocabulary: Sequence[str] = (),
    recover_from: str = "",
    options: Callable[[dict[str, str]], Sequence[str]] | None = None,
    contextual: bool = False,
) -> None:
    """Register a validator, and the closed set of values it accepts if it has one.

    The vocabulary is not only for rejecting. The intent classifier is shown it too:
    told a department must be one of four, it stops mapping "architecture" onto the
    nearest-sounding one it can think of — which is what happened, and which no
    downstream check can catch, because a substituted value is indistinguishable from
    something the caller actually said.
    """
    _VALIDATORS[name] = validator
    if contextual:
        _CONTEXTUAL.add(name)
    if vocabulary:
        _VOCABULARY[name] = tuple(vocabulary)
    if recover_from:
        _RECOVERY[name] = recover_from
    if options is not None:
        _OPTIONS[name] = options


def get(name: str) -> Validator | None:
    return _VALIDATORS.get(name)


def vocabulary(name: str) -> tuple[str, ...]:
    """Every value this validator accepts, or empty when the set is not closed."""
    return _VOCABULARY.get(name, ())


def options(name: str, known: dict[str, str] | None = None) -> tuple[str, ...]:
    """The choices for this slot right now, given what the call already knows.

    Some sets are fixed — the departments a hospital has — and some depend on the
    conversation so far: which doctors, once a department is chosen. Both end up in the
    same place, the prompt, because the alternative is a model reciting a list from
    memory. It invented "Dr. Mehta" and "Dr. Rao" for a hospital that employs neither.
    """
    dynamic = _OPTIONS.get(name)
    if dynamic is not None:
        return tuple(dynamic(known or {}))
    return vocabulary(name)


@dataclass(frozen=True)
class Rejection:
    slot: str
    said: str
    options: tuple[str, ...] = field(default=())
    reason: str = ""


def validate(
    rules: Sequence[tuple[str, str]],
    candidate: dict[str, str],
    known: dict[str, str] | None = None,
) -> tuple[dict[str, str], tuple[Rejection, ...]]:
    """Check freshly extracted slots against their rules.

    ``rules`` is (slot name, validator name) pairs. ``known`` is what the call has
    established so far, for the validators registered as contextual. Returns the slots
    that may be kept — in canonical form — and one rejection per value that failed.

    Fails open on an unknown validator name. A typo in agent configuration must not
    silently drop every value for that slot: a caller stuck in a loop that no prompt
    can escape is a far worse failure than an unchecked value reaching a tool, which
    still has its own guard.
    """
    by_slot = {slot: name for slot, name in rules}
    accepted: dict[str, str] = {}
    rejected: list[Rejection] = []

    for slot, said in candidate.items():
        if not said:
            continue
        validator = get(by_slot[slot]) if slot in by_slot else None
        if slot in by_slot and validator is None:
            log.warning("No validator registered for %r; accepting %r unchecked", by_slot[slot], slot)
        if validator is None:
            accepted[slot] = said
            continue

        name = by_slot[slot]
        verdict = validator(said, dict(known or {})) if name in _CONTEXTUAL else validator(said)
        if verdict.ok:
            accepted[slot] = verdict.value or said
        else:
            rejected.append(
                Rejection(slot=slot, said=said, options=verdict.options, reason=verdict.reason)
            )

    return accepted, tuple(rejected)


def recover(
    rules: Sequence[tuple[str, str]], text: str, known: dict[str, str]
) -> dict[str, str]:
    """Read slots straight out of what the caller said, where that is safe.

    A safety net under the classifier, not a replacement for it. Measured: two calls in
    six came back with the right intent and no slots at all for "I need a heart doctor
    appointment", and every one of those turns put the conversation a step out of line.
    The model is not deterministic and a demo cannot depend on it being lucky.

    Only for slots whose validator said how — see the modes above. Everything else is
    left to the classifier, because a wrong slot filled silently is worse than a missing
    one the agent will simply ask for again.
    """
    found: dict[str, str] = {}
    stripped = (text or "").strip()

    for slot, validator_name in rules:
        if slot in known or slot in found:
            continue
        mode = _RECOVERY.get(validator_name) or (
            RECOVER_ANYWHERE if vocabulary(validator_name) else ""
        )
        if not mode:
            continue
        if mode == RECOVER_BARE and re.search(r"[^\W\d_]", stripped, re.UNICODE):
            continue  # there are words in it, so it is a sentence, not a bare value

        validator = get(validator_name)
        verdict = validator(stripped) if validator else None
        if verdict is not None and verdict.ok and verdict.value:
            found[slot] = verdict.value
    return found


def note_for(rejection: Rejection) -> str:
    """The line the planner is told, so it corrects the caller in this same reply.

    Phrased as a fact, then an instruction, then an override. The override earns its
    words: the prompt already carries "Still needed: patient_name, department. Ask for
    the first one only", and against that the model reads a correction as optional
    background and asks for the name instead — measured, not guessed. A note that
    competes with an earlier instruction has to say it wins.

    Told only that the value was wrong, the model also tends to apologise at length or,
    worse, quietly substitute a value it likes better — which is how "architecture"
    once became "surgery".
    """
    said = rejection.said.strip()
    # A reason and a list of options are not alternatives. A slot is refused because it
    # named a day rather than a time, *and* the times are what the caller needs to hear
    # next; run through the options-only branch below, that turn came out as "tomorrow,
    # which does not exist", which is both wrong and baffling.
    if rejection.reason and rejection.options:
        listed = ", ".join(rejection.options)
        return (
            f"The caller's answer for {rejection.slot} ({said!r}) is not usable: "
            f"{rejection.reason}. Correcting this is the ONLY thing you do this turn — "
            "ignore any other instruction above about what to ask for next. In one short "
            f"sentence say so, name what is actually available ({listed}), and ask them "
            "to pick one. Do not choose for them."
        )
    if rejection.options:
        listed = ", ".join(rejection.options)
        return (
            f"The caller said {said!r} for {rejection.slot}, which does not exist. "
            "Correcting this is the ONLY thing you do this turn — ignore any other "
            "instruction above about what to ask for next. In one short sentence say "
            f"it is not one we have, name the real options ({listed}), and ask them to "
            "pick one. Do not choose for them and do not accept the value they gave."
        )
    return (
        f"The caller's {rejection.slot} ({said!r}) is not usable: {rejection.reason}. "
        "Correcting this is the ONLY thing you do this turn — ignore any other "
        "instruction above about what to ask for next. Say so in one short sentence "
        "and ask for it again. Do not guess a value."
    )
