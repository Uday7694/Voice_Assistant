"""Tool registry.

Every tool is timeout-bounded and carries a spoken fallback, because in a voice call a
failed tool still has to produce something sayable. Side-effecting tools are marked so
the flow can require a spoken confirmation before they run.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from . import slots as slot_rules
from .config import TOOL_TIMEOUT
from .hospital import MAX_DOCTORS_OFFERED
from .hospital import _day_from_words as day_from_words
from .hospital import _normalise as normalise
from .hospital import _ordinal_from_words as ordinal_from_words
from .hospital import _time_from_words as time_from_words
from .hospital import MAX_SLOTS_OFFERED
from .hospital import load as load_hospital
from .subagent import DeepSubagent, SubagentBusy

log = logging.getLogger(__name__)

Handler = Callable[..., Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler
    fallback_line: str
    side_effecting: bool = False
    timeout: float = TOOL_TIMEOUT

    def schema(self) -> dict[str, Any]:
        """OpenAI/Groq function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ToolOutcome:
    name: str
    ok: bool
    result: dict[str, Any] = field(default_factory=dict)
    ms: float = 0.0
    fallback_line: str = ""


@dataclass
class ToolRegistry:
    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> tuple[str, ...]:
        """Every registered tool name. Used to check one a model improvised."""
        return tuple(self._tools)

    def schemas_for(self, names: tuple[str, ...]) -> list[dict[str, Any]]:
        return [self._tools[n].schema() for n in names if n in self._tools]

    async def invoke(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        tool = self._tools.get(name)
        if tool is None:
            log.warning("Planner called unknown tool %r", name)
            return ToolOutcome(name=name, ok=False, fallback_line="Let me check that for you.")

        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(tool.handler(**arguments), timeout=tool.timeout)
            ok = True
        except asyncio.TimeoutError:
            log.warning("Tool %s timed out after %.1fs", name, tool.timeout)
            result, ok = {"error": "timeout"}, False
        except Exception as exc:  # noqa: BLE001 - a tool must not kill the call
            log.exception("Tool %s raised", name)
            result, ok = {"error": str(exc)}, False

        elapsed_ms = (time.perf_counter() - started) * 1000
        return ToolOutcome(
            name=name,
            ok=ok,
            result=result,
            ms=elapsed_ms,
            fallback_line="" if ok else tool.fallback_line,
        )


# --- Demo handlers --------------------------------------------------------
# Deterministic stand-ins for a real hospital system. Swap for HTTP calls later;
# the registry contract does not change.

def resolve_department(name: str) -> str | None:
    """Map what the caller said onto a real department, or None.

    Thin wrapper over the workbook so callers - the validator, the tools, the tests -
    do not each have to know where the data lives. The aliases ("heart", "synus",
    "ladies doctor") are a column in the Departments sheet, which is where the people
    who know what callers actually say can edit them.
    """
    return load_hospital().resolve_department(name)


def departments() -> list[str]:
    return sorted(load_hospital().departments)


# Registered so the flow can reject a wrong department in the turn it is spoken rather
# than three questions later when a tool finally sees it.
def _validate_department(said: str) -> slot_rules.SlotVerdict:
    resolved = resolve_department(said)
    if resolved:
        return slot_rules.SlotVerdict(ok=True, value=resolved)
    return slot_rules.SlotVerdict(
        ok=False, options=tuple(departments()), reason="no such department"
    )


def _validate_doctor(said: str) -> slot_rules.SlotVerdict:
    """Accept a doctor the hospital actually employs, in the hospital's spelling.

    Callers say "Ramesh garu" or "Dr Iyer"; the confirmation, the booking row and the
    SMS all have to say the same thing, so the workbook's spelling wins from here on.
    """
    hospital = load_hospital()
    doctor = hospital.resolve_doctor(said)
    if doctor is None:
        return slot_rules.SlotVerdict(ok=False, reason="no doctor by that name here")
    if not doctor.available:
        return slot_rules.SlotVerdict(
            ok=False, reason=f"{doctor.name} is {doctor.notes or 'not taking appointments'}"
        )
    return slot_rules.SlotVerdict(ok=True, value=doctor.name)


PHONE_DIGITS = 10


def _validate_phone(said: str) -> slot_rules.SlotVerdict:
    """Indian mobile numbers: ten digits, and the SMS goes to whatever is stored.

    Stored stripped of spaces, dashes and a +91 prefix, because the caller says it
    every way there is and the confirmation read-back has to be one of them.
    """
    digits = "".join(c for c in said if c.isdigit())
    digits = digits[2:] if len(digits) == 12 and digits.startswith("91") else digits
    if len(digits) == PHONE_DIGITS:
        return slot_rules.SlotVerdict(ok=True, value=digits)
    return slot_rules.SlotVerdict(
        ok=False, reason=f"a mobile number is {PHONE_DIGITS} digits, that was {len(digits)}"
    )


def _validate_slot(said: str, known: dict) -> slot_rules.SlotVerdict:
    """A chosen appointment slot: resolved to a real time, or refused with the options.

    Three answers to "ten, or half past twelve?" and only one of them used to work.

    A time — "half twelve", "మధ్యాహ్నం 12:30" — worked, but was stored in the caller's
    wording, so a time the schedule did not have survived all the way to the booking
    call at the end of the conversation.

    A position — "the first one", "మొదటిది" — did not. It named no time, so match_slot
    fell through to the first free slot in the sheet, which is only the right one by
    luck. It is now read as what it is: an index into the list the caller was just read.

    A day — "book it tomorrow" — is not an answer at all, and was the worst case, because
    it looked like one. The flow moved on to the phone number with no time chosen, and
    the caller heard what time their appointment was for the first time in the SMS.

    Whatever comes in, what is stored is the schedule's own wording, so the read-back at
    the confirm step and the booking at the end cannot say different things.
    """
    hospital = load_hospital()
    # What this caller can actually pick from, which is narrower than the whole
    # schedule by everything they have already told us.
    department = hospital.resolve_department(known.get("department", "")) or ""
    named = hospital.resolve_doctor(known.get("doctor", ""))
    offered = hospital.free_slots(
        department=department, doctor=named.name if named else "", limit=MAX_SLOTS_OFFERED
    )

    def refuse(reason: str) -> slot_rules.SlotVerdict:
        # Never a bare refusal. A caller told only that their answer was no good has
        # nothing to say next, and the times are the whole content of the correction.
        return slot_rules.SlotVerdict(
            ok=False, reason=reason, options=tuple(s.spoken() for s in offered)
        )

    key = normalise(said)
    if not key:
        return refuse("nothing was said")

    # Position before time, because the words overlap and position is the stronger
    # signal. "The first one" contains "one", and read as a clock first it booked
    # somebody one in the afternoon when they had picked ten in the morning.
    position = ordinal_from_words(key)
    if position is not None and offered:
        chosen = offered[-1] if position == -1 else offered[min(position, len(offered)) - 1]
        return slot_rules.SlotVerdict(ok=True, value=chosen.spoken())

    hour, _minute, _half = time_from_words(key)
    if hour is not None:
        chosen = hospital.match_slot(
            said, department=department, doctor=named.name if named else ""
        )
        if chosen is None:
            return refuse("that time is not one of the open ones")
        return slot_rules.SlotVerdict(ok=True, value=chosen.spoken())

    if day_from_words(key):
        return refuse(
            "that is a day, not a time. They have not picked one of the times you "
            "offered yet, so name those times and ask which one"
        )
    return refuse("it does not name a time, so there is nothing to book")


slot_rules.register("slot", _validate_slot, contextual=True)
slot_rules.register("department", _validate_department, vocabulary=departments())
def _doctor_options(known: dict) -> list[str]:
    """Who the caller can choose from, once a department is known.

    Empty before that, deliberately: reading thirteen names to somebody who has not
    said what is wrong with them is worse than asking the department first.
    """
    department = load_hospital().resolve_department(known.get("department", ""))
    if not department:
        return []
    return [d.name for d in load_hospital().doctors_in(department)]


slot_rules.register("doctor", _validate_doctor, options=_doctor_options)
slot_rules.register("phone", _validate_phone, recover_from=slot_rules.RECOVER_BARE)


async def _find_doctors(department: str = "", doctor: str = "") -> dict[str, Any]:
    """Who works here, and what they cost. Reads the Doctors sheet."""
    hospital = load_hospital()

    if doctor:
        found = hospital.resolve_doctor(doctor, hospital.resolve_department(department) or "")
        if found is None:
            return {"error": "unknown_doctor", "requested": doctor}
        return {"doctor": found.spoken()}

    resolved = hospital.resolve_department(department)
    if resolved is None:
        return {
            "error": "unknown_department",
            "requested": department,
            "known_departments": departments(),
        }

    available = hospital.doctors_in(resolved)
    if not available:
        away = [d.name for d in hospital.doctors if d.department == resolved]
        return {"department": resolved, "doctors": [], "none_available": True, "on_leave": away}
    return {
        "department": resolved,
        "doctors": [d.spoken() for d in available[:MAX_DOCTORS_OFFERED]],
    }


async def _check_availability(
    department: str = "", doctor: str = "", preferred_time: str = ""
) -> dict[str, Any]:
    """Open slots from the Slots sheet, soonest first."""
    hospital = load_hospital()
    resolved = hospital.resolve_department(department)
    if department and resolved is None:
        return {
            "error": "unknown_department",
            "requested": department,
            "known_departments": departments(),
        }

    named = hospital.resolve_doctor(doctor, resolved or "") if doctor else None
    if doctor and named is None:
        return {"error": "unknown_doctor", "requested": doctor}
    if named is not None and not named.available:
        return {
            "error": "doctor_unavailable",
            "doctor": named.name,
            "reason": named.notes or "not taking appointments",
            "other_doctors": [d.name for d in hospital.doctors_in(named.department)],
        }

    on = day_from_words(normalise(preferred_time)) if preferred_time else ""
    free = hospital.free_slots(
        department=resolved or "", doctor=named.name if named else "", on=on
    )
    if not free and on:
        # Asked for a day with nothing left on it. Offering the next open day is what a
        # person does; returning an empty list makes the model invent one.
        free = hospital.free_slots(department=resolved or "", doctor=named.name if named else "")

    return {
        "department": resolved or "",
        "doctor": named.name if named else "",
        "available_slots": [
            {"slot_id": s.slot_id, "doctor": s.doctor, "when": s.spoken()} for s in free
        ],
    }


async def _book_appointment(
    patient_name: str = "", department: str = "", doctor: str = "", slot: str = "", phone: str = ""
) -> dict[str, Any]:
    """Take a slot out of the schedule and write the booking."""
    if not (patient_name and slot and phone):
        return {"error": "patient_name, slot and phone are required"}

    hospital = load_hospital()
    resolved = hospital.resolve_department(department)
    if resolved is None:
        return {
            "error": "unknown_department",
            "requested": department,
            "known_departments": departments(),
        }

    named = hospital.resolve_doctor(doctor, resolved) if doctor else None
    chosen = hospital.match_slot(slot, department=resolved, doctor=named.name if named else "")
    if chosen is None:
        # The slot has gone, or was never offered. Saying so, with what is still open,
        # beats confirming a time the schedule does not have.
        return {
            "error": "slot_unavailable",
            "requested": slot,
            # This doctor's other times. Offering the department's next free slot
            # sends the caller to a stranger they did not choose.
            "available_slots": [
                s.spoken()
                for s in hospital.free_slots(
                    department=resolved, doctor=named.name if named else ""
                )
            ],
        }

    hospital.hold(chosen.slot_id)
    seat = named or hospital.resolve_doctor(chosen.doctor)
    reference = f"APT{abs(hash((patient_name, chosen.slot_id))) % 100000:05d}"
    return hospital.book(
        booked=True,
        reference=reference,
        patient_name=patient_name,
        department=chosen.department,
        doctor=chosen.doctor,
        slot=chosen.spoken(),
        date=chosen.day,
        time=chosen.at,
        phone=phone,
        room=seat.room if seat else "",
    )


async def _lookup_appointment(phone: str = "", reference: str = "") -> dict[str, Any]:
    """Find a booking in the Bookings sheet, or one made earlier in this process."""
    if not (phone or reference):
        return {"found": False, "error": "phone or reference is required"}

    row = load_hospital().find_booking(phone=phone, reference=reference)
    return {"found": True, **row} if row else {"found": False}


# --- deep-reasoning subagent ------------------------------------------------------


def _deep_reason_handler(subagent: DeepSubagent) -> Handler:
    """Start a background subagent job and return at once.

    Returning immediately is the whole point. The tool budget is 1.5 s and the subagent
    takes 6-25 s in practice, so waiting here would stall the call. The planner
    gets `status: working`, says a holding line, and the orchestrator delivers the
    answer on a later turn once the job lands.
    """

    async def handler(question: str = "", context: str = "") -> dict[str, Any]:
        question = question.strip()
        if not question:
            return {"error": "question is required"}
        if not subagent.configured:
            return {"error": "subagent unavailable"}

        try:
            job_id = subagent.start(question, context=context)
        except SubagentBusy:
            # Backpressure, not a failure. Saying so beats queueing work that will land
            # long after the caller has moved on.
            log.info("deep_reason refused: subagent queue full")
            return {
                "status": "busy",
                "note": "Say you are still working on the earlier question, then stop.",
            }

        return {
            "status": "working",
            "job_id": job_id,
            "note": "Tell the caller you are checking, then stop. Do not guess the answer.",
        }

    return handler


def deep_reason_tool(subagent: DeepSubagent) -> Tool:
    return Tool(
        name="deep_reason",
        description=(
            "Hand a hard question to a slower expert model: anything needing multi-step "
            "reasoning, long documents, or careful comparison. Returns immediately while "
            "the expert works, so tell the caller you are checking and wait for the answer. "
            "Do not use it for simple lookups or small talk."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The full question, self-contained.",
                },
                "context": {
                    "type": "string",
                    "description": "Any details from the call the expert needs.",
                },
            },
            "required": ["question"],
        },
        handler=_deep_reason_handler(subagent),
        fallback_line="I couldn't look into that just now.",
        # Starting a task is near-instant; this only guards against a wedged event loop.
        timeout=1.0,
    )


def build_default_registry(subagent: DeepSubagent | None = None) -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        Tool(
            name="check_availability",
            description=(
                "Open appointment slots, from the hospital schedule. Filter by "
                "department, by doctor, or by the day the caller asked for. Returns "
                "unknown_department with the real list if the department does not "
                "exist - read that list out instead of guessing. Never state a time "
                "this tool did not return."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "department": {"type": "string", "description": "Department name"},
                    "doctor": {
                        "type": "string",
                        "description": "Doctor's name, if the caller named one",
                    },
                    "preferred_time": {
                        "type": "string",
                        "description": "The day or time the caller asked for, in their words",
                    },
                },
                "required": ["department"],
            },
            handler=_check_availability,
            fallback_line="I couldn't pull up the schedule just now.",
        )
    )

    registry.register(
        Tool(
            name="find_doctors",
            description=(
                "Who works in a department, or the details of one doctor: "
                "qualification, experience, consultation fee, languages, room and OPD "
                "days. Use it when the caller asks who is available, what a visit "
                "costs, or about a doctor by name. Every fact about a doctor comes "
                "from here - never state one from memory."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "department": {"type": "string", "description": "Department name"},
                    "doctor": {"type": "string", "description": "A doctor's name"},
                },
                "required": [],
            },
            handler=_find_doctors,
            fallback_line="I couldn't pull up the doctor list just now.",
        )
    )

    registry.register(
        Tool(
            name="book_appointment",
            description=(
                "Confirm and book a slot. Only call after the caller has said yes to a "
                "specific time. Returns slot_unavailable with what is still open if the "
                "slot has gone in the meantime."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string"},
                    "department": {"type": "string"},
                    "doctor": {"type": "string", "description": "The doctor the slot belongs to"},
                    "slot": {"type": "string", "description": "Exact slot the caller agreed to"},
                    "phone": {"type": "string"},
                },
                # phone is required, not optional: close/ tells the caller an SMS is on
                # its way, and booking without a number makes that a promise the system
                # cannot keep.
                "required": ["patient_name", "slot", "phone"],
            },
            handler=_book_appointment,
            fallback_line="I couldn't complete the booking just now.",
        )
    )

    registry.register(
        Tool(
            name="lookup_appointment",
            description="Find an existing appointment by phone number or reference code.",
            parameters={
                "type": "object",
                "properties": {"phone": {"type": "string"}, "reference": {"type": "string"}},
            },
            handler=_lookup_appointment,
            fallback_line="I couldn't find that booking right now.",
        )
    )

    if subagent is not None and subagent.configured:
        registry.register(deep_reason_tool(subagent))

    return registry
