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

from .config import TOOL_TIMEOUT
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

_DEMO_SLOTS = {
    "cardiology": ["tomorrow 10:00 am", "tomorrow 3:30 pm", "Friday 11:00 am"],
    "orthopaedics": ["today 6:00 pm", "tomorrow 9:15 am"],
    "ent": ["today 4:00 pm", "tomorrow 11:30 am"],
    "general medicine": ["today 5:00 pm", "tomorrow 8:30 am", "tomorrow 12:00 pm"],
}

# What callers actually say, mapped to what the hospital calls it. Callers describe a
# body part or a symptom, not a department, and speech recognition mangles both.
_DEPARTMENT_ALIASES = {
    "heart": "cardiology", "cardiac": "cardiology", "cardio": "cardiology",
    "bone": "orthopaedics", "bones": "orthopaedics", "joint": "orthopaedics",
    "ortho": "orthopaedics", "orthopedics": "orthopaedics", "knee": "orthopaedics",
    "sinus": "ent", "synus": "ent", "sinuses": "ent", "ear": "ent", "nose": "ent",
    "throat": "ent", "e n t": "ent", "ent": "ent",
    "general": "general medicine", "physician": "general medicine",
    "fever": "general medicine", "gp": "general medicine",
}


def resolve_department(name: str) -> str | None:
    """Map what the caller said onto a real department, or None.

    None is the important half. The old behaviour handed back general medicine's slots
    under whatever name the caller used, so "synus" produced three real-looking times
    for a department that does not exist — and the model, having been given a
    department it did not recognise, renamed it "surgery" on the way out. Inventing
    availability is worse than admitting the department is unknown.
    """
    key = " ".join(name.strip().lower().split())
    if not key:
        return None
    if key in _DEMO_SLOTS:
        return key
    return _DEPARTMENT_ALIASES.get(key)


async def _check_availability(department: str = "", preferred_time: str = "") -> dict[str, Any]:
    resolved = resolve_department(department)
    if resolved is None:
        return {
            "error": "unknown_department",
            "requested": department,
            "known_departments": sorted(_DEMO_SLOTS),
        }
    return {"department": resolved, "available_slots": _DEMO_SLOTS[resolved]}


async def _book_appointment(
    patient_name: str = "", department: str = "", slot: str = "", phone: str = ""
) -> dict[str, Any]:
    if not (patient_name and slot and phone):
        return {"error": "patient_name, slot and phone are required"}

    # Validate here too. Availability and booking are separate calls, and a caller who
    # changed department in between would otherwise be booked into one that does not
    # exist.
    resolved = resolve_department(department)
    if resolved is None:
        return {
            "error": "unknown_department",
            "requested": department,
            "known_departments": sorted(_DEMO_SLOTS),
        }

    reference = f"APT{abs(hash((patient_name, slot))) % 100000:05d}"
    return {
        "booked": True,
        "reference": reference,
        "patient_name": patient_name,
        "department": resolved,
        "slot": slot,
        "phone": phone,
    }


async def _lookup_appointment(phone: str = "", reference: str = "") -> dict[str, Any]:
    if not (phone or reference):
        return {"found": False}
    return {
        "found": True,
        "reference": reference or "APT41822",
        "department": "cardiology",
        "slot": "tomorrow 10:00 am",
    }


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
                "List open appointment slots for a hospital department. Returns "
                "unknown_department with the real list if the department does not "
                "exist — read that list to the caller instead of guessing."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "department": {"type": "string", "description": "Department name"},
                    "preferred_time": {"type": "string", "description": "Caller's preference"},
                },
                "required": ["department"],
            },
            handler=_check_availability,
            fallback_line="I couldn't pull up the schedule just now.",
        )
    )

    registry.register(
        Tool(
            name="book_appointment",
            description="Confirm and book an appointment slot. Only call after the caller has said yes.",
            parameters={
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string"},
                    "department": {"type": "string"},
                    "slot": {"type": "string", "description": "Exact slot the caller agreed to"},
                    "phone": {"type": "string"},
                },
                # phone is required, not optional: close/ tells the caller an SMS is on
                # its way, and booking without a number makes that a promise the system
                # cannot keep.
                "required": ["patient_name", "slot", "phone"],
            },
            handler=_book_appointment,
            fallback_line="I wasn't able to confirm that booking.",
            side_effecting=True,
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
