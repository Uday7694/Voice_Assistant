"""Regressions for an invented department.

A caller said "synus". The tool returned general medicine's three slots labelled
"synus", the agent read them out as real availability, and by the confirmation step the
department had become "surgery" — a department the hospital does not have, offering
times that were never checked.
"""

from __future__ import annotations

import pytest

from brain.tools import ToolOutcome, build_default_registry, departments, resolve_department


# --- resolution ------------------------------------------------------------


@pytest.mark.parametrize(
    "said,expected",
    [
        ("cardiology", "cardiology"),
        ("Cardiology", "cardiology"),
        ("  heart  ", "cardiology"),
        ("bone", "orthopaedics"),
        ("orthopedics", "orthopaedics"),  # American spelling
        ("sinus", "ent"),
        ("synus", "ent"),  # the transcription that started this
        ("throat", "ent"),
        ("physician", "general medicine"),
    ],
)
def test_callers_words_map_onto_real_departments(said, expected):
    """Callers name a body part or a symptom, not a department."""
    assert resolve_department(said) == expected


@pytest.mark.parametrize("said", ["architecture", "surgery", "oncology", "", "   ", "asdf"])
def test_a_department_the_hospital_lacks_resolves_to_nothing(said):
    """None is the important half: it is what stops availability being invented."""
    assert resolve_department(said) is None


def test_every_alias_points_at_a_real_department():
    """The aliases are a column in the Departments sheet: a typo there is a bug here."""
    from brain.hospital import load

    hospital = load()

    for alias, target in hospital.aliases.items():
        assert target in hospital.departments, f"{alias} -> {target} is not a department"


# --- availability ----------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unknown_department_yields_no_slots():
    """The original bug: real-looking times for a department that does not exist."""
    outcome = await build_default_registry().invoke(
        "check_availability", {"department": "architecture"}
    )
    assert outcome.result["error"] == "unknown_department"
    assert "available_slots" not in outcome.result


@pytest.mark.asyncio
async def test_the_caller_is_offered_the_real_list_instead():
    outcome = await build_default_registry().invoke(
        "check_availability", {"department": "surgery"}
    )
    assert outcome.result["known_departments"] == departments()
    assert outcome.result["requested"] == "surgery"


@pytest.mark.asyncio
async def test_a_recognised_alias_returns_that_departments_own_slots():
    outcome = await build_default_registry().invoke(
        "check_availability", {"department": "synus"}
    )
    assert outcome.result["department"] == "ent"
    # Every slot offered belongs to an ENT doctor and exists in the schedule.
    from brain.hospital import load

    ent_doctors = {d.name for d in load().doctors_in("ent")}
    assert outcome.result["available_slots"]
    assert all(s["doctor"] in ent_doctors for s in outcome.result["available_slots"])


# --- booking ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_booking_into_a_department_that_does_not_exist_is_refused():
    """Availability and booking are separate calls; both have to validate."""
    outcome = await build_default_registry().invoke(
        "book_appointment",
        {"patient_name": "Amit", "slot": "tomorrow 10:00 am", "phone": "9900000000",
         "department": "surgery"},
    )
    assert outcome.result["error"] == "unknown_department"
    assert "reference" not in outcome.result


@pytest.mark.asyncio
async def test_a_successful_booking_records_the_canonical_department():
    """A booking made under the caller's word comes back under the hospital's.

    The slot is asked for rather than hardcoded. The workbook is generated with a
    random spread of taken slots, so naming a fixed time meant the test passed or
    failed on the dice: it spent this session red because "tomorrow 11:30 am"
    happened to be booked in that generation, while the tool was behaving perfectly
    and saying so.
    """
    registry = build_default_registry()
    available = await registry.invoke("check_availability", {"department": "synus"})
    offered = available.result.get("available_slots") or available.result.get("slots")
    assert offered, f"no ENT availability to book: {available.result}"

    outcome = await registry.invoke(
        "book_appointment",
        {"patient_name": "Amit", "slot": offered[0], "phone": "9900000000",
         "department": "synus"},
    )
    assert outcome.result.get("booked") is True, outcome.result
    assert outcome.result["department"] == "ent"


# --- slot write-back -------------------------------------------------------


def _brain():
    from brain.orchestrator import Brain

    return Brain.__new__(Brain)  # the helper under test is pure


def _session(**slots):
    from brain.models import Session

    return Session(agent_name="h", node_id="offer_slots").with_slots(slots)


def test_the_tools_answer_replaces_the_callers_wording():
    """Otherwise every later prompt carries "synus" as an established fact."""
    session = _session(department="synus")
    outcome = ToolOutcome(
        name="check_availability", ok=True, result={"department": "ent", "available_slots": []}
    )
    assert _brain()._adopt_canonical_slots(session, outcome).slots["department"] == "ent"


def test_a_failed_tool_call_does_not_rewrite_slots():
    """An unknown_department error carries the rejected wording, not a correction."""
    session = _session(department="synus")
    outcome = ToolOutcome(
        name="check_availability", ok=False, result={"requested": "synus"}
    )
    assert _brain()._adopt_canonical_slots(session, outcome).slots["department"] == "synus"


def test_unrelated_slots_are_left_alone():
    session = _session(department="ent", patient_name="Amit")
    outcome = ToolOutcome(name="check_availability", ok=True, result={"department": "ent"})
    assert _brain()._adopt_canonical_slots(session, outcome) is session
