"""The agent reads a schedule it did not invent.

Every doctor name, fee, room and time the agent says out loud comes from a row in
``data/hospital.xlsx``. These tests pin the parts of that contract a caller would
notice: that a real name resolves however it is spoken, that a time offered is a time
that exists, and that nothing is offered twice.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from brain.hospital import load
from brain.tools import build_default_registry, departments, resolve_department


@pytest.fixture()
def hospital():
    return load()


# --- the workbook itself ---------------------------------------------------


def test_the_workbook_has_every_field_a_receptionist_would_be_asked_for(hospital):
    doctor = hospital.doctors[0]
    for field in (
        "qualification", "designation", "experience_years", "consultation_fee",
        "languages", "room", "opd_days", "opd_start", "opd_end", "slot_minutes", "status",
    ):
        assert getattr(doctor, field) not in (None, ""), field


def test_every_doctor_belongs_to_a_department_that_exists(hospital):
    for doctor in hospital.doctors:
        assert doctor.department in hospital.departments, doctor.name


def test_every_slot_belongs_to_a_doctor_who_works_here(hospital):
    names = {d.name for d in hospital.doctors}
    for slot in hospital.slots[:200]:
        assert slot.doctor in names


def test_the_schedule_starts_today_and_is_not_all_free(hospital):
    """A schedule where everything is open never makes the agent say "that one has gone"."""
    assert hospital.slots
    assert min(s.day for s in hospital.slots) == date.today().isoformat()
    assert any(not s.free for s in hospital.slots)


# --- saying it out loud ----------------------------------------------------


def test_slots_are_spoken_the_way_a_person_says_them(hospital):
    spoken = hospital.free_slots(department="cardiology")[0].spoken()
    assert ":" in spoken and ("am" in spoken or "pm" in spoken)
    assert "T" not in spoken, "an ISO timestamp is not a spoken time"


def test_todays_slots_are_called_today(hospital):
    today = [s for s in hospital.slots if s.day == date.today().isoformat()]
    assert today and today[0].spoken().startswith("today")


# --- finding a doctor by whatever the caller said --------------------------


@pytest.mark.parametrize(
    "said,expected",
    [
        ("Dr. Ramesh Iyer", "Dr. Ramesh Iyer"),
        ("ramesh", "Dr. Ramesh Iyer"),
        ("Dr Iyer", "Dr. Ramesh Iyer"),
        ("doctor menon", "Dr. Kavya Menon"),
        ("sneha", "Dr. Sneha Reddy"),
    ],
)
def test_a_doctor_is_found_however_the_caller_says_the_name(hospital, said, expected):
    found = hospital.resolve_doctor(said)
    assert found is not None and found.name == expected


def test_a_doctor_who_does_not_work_here_is_not_invented(hospital):
    assert hospital.resolve_doctor("Dr. House") is None


def test_two_departments_can_each_have_a_rao(hospital):
    """The department narrows the search, so a shared surname does not pick the wrong one."""
    found = hospital.resolve_doctor("rao", department="ent")
    assert found is not None and found.department == "ent"


# --- what the caller means by a time ---------------------------------------


def test_tomorrow_ten_finds_a_slot_tomorrow_at_ten(hospital):
    slot = hospital.match_slot("tomorrow ten", department="cardiology")
    assert slot is not None
    assert slot.day != date.today().isoformat()
    assert slot.at.startswith("10")


def test_a_time_nobody_offers_is_not_matched(hospital):
    assert hospital.match_slot("tomorrow at 3 in the morning", department="cardiology") is None


# --- the tools -------------------------------------------------------------


@pytest.mark.asyncio
async def test_availability_only_returns_slots_that_exist():
    registry = build_default_registry()
    outcome = await registry.invoke("check_availability", {"department": "heart"})
    offered = {s["slot_id"] for s in outcome.result["available_slots"]}
    free = {s.slot_id for s in load().free_slots(department="cardiology", limit=99)}
    assert offered and offered <= free


@pytest.mark.asyncio
async def test_a_doctor_on_leave_is_not_offered_but_is_explained():
    registry = build_default_registry()
    outcome = await registry.invoke("find_doctors", {"department": "dermatology"})
    assert outcome.result["none_available"] is True
    assert outcome.result["on_leave"], "say who is away rather than that nobody works here"


@pytest.mark.asyncio
async def test_the_doctor_details_a_caller_asks_about_come_from_the_sheet():
    registry = build_default_registry()
    outcome = await registry.invoke("find_doctors", {"doctor": "Ramesh"})
    doctor = outcome.result["doctor"]
    assert doctor["consultation_fee"] == 800
    assert "Telugu" in doctor["languages"]
    assert doctor["room"] == "3A-12"


@pytest.mark.asyncio
async def test_booking_takes_the_slot_out_of_the_schedule():
    """The same time cannot be sold twice, which is the whole reason to hold it."""
    registry = build_default_registry()
    slot = load().free_slots(department="ent", limit=1)[0]

    booked = await registry.invoke(
        "book_appointment",
        {
            "patient_name": "Asha",
            "department": "ent",
            "doctor": slot.doctor,
            "slot": slot.spoken(),
            "phone": "9900000000",
        },
    )
    assert booked.result["booked"] is True
    assert booked.result["doctor"] == slot.doctor
    assert booked.result["room"], "the caller is told which room to go to"
    assert slot.slot_id not in {s.slot_id for s in load().free_slots(department="ent", limit=99)}


@pytest.mark.asyncio
async def test_a_slot_that_has_gone_is_refused_with_what_is_left():
    registry = build_default_registry()
    outcome = await registry.invoke(
        "book_appointment",
        {
            "patient_name": "Asha",
            "department": "cardiology",
            "slot": "next Tuesday at 4 in the morning",
            "phone": "9900000000",
        },
    )
    assert outcome.result["error"] == "slot_unavailable"
    assert outcome.result["available_slots"], "offer what is open rather than just refusing"


@pytest.mark.asyncio
async def test_an_existing_booking_is_found_by_phone_number():
    registry = build_default_registry()
    outcome = await registry.invoke("lookup_appointment", {"phone": "98480 12345"})
    assert outcome.result["found"] is True
    assert outcome.result["patient_name"] == "Ravi Kumar"


# --- the flow knows about the data -----------------------------------------


def test_the_department_vocabulary_comes_from_the_workbook():
    """Add a department to the sheet and the classifier is told about it."""
    from brain import slots as slot_mod

    assert set(slot_mod.vocabulary("department")) == set(departments())
    assert resolve_department("ladies doctor") == "gynaecology"


def test_a_doctor_the_caller_names_is_validated_at_capture():
    from brain import slots as slot_mod

    kept, rejected = slot_mod.validate((("doctor", "doctor"),), {"doctor": "ramesh"})
    assert kept["doctor"] == "Dr. Ramesh Iyer", "stored in the hospital's spelling"

    kept, rejected = slot_mod.validate((("doctor", "doctor"),), {"doctor": "Dr. House"})
    assert not kept and rejected[0].slot == "doctor"


def test_a_doctor_on_leave_is_refused_at_capture_with_the_reason():
    from brain import slots as slot_mod

    _, rejected = slot_mod.validate((("doctor", "doctor"),), {"doctor": "Nikhil Verma"})
    assert rejected and "leave" in rejected[0].reason


def test_the_flow_asks_who_before_it_asks_when():
    from brain.agents.hospital import HOSPITAL_AGENT

    node = HOSPITAL_AGENT.node("choose_doctor")
    assert node.required_slots == ("doctor",)
    assert "find_doctors" in node.allowed_tools
    assert any(t.to == "offer_slots" for t in node.transitions)


def test_the_tool_schemas_stay_json_serialisable():
    """They go into a prompt; a schema that cannot be serialised fails the whole turn."""
    registry = build_default_registry()
    for name in ("check_availability", "find_doctors", "book_appointment"):
        json.dumps(registry.get(name).schema())


# --- tool calls the model wrote as prose ------------------------------------


def test_a_tool_call_written_as_text_is_recovered_and_run():
    """Dropping it was the safety fix; running it is the one that books the appointment.

    Sarvam writes these often enough that discarding the turn cost real bookings: the
    caller gives a phone number, the model means to call book_appointment, and what
    comes back is pseudo-XML.
    """
    from brain.planner import _parse_text_tool_call

    registry = build_default_registry()
    call = _parse_text_tool_call(
        "<tool_call>check_availability\n"
        "<arg_key>department</arg_key>\n<arg_value>cardiology</arg_value>\n</tool_call>",
        registry,
    )
    assert call == {"name": "check_availability", "arguments": {"department": "cardiology"}}


def test_the_bracketless_variant_is_recovered_too():
    from brain.planner import _parse_text_tool_call

    call = _parse_text_tool_call(
        "<toolcall>book_appointment<argkey>patient_name</argkey><argvalue>Uday</argvalue>",
        build_default_registry(),
    )
    assert call["name"] == "book_appointment"
    assert call["arguments"]["patient_name"] == "Uday"


def test_a_tool_the_registry_does_not_have_is_not_guessed_at():
    """A model that invents a tool name has invented the arguments too."""
    from brain.planner import _parse_text_tool_call

    registry = build_default_registry()
    assert _parse_text_tool_call('{"name": "totally_made_up", "arguments": {"x": "1"}}', registry) is None
    # Ambiguous near-misses are dropped as well: "check_appointment" could be any of
    # three registered tools, and running the wrong one is worse than saying nothing.
    assert _parse_text_tool_call('{"name": "check_appointment", "arguments": {"phone": "1"}}', registry) is None


def test_ordinary_speech_is_not_mistaken_for_a_tool_call():
    from brain.planner import _parse_text_tool_call

    assert _parse_text_tool_call("Which department would you like?", build_default_registry()) is None


# --- stability: the flow does not depend on the model being lucky -----------


def test_a_department_the_classifier_missed_is_read_out_of_the_utterance():
    """Two calls in six came back with the right intent and no slots at all.

    The word "heart" is in the department sheet, so the flow can read it without asking
    the model twice.
    """
    from brain import slots as slot_mod
    from brain.agents.hospital import HOSPITAL_AGENT

    found = slot_mod.recover(HOSPITAL_AGENT.validation_pairs, "I need a heart doctor", {})
    assert found == {"department": "cardiology"}


def test_a_bare_phone_number_is_recovered_but_one_inside_a_sentence_is_not():
    """Fishing ten digits out of a sentence is how a booking gets the wrong number."""
    from brain import slots as slot_mod
    from brain.agents.hospital import HOSPITAL_AGENT

    rules = HOSPITAL_AGENT.validation_pairs
    assert slot_mod.recover(rules, "9949210999", {}) == {"phone": "9949210999"}
    assert slot_mod.recover(rules, "my number is 9949210999 ok", {}) == {}


def test_a_name_is_never_recovered_by_shape():
    """"Ramesh" is as likely to be the patient as the doctor."""
    from brain import slots as slot_mod
    from brain.agents.hospital import HOSPITAL_AGENT

    assert slot_mod.recover(HOSPITAL_AGENT.validation_pairs, "Ramesh", {}) == {}


def test_a_slot_already_known_is_not_overwritten():
    from brain import slots as slot_mod
    from brain.agents.hospital import HOSPITAL_AGENT

    found = slot_mod.recover(
        HOSPITAL_AGENT.validation_pairs, "I need a heart doctor", {"department": "ent"}
    )
    assert found == {}


def test_the_doctor_choices_depend_on_the_department_already_chosen():
    """It invented "Dr. Mehta" and "Dr. Rao" for a hospital that employs neither."""
    from brain import slots as slot_mod

    assert slot_mod.options("doctor", {"department": "cardiology"}) == (
        "Dr. Ramesh Iyer",
        "Dr. Sneha Reddy",
    )
    # Before a department is known, reading thirteen names to somebody who has not said
    # what is wrong with them is worse than asking the department first.
    assert slot_mod.options("doctor", {}) == ()


def test_the_planner_is_given_the_real_names_rather_than_asked_to_recall_them():
    from brain.agents.hospital import HOSPITAL_AGENT
    from brain.models import IntentResult, Session
    from brain.planner import build_messages

    messages = build_messages(
        HOSPITAL_AGENT,
        HOSPITAL_AGENT.node("choose_doctor"),
        Session(node_id="choose_doctor", slots={"department": "cardiology"}),
        IntentResult(name="provide_details", confidence=0.9),
        "who is there",
    )
    body = messages[-1]["content"]
    assert "Dr. Ramesh Iyer" in body and "never invent one" in body


def test_the_turns_instructions_are_the_last_thing_the_model_reads():
    """Sat above the history they lost to whatever the model felt like asking."""
    from brain.agents.hospital import HOSPITAL_AGENT
    from brain.models import IntentResult, Session
    from brain.planner import build_messages

    messages = build_messages(
        HOSPITAL_AGENT,
        HOSPITAL_AGENT.node("collect_booking"),
        Session(node_id="collect_booking", slots={"department": "cardiology"}),
        IntentResult(name="provide_details", confidence=0.9),
        "hello",
    )
    assert messages[-1]["role"] == "system"
    assert "Ask only for the patient name" in messages[-1]["content"]


@pytest.mark.asyncio
async def test_a_taken_slot_offers_that_doctors_other_times():
    """Offering the department's next free slot sends the caller to a stranger."""
    registry = build_default_registry()
    outcome = await registry.invoke(
        "book_appointment",
        {
            "patient_name": "Asha",
            "department": "cardiology",
            "doctor": "Dr. Sneha Reddy",
            "slot": "tomorrow at 4 in the morning",
            "phone": "9900000000",
        },
    )
    assert outcome.result["error"] == "slot_unavailable"
    hers = {s.spoken() for s in load().free_slots(doctor="Dr. Sneha Reddy", limit=99)}
    assert set(outcome.result["available_slots"]) <= hers


# --- a name said slightly wrong ---------------------------------------------


@pytest.mark.parametrize(
    "said,expected",
    [
        ("meeraa", "Dr. Meera Krishnan"),
        ("zoyaa", "Dr. Zoya Sheikh"),
        ("krishnan", "Dr. Meera Krishnan"),
    ],
)
def test_a_misspelt_name_still_finds_the_doctor(hospital, said, expected):
    """A caller offered two doctors typed "meeraa" and was asked to choose again, twice.

    An exact match is the wrong bar for a name a person is repeating back over a phone.
    """
    found = hospital.resolve_doctor(said, "gynaecology")
    assert found is not None and found.name == expected


def test_a_close_name_is_not_accepted_when_two_doctors_are_equally_close(hospital):
    """Accepting the wrong doctor is worse than asking again."""
    assert hospital.resolve_doctor("kavitha", "gynaecology") is None


@pytest.mark.parametrize("said,expected", [("suresh", "Dr. Suresh Babu"), ("ramesh", "Dr. Ramesh Iyer")])
def test_the_two_closest_real_names_are_still_told_apart(hospital, said, expected):
    """Ramesh and Suresh score 0.67 against each other; the floor sits above that."""
    found = hospital.resolve_doctor(said)
    assert found is not None and found.name == expected


def test_a_name_nobody_here_has_is_still_refused(hospital):
    assert hospital.resolve_doctor("Dr. Nobody", "gynaecology") is None


def test_the_second_unusable_answer_changes_the_question():
    """Re-reading a list somebody has already failed to pick from helps nobody."""
    from brain.orchestrator import _STUCK_ON

    note = _STUCK_ON.format(slot="doctor")
    assert "numbered choice" in note and "same words" in note
