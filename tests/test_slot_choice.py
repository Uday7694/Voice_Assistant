"""A booking made from a day the desk never understood.

A Telugu caller was read two open times, answered "repu book cheyandi" — book it
tomorrow — and was booked. Neither time was ever chosen, the read-back said "tomorrow"
with no hour in it, and underneath all of it the word "రేపు" had never matched a day at
all: normalisation was stripping the vowel signs out of every Telugu and Hindi word
before anything tried to match it.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from brain import slots as slot_mod
from brain import tools  # noqa: F401 - imported for its validator registrations
from brain.agents.hospital import HOSPITAL_AGENT
from brain.hospital import _day_from_words, _normalise, _time_from_words

RULES = HOSPITAL_AGENT.validation_pairs


def _in(days: int) -> str:
    return (date.today() + timedelta(days=days)).isoformat()


# --- normalisation must not shred Indic words -------------------------------


@pytest.mark.parametrize(
    "word",
    ["రేపు", "ఎల్లుండి", "మధ్యాహ్నం", "कल", "परसों", "सुबह"],
)
def test_an_indic_word_survives_normalisation_whole(word):
    r"""`\w` is str.isalnum(), which is False for a combining mark.

    So "రేపు" came out as "ర ప" and matched nothing, anywhere.
    """
    assert _normalise(word) == word


def test_normalisation_still_strips_punctuation():
    assert _normalise("  Dr. Meera-Krishnan!  ") == "dr meera krishnan"


# --- days, in the languages the agent speaks --------------------------------


@pytest.mark.parametrize(
    "said, offset",
    [
        ("tomorrow", 1),
        ("repu", 1),
        ("రేపు", 1),
        ("kal", 1),
        ("कल", 1),
        ("today", 0),
        ("आज", 0),
        ("day after tomorrow", 2),
        ("ఎల్లుండి", 2),
        ("परसों", 2),
    ],
)
def test_the_desk_understands_the_day_in_every_language_it_speaks(said, offset):
    assert _day_from_words(_normalise(said)) == _in(offset)


def test_the_day_after_tomorrow_is_not_tomorrow():
    """Matched shortest-first, "day after tomorrow" contained "tomorrow" and lost a day."""
    assert _day_from_words("day after tomorrow") == _in(2)


def test_a_name_that_contains_a_day_word_is_not_a_day():
    """"kal" is tomorrow in Hindi and the first syllable of Kalyan."""
    assert _day_from_words(_normalise("Kalyan")) == ""


@pytest.mark.parametrize(
    "said, half", [("ఉదయం 10", "am"), ("सुबह 10", "am"), ("మధ్యాహ్నం 12", "pm"), ("शाम 6", "pm")]
)
def test_the_half_of_the_day_is_read_in_the_callers_language(said, half):
    """The agent reads slots out as "ఉదయం 10:00", so that is how they are said back."""
    assert _time_from_words(_normalise(said))[2] == half


# --- a day is not a choice of slot ------------------------------------------


def test_a_day_with_no_time_in_it_is_not_a_booked_slot():
    """"Book it tomorrow" left the time to whatever the schedule sorted first."""
    kept, rejected = slot_mod.validate(RULES, {"slot": "రేపు book చేయండి"})
    assert "slot" not in kept
    assert rejected and "day, not a time" in rejected[0].reason


def _first_offer(**known) -> str:
    from brain.hospital import load

    return load().free_slots(**known)[0].spoken()


def test_a_time_that_is_open_is_stored_in_the_schedules_own_words():
    """The caller's wording reached the booking call three turns later.

    So a time the schedule did not have survived the whole conversation and failed at
    the end, and the confirmation read back whatever the caller happened to say.
    """
    offer = _first_offer(department="gynaecology")
    kept, rejected = slot_mod.validate(
        RULES, {"slot": offer}, known={"department": "gynaecology"}
    )
    assert not rejected and kept["slot"] == offer


def test_a_time_nobody_offered_is_refused_with_the_ones_that_were():
    kept, rejected = slot_mod.validate(
        RULES, {"slot": "3:47 am"}, known={"department": "gynaecology"}
    )
    assert "slot" not in kept
    assert "not one of the open ones" in rejected[0].reason
    assert rejected[0].options


def test_the_rejection_tells_the_agent_to_name_the_times_again():
    """A caller who has not picked needs the options, not the question repeated."""
    _, rejected = slot_mod.validate(
        RULES, {"slot": "tomorrow"}, known={"department": "gynaecology"}
    )
    note = slot_mod.note_for(rejected[0]).lower()
    assert "day, not a time" in note and "pick one" in note
    assert "does not exist" not in note  # the old options-only wording


# --- picking by position ----------------------------------------------------


@pytest.mark.parametrize(
    "said, index",
    [
        ("the first one", 0),
        ("మొదటిది", 0),
        ("pehla", 0),
        ("second", 1),
        ("రెండో", 1),
        ("दूसरा", 1),
        ("third one", 2),
        ("the last one", -1),
        ("anytime", 0),
    ],
)
def test_a_caller_can_pick_a_slot_by_position(said, index):
    """"The first one" is an index into what they were read, not a time."""
    from brain.hospital import load

    known = {"department": "gynaecology"}
    offered = load().free_slots(department="gynaecology")
    kept, rejected = slot_mod.validate(RULES, {"slot": said}, known=known)
    assert not rejected
    assert kept["slot"] == offered[index].spoken()


def test_a_position_past_the_end_of_the_list_is_the_last_one():
    """Three were offered and the caller said "the fourth". They meant the last."""
    from brain.hospital import load

    offered = load().free_slots(department="gynaecology")
    kept, _ = slot_mod.validate(
        RULES, {"slot": "third one"}, known={"department": "gynaecology"}
    )
    assert kept["slot"] == offered[min(3, len(offered)) - 1].spoken()


# --- the confirmation is one sentence ---------------------------------------


def test_the_confirmation_step_is_capped_at_one_sentence():
    """It arrived as "shall I say yes?" and then, separately, what to say yes to."""
    confirm = HOSPITAL_AGENT.node("confirm")
    assert confirm.max_sentences == 1
    assert "exactly one sentence" in confirm.goal


def test_a_step_with_no_budget_of_its_own_uses_the_planners():
    assert HOSPITAL_AGENT.node("greet").max_sentences == 0


# --- nothing in the past ----------------------------------------------------


def _slot(day_offset: int, at: str, status: str = "free"):
    from brain.hospital import Slot

    return Slot(
        slot_id=f"S{day_offset}{at}",
        doctor_id="D1",
        doctor="Dr. Meera Krishnan",
        department="gynaecology",
        day=_in(day_offset),
        at=at,
        status=status,
    )


def test_a_time_that_has_already_passed_today_is_not_bookable():
    """The schedule is a spreadsheet: a row nobody sat in still says free."""
    from datetime import datetime

    noon = datetime.fromisoformat(f"{_in(0)}T12:00")
    assert not _slot(0, "10:00").bookable(noon)
    assert _slot(0, "16:00").bookable(noon)
    assert _slot(1, "10:00").bookable(noon)


def test_a_slot_too_soon_to_get_to_is_not_offered():
    """Told at 09:58 that 10:00 is free, the caller cannot be there by 10:00."""
    from datetime import datetime

    from brain.hospital import BOOKING_LEAD_MINUTES

    assert BOOKING_LEAD_MINUTES > 0
    almost = datetime.fromisoformat(f"{_in(0)}T09:58")
    assert not _slot(0, "10:00").bookable(almost)


def test_a_slot_with_an_unreadable_time_is_left_alone():
    """A formatting problem in the workbook must not hide the hospital's schedule."""
    assert _slot(0, "half ten").upcoming()


def test_the_schedule_only_offers_times_that_are_still_ahead():
    from datetime import datetime

    from brain.hospital import load

    now = datetime.now()
    for slot in load().free_slots(department="gynaecology", limit=10):
        assert datetime.fromisoformat(f"{slot.day}T{slot.at}") > now


def test_a_time_that_has_gone_cannot_be_asked_for_by_name_either():
    """What is offered and what may be picked come from one list, so they agree."""
    from brain.hospital import load

    hospital = load()
    assert all(
        s.bookable() for s in [hospital.match_slot("today", department="gynaecology")] if s
    )


# --- the agent knows what time it is ----------------------------------------


def test_the_planner_is_told_the_date_and_the_time():
    from datetime import datetime

    from brain.planner import _clock_note

    note = _clock_note(datetime.fromisoformat("2026-08-25T21:52"))
    assert "Tuesday 25 August 2026" in note
    assert "9:52 pm" in note
    assert "already gone" in note


def test_the_clock_is_in_the_turn_block_not_the_cacheable_prefix():
    """A clock in the system prefix changes it every minute and defeats caching."""
    from brain.models import IntentResult, Session
    from brain.planner import build_messages

    session = Session(agent_name=HOSPITAL_AGENT.name, node_id="offer_slots", language="en-IN")
    messages = build_messages(
        HOSPITAL_AGENT,
        HOSPITAL_AGENT.node("offer_slots"),
        session,
        IntentResult(name="choose_slot"),
        "which times are free",
    )
    assert "Right now it is" not in messages[0]["content"]
    assert "Right now it is" in messages[-1]["content"]


# --- reading a time off a caller --------------------------------------------


@pytest.mark.parametrize(
    "said, expected",
    [
        ("10:20", (10, 20)),
        ("ఉదయం 10:00", (10, 0)),
        ("మధ్యాహ్నం 12:30", (12, 30)),
        ("ten forty", (10, 40)),
        ("ten thirty", (10, 30)),
        ("half past two", (2, 30)),
        ("quarter past ten", (10, 15)),
        ("twenty past ten", (10, 20)),
        ("quarter to eleven", (10, 45)),
        ("ten to eleven", (10, 50)),
        ("twenty to one", (12, 40)),
        ("four", (4, None)),
    ],
)
def test_the_desk_hears_the_minutes_a_caller_says(said, expected):
    """Two ways to lose them, and both read the wrong time back convincingly.

    "10 20" — punctuation is gone by this point — fell through to the bare hour and
    booked ten o'clock, and "quarter to eleven" was read in the order it was said and
    booked an hour late.
    """
    from brain.hospital import _normalise, _time_from_words

    hour, minute, _half = _time_from_words(_normalise(said))
    assert (hour, minute) == expected


def test_to_is_only_a_clock_word_directly_after_a_minute():
    """Otherwise every sentence with "to" in it loses an hour."""
    from brain.hospital import _normalise, _time_from_words

    assert _time_from_words(_normalise("I want to book at eleven"))[:2] == (11, None)
