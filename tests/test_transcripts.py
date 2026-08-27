"""What gets written down, and what must never stop a call.

The log exists to be training data later, which sets the bar: a row has to be readable
on its own — who called, in what language, at which step, what they said, what the agent
said back — because whoever trains on it will not have this codebase in front of them.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from brain import transcripts
from brain.models import IntentResult, Role, Session


@pytest.fixture()
def session():
    return Session(
        agent_name="apollo_front_desk",
        node_id="collect_booking",
        language="te-IN",
        metadata={"channel": "talk"},
    ).with_turn(Role.USER, "naku cardiology appointment kavali")


def _lines(directory) -> list[dict]:
    path = directory / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --- the shape of a day's file ---------------------------------------------


def test_a_day_gets_its_own_file(transcripts_in_tmp, session):
    transcripts.call_started(session)
    assert (transcripts_in_tmp / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl").exists()


def test_a_turn_records_both_halves_of_the_exchange(transcripts_in_tmp, session):
    transcripts.turn(
        session,
        user_text="naku cardiology appointment kavali",
        agent_text="రోగి పేరు చెప్పండి.",
        intent=IntentResult(name="book_appointment", confidence=0.95),
    )
    record = _lines(transcripts_in_tmp)[0]
    assert record["user"] == "naku cardiology appointment kavali"
    assert record["agent_said"] == "రోగి పేరు చెప్పండి."
    assert record["intent"] == {"name": "book_appointment", "confidence": 0.95}


def test_every_row_says_who_when_and_where_in_the_call(transcripts_in_tmp, session):
    """A row has to stand on its own: nobody training on this has the codebase."""
    transcripts.turn(session, user_text="hi", agent_text="hello")
    record = _lines(transcripts_in_tmp)[0]
    for field in ("at", "session", "caller", "agent", "channel", "language", "node", "slots"):
        assert field in record, field


def test_indian_scripts_are_written_as_themselves_not_as_escapes(transcripts_in_tmp, session):
    transcripts.turn(session, user_text="మీరు మనిషినా", agent_text="నేను మీరా")
    path = transcripts_in_tmp / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    assert "మీరు మనిషినా" in path.read_text(encoding="utf-8")


def test_tool_calls_and_latency_ride_along_with_the_turn(transcripts_in_tmp, session):
    transcripts.turn(
        session,
        user_text="cardiology",
        agent_text="Tomorrow ten?",
        tools=[{"name": "check_availability", "ok": True, "ms": 3}],
        stage_ms={"intent": 1052.4, "turn_total": 2021.9},
    )
    record = _lines(transcripts_in_tmp)[0]
    assert record["tools"][0]["name"] == "check_availability"
    assert record["latency_ms"]["turn_total"] == 2022


# --- who called ------------------------------------------------------------


def test_a_caller_is_their_number_once_they_have_given_it(transcripts_in_tmp, session):
    """The number is the only identifier that survives a caller ringing back."""
    transcripts.turn(session.with_slots({"phone": "9949210999"}), user_text="x", agent_text="y")
    assert _lines(transcripts_in_tmp)[0]["caller"] == "9949210999"


def test_before_that_the_session_stands_in_for_them(transcripts_in_tmp, session):
    transcripts.turn(session, user_text="x", agent_text="y")
    assert _lines(transcripts_in_tmp)[0]["caller"] == session.session_id


def test_redaction_keeps_same_caller_answerable_and_which_caller_not(
    transcripts_in_tmp, session, monkeypatch
):
    """For any deployment where the log leaves the building."""
    monkeypatch.setattr(transcripts, "REDACT", True)
    with_phone = session.with_slots({"phone": "+91 99492 10999"})
    transcripts.turn(with_phone, user_text="x", agent_text="y")
    transcripts.turn(with_phone, user_text="x", agent_text="y")

    callers = [row["caller"] for row in _lines(transcripts_in_tmp)]
    assert "9949210999" not in callers[0]
    assert callers[0].startswith("sha256:")
    assert callers[0] == callers[1], "the same caller is still recognisable as the same"


# --- endings ---------------------------------------------------------------


def test_the_end_of_a_call_records_how_it_finished(transcripts_in_tmp, session):
    transcripts.call_ended(session, reason="flow reached a terminal step", outcome="completed")
    record = _lines(transcripts_in_tmp)[-1]
    assert record["type"] == "call_end"
    assert record["outcome"] == "completed"
    assert record["turns"] == len(session.history)


def test_an_escalated_call_is_marked_as_one(transcripts_in_tmp, session):
    escalated = session.model_copy(update={"escalated": True, "flagged": True, "flag_reason": "abuse"})
    transcripts.call_ended(escalated, reason="4 abusive turns", outcome="escalated")
    record = _lines(transcripts_in_tmp)[-1]
    assert record["escalated"] is True and record["flag_reason"] == "abuse"


# --- reading it back -------------------------------------------------------


def test_a_day_groups_into_calls(transcripts_in_tmp, session, monkeypatch):
    other = Session(agent_name="apollo_front_desk", node_id="greet")
    transcripts.turn(session, user_text="a", agent_text="b")
    transcripts.turn(other, user_text="c", agent_text="d")
    transcripts.turn(session, user_text="e", agent_text="f")

    grouped = transcripts.calls()
    assert len(grouped[session.session_id]) == 2
    assert len(grouped[other.session_id]) == 1


def test_a_half_written_line_does_not_lose_the_rest_of_the_day(transcripts_in_tmp, session):
    """A process killed mid-append must not make Tuesday unreadable."""
    transcripts.turn(session, user_text="a", agent_text="b")
    path = transcripts_in_tmp / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"type": "turn", "user": "cut off mid-w')

    assert len(transcripts.read()) == 1


def test_reading_a_day_that_never_happened_is_empty_not_an_error():
    assert transcripts.read("1999-01-01") == []


# --- never at the caller's expense ------------------------------------------


def test_a_log_that_cannot_be_written_does_not_end_the_call(transcripts_in_tmp, session, monkeypatch):
    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(transcripts.Path, "mkdir", explode)
    transcripts.turn(session, user_text="x", agent_text="y")  # must not raise


def test_an_odd_value_is_stringified_rather_than_losing_the_row(transcripts_in_tmp, session):
    """Losing a whole exchange over one unprintable field is the worse trade."""
    transcripts.write({"type": "turn", "thing": object()})
    assert "object" in transcripts.read()[0]["thing"]


def test_a_record_that_cannot_be_serialised_at_all_is_dropped_not_raised(transcripts_in_tmp):
    circular: dict = {"type": "turn"}
    circular["self"] = circular
    transcripts.write(circular)  # must not raise
    assert transcripts.read() == []


def test_logging_can_be_turned_off_entirely(transcripts_in_tmp, session, monkeypatch):
    monkeypatch.setattr(transcripts, "ENABLED", False)
    transcripts.turn(session, user_text="x", agent_text="y")
    assert not (transcripts_in_tmp / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl").exists()


# --- the orchestrator actually calls it -------------------------------------


@pytest.mark.asyncio
async def test_a_whole_call_lands_in_the_log(transcripts_in_tmp):
    from brain.agents.hospital import HOSPITAL_AGENT
    from brain.orchestrator import Brain

    class StubLLM:
        provider = type("P", (), {"name": "stub", "planner_model": "m", "fast_model": "m"})()
        fallbacks = ()

        async def json_call(self, *args, **kwargs):
            return {"intent": "book_appointment", "confidence": 0.9, "slots": {}}

        async def stream_chat(self, *args, **kwargs):
            yield {"type": "text", "text": "Which department?"}

        async def aclose(self):
            return None

    brain = Brain(HOSPITAL_AGENT, llm=StubLLM(), deep_reason=False)
    session = brain.start(channel="talk")
    async for _ in brain.handle(session.session_id, "I need an appointment"):
        pass

    records = transcripts.read()
    assert [r["type"] for r in records] == ["call_start", "turn"]
    assert records[1]["user"] == "I need an appointment"
    assert records[1]["agent_said"] == "Which department?"
    assert records[1]["channel"] == "talk"
