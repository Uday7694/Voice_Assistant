"""Turns that must be read the same way every time.

Every case here is drawn from one live Hindi call that went wrong: the caller was
understood word for word and still ended up transferred to a person. Each test pins one
link in that chain, in the deterministic layer, so the answer no longer depends on which
provider happened to reply.
"""

from __future__ import annotations

from brain.agents import HOSPITAL_AGENT
from brain.flow import next_node
from brain.intent import _coerce
from brain.models import IntentResult, Session
from brain.orchestrator import needs_scope_line


def _node(node_id: str):
    return HOSPITAL_AGENT.node(node_id)


# --- a label the model could not choose, from a value it clearly heard ------


def test_unknown_carrying_a_flow_slot_is_read_as_provide_details():
    payload = {"intent": "unknown", "confidence": 0.2, "slots": {"patient_name": "amit ji"}}
    result = _coerce(payload, _node("greet"), HOSPITAL_AGENT)
    assert result.name == "provide_details"
    assert result.slots["patient_name"] == "amit ji"


def test_promoted_intent_is_confident_enough_to_count_as_a_match():
    payload = {"intent": "unknown", "confidence": 0.0, "slots": {"department": "cardiology"}}
    assert _coerce(payload, _node("greet"), HOSPITAL_AGENT).is_confident


def test_unknown_without_any_flow_slot_stays_unknown():
    payload = {"intent": "unknown", "confidence": 0.2, "slots": {"mood": "annoyed"}}
    assert _coerce(payload, _node("greet"), HOSPITAL_AGENT).name == "unknown"


def test_a_named_intent_is_never_overwritten_by_its_slots():
    payload = {"intent": "out_of_scope", "confidence": 0.9, "slots": {"patient_name": "amit"}}
    assert _coerce(payload, _node("greet"), HOSPITAL_AGENT).name == "out_of_scope"


# --- the scope line belongs to the turn it describes ------------------------


def test_a_confident_request_is_answered_even_after_two_missed_turns():
    booking = IntentResult(name="book_appointment", confidence=0.9)
    assert not needs_scope_line(booking, no_match_streak=2)


def test_an_off_topic_turn_still_hears_what_the_desk_is_for():
    assert needs_scope_line(IntentResult(name="out_of_scope", confidence=0.9), 0)


def test_a_third_unreadable_turn_still_hears_what_the_desk_is_for():
    assert needs_scope_line(IntentResult(name="unknown", confidence=0.1), 2)


# --- a turn budget must not outrank what the caller just said ---------------


def test_a_clear_request_on_the_last_turn_is_acted_on_not_escalated():
    """The live failure: understood perfectly, transferred anyway."""
    session = Session(agent_name=HOSPITAL_AGENT.name, node_id="greet", turns_in_node=9)
    booking = IntentResult(name="book_appointment", confidence=0.9)
    decision = next_node(HOSPITAL_AGENT, session, booking)
    assert not decision.force_escalate
    assert decision.node_id == "collect_booking"


def test_a_spent_budget_still_escalates_when_the_turn_leads_nowhere():
    session = Session(agent_name=HOSPITAL_AGENT.name, node_id="greet", turns_in_node=9)
    decision = next_node(HOSPITAL_AGENT, session, IntentResult(name="unknown", confidence=0.1))
    assert decision.force_escalate
    assert decision.reason == "stuck in greet"


def test_an_unconfident_match_does_not_buy_another_turn():
    """is_confident already gates transitions; the budget must still apply below it."""
    session = Session(agent_name=HOSPITAL_AGENT.name, node_id="greet", turns_in_node=9)
    unsure = IntentResult(name="book_appointment", confidence=0.2)
    assert next_node(HOSPITAL_AGENT, session, unsure).force_escalate


# --- what the classifier is told -------------------------------------------


def test_the_classifier_prompt_carries_the_purpose_not_the_whole_persona():
    """Every character is spent on every turn, against a token-per-minute ceiling."""
    from brain.intent import _prompt_for

    session = Session(agent_name=HOSPITAL_AGENT.name, node_id="collect_booking")
    context = _prompt_for(_node("collect_booking"), HOSPITAL_AGENT, session, "Amit")[1]["content"]
    assert HOSPITAL_AGENT.purpose in context
    assert "warm, brisk, and respectful" not in context


def test_the_prompt_still_names_the_slots_and_their_allowed_values():
    from brain.intent import _prompt_for

    session = Session(agent_name=HOSPITAL_AGENT.name, node_id="collect_booking")
    context = _prompt_for(_node("collect_booking"), HOSPITAL_AGENT, session, "Amit")[1]["content"]
    assert "patient_name" in context
    assert "cardiology" in context


def test_the_prompt_stays_inside_the_rate_limited_budget():
    """Groq's tier allows 8000 tokens/minute; a turn that costs more throttles the call."""
    from brain.intent import _SYSTEM, _prompt_for

    session = Session(agent_name=HOSPITAL_AGENT.name, node_id="collect_booking")
    messages = _prompt_for(_node("collect_booking"), HOSPITAL_AGENT, session, "Amit")
    assert sum(len(m["content"]) for m in messages) < 3600
    assert "SLOTS ARE THE POINT" in _SYSTEM
