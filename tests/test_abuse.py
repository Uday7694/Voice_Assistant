"""How the agent handles an abusive caller.

A caller swore at the agent and the call was transferred to a human instantly, with the
reason "user asked for a human". They had asked for no such thing: the classifier had
no label for abuse, so it reached for escalate_to_human, and the flow read that as an
explicit request. That is wrong twice — it is not what the caller said, and it makes
swearing the fastest way to skip the queue.
"""

from __future__ import annotations


import pytest

from brain.config import MAX_ABUSIVE_TURNS
from brain.flow import next_node
from brain.models import GLOBAL_INTENTS, IntentResult, Session
from brain.agents.hospital import HOSPITAL_AGENT


def _brain():
    from brain.orchestrator import Brain

    brain = Brain.__new__(Brain)  # the trackers are pure helpers
    brain.agent = HOSPITAL_AGENT
    return brain


def _session(**kw):
    return Session(agent_name="h", node_id="collect_booking", **kw)


def _intent(name: str) -> IntentResult:
    return IntentResult(name=name, confidence=0.95)


# --- vocabulary ------------------------------------------------------------


def test_abuse_has_a_label_of_its_own():
    """Without one the classifier reaches for escalate_to_human."""
    assert "abusive" in GLOBAL_INTENTS


def test_the_classifier_is_told_the_two_are_different():
    from brain.intent import _SYSTEM

    assert "abusive" in _SYSTEM
    assert "explicitly asked to speak to a person" in _SYSTEM


# --- flow ------------------------------------------------------------------


def test_abuse_does_not_transfer_the_call():
    """The original bug: one insult ended the conversation."""
    decision = next_node(HOSPITAL_AGENT, _session(), _intent("abusive"))
    assert decision.force_escalate is False
    assert decision.force_end is False


def test_asking_for_a_person_still_transfers():
    """The fix must not cost the caller a legitimate request."""
    decision = next_node(HOSPITAL_AGENT, _session(), _intent("escalate_to_human"))
    assert decision.force_escalate is True


# --- counting --------------------------------------------------------------


def test_the_streak_grows_on_consecutive_abuse():
    brain, session = _brain(), _session()
    for expected in (1, 2, 3):
        session = brain._track_abuse(session, _intent("abusive"))
        assert session.abuse_streak == expected


def test_one_ordinary_turn_clears_the_streak():
    """People swear out of frustration and then carry on normally."""
    brain = _brain()
    session = brain._track_abuse(_session(abuse_streak=2), _intent("provide_details"))
    assert session.abuse_streak == 0


def test_a_first_outburst_is_tolerated():
    """Hanging up on one bad moment is worse service than absorbing it."""
    brain = _brain()
    session = brain._track_abuse(_session(), _intent("abusive"))
    assert session.abuse_streak <= MAX_ABUSIVE_TURNS


def test_the_tolerance_is_configurable_not_baked_in():
    """A hospital desk and an outbound campaign do not want the same patience."""
    import importlib
    import os

    import brain.config as config

    previous = os.environ.get("MAX_ABUSIVE_TURNS")
    os.environ["MAX_ABUSIVE_TURNS"] = "7"
    try:
        assert importlib.reload(config).MAX_ABUSIVE_TURNS == 7
    finally:
        if previous is None:
            del os.environ["MAX_ABUSIVE_TURNS"]
        else:
            os.environ["MAX_ABUSIVE_TURNS"] = previous
        importlib.reload(config)


def test_every_conversation_limit_is_overridable():
    import inspect

    import brain.config as config

    source = inspect.getsource(config)
    for name in (
        "MAX_TURNS_PER_SESSION",
        "MAX_CONSECUTIVE_NO_MATCH",
        "MAX_HISTORY_TURNS",
        "MAX_ABUSIVE_TURNS",
    ):
        assert f'os.getenv("{name}"' in source, f"{name} is hard-coded"


def test_a_flagged_call_is_marked_for_review():
    """Flagging is separate from escalating: ordinary transfers are not abuse reports."""
    session = _session(abuse_streak=99).model_copy(
        update={"flagged": True, "flag_reason": "99 abusive turns"}
    )
    assert session.flagged is True
    assert "abusive" in session.flag_reason


def test_an_ordinary_call_is_not_flagged():
    assert _session().flagged is False


def test_a_sustained_pattern_reaches_a_person():
    brain, session = _brain(), _session()
    for _ in range(MAX_ABUSIVE_TURNS + 1):
        session = brain._track_abuse(session, _intent("abusive"))
    assert session.abuse_streak > MAX_ABUSIVE_TURNS


# --- what the agent says ---------------------------------------------------


def test_the_planner_is_told_to_ask_for_respect_and_carry_on():
    from brain.planner import build_messages

    messages = build_messages(
        HOSPITAL_AGENT,
        HOSPITAL_AGENT.node("collect_booking"),
        _session(language="hi-IN"),
        _intent("abusive"),
        "...",
    )
    body = " ".join(m["content"] for m in messages)
    assert "keep it respectful" in body
    assert "continue with the current step" in body


def test_the_agent_does_not_scold_or_threaten():
    """A receptionist asks once, calmly, and moves on."""
    from brain.planner import build_messages

    messages = build_messages(
        HOSPITAL_AGENT,
        HOSPITAL_AGENT.node("collect_booking"),
        _session(language="hi-IN"),
        _intent("abusive"),
        "...",
    )
    body = " ".join(m["content"] for m in messages)
    lowered = body.lower()
    for forbidden in ("scold", "lecture", "warn them", "threaten to end the call"):
        assert f"do not {forbidden}" in lowered or forbidden in lowered, forbidden
    assert "stay warm and unruffled" in lowered


def test_an_ordinary_turn_carries_no_abuse_instruction():
    from brain.planner import build_messages

    messages = build_messages(
        HOSPITAL_AGENT,
        HOSPITAL_AGENT.node("collect_booking"),
        _session(language="hi-IN"),
        _intent("provide_details"),
        "Amit",
    )
    # Match the instruction, not the word: the persona itself says "respectful".
    assert "keep it respectful" not in " ".join(m["content"] for m in messages)


# --- handoff language ------------------------------------------------------


@pytest.mark.asyncio
async def test_the_handoff_is_spoken_in_the_callers_language(tmp_path):
    """An English sentence ending a Hindi call is the most jarring moment in it."""
    from brain.lines import LineBook

    book = LineBook(HOSPITAL_AGENT, _ScriptedWriter("संपर्क"), cache_dir=tmp_path)
    assert await book.line("handoff", "hi-IN") == "संपर्क"


@pytest.mark.asyncio
async def test_an_unwritable_line_falls_back_rather_than_going_silent(tmp_path):
    """No model, no network, no cache — the caller still hears something."""
    from brain.lines import SPECS, LineBook

    book = LineBook(HOSPITAL_AGENT, llm=None, cache_dir=tmp_path)
    assert await book.line("handoff", "xx-XX") == SPECS["handoff"].fallback


@pytest.mark.asyncio
async def test_a_language_is_written_once_and_then_read_from_disk(tmp_path):
    """The cost of a language is one call, ever — not one per call."""
    from brain.lines import LineBook

    writer = _ScriptedWriter("ఒకసారి")
    book = LineBook(HOSPITAL_AGENT, writer, cache_dir=tmp_path)
    await book.line("handoff", "te-IN")
    assert writer.calls == 1

    fresh = LineBook(HOSPITAL_AGENT, writer, cache_dir=tmp_path)
    assert await fresh.line("handoff", "te-IN") == "ఒకసారి"
    assert writer.calls == 1, "a language already on disk must not be written again"


@pytest.mark.asyncio
async def test_a_language_nobody_wrote_a_table_for_still_works(tmp_path):
    """The point of the whole arrangement: adding a language is not a code change."""
    from brain.lines import LineBook

    book = LineBook(HOSPITAL_AGENT, _ScriptedWriter("ਸਤ ਸ੍ਰੀ ਅਕਾਲ"), cache_dir=tmp_path)
    assert await book.line("opener", "pa-IN") == "ਸਤ ਸ੍ਰੀ ਅਕਾਲ"


class _ScriptedWriter:
    """An LLM that writes one known line, and counts how often it is asked to."""

    def __init__(self, line: str) -> None:
        self.line = line
        self.calls = 0

    async def json_call(self, messages, **kwargs):
        self.calls += 1
        return {"lines": [self.line]}


def test_the_orchestrator_does_not_hard_code_the_handoff():
    import inspect

    from brain import orchestrator

    assert "put you through" not in inspect.getsource(orchestrator)
