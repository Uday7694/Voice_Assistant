"""Subagent tests. No network: the HTTP call is stubbed at _blocking_call."""

from __future__ import annotations

import asyncio
import time

import pytest

from brain.subagent import DeepSubagent
from brain.tools import build_default_registry


class FakeSubagent(DeepSubagent):
    """Real job machinery, fake transport."""

    def __init__(self, *, reply="the answer", fail_times=0, delay=0.0, permanent=False):
        super().__init__(api_key="test-key")
        self.reply, self.delay = reply, delay
        self.fail_times, self.calls, self.permanent = fail_times, 0, permanent

    def _client_or_raise(self):  # never build a real client
        return None

    def _blocking_call(self, model, prompt, timeout):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)  # real blocking work, in the worker thread
        if self.calls <= self.fail_times:
            error = RuntimeError(f"boom {self.calls}")
            error.status_code = 404 if self.permanent else 429
            raise error
        return self.reply


@pytest.mark.asyncio
async def test_start_returns_without_awaiting_the_model():
    sub = FakeSubagent(delay=0.2)
    job_id = sub.start("a hard question")
    assert sub.pending == 1
    assert sub.collect() == []  # nothing finished yet
    results = await sub.drain()
    assert [r.id for r in results] == [job_id]
    assert results[0].ok and results[0].text == "the answer"


@pytest.mark.asyncio
async def test_deep_reason_tool_does_not_block_the_turn():
    sub = FakeSubagent()
    registry = build_default_registry(sub)
    outcome = await registry.invoke("deep_reason", {"question": "why?"})
    assert outcome.ok
    assert outcome.result["status"] == "working"
    assert outcome.ms < 50, f"tool blocked for {outcome.ms:.0f} ms"
    await sub.aclose()


@pytest.mark.asyncio
async def test_retries_transient_failure_then_succeeds(monkeypatch):
    monkeypatch.setattr("brain.subagent.RETRY_BACKOFF", (0.01, 0.01))
    sub = FakeSubagent(fail_times=1)
    assert await sub.run("q") == "the answer"
    assert sub.calls == 2


@pytest.mark.asyncio
async def test_permanent_failure_is_not_retried_on_the_same_model(monkeypatch):
    monkeypatch.setattr("brain.subagent.RETRY_BACKOFF", (0.01, 0.01))
    sub = FakeSubagent(fail_times=99, permanent=True)
    with pytest.raises(RuntimeError):
        await sub.run("q")
    assert sub.calls == 2, "one attempt per model, no retries on 404"


@pytest.mark.asyncio
async def test_failed_job_is_reported_not_raised():
    sub = FakeSubagent(fail_times=99, permanent=True)
    sub.start("q")
    results = await sub.drain()
    assert len(results) == 1 and not results[0].ok and results[0].error


@pytest.mark.asyncio
async def test_tool_absent_when_no_api_key():
    registry = build_default_registry(DeepSubagent(api_key=""))
    assert registry.get("deep_reason") is None


@pytest.mark.asyncio
async def test_aclose_cancels_outstanding_jobs():
    sub = FakeSubagent(delay=5.0)
    sub.start("q")
    await sub.aclose()
    assert sub.pending == 0


# --- regression: reported duration ----------------------------------------


@pytest.mark.asyncio
async def test_duration_excludes_time_spent_waiting_to_be_collected():
    """A job finishing on turn one must not be billed for turns two and three.

    ``ms`` previously ran to collection time, reporting ~1000 ms for a job that took
    under 50 ms. It is the number used to judge whether a subagent is too slow.
    """
    sub = FakeSubagent()
    sub.start("q")
    await asyncio.sleep(0.05)  # the job completes here
    await asyncio.sleep(0.60)  # the caller keeps talking for two more turns
    result = sub.collect()[0]
    assert result.ok
    assert result.ms < 200, f"duration inflated to {result.ms:.0f} ms by collection delay"


# --- orchestrator handover -------------------------------------------------


def _bare_brain(subagent):
    from brain.orchestrator import Brain

    brain = Brain.__new__(Brain)  # the helpers under test need no LLM
    brain.subagent = subagent
    return brain


def _session():
    from brain.models import Session

    return Session(agent_name="t", node_id="collect_booking")


def _start_for(sub, session, brief, **kw):
    """Start a job owned by ``session``, the way the orchestrator does."""
    from brain.subagent import current_session

    current_session.set(session.session_id)
    return sub.start(brief, **kw)


async def _settle(sub, timeout=2.0):
    """Wait for every job to finish without collecting it.

    ``drain()`` waits *and* collects, which empties the pool the orchestrator helper is
    meant to read from.
    """
    deadline = time.perf_counter() + timeout
    while sub.pending and time.perf_counter() < deadline:
        await asyncio.sleep(0.01)
    assert not sub.pending, "jobs did not finish"


@pytest.mark.asyncio
async def test_finished_answer_reaches_the_planner_as_a_slot():
    sub = FakeSubagent(reply="Cardiology is the right department.")
    brain = _bare_brain(sub)
    session = _session()
    _start_for(sub, session, "which department?")
    await _settle(sub)

    session = brain._absorb_subagent(session)
    assert session.slots[brain.EXPERT_SLOT] == "Cardiology is the right department."


@pytest.mark.asyncio
async def test_answer_is_consumed_after_one_turn():
    """Answers belong to the question that produced them and go stale at once."""
    sub = FakeSubagent(reply="an answer")
    brain = _bare_brain(sub)
    session = _session()
    _start_for(sub, session, "q")
    await _settle(sub)

    session = brain._absorb_subagent(session)
    assert brain.EXPERT_SLOT in session.slots
    cleared = brain._clear_expert_answer(session)
    assert brain.EXPERT_SLOT not in cleared.slots


@pytest.mark.asyncio
async def test_clearing_leaves_every_other_slot_untouched():
    sub = FakeSubagent()
    brain = _bare_brain(sub)
    session = _session().with_slots({"patient_name": "Asha", brain.EXPERT_SLOT: "x"})
    cleared = brain._clear_expert_answer(session)
    assert cleared.slots == {"patient_name": "Asha"}


@pytest.mark.asyncio
async def test_a_failed_job_is_surfaced_rather_than_dropped():
    """The caller was told the agent was checking; silence strands them."""
    sub = FakeSubagent(fail_times=99, permanent=True)
    brain = _bare_brain(sub)
    session = _session()
    _start_for(sub, session, "q")
    await _settle(sub)

    session = brain._absorb_subagent(session)
    assert session.slots[brain.EXPERT_SLOT] == brain.EXPERT_FAILED


@pytest.mark.asyncio
async def test_no_finished_jobs_leaves_the_session_alone():
    brain = _bare_brain(FakeSubagent())
    session = _session()
    assert brain._absorb_subagent(session) is session


# --- backpressure ----------------------------------------------------------


@pytest.mark.asyncio
async def test_repeating_an_in_flight_question_reuses_the_job():
    """The planner re-asks every turn while waiting; each duplicate costs real quota."""
    sub = FakeSubagent(delay=0.3)
    first = sub.start("which department?")
    for _ in range(11):
        assert sub.start("which department?") == first
    assert len(sub._jobs) == 1
    await sub.aclose()


@pytest.mark.asyncio
async def test_dedupe_ignores_case_and_surrounding_space():
    sub = FakeSubagent(delay=0.3)
    first = sub.start("Which Department?")
    assert sub.start("  which department?  ") == first
    await sub.aclose()


@pytest.mark.asyncio
async def test_a_different_question_starts_its_own_job():
    sub = FakeSubagent(delay=0.3)
    assert sub.start("question one") != sub.start("question two")
    assert len(sub._jobs) == 2
    await sub.aclose()


@pytest.mark.asyncio
async def test_same_question_with_different_context_is_not_deduped():
    sub = FakeSubagent(delay=0.3)
    a = sub.start("which department?", context="caller is 54")
    b = sub.start("which department?", context="caller is 12")
    assert a != b
    await sub.aclose()


@pytest.mark.asyncio
async def test_queue_is_capped(monkeypatch):
    from brain.subagent import SubagentBusy

    monkeypatch.setattr("brain.subagent.MAX_QUEUED_JOBS", 3)
    sub = FakeSubagent(delay=0.3)
    for i in range(3):
        sub.start(f"question {i}")
    with pytest.raises(SubagentBusy):
        sub.start("one too many")
    await sub.aclose()


@pytest.mark.asyncio
async def test_tool_reports_busy_rather_than_raising(monkeypatch):
    monkeypatch.setattr("brain.subagent.MAX_QUEUED_JOBS", 1)
    sub = FakeSubagent(delay=0.3)
    registry = build_default_registry(sub)
    await registry.invoke("deep_reason", {"question": "first"})
    outcome = await registry.invoke("deep_reason", {"question": "second"})
    assert outcome.ok and outcome.result["status"] == "busy"
    await sub.aclose()


# --- prompt safety ---------------------------------------------------------


def test_failure_text_is_a_fact_not_an_instruction():
    """Slots render under "Known details" and may be spoken verbatim.

    An instruction placed there gets read out as the agent's own stage directions.
    """
    from brain.orchestrator import Brain

    text = Brain.EXPERT_FAILED.lower()
    for directive in ("apologise", "apologize", "offer a human", "tell the caller", "say "):
        assert directive not in text, f"{directive!r} is an instruction, not a fact"
    assert text.rstrip().endswith(".")


# --- session isolation -----------------------------------------------------


@pytest.mark.asyncio
async def test_one_caller_never_receives_another_callers_answer():
    """One Brain serves every concurrent caller from a single job pool.

    An unfiltered drain hands whoever speaks next whatever finished most recently. In a
    hospital flow that is one caller hearing another caller's medical answer.
    """
    sub = FakeSubagent(reply="Caller A needs cardiology for a stent follow-up.")
    brain = _bare_brain(sub)

    caller_a, caller_b = _session(), _session()
    _start_for(sub, caller_a, "what does caller A need?")
    await _settle(sub)

    assert brain.EXPERT_SLOT not in brain._absorb_subagent(caller_b).slots
    assert brain.EXPERT_SLOT in brain._absorb_subagent(caller_a).slots


@pytest.mark.asyncio
async def test_dedupe_does_not_reuse_another_callers_job():
    sub = FakeSubagent(delay=0.3)
    a, b = _session(), _session()
    assert _start_for(sub, a, "same question") != _start_for(sub, b, "same question")
    await sub.aclose()


@pytest.mark.asyncio
async def test_ending_a_call_releases_its_queue_slot():
    """A job in flight at hangup is never collected and would hold a slot forever."""
    sub = FakeSubagent(delay=5.0)
    brain = _bare_brain(sub)
    session = _session()
    _start_for(sub, session, "q")
    assert len(sub._jobs) == 1

    brain._release_subagent(session)
    assert len(sub._jobs) == 0
    await sub.aclose()
