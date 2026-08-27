"""Which provider answers when the first one does not, and how long that may take.

Both questions had wrong answers. With a Sarvam planner and a Groq classifier, the
"fallback" was resolved against the *planner's* provider and came out as Groq — so a
Groq classifier timeout was retried on Groq, with the same budget and the same
overloaded endpoint. And nothing bounded the sequence: a 3 s timeout could be followed
by a 7.5 s scaled retry, and the caller heard eleven seconds of nothing before the turn
degraded to "unknown" anyway.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from brain.config import FAST_TIMEOUT, FAST_TOTAL_TIMEOUT, PLANNER_TIMEOUT, PLANNER_TOTAL_TIMEOUT
from brain.llm import MIN_ATTEMPT_SECONDS, LLMClient, LLMTimeout, _attempt_budget
from brain.providers import GROQ, NVIDIA, SARVAM, Provider, fallback_chain


def _client(primary: Provider, fast: Provider, fallbacks: tuple[Provider, ...]) -> LLMClient:
    client = LLMClient.__new__(LLMClient)  # no network, no keys
    client.provider = primary
    client.fast = fast
    client.fallbacks = fallbacks
    return client


# --- ordering --------------------------------------------------------------


def test_a_provider_is_never_its_own_fallback():
    """The bug, exactly: Groq classifier times out, Groq is retried."""
    client = _client(SARVAM, GROQ, (GROQ, NVIDIA, SARVAM))
    assert client._chain(GROQ).count(GROQ) == 1
    assert client._chain(GROQ)[0] is GROQ


def test_the_classifier_falls_back_to_a_different_provider():
    client = _client(SARVAM, GROQ, (GROQ, NVIDIA, SARVAM))
    chain = client._chain(GROQ)
    assert [p.name for p in chain] == ["groq", "nvidia", "sarvam"]


def test_the_planner_starts_on_its_own_provider_whatever_the_chain_order():
    """Sarvam is the planner for the quality of its Indian languages, not its speed."""
    client = _client(SARVAM, GROQ, (GROQ, NVIDIA, SARVAM))
    assert client._chain(SARVAM)[0] is SARVAM


def test_fallbacks_are_ordered_by_measured_speed():
    assert GROQ.rank < NVIDIA.rank < SARVAM.rank
    ranked = [p.name for p in fallback_chain()]
    assert ranked == sorted(ranked, key=lambda n: {"groq": 10, "nvidia": 20, "sarvam": 30}[n])


# --- budgets ---------------------------------------------------------------


def test_the_first_attempt_cannot_eat_the_whole_ceiling():
    """Otherwise the fallback it exists for never gets a request."""
    budget = _attempt_budget(PLANNER_TIMEOUT, SARVAM, PLANNER_TOTAL_TIMEOUT, is_last=False)
    assert budget < PLANNER_TOTAL_TIMEOUT


def test_the_last_attempt_may_use_what_is_left():
    assert _attempt_budget(FAST_TIMEOUT, GROQ, 1.2, is_last=True) == pytest.approx(1.2)


def test_a_slow_provider_still_gets_its_scale_when_there_is_room():
    assert _attempt_budget(1.0, NVIDIA, 60.0, is_last=True) == pytest.approx(NVIDIA.timeout_scale)


def test_no_attempt_is_made_with_a_hopeless_sliver_of_time():
    assert _attempt_budget(FAST_TIMEOUT, GROQ, 0.01, is_last=True) == MIN_ATTEMPT_SECONDS


def test_the_classifier_ceiling_leaves_room_for_more_than_one_provider():
    assert FAST_TOTAL_TIMEOUT > FAST_TIMEOUT
    assert PLANNER_TOTAL_TIMEOUT > PLANNER_TIMEOUT


# --- the sequence ----------------------------------------------------------


@pytest.mark.asyncio
async def test_the_second_provider_answers_when_the_first_fails():
    client = _client(SARVAM, GROQ, (GROQ, NVIDIA, SARVAM))
    tried: list[str] = []

    async def fail_once(provider, budget):
        tried.append(provider.name)
        if len(tried) == 1:
            raise RuntimeError("rate limited")
        return "answer"

    assert await client._with_fallback(fail_once, what="probe", primary=GROQ) == "answer"
    assert tried == ["groq", "nvidia"]


@pytest.mark.asyncio
async def test_the_whole_sequence_stays_inside_its_ceiling():
    client = _client(SARVAM, GROQ, (GROQ, NVIDIA, SARVAM))

    async def never_answers(provider, budget):
        await asyncio.sleep(budget * 10)

    started = time.monotonic()
    with pytest.raises(LLMTimeout):
        await client._with_fallback(
            never_answers, what="probe", primary=GROQ, timeout=0.4, ceiling=1.2
        )
    assert time.monotonic() - started < 2.0


@pytest.mark.asyncio
async def test_an_attempt_that_ignores_its_budget_is_cut_off_anyway():
    """The ceiling cannot depend on every call site remembering to apply it."""
    client = _client(SARVAM, GROQ, (GROQ,))

    async def ignores_budget(provider, budget):
        await asyncio.sleep(30)

    started = time.monotonic()
    with pytest.raises(LLMTimeout):
        await client._with_fallback(
            ignores_budget, what="probe", primary=GROQ, timeout=0.5, ceiling=0.6
        )
    assert time.monotonic() - started < 2.0


# --- answering with nothing -------------------------------------------------


@pytest.mark.asyncio
async def test_a_provider_that_answers_with_nothing_is_not_the_end_of_the_sequence():
    """NVIDIA returns a well-formed {"lines": []} for a writing task it cannot do.

    Treated as success that ends the chain, the language silently gets the English
    fallback while a provider that could have written the line goes untried.
    """
    client = _client(SARVAM, GROQ, (GROQ, NVIDIA, SARVAM))
    tried: list[str] = []

    async def empty_then_full(provider, budget):
        tried.append(provider.name)
        return [] if len(tried) == 1 else ["a line"]

    result = await client._with_fallback(
        empty_then_full, what="probe", primary=GROQ, accept=bool
    )
    assert result == ["a line"]
    assert tried == ["groq", "nvidia"]


@pytest.mark.asyncio
async def test_nothing_from_every_provider_raises_rather_than_returning_junk():
    client = _client(SARVAM, GROQ, (GROQ, NVIDIA))

    async def always_empty(provider, budget):
        return []

    with pytest.raises(Exception):
        await client._with_fallback(always_empty, what="probe", primary=GROQ, accept=bool)


def test_writing_a_language_gets_a_budget_that_is_not_the_classifier_s():
    """It happens once per language, off the critical path, and is cached forever."""
    from brain.config import LINE_TIMEOUT, LINE_TOTAL_TIMEOUT

    assert LINE_TIMEOUT > FAST_TIMEOUT
    assert LINE_TOTAL_TIMEOUT > FAST_TOTAL_TIMEOUT
