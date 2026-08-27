"""Turn assembly from a transcriber's event stream.

No device and no network: a fake ear replays event sequences recorded from the live
service, which is the only part of this worth testing offline — the ordering of those
events is the thing that is easy to get wrong and impossible to guess.
"""

from __future__ import annotations

import asyncio

import pytest

from brain.speech.listen import Listener
from brain.speech.types import SpeechEvent


class FakeEar:
    def __init__(self, events: list[SpeechEvent]) -> None:
        self._events = events

    async def events(self):
        for event in self._events:
            yield event

    async def feed(self, pcm: bytes) -> None:  # pragma: no cover - never called here
        pass


async def _turns_from(events: list[SpeechEvent], **kwargs) -> list[str]:
    listener = Listener(**kwargs)
    listener._ear = FakeEar(events)
    await listener._read()
    collected = []
    while not listener._turns.empty():
        collected.append(listener._turns.get_nowait())
    return collected


def _start() -> SpeechEvent:
    return SpeechEvent(kind="speech_start")


def _end() -> SpeechEvent:
    return SpeechEvent(kind="speech_end")


def _said(text: str) -> SpeechEvent:
    return SpeechEvent(kind="transcript", text=text)


@pytest.mark.asyncio
async def test_a_transcript_arriving_after_the_end_signal_still_becomes_a_turn():
    """The live ordering: start, end, then the text. Measured, not assumed."""
    events = [_start(), _end(), _said("मुझे appointment चाहिए")]
    assert await _turns_from(events) == ["मुझे appointment चाहिए"]


@pytest.mark.asyncio
async def test_a_transcript_arriving_before_the_end_signal_still_becomes_a_turn():
    events = [_start(), _said("book an appointment"), _end()]
    assert await _turns_from(events) == ["book an appointment"]


@pytest.mark.asyncio
async def test_a_pause_mid_sentence_is_one_turn_not_two():
    events = [_start(), _said("cardiology"), _said("tomorrow morning"), _end()]
    assert await _turns_from(events) == ["cardiology tomorrow morning"]


@pytest.mark.asyncio
async def test_two_utterances_are_two_turns():
    events = [_start(), _end(), _said("yes"), _start(), _end(), _said("Uday")]
    assert await _turns_from(events) == ["yes", "Uday"]


@pytest.mark.asyncio
async def test_silence_produces_no_turn():
    assert await _turns_from([_start(), _end()]) == []


@pytest.mark.asyncio
async def test_voice_activity_fires_the_barge_in_callback_before_any_transcript():
    fired: list[str] = []

    def on_start() -> None:
        fired.append("interrupt")

    events = [_start(), _end(), _said("stop")]
    await _turns_from(events, on_speech_start=on_start)
    assert fired == ["interrupt"]
