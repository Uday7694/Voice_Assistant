"""Turn-taking controller tests.

No audio, no network, no API key. The point is to pin down the behaviour that is
expensive to debug through a live phone call: who holds the floor, what happens on
interruption, and in what order.
"""

from __future__ import annotations

import asyncio

import pytest

from brain.speech.sarvam import SpeechEvent
from brain.voice.controller import State, TurnController
from tests.fakes import FakeBrain, FakeSynth, RecordingSink, ScriptedSource


def build(script, brain=None, **kwargs):
    synth = FakeSynth()
    sink = RecordingSink()
    controller = TurnController(
        brain or FakeBrain(),
        "s1",
        ScriptedSource(script),
        synth,
        sink,
        filler=kwargs.pop("filler", None),  # off unless a test asks for it
        **kwargs,
    )
    return controller, synth, sink


def said(text: str) -> SpeechEvent:
    return SpeechEvent(kind="transcript", text=text)


SPEECH_START = SpeechEvent(kind="speech_start")


# --- the ordinary path ----------------------------------------------------


@pytest.mark.asyncio
async def test_a_transcript_produces_spoken_audio():
    brain = FakeBrain({"hello": ["Good morning.", "How can I help?"]})
    controller, synth, sink = build([said("hello")], brain)

    await controller.run()

    assert synth.spoken == ["Good morning.", "How can I help?"]
    assert sink.transcript == "Good morning. Good morning. Good morning. How can I help? How can I help? How can I help?"
    assert controller.state is State.ENDED


@pytest.mark.asyncio
async def test_silence_alone_never_starts_a_turn():
    brain = FakeBrain()
    controller, _, sink = build([SPEECH_START, SpeechEvent(kind="speech_end")], brain)

    await controller.run()

    assert brain.heard == []
    assert sink.played == []


@pytest.mark.asyncio
async def test_empty_transcript_is_ignored():
    brain = FakeBrain()
    controller, _, _ = build([said("   ")], brain)

    await controller.run()

    assert brain.heard == []


# --- barge-in -------------------------------------------------------------


@pytest.mark.asyncio
async def test_barge_in_clears_the_transport_buffer():
    brain = FakeBrain({"hi": ["A long reply that keeps going.", "And going."]})
    controller, _, sink = build(
        [said("hi"), 0.05, SPEECH_START], brain, barge_in_grace=0.0
    )

    await controller.run()

    assert controller.barge_ins == 1
    assert sink.clears == 1


@pytest.mark.asyncio
async def test_barge_in_stops_playback_before_clearing():
    """Ordering matters: clearing first, then still writing frames, replays audio."""
    brain = FakeBrain({"hi": ["One.", "Two.", "Three."]})
    controller, _, sink = build(
        [said("hi"), 0.05, SPEECH_START], brain, barge_in_grace=0.0
    )

    await controller.run()

    assert "clear" in sink.events
    # Nothing may be played after the buffer was cleared.
    assert sink.events[-1] == "clear"


@pytest.mark.asyncio
async def test_barge_in_abandons_the_rest_of_the_reply():
    brain = FakeBrain({"hi": ["One.", "Two.", "Three.", "Four."]})
    controller, synth, sink = build(
        [said("hi"), 0.05, SPEECH_START], brain, barge_in_grace=0.0
    )

    await controller.run()

    assert len(sink.played) < 4 * synth.frames_per_sentence


@pytest.mark.asyncio
async def test_barge_in_returns_the_floor_to_the_caller():
    brain = FakeBrain({"hi": ["Talking.", "Still talking."]})
    controller, _, _ = build([said("hi"), 0.05, SPEECH_START], brain, barge_in_grace=0.0)

    await controller.run()

    assert controller.state is State.ENDED  # run() finished cleanly after the interrupt


@pytest.mark.asyncio
async def test_speech_inside_the_grace_window_is_not_a_barge_in():
    """The tail of our own audio must not interrupt us."""
    brain = FakeBrain({"hi": ["One.", "Two."]})
    controller, _, sink = build([said("hi"), 0.02, SPEECH_START], brain, barge_in_grace=5.0)

    await controller.run()

    assert controller.barge_ins == 0
    assert sink.clears == 0


@pytest.mark.asyncio
async def test_speech_start_while_idle_is_not_a_barge_in():
    controller, _, sink = build([SPEECH_START, said("hello")], barge_in_grace=0.0)

    await controller.run()

    assert controller.barge_ins == 0
    assert sink.clears == 0


@pytest.mark.asyncio
async def test_interrupting_transcript_replaces_the_previous_turn():
    brain = FakeBrain({"first": ["Answering first.", "More."], "second": ["Answering second."]})
    controller, synth, _ = build(
        [said("first"), 0.02, said("second")], brain, barge_in_grace=0.0
    )

    await controller.run()

    assert brain.heard == ["first", "second"]
    assert "Answering second." in synth.spoken


# --- filler ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_filler_plays_when_the_planner_is_slow():
    brain = FakeBrain({"hi": ["Finally here."]}, think=0.2)
    controller, synth, _ = build(
        [said("hi")], brain, filler="One moment.", filler_after=0.05
    )

    await controller.run()

    assert "One moment." in synth.spoken
    assert "Finally here." in synth.spoken


@pytest.mark.asyncio
async def test_filler_is_suppressed_when_the_planner_is_quick():
    brain = FakeBrain({"hi": ["Instant."]})
    controller, synth, _ = build(
        [said("hi")], brain, filler="One moment.", filler_after=5.0
    )

    await controller.run()

    assert "One moment." not in synth.spoken


# --- ending ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_event_finishes_the_call_after_the_last_line():
    brain = FakeBrain({"bye": ["Goodbye."]}, end_after="bye")
    controller, synth, _ = build([said("bye"), 1.0, said("still there?")], brain)

    await controller.run()

    assert synth.spoken == ["Goodbye."]
    assert brain.heard == ["bye"]
    assert controller.state is State.ENDED


@pytest.mark.asyncio
async def test_escalation_finishes_the_call():
    brain = FakeBrain({"human": ["Connecting you."]}, escalate_after="human")
    controller, _, _ = build([said("human"), 1.0, said("hello?")], brain)

    await controller.run()

    assert brain.heard == ["human"]


# --- resource hygiene -----------------------------------------------------


@pytest.mark.asyncio
async def test_a_barge_in_does_not_leak_the_synthesis_socket():
    brain = FakeBrain({"hi": ["One.", "Two.", "Three."]})
    controller, synth, _ = build([said("hi"), 0.05, SPEECH_START], brain, barge_in_grace=0.0)

    await controller.run()
    await asyncio.sleep(0.05)

    assert synth.open_sockets == 0


@pytest.mark.asyncio
async def test_turns_do_not_overlap_synthesis_sockets():
    brain = FakeBrain({"a": ["First."], "b": ["Second."]})
    controller, synth, _ = build([said("a"), 0.2, said("b")], brain)

    await controller.run()

    assert synth.max_open_sockets == 1


@pytest.mark.asyncio
async def test_caller_can_interrupt_the_filler_phrase():
    """While a holding phrase plays the agent is talking, so it must yield the floor."""
    brain = FakeBrain({"hi": ["Eventually."]}, think=0.5)
    controller, _, sink = build(
        [said("hi"), 0.1, SPEECH_START],
        brain,
        filler="One moment.",
        filler_after=0.01,
        barge_in_grace=0.0,
    )

    await controller.run()

    assert controller.barge_ins == 1
    assert sink.clears == 1
