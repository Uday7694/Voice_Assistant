"""L0 — the turn-taking controller.

Owns the conversational floor. This is the layer that decides *when* the agent speaks,
listens, or shuts up; the LLM only decides *what* gets said. Keeping that boundary is
what separates an agent that feels responsive from one that talks over people.

    IDLE -> LISTENING -> THINKING -> SPEAKING -> (barge-in) -> LISTENING

It talks to three narrow interfaces so it can be exercised with fakes — no microphone,
no Sarvam key, no network. The barge-in path in particular is far too important to leave
untested until there is audio.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from enum import Enum
from typing import AsyncIterator, Protocol

from ..models import EndEvent, EscalateEvent, SayEvent
from ..speech.types import SpeechEvent

log = logging.getLogger(__name__)


class State(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ENDED = "ended"


class SpeechSource(Protocol):
    """Anything that emits speech_start / speech_end / transcript. `Ear` satisfies this."""

    def events(self) -> AsyncIterator[SpeechEvent]: ...


class Synthesizer(Protocol):
    """Anything that turns streamed text into streamed audio. `Mouth` satisfies this."""

    def say(self, chunks: AsyncIterator[str]) -> AsyncIterator[bytes]: ...


class AudioSink(Protocol):
    """The transport: a LiveKit track, an Exotel socket, a browser, a test recorder."""

    async def play(self, chunk: bytes) -> None: ...

    async def clear(self) -> None:
        """Drop audio already queued downstream.

        Stopping generation is not enough. Whatever is sitting in the transport's jitter
        buffer will still play, and the caller hears the agent talk over them.
        """


# Ignore speech detected in the first moments of our own speech. Without a grace period
# the tail of the agent's own audio — or room echo on a device with weak cancellation —
# retriggers barge-in immediately and the agent cuts itself off mid-greeting.
# The cost is that a caller cannot interrupt within this window; keep it short.
BARGE_IN_GRACE = 0.25

# If the planner has produced nothing sayable by now, play a holding phrase. Silence
# past roughly this point reads as a dropped call.
#
# Tuned against measurement, not intuition. planner_first_sentence runs 300-600 ms on a
# plain turn and 1200-2000 ms when a tool round is involved. A 0.6 s threshold therefore
# fired on almost every turn, and an agent that says "One moment." before each sentence
# sounds worse than one that is briefly quiet. At 1.2 s it fires only on the tool rounds,
# which is exactly where the silence is long enough to worry a caller.
FILLER_AFTER = 1.2
DEFAULT_FILLER = "One moment."


class TurnController:
    def __init__(
        self,
        brain,
        session_id: str,
        source: SpeechSource,
        synth: Synthesizer,
        sink: AudioSink,
        *,
        filler: str | None = DEFAULT_FILLER,
        filler_after: float = FILLER_AFTER,
        barge_in_grace: float = BARGE_IN_GRACE,
        clock=time.monotonic,
    ) -> None:
        self.brain = brain
        self.session_id = session_id
        self.source = source
        self.synth = synth
        self.sink = sink
        self.filler = filler
        self.filler_after = filler_after
        self.barge_in_grace = barge_in_grace
        self._clock = clock

        self.state = State.IDLE
        self.barge_ins = 0
        self._turn: asyncio.Task | None = None
        self._interrupted = asyncio.Event()
        self._speaking_since: float | None = None
        self._ending = False

    # --- main loop ---------------------------------------------------------

    async def run(self) -> None:
        """Drive the conversation until the brain ends it or the source stops."""
        self.state = State.LISTENING

        async for event in self.source.events():
            # Checked before handling, not after: once the brain has ended or escalated
            # the call, anything still arriving from the caller must be ignored rather
            # than started as a fresh turn.
            if self._ending:
                break

            if event.kind == "speech_start":
                await self._on_speech_start()
            elif event.kind == "transcript" and event.text.strip():
                await self._on_transcript(event.text)

        # The source ending is not an interruption — let the reply in flight finish
        # speaking. A real hangup cancels this whole task from the transport side,
        # which cancels the turn with it.
        await self._finish_turn()
        self.state = State.ENDED

    async def _on_speech_start(self) -> None:
        if self.state is not State.SPEAKING:
            return
        if not self._past_grace():
            log.debug("Ignoring speech_start inside barge-in grace window")
            return
        await self._barge_in()

    async def _on_transcript(self, text: str) -> None:
        # A transcript arriving mid-reply means the caller talked over us and finished a
        # thought. Their turn wins: abandon whatever we were saying and answer this.
        await self._cancel_turn()
        self._interrupted.clear()
        self._turn = asyncio.create_task(self._run_turn(text))

    def _past_grace(self) -> bool:
        if self._speaking_since is None:
            return True
        return (self._clock() - self._speaking_since) >= self.barge_in_grace

    # --- barge-in ----------------------------------------------------------

    async def _barge_in(self) -> None:
        """Yield the floor, in the only order that actually sounds right.

        1. stop feeding the sink, so nothing further is written
        2. clear what the transport has already buffered
        3. cancel the turn, which closes the TTS socket and stops generation

        Doing (3) first would leave buffered audio playing after the model stopped —
        the agent keeps talking for a second after being interrupted.
        """
        self._interrupted.set()
        await self.sink.clear()
        await self._cancel_turn()

        self.barge_ins += 1
        self._speaking_since = None
        self.state = State.LISTENING
        log.info("Barge-in #%d", self.barge_ins)

    async def _finish_turn(self) -> None:
        """Wait for the current turn to finish speaking, without cancelling it."""
        turn, self._turn = self._turn, None
        if turn is None:
            return
        with suppress(asyncio.CancelledError):
            await turn

    async def _cancel_turn(self) -> None:
        turn, self._turn = self._turn, None
        if turn is None or turn.done():
            return
        turn.cancel()
        with suppress(asyncio.CancelledError):
            await turn

    # --- one turn ----------------------------------------------------------

    async def _run_turn(self, text: str) -> None:
        self.state = State.THINKING
        self._speaking_since = None

        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        speaker = asyncio.create_task(self._stream_audio(_drain(sentences)))
        filler = asyncio.create_task(self._play_filler_if_slow())

        try:
            async for event in self.brain.handle(self.session_id, text):
                if isinstance(event, SayEvent):
                    filler.cancel()
                    await sentences.put(event.text)
                elif isinstance(event, (EndEvent, EscalateEvent)):
                    self._ending = True

            await sentences.put(None)
            await speaker
        finally:
            filler.cancel()
            with suppress(asyncio.CancelledError):
                await filler
            if not speaker.done():
                speaker.cancel()
                with suppress(asyncio.CancelledError):
                    await speaker
            if self.state is State.SPEAKING:
                self.state = State.LISTENING
                self._speaking_since = None

    async def _stream_audio(self, sentences: AsyncIterator[str]) -> None:
        await self._emit(self.synth.say(sentences))

    async def _emit(self, audio: AsyncIterator[bytes]) -> None:
        """Play generated audio, holding the floor while it does.

        Every path that produces sound goes through here, filler included. A path that
        plays audio without marking the state SPEAKING leaves the agent audibly talking
        while the controller still thinks it is idle — and it will ignore a caller who
        interrupts.
        """
        async for chunk in audio:
            if self._interrupted.is_set():
                break
            if self.state is not State.SPEAKING:
                self.state = State.SPEAKING
                self._speaking_since = self._clock()
            await self.sink.play(chunk)

    async def _play_filler_if_slow(self) -> None:
        """Hold the line if the planner is taking too long.

        Deliberately bypasses the sentence queue: this audio must not be part of the
        reply being synthesised, or it would delay the real answer behind it.
        """
        if not self.filler:
            return
        await asyncio.sleep(self.filler_after)

        log.info("Planner slow; playing filler")
        await self._emit(self.synth.say(_once(self.filler)))


async def _drain(queue: asyncio.Queue) -> AsyncIterator[str]:
    """Turn a queue into the async iterator the synthesizer expects."""
    while True:
        item = await queue.get()
        if item is None:
            return
        yield item


async def _once(text: str) -> AsyncIterator[str]:
    yield text
