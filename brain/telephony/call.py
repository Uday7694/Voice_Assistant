"""One phone call: carrier socket at one end, the brain at the other.

This is the same conversation `talk.py` holds, with the sound card replaced by a
WebSocket. That is the whole point of the shape the rest of the system was built in —
the ear takes PCM frames from anywhere, the mouth produces PCM frames for anywhere, and
neither knows whether the far end is a headset or a caller in Coimbatore.

What a phone call adds is that the caller cannot be asked to wait. Two things follow.
The agent's audio is paced out at the rate the line consumes it rather than pushed as
fast as it is synthesised, because a carrier given six seconds of audio in one burst will
play all six even after the caller interrupts. And voice activity has to cut the agent
off inside a frame or two, which means the paced sender must be cancellable at any point.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

from ..models import EndEvent, EscalateEvent, SayEvent
from ..orchestrator import Brain
from ..speech.fillers import Backchannel
from ..speech.sarvam import Ear, Mouth, SarvamError, warm_tts
from ..speech.types import TELEPHONY_SAMPLE_RATE, WEB_SAMPLE_RATE
from . import codec
from .carriers import Carrier, Frame

log = logging.getLogger(__name__)

# Enough digits to be a number a person would slow down for.
DIGITS_IN_A_READBACK = 4


def _kind_of(text: str) -> str:
    """Deliver a line the way its content deserves."""
    lowered = text.lower()
    if "sorry" in lowered or "apolog" in lowered:
        return "apology"
    if sum(char.isdigit() for char in text) >= DIGITS_IN_A_READBACK:
        return "readback"
    return "ask"

# How much synthesised audio to hand the carrier ahead of real time. Some buffer is
# needed or a slow synthesis round trip becomes a gap in the middle of a word; too much
# and an interrupted agent keeps talking out of the carrier's queue. Two frames is the
# smallest amount that survives a hiccup.
LEAD_FRAMES = 2


class Socket(Protocol):
    """What the call needs from a WebSocket, and nothing more.

    Declared structurally so the call loop can be driven by a test double that is a few
    lines long rather than by a real carrier connection.
    """

    async def send(self, message: str) -> None: ...
    def __aiter__(self): ...


@dataclass
class CallStats:
    """What the call cost and how it felt, for the log line at the end."""

    turns: int = 0
    interruptions: int = 0
    caller_seconds: float = 0.0
    agent_characters: int = 0
    first_audio_ms: list[float] = field(default_factory=list)

    @property
    def median_first_audio_ms(self) -> float:
        if not self.first_audio_ms:
            return 0.0
        ordered = sorted(self.first_audio_ms)
        return ordered[len(ordered) // 2]


class Call:
    """A live call. One instance per connection; never reused."""

    def __init__(
        self,
        socket: Socket,
        carrier: Carrier,
        brain: Brain,
        *,
        language: str = "en-IN",
        session_id: str = "",
    ) -> None:
        self.socket = socket
        self.carrier = carrier
        self.brain = brain
        self.language = language
        self.session_id = session_id
        self.call_id = ""
        self.stats = CallStats()

        self._backchannel: Backchannel | None = None
        self._ear: Ear | None = None
        self._speaking: asyncio.Task | None = None
        self._turns: asyncio.Queue[str] = asyncio.Queue()
        self._ended = asyncio.Event()

    # --- the call ---------------------------------------------------------

    async def run(self) -> CallStats:
        """Hold the call until either side hangs up."""
        # Open a synthesis connection now, while the carrier is still exchanging setup
        # frames. It absorbs the TLS and WebSocket handshake — measured at 850 ms on the
        # first utterance of a process — so the greeting starts when the agent is ready
        # to talk rather than when the socket is ready to carry it.
        warm_tts()

        # The words a person makes while they think. Written once per language and
        # cached to disk as audio, so saying one costs a file read rather than a round
        # trip — which is the only reason it can go out before the planner has started.
        try:
            await self.brain.lines.warm(self.language)
            self._backchannel = await Backchannel.for_call(self.brain.lines, self.language)
        except Exception:  # noqa: BLE001 - a call without fillers is still a call
            log.warning("Backchannel unavailable; turns will start silent", exc_info=True)

        if not self.session_id:
            session = self.brain.start(channel=f"phone:{self.carrier.name}")
            session = session.model_copy(update={"language": self.language})
            self.brain.store.put(session)
            self.session_id = session.session_id

        async with Ear(language=self.language, sample_rate=WEB_SAMPLE_RATE) as ear:
            self._ear = ear
            workers = [
                asyncio.create_task(self._read_carrier(), name="carrier-in"),
                asyncio.create_task(self._read_ear(), name="ear"),
                asyncio.create_task(self._converse(), name="brain"),
            ]
            try:
                await self._ended.wait()
            finally:
                for task in workers:
                    task.cancel()
                for task in workers:
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):  # noqa: BLE001 - teardown
                        pass
                await self._stop_speaking()

        log.info(
            "Call ended: %d turns, %d interruptions, %.0f ms median time to first audio",
            self.stats.turns,
            self.stats.interruptions,
            self.stats.median_first_audio_ms,
        )
        return self.stats

    # --- caller audio in --------------------------------------------------

    async def _read_carrier(self) -> None:
        """Carrier frames to the ear, for the length of the call."""
        async for message in self.socket:
            frame = self.carrier.parse(message)
            if frame.kind == "start":
                self.call_id = frame.call_id or self.call_id
                log.info("Call %s connected on %s", self.call_id, self.carrier.name)
            elif frame.kind == "audio" and frame.audio:
                self.stats.caller_seconds += len(frame.audio) / TELEPHONY_SAMPLE_RATE
                if self._ear is not None:
                    await self._ear.feed(
                        codec.from_telephony(frame.audio, WEB_SAMPLE_RATE)
                    )
            elif frame.kind == "stop":
                log.info("Caller hung up")
                self._ended.set()
                return
        # The socket closed without a stop frame, which is a dropped call.
        self._ended.set()

    async def _read_ear(self) -> None:
        """Transcripts to turns, and voice activity straight to barge-in."""
        assert self._ear is not None
        buffer: list[str] = []
        stopped = False
        async for event in self._ear.events():
            if event.kind == "speech_start":
                # Before any transcript exists, which is the only moment early enough to
                # matter: by the time there are words the caller has been talking over
                # the agent for a second.
                if self._speaking is not None and not self._speaking.done():
                    self.stats.interruptions += 1
                    log.info("Caller interrupted")
                    await self._stop_speaking()
                stopped = False
            elif event.kind == "transcript" and event.text.strip():
                buffer.append(event.text.strip())
                if stopped:
                    self._turns.put_nowait(" ".join(buffer))
                    buffer, stopped = [], False
            elif event.kind == "speech_end":
                stopped = True
                if buffer:
                    self._turns.put_nowait(" ".join(buffer))
                    buffer, stopped = [], False

    # --- the conversation -------------------------------------------------

    async def _converse(self) -> None:
        """Take turns until the brain ends the call."""
        while not self._ended.is_set():
            said = await self._turns.get()
            if not said:
                continue
            self.stats.turns += 1
            log.info("Caller: %s", said)

            started = time.perf_counter()
            first_audio = True

            # Before the classifier has run, let alone the planner. The measured gap
            # between a caller stopping and the agent having words is 1.1-2.0 s, and
            # there is no version of that a person would sit through in silence — people
            # say "right" while they are still thinking, and the silence is the tell.
            # This is cached audio, so it starts in the time it takes to read a file.
            if self._backchannel is not None and self._backchannel.should_ack(said):
                await self.say(self._backchannel.ack(), kind="filler", cache=True)

            try:
                async for event in self.brain.handle(self.session_id, said):
                    if isinstance(event, SayEvent):
                        log.info("Agent: %s", event.text)
                        self.stats.agent_characters += len(event.text)
                        await self.say(event.text, kind=_kind_of(event.text))
                        if first_audio:
                            self.stats.first_audio_ms.append(
                                (time.perf_counter() - started) * 1000
                            )
                            first_audio = False
                    elif isinstance(event, (EndEvent, EscalateEvent)):
                        # Let the closing line finish before the line drops; hanging up
                        # mid-sentence is how a caller learns they were talking to
                        # software.
                        await self._await_speech()
                        self._ended.set()
                        return
            except Exception:  # noqa: BLE001 - one bad turn must not drop the call
                log.exception("Turn failed")

    # --- agent audio out --------------------------------------------------

    async def say(self, text: str, *, kind: str = "ask", cache: bool = False) -> None:
        """Speak one line, paced to the line, interruptible at any frame."""
        await self._stop_speaking()
        self._speaking = asyncio.create_task(self._stream(text, kind, cache), name="say")
        try:
            await self._speaking
        except asyncio.CancelledError:
            pass

    async def _stream(self, text: str, kind: str = "ask", cache: bool = False) -> None:
        """Synthesise and pace out to the carrier.

        Paced deliberately. Handing the carrier a whole utterance at once means it plays
        the whole utterance, and a `clear` — where the carrier supports one at all — is a
        round trip away. Sending at roughly the rate the line consumes keeps at most a
        frame or two beyond the point of interruption.
        """
        mouth = Mouth(
            language=self.language,
            kind=kind,
            markers=self._backchannel.markers if self._backchannel else (),
            codec="linear16",
            sample_rate=TELEPHONY_SAMPLE_RATE,
        )
        frame_bytes = int(TELEPHONY_SAMPLE_RATE * self.carrier.frame_ms / 1000) * 2
        pending = bytearray()
        sent_frames = 0
        started = time.perf_counter()

        try:
            async for chunk in mouth.say_once(text, cache=cache):
                pending.extend(chunk)
                while len(pending) >= frame_bytes:
                    frame, pending = pending[:frame_bytes], pending[frame_bytes:]
                    await self._send_audio(bytes(frame))
                    sent_frames += 1
                    await self._pace(sent_frames, started)
            if pending:
                await self._send_audio(bytes(pending))
        except SarvamError as exc:
            # Out of credits, or a bad voice/model pairing. The caller is on the line, so
            # say nothing rather than dropping them; the turn is already logged.
            log.warning("Speech unavailable mid-call: %s", exc)

    async def _pace(self, sent_frames: int, started: float) -> None:
        """Hold until the line has caught up, less the lead."""
        due = started + (sent_frames - LEAD_FRAMES) * self.carrier.frame_ms / 1000
        delay = due - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)

    async def _send_audio(self, pcm: bytes) -> None:
        mulaw = codec.to_telephony(pcm, TELEPHONY_SAMPLE_RATE)
        await self.socket.send(self.carrier.audio_message(mulaw, self.call_id))

    async def _stop_speaking(self) -> None:
        """Cut the agent off, and drop whatever the carrier has already buffered."""
        task, self._speaking = self._speaking, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - teardown
                pass
        clear = self.carrier.clear_message(self.call_id)
        if clear:
            await self.socket.send(clear)

    async def _await_speech(self) -> None:
        if self._speaking is not None and not self._speaking.done():
            try:
                await self._speaking
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
