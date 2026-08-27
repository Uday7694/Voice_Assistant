"""Voice Desk: a browser console for holding calls with the agent and watching it think.

Voice to voice. You speak, the same ear a phone call uses transcribes it, the brain takes
the turn, and the reply is synthesised straight back — streamed, so it starts playing
while the rest is still being written. Typing does the same thing without spending an STT
credit, which is how you re-run one awkward sentence twenty times.

The right-hand panel is the reason this exists rather than a chat window. Every turn the
brain emits what it decided — intent and confidence, values captured, step moved to,
tools called, milliseconds per stage — and none of that reaches a caller. Read together
with what was said, it turns "the agent got that wrong" into "the classifier read it as
out_of_scope at 0.9", which is a bug report.

One Brain per process, one session per browser tab.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
import time
from pathlib import Path

import gradio as gr
import numpy as np

from brain.agents.hospital import HOSPITAL_AGENT
from brain.models import (
    EndEvent,
    ErrorEvent,
    EscalateEvent,
    IntentEvent,
    SayEvent,
    TimingEvent,
    ToolEvent,
    TransitionEvent,
)
from brain.orchestrator import Brain
from brain.speech import Mouth, SarvamError
from brain.speech.sarvam import warm_tts
from brain.speech.types import TTS_SAMPLE_RATE, WEB_SAMPLE_RATE
from ui import player, trace
from ui.transcribe import resample, to_pcm, transcribe

log = logging.getLogger(__name__)

STYLES = (Path(__file__).parent / "styles.css").read_text(encoding="utf-8")

# Whether the browser can be fed audio a slice at a time.
#
# Gradio encodes each streamed chunk as AAC in an ADTS container, and the only thing that
# produces one is ffmpeg, with ffprobe beside it to read back what it wrote. Neither
# ships with Gradio or with pip.
#
# Absent them the console still talks, a sentence at a time — a real loss, since the
# first word then waits for the last one to be synthesised, but not a reason to refuse to
# start. Install ffmpeg and this turns itself on.
FFMPEG_READY = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))

# Frames of synthesised audio per streamed slice. A frame is what Bulbul emits at a time;
# a dozen is a fraction of a second, small enough that playback starts almost at once and
# large enough not to spend the turn in HTTP round trips.
CHUNK_FRAMES = 12

# Fraction of a sentence's own duration to wait before handing over the next one, when
# slices are not available. Under 1.0 deliberately: an autoplaying player replaces what
# it is playing, so arriving a little early is a clipped final syllable, and arriving
# late is a gap. The overlap is where a person breathes.
SENTENCE_PACING = 0.92

# Pause between two sentences of one reply.
GAP_BETWEEN_SENTENCES_MS = 220


# Below this peak level a clip is silence, not speech. Generous: a quiet room recorded at
# a low input gain still peaks well above it, while a muted or unpermitted microphone
# yields a flat zero.
SILENCE_LEVEL = 200

# Shorter than this and there is nothing to transcribe — a stray click on record, or a
# clip that failed to capture.
MIN_CLIP_SECONDS = 0.3


def _level(pcm: bytes) -> int:
    """Peak sample of some 16-bit audio, 0-32767."""
    if not pcm:
        return 0
    return int(np.abs(np.frombuffer(pcm, dtype=np.int16)).max())


# Enough digits to be a number worth slowing down for: a reference code or a phone
# number, not "2 pm".
DIGITS_IN_A_READBACK = 4


def _kind_of(text: str) -> str:
    """Deliver a line the way its content deserves. Mirrors talk.py."""
    lowered = text.lower()
    if "sorry" in lowered or "apolog" in lowered or "माफ़" in text:
        return "apology"
    if sum(char.isdigit() for char in text) >= DIGITS_IN_A_READBACK:
        return "readback"
    return "ask"


def _last_agent_lines(history: list) -> list[str]:
    """The agent's reply at the end of the transcript: everything after the last caller turn."""
    lines: list[str] = []
    for message in reversed(history or []):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            break
        content = str(message.get("content") or "").strip()
        if content:
            lines.append(content)
    return list(reversed(lines))


class Console:
    """The app's state: one brain, many browser sessions."""

    def __init__(self) -> None:
        self.brain = Brain(HOSPITAL_AGENT)

    # --- session ---------------------------------------------------------

    async def open_call(self, language: str):
        """Start a call and speak the opener.

        The opener is cached on disk after the first call in a language, so this is
        usually a file read rather than a round trip.
        """
        # Before anything is synthesised: the handshake costs 850 ms on the first
        # utterance and none of it has to be spent while the caller waits.
        warm_tts()

        session = self.brain.start(channel="console")
        session = session.model_copy(update={"language": language})
        self.brain.store.put(session)

        await self.brain.lines.warm(language)
        opener = await self.brain.lines.line("opener", language)
        history = [{"role": "assistant", "content": opener}]
        yield session.session_id, history, trace.render({}), player.SILENT
        yield (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            await self._player_for([opener], language, kind="greeting", cache=True),
        )

    # --- one turn --------------------------------------------------------

    async def take_turn(self, session_id: str, language: str, said: str, history: list):
        """Run a turn: transcript and telemetry out, sentences to speak handed on.

        Four outputs, the last of them the voice. The transcript and the telemetry stream
        as the turn happens; the audio arrives once the reply is written, because it is
        embedded in the page rather than streamed (see ui/player.py for why).
        """
        history = list(history) + [{"role": "user", "content": said}]
        turn: dict = {"heard": said, "tools": []}
        yield history, trace.render(turn), "", player.SILENT

        started = time.perf_counter()
        spoken: list[str] = []
        note = ""

        try:
            async for event in self.brain.handle(session_id, said):
                if isinstance(event, SayEvent):
                    spoken.append(event.text)
                    history = history + [{"role": "assistant", "content": event.text}]
                    turn["total_ms"] = (time.perf_counter() - started) * 1000
                    # The text appears before its audio, deliberately: reading the reply
                    # while it is being spoken is how you catch a wrong word.
                    yield history, trace.render(turn), gr.skip(), gr.skip()
                elif isinstance(event, IntentEvent):
                    turn["intent"] = event.intent.name
                    turn["confidence"] = event.intent.confidence
                    turn["slots"] = dict(event.intent.slots)
                elif isinstance(event, TransitionEvent):
                    turn["node"] = event.to_node
                    if event.from_node != event.to_node:
                        turn["moved"] = f"{event.from_node} → {event.to_node}"
                elif isinstance(event, ToolEvent):
                    turn["tools"].append({"name": event.name, "ok": event.ok, "ms": event.ms})
                elif isinstance(event, TimingEvent):
                    turn["stage_ms"] = dict(event.stage_ms)
                elif isinstance(event, EscalateEvent):
                    note = f"Escalated to a human: {event.reason}"
                elif isinstance(event, EndEvent):
                    note = f"Call ended: {event.reason}"
                elif isinstance(event, ErrorEvent):
                    note = f"Error: {event.message}"
        except Exception as exc:  # noqa: BLE001 - a failed turn must not kill the console
            log.exception("Turn failed")
            note = f"{type(exc).__name__}: {exc}"

        # The session is the authority on what is known by the end of the turn: the
        # intent carries only what this utterance added, and a tool may have corrected it.
        session = self.brain.store.get(session_id)
        if session is not None:
            turn["slots"] = dict(session.slots)
            turn.setdefault("node", session.node_id)
        turn["total_ms"] = (time.perf_counter() - started) * 1000
        turn["note"] = note
        if not spoken and not note:
            turn["note"] = "The agent said nothing this turn."
        yield history, trace.render(turn), "", await self._player_for(spoken, language)

    async def _player_for(
        self, lines: list[str], language: str, *, kind: str = "", cache: bool = False
    ) -> str:
        """Everything the agent said this turn, as one autoplaying element."""
        pcm = bytearray()
        for text in lines:
            async for chunk in self._voice(
                text, language, kind=kind or _kind_of(text), cache=cache
            ):
                pcm.extend(chunk[1].tobytes())
            # A beat between sentences. Concatenated sample to sample they run together
            # into one breathless line.
            pcm.extend(bytes(int(TTS_SAMPLE_RATE * GAP_BETWEEN_SENTENCES_MS / 1000) * 2))
        if not pcm:
            return player.SILENT
        samples = np.frombuffer(bytes(pcm), dtype=np.int16)
        log.info("Speaking %.2fs of %s", len(samples) / TTS_SAMPLE_RATE, language)
        return player.element(TTS_SAMPLE_RATE, samples, token=uuid.uuid4().hex[:8])

    # --- speech out ------------------------------------------------------

    async def _voice(self, text: str, language: str, *, kind: str, cache: bool = False):
        """Audio for one spoken line.

        The only place that talks to the synthesiser, so the two delivery modes differ in
        exactly one thing — how finely the audio is cut — and every caller gets whichever
        one the browser can play.
        """
        mouth = Mouth(language=language, kind=kind, sample_rate=TTS_SAMPLE_RATE)
        buffer = bytearray()
        frames = 0
        try:
            async for frame in mouth.say_once(text, cache=cache):
                buffer.extend(frame)
                frames += 1
                if FFMPEG_READY and frames >= CHUNK_FRAMES:
                    yield TTS_SAMPLE_RATE, np.frombuffer(bytes(buffer), dtype=np.int16)
                    buffer, frames = bytearray(), 0
        except SarvamError as exc:
            # Out of credits, or a bad speaker/model pairing. The text is already on
            # screen; losing the audio is not worth ending the call over.
            log.warning("Speech unavailable: %s", exc)
            return
        if buffer:
            yield TTS_SAMPLE_RATE, np.frombuffer(bytes(buffer), dtype=np.int16)

    # --- speech in -------------------------------------------------------

    async def heard(self, clip, language: str) -> str:
        """A recorded clip to the words in it.

        Loud about what it received. A clip that transcribes to nothing has three very
        different causes — a muted or unpermitted microphone (silence), a conversion that
        wrecked the samples (loud noise), or speech the model genuinely could not read —
        and they are indistinguishable from an empty string. The level and the duration
        tell them apart in one line of log.
        """
        if clip is None:
            log.warning("Microphone returned nothing at all")
            return ""

        sample_rate, samples = clip
        rate, pcm = to_pcm(sample_rate, samples)
        ready = resample(pcm, rate, WEB_SAMPLE_RATE)
        seconds = len(ready) / (WEB_SAMPLE_RATE * 2)
        log.info(
            "Heard %.2fs at %d Hz (%s), level %d/32767 after conversion",
            seconds,
            sample_rate,
            getattr(samples, "dtype", "?"),
            _level(ready),
        )

        if seconds < MIN_CLIP_SECONDS:
            log.warning("Clip too short to transcribe (%.2fs)", seconds)
            return ""
        if _level(ready) < SILENCE_LEVEL:
            log.warning("Clip is effectively silent; check the microphone and its level")
            return ""

        text = await transcribe(ready, language)
        log.info("Transcribed: %r", text)
        return text
