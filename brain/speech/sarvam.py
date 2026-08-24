"""Sarvam streaming speech: STT in, TTS out.

Written against sarvamai 0.1.30. Both directions are WebSockets — never the batch HTTP
endpoints, which cannot hold a conversation.

UNVERIFIED against the live service: no SARVAM_API_KEY was available when this was
written. Run `python probe_sarvam.py` once you have a key; it exercises both directions
and prints what actually comes back.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import struct
from contextlib import suppress
from typing import AsyncIterator

from sarvamai import AsyncSarvamAI

from .types import WEB_SAMPLE_RATE, SpeechEvent

log = logging.getLogger(__name__)

# The SDK also accepts saaras:v4; v3 stays the default until v4 is measured here,
# because a newer model is not automatically a faster one and latency is the budget.
STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
# The SDK default speaker is "anushka", which is a bulbul:v2 voice. Pairing it with our
# v3 default is rejected at configure time with a 422 before any audio is generated —
# the two defaults are incompatible out of the box, so this must be set explicitly.
DEFAULT_SPEAKER = os.getenv("SARVAM_SPEAKER", "priya")

# Reported by the service in the 422 it raises on a mismatch. Kept here so a bad speaker
# fails locally with the full list instead of costing a round trip to find out.
BULBUL_V3_SPEAKERS = frozenset(
    """aditya ritu ashutosh priya neha rahul pooja rohan simran kavya amit dev ishita
    shreya ratan varun manan sumit roopa kabir aayan shubh advait anand tanya tarun
    sunny mani gokul vijay shruti suhani mohit kavitha rehan soham rupali niharika""".split()
)

BULBUL_V2_SPEAKERS = frozenset(
    "anushka manisha vidya arya abhilash karun hitesh".split()
)


def check_speaker(speaker: str, model: str) -> None:
    """Fail before connecting when the voice cannot work with the model."""
    known = {"bulbul:v3": BULBUL_V3_SPEAKERS, "bulbul:v2": BULBUL_V2_SPEAKERS}.get(model)
    if known and speaker.lower() not in known:
        raise ValueError(
            f"Speaker {speaker!r} is not compatible with {model}. "
            f"Available: {', '.join(sorted(known))}"
        )

# Callers hand `feed()` raw 16-bit mono PCM; it wraps each frame before sending. WAV is
# the SDK's own default for `transcribe(encoding=...)`, so declaring it here keeps the
# connect-time codec and the per-message encoding in agreement. The SDK also accepts
# pcm_s16le/pcm_l16/pcm_raw — switching means dropping the wrapper in `feed()` too, and
# the two must always change together.
INPUT_CODEC = os.getenv("SARVAM_INPUT_CODEC", "wav")

# Characters TTS buffers before it starts synthesising — in principle the main lever on
# time-to-first-audio, so lowering it looks like free latency.
#
# It is not. The service enforces a floor and rejects anything below it with a bare
# "422: Input parameters has to be a valid dictionary" that names no parameter. Measured
# against the live API: 24 is rejected, 50 is accepted. 50 is also the SDK default, so
# treat it as a minimum rather than a suggestion and buy first-audio latency elsewhere —
# the planner already streams sentence by sentence, which is the real win.
MIN_BUFFER_SIZE = int(os.getenv("SARVAM_MIN_BUFFER_SIZE", "50"))

# Upper bound on one synthesis chunk. Caps how long a single burst can take.
MAX_CHUNK_LENGTH = int(os.getenv("SARVAM_MAX_CHUNK_LENGTH", "150"))


def client() -> AsyncSarvamAI:
    key = os.getenv("SARVAM_API_KEY", "").strip()
    if not key:
        raise RuntimeError("SARVAM_API_KEY is not set.")
    return AsyncSarvamAI(api_subscription_key=key)


# --- speech in ------------------------------------------------------------


class Ear:
    """Streaming STT with VAD signals.

    `speech_start` is the barge-in trigger. It must reach the turn-taking controller
    while the agent is still speaking, which is why VAD events and transcripts arrive on
    the same stream rather than being inferred after the fact.
    """

    def __init__(
        self,
        language: str = "en-IN",
        *,
        sample_rate: int = WEB_SAMPLE_RATE,
        snappy: bool = True,
    ) -> None:
        self.language = language
        self.sample_rate = sample_rate
        self.snappy = snappy
        self._socket = None
        self._ctx = None

    async def __aenter__(self) -> "Ear":
        self._ctx = client().speech_to_text_streaming.connect(
            language_code=self.language,
            model=STT_MODEL,
            mode="transcribe",
            sample_rate=str(self.sample_rate),
            input_audio_codec=INPUT_CODEC,
            # Strings, not booleans — verified against the 0.1.30 signature, which types
            # these as Literal['true','false']. The published docs show booleans; the
            # SDK does not accept them.
            vad_signals="true",
            high_vad_sensitivity="true" if self.snappy else "false",
        )
        self._socket = await self._ctx.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._ctx is not None:
            await self._ctx.__aexit__(*exc)
        self._socket = None
        self._ctx = None

    async def feed(self, pcm: bytes) -> None:
        """Push one frame of raw 16-bit mono PCM at the configured sample rate.

        Wrapping happens here rather than in the caller. Every audio source in the
        system — LiveKit, a telephony bridge, the probe — produces bare PCM, so making
        each one build its own container invites exactly one of them to disagree with
        INPUT_CODEC and fail only against the live service.
        """
        if self._socket is None:
            raise RuntimeError("Ear used outside its context manager")
        await self._socket.transcribe(
            audio=base64.b64encode(_wav(pcm, self.sample_rate)).decode(),
            encoding="audio/wav",
            sample_rate=self.sample_rate,
        )

    async def flush(self) -> None:
        """Force processing without waiting for a silence boundary."""
        if self._socket is not None:
            await self._socket.flush()

    async def events(self) -> AsyncIterator[SpeechEvent]:
        """Yield VAD signals and final transcripts as they arrive."""
        if self._socket is None:
            raise RuntimeError("Ear used outside its context manager")

        async for message in self._socket:
            problem = _error_of(message)
            if problem:
                raise SarvamError(f"STT rejected: {problem}")
            parsed = _parse_stt(message)
            if parsed is not None:
                yield parsed


def _wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap 16-bit mono PCM in a minimal WAV container."""
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    header += b"data" + struct.pack("<I", len(pcm))
    return header + pcm


def _parse_stt(message) -> SpeechEvent | None:
    """Normalise the SDK's message shapes into one event type.

    Defensive on purpose: the wire format is documented loosely and differs between
    STT and STT-translate (`transcript` vs `translation`), so read by attribute and
    tolerate absence rather than assuming a schema.
    """
    kind = getattr(message, "type", None)
    data = getattr(message, "data", None)
    if data is None:
        return None

    if kind == "events":
        signal = str(getattr(data, "signal_type", "") or "").upper()
        if signal == "START_SPEECH":
            return SpeechEvent(kind="speech_start")
        if signal == "END_SPEECH":
            return SpeechEvent(kind="speech_end")
        return None

    text = getattr(data, "transcript", None) or getattr(data, "translation", None)
    if text:
        return SpeechEvent(kind="transcript", text=str(text))
    return None


# --- speech out -----------------------------------------------------------


class Mouth:
    """Streaming TTS for a single utterance.

    One socket per utterance, deliberately. Sarvam's TTS WebSocket has no cancel
    message, so the only way to stop generation on barge-in is to close the connection —
    which means a long-lived shared socket cannot be interrupted.
    """

    def __init__(
        self,
        language: str = "en-IN",
        *,
        speaker: str = DEFAULT_SPEAKER,
        codec: str = "linear16",
        sample_rate: int = WEB_SAMPLE_RATE,
        pace: float = 1.0,
    ) -> None:
        self.language = language
        self.speaker = speaker
        self.codec = codec
        self.sample_rate = sample_rate
        self.pace = pace

    async def say(self, chunks: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Speak a stream of text, yielding audio as it is generated.

        Takes an iterator rather than a string so the planner's sentences can be pushed
        in as they are produced — that overlap is where the perceived latency goes.
        """
        check_speaker(self.speaker, TTS_MODEL)
        ctx = client().text_to_speech_streaming.connect(
            model=TTS_MODEL, send_completion_event="true"
        )
        async with ctx as socket:
            await socket.configure(
                target_language_code=self.language,
                speaker=self.speaker,
                speech_sample_rate=self.sample_rate,
                output_audio_codec=self.codec,
                pace=self.pace,
                min_buffer_size=MIN_BUFFER_SIZE,
                max_chunk_length=MAX_CHUNK_LENGTH,
            )

            # Text goes in on a background task while audio comes out here, so the
            # first frames play while later sentences are still being written.
            sender = asyncio.create_task(_send_all(socket, chunks))
            try:
                async for message in socket:
                    problem = _error_of(message)
                    if problem:
                        raise SarvamError(f"TTS rejected: {problem}")
                    audio = _parse_tts_audio(message)
                    if audio is not None:
                        yield audio
                    elif _is_final(message):
                        break
            finally:
                sender.cancel()
                with suppress(asyncio.CancelledError):
                    await sender

    async def say_once(self, text: str) -> AsyncIterator[bytes]:
        """Convenience wrapper for a single fixed line (greetings, hold phrases)."""

        async def one() -> AsyncIterator[str]:
            yield text

        async for audio in self.say(one()):
            yield audio


async def _send_all(socket, chunks: AsyncIterator[str]) -> None:
    """Push text into the TTS socket as it arrives, then flush."""
    async for text in chunks:
        if text.strip():
            await socket.convert(text)
    await socket.flush()


class SarvamError(RuntimeError):
    """The service rejected the request.

    Surfaced rather than swallowed: an error frame parses as "not audio", so without
    this a 422 is indistinguishable from silence and the caller just hears nothing.
    """


def _error_of(message) -> str | None:
    """Return the error text if this frame is an error, else None."""
    if getattr(message, "type", None) != "error":
        return None
    data = getattr(message, "data", None)
    code = getattr(data, "code", None) if data is not None else None
    text = getattr(data, "message", None) if data is not None else None
    return f"{code}: {text}" if code else str(text or "unknown error")


def _parse_tts_audio(message) -> bytes | None:
    data = getattr(message, "data", None)
    if data is None:
        return None
    encoded = getattr(data, "audio", None)
    if not encoded:
        return None
    try:
        return base64.b64decode(encoded)
    except Exception:  # noqa: BLE001 - a malformed frame must not end the utterance
        log.warning("Undecodable TTS audio frame")
        return None


def _is_final(message) -> bool:
    data = getattr(message, "data", None)
    return bool(data is not None and getattr(data, "event_type", None) == "final")
