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
from dataclasses import replace
from typing import AsyncIterator

from sarvamai import AsyncSarvamAI

from . import cache as audio_cache
from .prosody import speakable
from .warm import SocketWarmer
from .types import TTS_SAMPLE_RATE, WEB_SAMPLE_RATE, SpeechEvent
from .voice import VoiceProfile, profile_for

log = logging.getLogger(__name__)

# The SDK also accepts saaras:v4; v3 stays the default until v4 is measured here,
# because a newer model is not automatically a faster one and latency is the budget.
STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
# The SDK default speaker is "anushka", which is a bulbul:v2 voice. Pairing it with our
# v3 default is rejected at configure time with a 422 before any audio is generated —
# the two defaults are incompatible out of the box, so this must be set explicitly.
DEFAULT_SPEAKER = os.getenv("SARVAM_SPEAKER", "ishita")

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

# Only meaningful for a compressed codec (mp3/opus); linear16 carries its own fixed
# rate. Sent regardless so switching codec does not silently drop to the SDK default.
TTS_BITRATE = os.getenv("SARVAM_TTS_BITRATE", "128k")


def _tts_connect():
    """Open a TTS socket. One place, so the warmer and a cold utterance agree."""
    return client().text_to_speech_streaming.connect(
        model=TTS_MODEL, send_completion_event="true"
    )


# Process-wide, because connections are per-process and a call does not own the pool.
# Idle until `warm_tts()` is called, so nothing is opened by merely importing this.
TTS_WARMER = SocketWarmer(_tts_connect)


def warm_tts() -> None:
    """Start keeping TTS connections ready.

    Call it once the event loop is running and a call is plausible - at the top of a
    call, or when the bridge accepts a connection. Costs one socket per pool slot and
    saves roughly 600 ms on the first thing the agent says.
    """
    TTS_WARMER.start()


async def close_tts_warmer() -> None:
    await TTS_WARMER.aclose()


# One SDK client per event loop, not one per connection.
#
# Constructing AsyncSarvamAI measured 357 ms — before a single packet moves, on every
# utterance the agent speaks and every time the ear opens. It builds the HTTP client
# stack underneath, and that work is identical every time. Reusing it is the largest
# single latency win in the speech path and costs nothing but this dictionary.
#
# Keyed by loop rather than kept in one global because the client binds to the loop that
# made it: a client built under one asyncio.run and used under the next fails in ways
# that look like network flakiness. Tests create a loop per test, which is exactly where
# that would have bitten.
_CLIENTS: dict[int, AsyncSarvamAI] = {}


def client() -> AsyncSarvamAI:
    key = os.getenv("SARVAM_API_KEY", "").strip()
    if not key:
        raise RuntimeError("SARVAM_API_KEY is not set.")
    try:
        loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        # Called outside a loop — no reuse to be had, and no loop to key on.
        return AsyncSarvamAI(api_subscription_key=key)

    existing = _CLIENTS.get(loop_id)
    if existing is None:
        existing = AsyncSarvamAI(api_subscription_key=key)
        _CLIENTS[loop_id] = existing
    return existing


def forget_clients() -> None:
    """Drop cached clients. For tests, and for a process changing its API key."""
    _CLIENTS.clear()


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

    That per-utterance socket is also what makes prosody possible: pace, pitch and
    loudness are connection settings, so each line can be delivered differently
    depending on what it is doing. Pass ``kind`` ("greeting", "readback", "apology",
    "filler") and the voice adjusts the way a person's would.
    """

    def __init__(
        self,
        language: str = "en-IN",
        *,
        voice: str = "meera",
        kind: str = "ask",
        profile: VoiceProfile | None = None,
        markers: tuple[str, ...] = (),
        speaker: str | None = None,
        codec: str = "linear16",
        sample_rate: int = TTS_SAMPLE_RATE,
        pace: float | None = None,
    ) -> None:
        self.language = language
        self.kind = kind
        # Words that read as throat-clearing in this call's language, so prosody knows
        # where a beat belongs. Supplied by the caller because only the line book knows
        # them, and only for the language actually being spoken.
        self.markers = markers
        # An explicit profile wins; otherwise the named voice decides, shaped by kind.
        # `speaker` and `pace` stay as overrides for probes and A/B listening, where
        # naming one voice against one line is the entire point.
        base = profile or profile_for(voice, language, kind)
        if speaker:
            base = replace(base, speaker=speaker)
        if pace is not None:
            base = replace(base, pace=pace)
        self.profile = base
        self.codec = codec
        self.sample_rate = sample_rate

    @property
    def speaker(self) -> str:
        return self.profile.speaker

    @property
    def pace(self) -> float:
        return self.profile.pace

    async def say(self, chunks: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Speak a stream of text, yielding audio as it is generated.

        Takes an iterator rather than a string so the planner's sentences can be pushed
        in as they are produced — that overlap is where the perceived latency goes.
        """
        check_speaker(self.profile.speaker, TTS_MODEL)

        # A pre-opened connection when one is ready, otherwise open one here and wait.
        # The utterance owns whichever it gets and closes it either way: barge-in works
        # by closing the socket, so a shared one could not be interrupted.
        warmed = TTS_WARMER.take()
        ctx = warmed[0] if warmed else _tts_connect()
        socket = warmed[1] if warmed else await ctx.__aenter__()

        try:
            await socket.configure(
                target_language_code=self.language,
                speaker=self.profile.speaker,
                speech_sample_rate=self.sample_rate,
                output_audio_codec=self.codec,
                output_audio_bitrate=TTS_BITRATE,
                pace=self.profile.pace,
                pitch=self.profile.pitch,
                loudness=self.profile.loudness,
                min_buffer_size=MIN_BUFFER_SIZE,
                max_chunk_length=MAX_CHUNK_LENGTH,
            )

            # Text goes in on a background task while audio comes out here, so the
            # first frames play while later sentences are still being written.
            sender = asyncio.create_task(_send_all(socket, chunks, self.language, self.markers))
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
        finally:
            # Always, and on every path: a normal finish, an error, and above all
            # cancellation, which is what barge-in is. Closing the socket is both how the
            # connection is released and how Bulbul is told to stop talking, so a missed
            # close here is a leaked connection and an agent that will not shut up.
            with suppress(Exception):
                await ctx.__aexit__(None, None, None)

    async def say_once(self, text: str, *, cache: bool = False) -> AsyncIterator[bytes]:
        """Speak one fixed line (greeting, filler, handoff).

        ``cache=True`` is for lines that are byte-identical on every call. The first
        one pays for synthesis; every later one replays from disk at no credit cost and
        no round trip, which is what makes a filler cheap enough to speak in the gap
        before the planner has produced anything.
        """
        if not cache:
            async for audio in self.say(_one(text)):
                yield audio
            return

        cache_key = audio_cache.key(
            text,
            speaker=self.profile.speaker,
            language=self.language,
            sample_rate=self.sample_rate,
            pace=self.profile.pace,
            pitch=self.profile.pitch,
            loudness=self.profile.loudness,
        )
        stored = audio_cache.load(cache_key)
        if stored is not None:
            yield stored
            return

        collected = bytearray()
        async for audio in self.say(_one(text)):
            collected.extend(audio)
            yield audio
        audio_cache.store(cache_key, bytes(collected))


async def _one(text: str) -> AsyncIterator[str]:
    yield text


async def _send_all(
    socket, chunks: AsyncIterator[str], language: str, markers: tuple[str, ...] = ()
) -> None:
    """Push text into the TTS socket as it arrives, then flush.

    Prosody is applied here rather than in the planner. The planner writes what the
    agent means; how it is broken up to be *said* is a property of speech, and putting
    it here means every caller of Mouth gets it without knowing about it.
    """
    async for text in chunks:
        spoken = speakable(text, language, markers)
        if spoken:
            await socket.convert(spoken)
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
