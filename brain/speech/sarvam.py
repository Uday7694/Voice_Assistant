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
from contextlib import suppress
from dataclasses import dataclass
from typing import AsyncIterator, Literal

from sarvamai import AsyncSarvamAI

log = logging.getLogger(__name__)

# Telephony is 8 kHz; browser capture is 16 kHz. Sarvam accepts both, and the value must
# match between the connect call and every audio frame.
TELEPHONY_SAMPLE_RATE = 8000
WEB_SAMPLE_RATE = 16000

STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
DEFAULT_SPEAKER = os.getenv("SARVAM_SPEAKER", "anushka")


def client() -> AsyncSarvamAI:
    key = os.getenv("SARVAM_API_KEY", "").strip()
    if not key:
        raise RuntimeError("SARVAM_API_KEY is not set.")
    return AsyncSarvamAI(api_subscription_key=key)


# --- speech in ------------------------------------------------------------


@dataclass(frozen=True)
class SpeechEvent:
    """What the ear reports upward."""

    kind: Literal["speech_start", "speech_end", "transcript"]
    text: str = ""


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
            # Both are strings in this SDK, not booleans.
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
        """Push one frame of 16-bit mono PCM at the configured sample rate."""
        if self._socket is None:
            raise RuntimeError("Ear used outside its context manager")
        await self._socket.transcribe(
            audio=base64.b64encode(pcm).decode(),
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
            parsed = _parse_stt(message)
            if parsed is not None:
                yield parsed


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
            )

            # Text goes in on a background task while audio comes out here, so the
            # first frames play while later sentences are still being written.
            sender = asyncio.create_task(_send_all(socket, chunks))
            try:
                async for message in socket:
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
