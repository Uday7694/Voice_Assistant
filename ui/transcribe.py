"""One recorded clip in, one transcript out.

The console records a whole utterance in the browser and sends it up as a finished clip,
which is a different shape from the phone call `Listener` serves: there is no barge-in to
support and no partial to act on, only a blob of audio and a question about what it says.

So the clip is fed through the same streaming ear a call uses, in real frame-sized
pieces, and the events are collected rather than reacted to. Using the streaming path for
a batch job looks roundabout and is deliberate: it is the code the phone call runs, so
what the console hears is what a caller would be heard as — same model, same sample rate,
same framing, same failure modes.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from brain.speech.sarvam import Ear
from brain.speech.types import WEB_SAMPLE_RATE

log = logging.getLogger(__name__)

# How long to wait for the transcript after the audio has all been sent. The service
# emits speech_end first and the text a beat later, so cutting this too fine returns
# nothing for a clip that was transcribed perfectly well.
TAIL_TIMEOUT = 8.0

# Frames are sent at the rate they would arrive on a call rather than as fast as the
# socket will take them. The service's voice-activity detection is timing-based; firing a
# ten-second clip at it in fifty milliseconds produces one long "still talking".
FRAME_MS = 100


def to_pcm(sample_rate: int, samples: np.ndarray) -> tuple[int, bytes]:
    """Browser audio to the mono 16-bit PCM every other layer speaks."""
    audio = np.asarray(samples)
    if audio.ndim > 1:
        # Gradio hands back (frames, channels) for a stereo device. Average rather than
        # take channel 0: a headset with one dead channel is otherwise silent.
        audio = audio.mean(axis=1)

    if audio.dtype.kind == "f":
        # Float input is nominally -1..1 but clips over it; scaling without the clamp
        # wraps a loud syllable round to full-scale noise.
        audio = np.clip(audio, -1.0, 1.0) * 32767.0
    elif audio.dtype == np.int32:
        audio = audio / 65536.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) * 256.0

    return sample_rate, audio.astype(np.int16).tobytes()


def resample(pcm: bytes, source_rate: int, target_rate: int = WEB_SAMPLE_RATE) -> bytes:
    """Nearest-sample rate conversion.

    Deliberately the cheap kind. Browsers hand over 44.1 or 48 kHz and the ear wants 16,
    which is a downsample — the anti-aliasing a proper resampler buys back sits above
    8 kHz, where speech recognition is not listening. Measured against the live service,
    transcripts came back identical to a windowed-sinc conversion.
    """
    if source_rate == target_rate or not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16)
    count = int(len(samples) * target_rate / source_rate)
    if count <= 0:
        return b""
    index = (np.arange(count) * (source_rate / target_rate)).astype(np.int64)
    return samples[np.clip(index, 0, len(samples) - 1)].tobytes()


async def transcribe(pcm: bytes, language: str, *, sample_rate: int = WEB_SAMPLE_RATE) -> str:
    """What was said in this clip, or "" if nothing was."""
    if not pcm:
        return ""

    frame = int(sample_rate * FRAME_MS / 1000) * 2
    heard: list[str] = []
    done = asyncio.Event()

    async with Ear(language=language, sample_rate=sample_rate) as ear:

        async def push() -> None:
            for start in range(0, len(pcm), frame):
                await ear.feed(pcm[start : start + frame])
                await asyncio.sleep(FRAME_MS / 1000)
            # A beat of silence, then a flush: together they tell the service the caller
            # has stopped rather than paused, which is what releases the transcript.
            await ear.feed(bytes(frame))
            await ear.flush()

        async def read() -> None:
            # The text can land either side of the end-of-speech signal — measured, it
            # usually lands after — so the clip is finished when both are in, not when
            # whichever came first arrived. Waiting on the end signal alone returns
            # nothing; stopping at the first transcript truncates anyone who paused.
            stopped = False
            async for event in ear.events():
                if event.kind == "transcript" and event.text.strip():
                    heard.append(event.text.strip())
                    if stopped:
                        break
                elif event.kind == "speech_end":
                    stopped = True
                    if heard:
                        break
            done.set()

        sender = asyncio.create_task(push())
        reader = asyncio.create_task(read())
        try:
            await asyncio.wait_for(
                done.wait(), timeout=len(pcm) / (sample_rate * 2) + TAIL_TIMEOUT
            )
        except asyncio.TimeoutError:
            log.warning("Transcription timed out with %d segment(s) so far", len(heard))
        finally:
            for task in (sender, reader):
                task.cancel()

    return " ".join(heard)
