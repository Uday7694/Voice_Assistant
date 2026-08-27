"""The local sound card: streaming playback out, streaming capture in.

This is the media layer for running the agent on a laptop. In production the same two
roles are filled by the telephony bridge — audio arrives as RTP frames and leaves the
same way — so nothing above this file knows a sound card exists. It trades in bare
16-bit mono PCM in both directions, which is what every other audio source in this
system already speaks.

Streaming rather than clip-at-a-time, and that distinction is the entire point of the
module. Writing a WAV per sentence and handing it to a blocking player costs the length
of the clip before the next one can start, so a two-sentence reply has a seam in the
middle of it; worse, nothing can interrupt a blocking call, so the caller cannot talk
over the agent. Here the audio device pulls from a buffer that the synthesiser fills as
frames arrive, which means the first frame plays while the last is still being
generated, and cutting the agent off is one method call that empties the buffer.

sounddevice (PortAudio) is the dependency. It ships prebuilt wheels on Windows, macOS
and Linux, and is the only part of the system that touches hardware.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import AsyncIterator

import sounddevice as sd

from .types import WEB_SAMPLE_RATE

log = logging.getLogger(__name__)

# How much audio the device asks for at a time. Small enough that stopping playback is
# heard as immediate — one block is the worst-case delay on a barge-in — and large
# enough not to spend the call in callbacks. 20 ms is the usual telephony frame.
BLOCK_MS = 20


def _block_bytes(sample_rate: int, milliseconds: int = BLOCK_MS) -> int:
    return int(sample_rate * milliseconds / 1000) * 2


class Playback:
    """A speaker you can push PCM into and cut off mid-sentence.

    ``write`` never blocks: it appends to a buffer that PortAudio drains on its own
    thread. Underruns are filled with silence rather than treated as errors, because a
    planner that is still writing is the normal case, not a fault.
    """

    def __init__(self, sample_rate: int = WEB_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._stream: sd.RawOutputStream | None = None
        # Set whenever the buffer empties, so `drain` can wait instead of polling hard.
        self._drained = threading.Event()
        self._drained.set()

    def __enter__(self) -> "Playback":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        if self._stream is not None:
            return
        block = _block_bytes(self.sample_rate) // 2
        self._stream = sd.RawOutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=block,
            callback=self._fill,
        )
        self._stream.start()

    def _fill(self, out, frames, _time, status) -> None:
        """PortAudio's callback. Runs on its own thread; must not block or allocate much."""
        if status:
            log.debug("Playback stream status: %s", status)
        wanted = frames * 2
        with self._lock:
            take = min(wanted, len(self._buffer))
            if take:
                out[:take] = bytes(self._buffer[:take])
                del self._buffer[:take]
            if take < wanted:
                # Underrun. Silence, not an error: the synthesiser is still upstream.
                out[take:wanted] = b"\x00" * (wanted - take)
            if not self._buffer:
                self._drained.set()

    def write(self, pcm: bytes) -> None:
        if not pcm:
            return
        with self._lock:
            self._buffer.extend(pcm)
            self._drained.clear()

    def stop(self) -> None:
        """Drop everything not yet played. This is barge-in.

        The buffer is emptied rather than the stream closed: the device stays open and
        ready, so the next thing the agent says starts without a device round trip.
        """
        with self._lock:
            self._buffer.clear()
            self._drained.set()

    @property
    def busy(self) -> bool:
        with self._lock:
            return bool(self._buffer)

    async def drain(self) -> None:
        """Wait until everything written has been played."""
        while True:
            if self._drained.wait(timeout=0):
                break
            await asyncio.sleep(BLOCK_MS / 1000)
        # The buffer is empty, but the device still holds roughly one block that has
        # been handed over and not yet reached the speaker. Without this the last
        # syllable of a reply is cut off by whatever happens next.
        await asyncio.sleep(BLOCK_MS / 1000 * 2)

    def close(self) -> None:
        self.stop()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class Microphone:
    """The default input device, as an async iterator of PCM frames.

    Capture runs continuously from the moment it opens, including while the agent is
    speaking — that overlap is what makes barge-in possible at all. A microphone that
    is only opened when it is the caller's turn cannot hear them interrupt.
    """

    def __init__(self, sample_rate: int = WEB_SAMPLE_RATE, *, frame_ms: int = 100) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._stream: sd.RawInputStream | None = None

    def __enter__(self) -> "Microphone":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=_block_bytes(self.sample_rate, self.frame_ms) // 2,
            callback=self._capture,
        )
        self._stream.start()

    def _capture(self, data, _frames, _time, status) -> None:
        if status:
            log.debug("Capture stream status: %s", status)
        # A plain thread-safe queue rather than an asyncio one: this runs on PortAudio's
        # thread, which has no event loop and must never wait for one.
        self._queue.put(bytes(data))

    async def frames(self) -> AsyncIterator[bytes]:
        """Yield captured frames until the microphone is closed."""
        loop = asyncio.get_running_loop()
        while self._stream is not None:
            frame = await loop.run_in_executor(None, self._next_frame)
            if frame:
                yield frame

    def _next_frame(self) -> bytes:
        try:
            return self._queue.get(timeout=0.2)
        except queue.Empty:
            return b""

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
