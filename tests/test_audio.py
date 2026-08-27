"""The playback buffer, without a sound card.

PortAudio's callback is just a function that asks for bytes, so the buffer it drains can
be exercised by calling it directly. That covers the part that matters — what happens to
audio that has been written but not yet played when the caller interrupts.
"""

from __future__ import annotations

import pytest

sd = pytest.importorskip("sounddevice", reason="audio device layer needs sounddevice")

from brain.speech.audio import Playback, _block_bytes  # noqa: E402


def _drained(playback: Playback, frames: int) -> bytes:
    """One turn of PortAudio's callback, returning what would reach the speaker."""
    out = bytearray(frames * 2)
    playback._fill(memoryview(out), frames, None, None)
    return bytes(out)


def test_written_audio_is_handed_to_the_device_in_order():
    playback = Playback(sample_rate=16000)
    playback.write(b"\x01\x02" * 4)
    playback.write(b"\x03\x04" * 4)
    assert _drained(playback, 8) == b"\x01\x02" * 4 + b"\x03\x04" * 4


def test_an_underrun_is_filled_with_silence_not_an_error():
    """The synthesiser is still upstream; that is the normal case, not a fault."""
    playback = Playback(sample_rate=16000)
    playback.write(b"\x01\x02")
    assert _drained(playback, 4) == b"\x01\x02" + b"\x00" * 6


def test_interrupting_drops_audio_that_has_not_been_played():
    playback = Playback(sample_rate=16000)
    playback.write(b"\x01\x02" * 100)
    playback.stop()
    assert not playback.busy
    assert _drained(playback, 4) == b"\x00" * 8


def test_the_device_is_ready_again_after_an_interruption():
    """stop() empties the buffer; it must not wedge the stream for the next reply."""
    playback = Playback(sample_rate=16000)
    playback.write(b"\x01\x02" * 100)
    playback.stop()
    playback.write(b"\x05\x06" * 2)
    assert _drained(playback, 2) == b"\x05\x06" * 2


def test_busy_reports_whether_anything_is_still_waiting_to_be_heard():
    playback = Playback(sample_rate=16000)
    assert not playback.busy
    playback.write(b"\x01\x02" * 4)
    assert playback.busy
    _drained(playback, 4)
    assert not playback.busy


@pytest.mark.asyncio
async def test_drain_returns_once_the_buffer_is_empty():
    playback = Playback(sample_rate=16000)
    playback.write(b"\x01\x02" * 4)
    _drained(playback, 4)
    await playback.drain()   # must not hang


def test_a_block_is_twenty_milliseconds_of_sixteen_bit_mono():
    assert _block_bytes(16000, 20) == 640
    assert _block_bytes(24000, 20) == 960
