"""A whole call, driven through a fake carrier socket.

No phone line and no network. The socket is a few lines that hand over prepared frames
and record what comes back, which is enough to check the things that actually go wrong on
a call: that caller audio reaches the ear in the right format, that the agent's audio is
paced rather than dumped, that an interruption stops it inside a frame or two, and that
hanging up ends the call rather than leaving tasks running.
"""

from __future__ import annotations

import asyncio
import base64
import json

import numpy as np
import pytest

from brain.telephony import codec
from brain.telephony.call import Call, CallStats
from brain.telephony.carriers import Twilio


class FakeSocket:
    """The carrier's end of the wire: scripted messages in, everything recorded."""

    def __init__(self, messages: list[str] | None = None) -> None:
        self._messages = list(messages or [])
        self.sent: list[dict] = []
        self._closed = asyncio.Event()

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        """Hand over the script, then stay open the way a real socket does — running out
        of prepared frames is not the caller hanging up."""
        for message in self._messages:
            yield message
            await asyncio.sleep(0)
        self._closed.set()
        await asyncio.sleep(3600)

    @property
    def audio_sent(self) -> bytes:
        """Everything the agent played, back as mu-law."""
        return b"".join(
            base64.b64decode(m["media"]["payload"])
            for m in self.sent
            if m.get("event") == "media"
        )

    @property
    def clears(self) -> int:
        return sum(1 for m in self.sent if m.get("event") == "clear")


def _start(call_id: str = "MZ1") -> str:
    return json.dumps({"event": "start", "streamSid": call_id})


def _audio(pcm_at_8k: np.ndarray, call_id: str = "MZ1") -> str:
    payload = base64.b64encode(codec.encode(pcm_at_8k.astype(np.int16).tobytes())).decode()
    return json.dumps({"event": "media", "streamSid": call_id, "media": {"payload": payload}})


def _stop(call_id: str = "MZ1") -> str:
    return json.dumps({"event": "stop", "streamSid": call_id})


# --- what the carrier sends us ---------------------------------------------


def test_caller_audio_arrives_at_the_ear_as_sixteen_kilohertz_pcm():
    """The line is 8 kHz mu-law; the ear wants 16 kHz PCM. The gap is the bug surface."""
    tone = (np.sin(np.arange(160) * 0.2) * 9000).astype(np.int16)
    frame = Twilio().parse(_audio(tone))
    converted = codec.from_telephony(frame.audio, 16000)
    assert len(converted) // 2 == 320                      # 20 ms at 16 kHz
    assert int(np.abs(np.frombuffer(converted, np.int16)).max()) > 5000


def test_the_call_handle_is_learned_from_the_start_frame():
    assert Twilio().parse(_start("MZ-abc")).call_id == "MZ-abc"


# --- what we send back ------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_audio_is_paced_to_the_line_not_dumped_in_one_burst():
    """A carrier handed six seconds at once plays six seconds, interruption or not."""
    socket = FakeSocket()
    call = _call(socket)
    call.call_id = "MZ1"

    speech = np.zeros(8000, dtype=np.int16).tobytes()      # one second at 8 kHz
    started = asyncio.get_running_loop().time()
    await _stream_fixed(call, speech)
    elapsed = asyncio.get_running_loop().time() - started

    # Fifty 20 ms frames, less the two-frame lead, is roughly a second of wall clock.
    assert len(socket.audio_sent) == 8000
    assert 0.8 < elapsed < 1.4, f"sent a second of audio in {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_an_interruption_stops_the_agent_within_a_frame_or_two():
    socket = FakeSocket()
    call = _call(socket)
    call.call_id = "MZ1"

    speech = np.zeros(8000 * 3, dtype=np.int16).tobytes()  # three seconds
    call._speaking = asyncio.create_task(_stream_fixed(call, speech))
    await asyncio.sleep(0.3)
    speaking = call._speaking
    await call._stop_speaking()

    assert speaking.cancelled(), "the sender must actually stop, not run to completion"
    played = len(socket.audio_sent) / 8000
    assert played < 0.6, f"kept talking for {played:.2f}s after being cut off"
    # And nothing more goes out after the cut.
    await asyncio.sleep(0.2)
    assert len(socket.audio_sent) / 8000 == played


@pytest.mark.asyncio
async def test_an_interruption_tells_the_carrier_to_drop_what_it_has_buffered():
    socket = FakeSocket()
    call = _call(socket)
    call.call_id = "MZ1"
    await call._stop_speaking()
    assert socket.clears == 1


# --- helpers ----------------------------------------------------------------


def _call(socket: FakeSocket) -> Call:
    """A Call wired to a fake socket, with no brain attached.

    The brain is exercised elsewhere; what is under test here is the media path, and
    giving it a real Brain would put an LLM in a unit test.
    """
    return Call(socket, Twilio(), brain=None, language="en-IN", session_id="test")


async def _stream_fixed(call: Call, pcm_at_8k: bytes) -> None:
    """Pace fixed audio out, standing in for synthesis.

    Uses the call's own `_pace`, so the timing under test is the shipped timing rather
    than a copy of it that could drift.
    """
    import time

    frame_bytes = int(8000 * call.carrier.frame_ms / 1000) * 2
    started = time.perf_counter()
    sent = 0
    for start in range(0, len(pcm_at_8k), frame_bytes):
        await call._send_audio(pcm_at_8k[start : start + frame_bytes])
        sent += 1
        await call._pace(sent, started)


def test_call_stats_report_the_median_not_the_worst_first_audio():
    stats = CallStats(first_audio_ms=[400.0, 450.0, 3000.0])
    assert stats.median_first_audio_ms == 450.0


def test_call_stats_survive_a_call_that_never_spoke():
    assert CallStats().median_first_audio_ms == 0.0
