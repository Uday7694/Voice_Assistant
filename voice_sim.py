"""Simulate a call: real brain, real turn-taking, printed audio instead of sound.

Proves the controller against the actual `Brain.handle()` contract, and lets you watch
barge-in behave without a microphone or a Sarvam key.

    python voice_sim.py
"""

from __future__ import annotations

import asyncio
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from brain.agents import HOSPITAL_AGENT  # noqa: E402
from brain.orchestrator import Brain  # noqa: E402
from brain.speech.sarvam import SpeechEvent  # noqa: E402
from brain.voice.controller import TurnController  # noqa: E402



class PrintingSynth:
    """Stands in for Bulbul: prints each sentence, emits one frame per word."""

    def say(self, chunks):
        return self._say(chunks)

    async def _say(self, chunks):
        async for text in chunks:
            print(f"  [{_ms()}] speaking: {text}")
            for word in text.split():
                await asyncio.sleep(0.04)  # rough word-rate pacing
                yield word.encode()


class PrintingSink:
    def __init__(self) -> None:
        self.played: list[bytes] = []

    async def play(self, chunk: bytes) -> None:
        self.played.append(chunk)

    async def clear(self) -> None:
        print(f"  [{_ms()}] >>> CLEAR transport buffer (barge-in)")


class Caller:
    """A scripted caller. Floats are pauses; strings are what they say."""

    def __init__(self, script) -> None:
        self.script = script

    async def events(self):
        for item in self.script:
            if isinstance(item, float):
                await asyncio.sleep(item)
            elif item == "<interrupt>":
                print(f"  [{_ms()}] caller starts talking over the agent")
                yield SpeechEvent(kind="speech_start")
            else:
                print(f"\n[{_ms()}] caller: {item}")
                yield SpeechEvent(kind="transcript", text=item)


_T0 = time.monotonic()


def _ms() -> str:
    return f"{(time.monotonic() - _T0) * 1000:6.0f}ms"


SCRIPT = [
    "<call connected>",
    2.5,
    "hi, I want to book an appointment with a heart doctor",
    2.5,
    "my name is Asha Verma",
    1.2,
    "<interrupt>",          # cut the agent off mid-sentence
    0.3,
    "actually make it tomorrow morning",
    3.0,
]


async def main() -> int:
    brain = Brain(HOSPITAL_AGENT)
    session = brain.start(channel="sim")
    sink = PrintingSink()

    controller = TurnController(
        brain,
        session.session_id,
        Caller(SCRIPT),
        PrintingSynth(),
        sink,
        filler="One moment.",
        filler_after=0.8,
        barge_in_grace=0.2,
    )

    await controller.run()

    state = brain.get(session.session_id)
    print(f"\nbarge-ins: {controller.barge_ins}  final state: {controller.state.value}")
    print(f"node={state.node_id} slots={state.slots}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
