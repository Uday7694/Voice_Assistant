"""Test doubles for the turn-taking controller.

They make the conversation deterministic: a scripted ear, a synthesizer that emits a
known number of audio frames with controllable pacing, and a sink that records exactly
what was played and when it was cleared.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from brain.models import EndEvent, EscalateEvent, SayEvent
from brain.speech.sarvam import SpeechEvent


class ScriptedSource:
    """Emits a fixed sequence of speech events, with optional delays between them."""

    def __init__(self, script: list[SpeechEvent | float]) -> None:
        self.script = script

    async def events(self) -> AsyncIterator[SpeechEvent]:
        for item in self.script:
            if isinstance(item, float):
                await asyncio.sleep(item)
                continue
            yield item


class FakeSynth:
    """Turns each sentence into a few labelled audio frames.

    `frame_delay` simulates synthesis pacing so a barge-in can land mid-utterance.
    """

    def __init__(self, frames_per_sentence: int = 3, frame_delay: float = 0.01) -> None:
        self.frames_per_sentence = frames_per_sentence
        self.frame_delay = frame_delay
        self.spoken: list[str] = []
        self.open_sockets = 0
        self.max_open_sockets = 0

    async def say(self, chunks: AsyncIterator[str]) -> AsyncIterator[bytes]:
        self.open_sockets += 1
        self.max_open_sockets = max(self.max_open_sockets, self.open_sockets)
        try:
            async for text in chunks:
                self.spoken.append(text)
                for index in range(self.frames_per_sentence):
                    await asyncio.sleep(self.frame_delay)
                    yield f"{text}#{index}".encode()
        finally:
            self.open_sockets -= 1


class RecordingSink:
    """Records played frames and clear() calls, in order."""

    def __init__(self) -> None:
        self.played: list[bytes] = []
        self.events: list[str] = []
        self.clears = 0

    async def play(self, chunk: bytes) -> None:
        self.played.append(chunk)
        self.events.append(f"play:{chunk.decode()}")

    async def clear(self) -> None:
        self.clears += 1
        self.events.append("clear")

    @property
    def transcript(self) -> str:
        return " ".join(c.decode().split("#")[0] for c in self.played)


class FakeBrain:
    """Replays canned agent replies, with controllable thinking time."""

    def __init__(
        self,
        replies: dict[str, list[str]] | None = None,
        *,
        think: float = 0.0,
        end_after: str | None = None,
        escalate_after: str | None = None,
    ) -> None:
        self.replies = replies or {}
        self.think = think
        self.end_after = end_after
        self.escalate_after = escalate_after
        self.heard: list[str] = []

    async def handle(self, session_id: str, text: str):
        self.heard.append(text)
        if self.think:
            await asyncio.sleep(self.think)

        for sentence in self.replies.get(text, [f"You said {text}."]):
            yield SayEvent(text=sentence)

        if self.escalate_after == text:
            yield EscalateEvent(reason="test")
        if self.end_after == text:
            yield EndEvent(reason="test")
