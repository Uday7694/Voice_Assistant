"""Backchannels: the tiny sound a person makes before they answer.

Two problems, one fix. The human one is that silence between the caller finishing and
the agent starting is the tell — people do not go quiet for 800 ms and then deliver a
complete sentence, they say "right" while they are still thinking. The engineering one
is that the planner needs those 800 ms and cannot be made to need none.

So the gap gets filled with a syllable that costs nothing: the lines are fixed, so their
audio is synthesised once and cached, and every later call replays it at zero latency
and zero credits. That is the whole trick — perceived time to first audio becomes a
cache read, not a model call.

The words themselves come from the line book, which writes them in whatever language
the call is in. Nothing here knows any language: this module decides *when* a
backchannel is worth saying and *which* of the available ones, and both of those
questions have the same answer in every language.
"""

from __future__ import annotations

import random
from typing import Sequence

# What a person says when there is nothing to say yet. Used only when the line book has
# nothing — no model, no cache — and a caller is already waiting.
FALLBACK_ACK = ("Right.",)
FALLBACK_HOLD = ("One moment.",)


class Backchannel:
    """Picks fillers, never the same one twice running.

    Repetition is what exposes a canned line. Three "Right."s in a row is worse than no
    filler at all, so the last choice is remembered per instance — which means one
    instance per call, not one per process.
    """

    # How often a two-word-plus turn earns an acknowledgement. Not 1.0: a receptionist
    # who says "right" before every single sentence is as obviously scripted as one who
    # never does. Variation is the thing being bought, not the filler itself.
    ACK_RATE = 0.45
    MIN_WORDS_FOR_ACK = 2

    def __init__(
        self,
        acks: Sequence[str] = (),
        holds: Sequence[str] = (),
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.acks = tuple(acks) or FALLBACK_ACK
        self.holds = tuple(holds) or FALLBACK_HOLD
        self._rng = rng or random.Random()
        self._last: dict[str, str] = {}
        self._acked = False

    @classmethod
    async def for_call(cls, book, language: str, *, rng: random.Random | None = None) -> "Backchannel":
        """Build one for a call, taking its words from the line book."""
        return cls(
            await book.variants("ack", language),
            await book.variants("hold", language),
            rng=rng,
        )

    def should_ack(self, user_text: str) -> bool:
        """Whether this turn deserves a backchannel before the real reply.

        Never twice running, and never after a one-word answer — acknowledging "yes"
        makes the agent sound like it is stalling, which is exactly the impression the
        filler exists to avoid.
        """
        if self._acked:
            self._acked = False
            return False
        if len(user_text.split()) < self.MIN_WORDS_FOR_ACK:
            return False
        self._acked = self._rng.random() < self.ACK_RATE
        return self._acked

    def ack(self) -> str:
        return self._pick(self.acks, "ack")

    def hold(self) -> str:
        return self._pick(self.holds, "hold")

    @property
    def markers(self) -> tuple[str, ...]:
        """Every filler word this call can use.

        Handed to the prosody layer, which needs to know which opening words are
        throat-clearing rather than content — those are exactly the words that need a
        beat after them. Derived rather than listed, so it is right in a language
        nobody wrote a list for.
        """
        return tuple(line.strip(" .,!?।") for line in self.acks + self.holds if line.strip())

    def _pick(self, options: tuple[str, ...], key: str) -> str:
        fresh = [o for o in options if o != self._last.get(key)] or list(options)
        choice = self._rng.choice(fresh)
        self._last[key] = choice
        return choice
