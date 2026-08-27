"""Voice profiles: who the agent sounds like, and how that changes line by line.

A single fixed speaker at a single fixed pace is most of why a synthetic voice reads as
synthetic. Real desk staff do not deliver a greeting, a digit read-back and an apology
at the same speed or the same pitch — the greeting is unhurried, digits slow down, an
apology drops. Bulbul exposes pace, pitch and loudness per connection, and `Mouth`
opens one connection per utterance anyway (there is no cancel message, so barge-in has
to close the socket), which means every utterance can carry its own settings for free.

Kept out of `flow.py` on purpose: an agent names a voice, it does not describe one, so
the flow graph stays independent of whichever speech vendor is behind it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

# What a line is for. The planner does not choose this; the orchestrator knows which
# path produced the text, which is exactly the information a human speaker has.
Kind = str  # "greeting" | "ask" | "readback" | "apology" | "close" | "filler"


@dataclass(frozen=True)
class VoiceProfile:
    speaker: str
    pace: float = 1.0
    pitch: float = 0.0
    loudness: float = 1.0

    def for_kind(self, kind: Kind) -> "VoiceProfile":
        """The same voice, adjusted for what this line is doing.

        Very small numbers. Past a few percent on pace, Bulbul stops sounding like a
        person changing gear and starts sounding like a tape at the wrong speed.
        """
        shift = _SHIFTS.get(kind)
        if shift is None:
            return self
        pace, pitch, loud = shift
        return replace(
            self,
            pace=round(self.pace * pace, 3),
            pitch=round(self.pitch + pitch, 3),
            loudness=round(self.loudness * loud, 3),
        )


# (pace multiplier, pitch offset, loudness multiplier)
#
# Deliberately a narrow band, and narrower than the first version of this table. Bulbul's
# pace parameter changes the rate of the *whole* utterance uniformly, which is not what a
# person does — a person slows down over the digits and keeps the rest at one speed. Set
# wide (0.88 for a read-back, 1.08 for a filler) the effect is not "she is being careful
# with the number", it is a tape running at the wrong speed, and consecutive lines lurch
# between fast and slow.
#
# So the rate barely moves, and the human-sounding part is bought where it actually
# lives: in the pauses between and inside sentences. See `prosody` and `gap_after`.
_SHIFTS: dict[str, tuple[float, float, float]] = {
    "greeting": (0.98, 0.0, 1.0),
    # Digits and reference codes. A touch slower and a touch louder, the way a person
    # reads a number off a screen — audible as care, not as a change of speed.
    "readback": (0.96, -0.02, 1.04),
    "apology": (0.97, -0.05, 0.97),
    "close": (0.98, 0.0, 1.0),
    # Backchannels are thrown away, not delivered: slightly quicker, clearly quieter.
    # Loudness carries this one, because it is what makes a filler read as an aside
    # rather than as a statement the caller has to answer.
    "filler": (1.03, 0.0, 0.88),
}


# Bulbul v3 speakers are multilingual: the same voice carries every language the model
# supports, so a voice is one setting, not a table with a row per language. An override
# exists for the case a listen test actually turns one up — a voice that reads warm in
# Hindi and clipped in Telugu — and it stays empty until that happens rather than being
# pre-filled with guesses nobody has heard.
# Pace 1.2: bulbul:v3's own rate reads slower than a hospital desk actually speaks, so
# the baseline is lifted globally and the per-kind table below still shapes it line by
# line. Overridable, because the right rate is a listening decision, not a constant.
DEFAULT = VoiceProfile(
    speaker=os.getenv("SARVAM_SPEAKER", "ishita"),
    pace=float(os.getenv("SARVAM_PACE", "1.2")),
)

VOICES: dict[str, VoiceProfile] = {"meera": DEFAULT}

# {voice name: {language tag: profile}}. Empty by design. Fill a row only after
# listening to that voice in that language and preferring something else.
OVERRIDES: dict[str, dict[str, VoiceProfile]] = {}


def profile_for(voice: str, language: str, kind: Kind = "ask") -> VoiceProfile:
    """The profile for a named voice in a language, shaped for what is being said."""
    base = OVERRIDES.get(voice, {}).get(language) or VOICES.get(voice) or DEFAULT
    return base.for_kind(kind)
