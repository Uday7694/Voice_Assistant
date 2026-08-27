"""Shared speech types.

Lives apart from any provider so the turn-taking controller depends on the contract
rather than on a particular speech vendor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Telephony is 8 kHz; browser capture is 16 kHz. The value must match between session
# setup and every audio frame.
TELEPHONY_SAMPLE_RATE = 8000
WEB_SAMPLE_RATE = 16000

# Output rate for synthesis. Independent of the capture rate above: the ear only needs
# 16 kHz to transcribe, but bulbul:v3 renders at 24 kHz, and downsampling its output to
# match the microphone throws away the top of the voice for nothing.
TTS_SAMPLE_RATE = 24000


@dataclass(frozen=True)
class SpeechEvent:
    """What the ear reports upward.

    ``speech_start`` is the barge-in trigger and must reach the controller while the
    agent is still speaking. A provider that does not emit voice activity natively has
    to derive it locally rather than infer it after a transcript arrives — by then the
    caller has been talking over the agent for a second or more.
    """

    kind: Literal["speech_start", "speech_end", "transcript"]
    text: str = ""


LANGUAGE_NAMES = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "te-IN": "Telugu",
    "ta-IN": "Tamil",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "bn-IN": "Bengali",
    "gu-IN": "Gujarati",
    "pa-IN": "Punjabi",
    "or-IN": "Odia",
    "as-IN": "Assamese",
}


def bare_language(tag: str) -> str:
    """'hi-IN' -> 'hi'. For services that key on the bare ISO code; the brain uses BCP-47."""
    return (tag or "en-IN").split("-")[0].lower()
