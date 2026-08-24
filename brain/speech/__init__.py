"""Speech I/O adapters.

Sarvam streaming STT/TTS, behind the ``Ear`` / ``Mouth`` protocols the turn-taking
controller depends on. Both directions are WebSockets: voice-activity signals arrive
while the caller is still talking, which is what drives barge-in, and TTS audio arrives
frame by frame rather than as one finished clip.

A second provider slots in here by satisfying the same two protocols — the controller
imports from this module, never from a provider directly.
"""

from __future__ import annotations

from .sarvam import Ear, Mouth, SarvamError
from .types import (
    LANGUAGE_NAMES,
    TELEPHONY_SAMPLE_RATE,
    WEB_SAMPLE_RATE,
    SpeechEvent,
    bare_language,
)

PROVIDER = "sarvam"

__all__ = [
    "Ear",
    "Mouth",
    "SarvamError",
    "SpeechEvent",
    "PROVIDER",
    "LANGUAGE_NAMES",
    "TELEPHONY_SAMPLE_RATE",
    "WEB_SAMPLE_RATE",
    "bare_language",
]
