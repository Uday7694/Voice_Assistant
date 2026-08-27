"""On-disk audio cache for lines that never change.

The opener, the fillers and the handoff line are spoken on every single call. Paying
Bulbul to generate identical audio each time costs credits and, worse, costs the one
part of the call where latency is most visible — the first second.

Keyed on everything that can change the waveform, so a voice or pace change misses the
cache rather than replaying the old sound. Stores raw PCM, not WAV: the container is
cheap to rebuild and the sample rate is already in the key.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

CACHE_DIR = Path(os.getenv("VOICE_CACHE_DIR", ".cache/voice"))


def key(text: str, *, speaker: str, language: str, sample_rate: int, pace: float,
        pitch: float = 0.0, loudness: float = 1.0) -> str:
    raw = f"{text}|{speaker}|{language}|{sample_rate}|{pace}|{pitch}|{loudness}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def load(cache_key: str) -> bytes | None:
    path = CACHE_DIR / f"{cache_key}.pcm"
    try:
        return path.read_bytes()
    except OSError:
        return None


def store(cache_key: str, pcm: bytes) -> None:
    """Best effort. A cache that cannot write must not end a call."""
    if not pcm:
        return
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Write beside the target and rename, so a crash mid-write cannot leave a
        # truncated clip that every later call happily replays.
        tmp = CACHE_DIR / f"{cache_key}.part"
        tmp.write_bytes(pcm)
        tmp.replace(CACHE_DIR / f"{cache_key}.pcm")
    except OSError:
        pass
