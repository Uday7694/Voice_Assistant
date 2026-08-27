"""The agent's voice, as a plain HTML audio element.

Gradio's own Audio component was the obvious thing to use and did not work here. In this
build it plays the first clip a page is given and never a second one: the audio is
synthesised, the browser fetches it, the player does not move — no error on either side,
which is the worst kind of failure to own. Five arrangements were tried (streamed slices,
a dedicated speaking event, state handoff, audio as the final yield, a whole version
downgrade) and every one behaved the same way.

So the player is ours. A `<audio autoplay>` written into an HTML block, carrying its
audio inline as a data URI, replaced wholesale each turn. A new element cannot be holding
a previous stream, which is the entire bug, and there is nothing between the samples and
the speaker but the browser.

The cost is that a reply is delivered whole rather than in slices: base64 in the page
means the audio has to exist before it can be sent. On a hospital desk reply — one or two
sentences — that is a fraction of a second against a component that does not play.
"""

from __future__ import annotations

import base64
import struct

import numpy as np

# A turn's audio is embedded in the page rather than served from a URL, so it cannot be
# stale, cached, or raced against the next turn. Base64 costs a third in size; a
# ten-second reply is still well under a megabyte.
SILENT = '<div class="vd-quiet">…</div>'


def wav_bytes(sample_rate: int, samples: np.ndarray) -> bytes:
    """16-bit mono WAV. The one container every browser plays without negotiation."""
    pcm = np.asarray(samples, dtype=np.int16).tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    return header + b"data" + struct.pack("<I", len(pcm)) + pcm


def element(sample_rate: int, samples: np.ndarray, *, token: str = "") -> str:
    """An autoplaying player for this clip.

    ``token`` goes in the element's key so that two identical replies in a row are still
    two different elements. Without it the browser sees the same node and the same source
    and does nothing, so the agent goes silent exactly when it repeats itself.
    """
    if samples is None or len(samples) == 0:
        return SILENT

    audio = base64.b64encode(wav_bytes(sample_rate, samples)).decode()
    seconds = len(samples) / sample_rate
    return (
        f'<div class="vd-quiet" data-turn="{token}">'
        f'<audio autoplay controls preload="auto" src="data:audio/wav;base64,{audio}">'
        "</audio>"
        f"<span>{seconds:.1f}s</span>"
        "</div>"
    )
