"""G.711 mu-law and rate conversion: the phone network's audio, and ours.

Telephony is 8 kHz, 8-bit mu-law, and has been since 1972. Everything above this file
works in 16-bit linear PCM — the ear at 16 kHz, the mouth at 24 kHz — so every frame in
either direction crosses this boundary, fifty times a second, for the length of the call.
That budget is why the conversions are table-driven rather than computed per sample: a
lookup over a numpy array is one indexing operation for a whole frame.

Written without `audioop`, which does exactly this and was removed from the standard
library in Python 3.13. Keeping our own costs eighty lines and removes a countdown.
"""

from __future__ import annotations

import numpy as np

MULAW_BIAS = 0x84
MULAW_CLIP = 32635

# Full 8-bit domain, decoded once at import. Decoding a frame is then a single lookup.
_DECODE_TABLE: np.ndarray


def _build_decode_table() -> np.ndarray:
    codes = np.arange(256, dtype=np.uint8)
    inverted = ~codes & 0xFF
    sign = inverted & 0x80
    exponent = (inverted >> 4) & 0x07
    mantissa = inverted & 0x0F
    magnitude = ((mantissa.astype(np.int32) << 3) + MULAW_BIAS) << exponent
    magnitude -= MULAW_BIAS
    samples = np.where(sign != 0, -magnitude, magnitude)
    return samples.astype(np.int16)


_DECODE_TABLE = _build_decode_table()


def decode(mulaw: bytes) -> bytes:
    """One mu-law frame to 16-bit linear PCM."""
    if not mulaw:
        return b""
    codes = np.frombuffer(mulaw, dtype=np.uint8)
    return _DECODE_TABLE[codes].tobytes()


# Segment boundaries from the G.711 reference encoder, on the 14-bit scale it works in.
# The obvious shortcut — take the exponent from the highest set bit — disagrees with the
# standard by one code on about seven samples in a thousand, always at a segment
# boundary. Inaudible, and still the wrong answer: carriers and conformance tools compare
# against the reference, and "close enough" is not a property you want to discover during
# an integration.
_SEGMENT_ENDS = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF], dtype=np.int32)

# The whole 16-bit input domain, encoded once at import. 64 KB of table turns encoding a
# frame into one indexing operation, which at fifty frames a second for the length of a
# call is worth more than the memory.
_ENCODE_TABLE: np.ndarray


def _encode_values(samples: np.ndarray) -> np.ndarray:
    """The reference encoder, vectorised. Operates on the 14-bit scale."""
    narrowed = samples.astype(np.int32) >> 2
    negative = narrowed < 0
    magnitude = np.minimum(np.abs(narrowed), MULAW_CLIP >> 2) + (MULAW_BIAS >> 2)

    # searchsorted with 'left' is exactly the reference's linear search for the first
    # boundary the magnitude does not exceed.
    segment = np.searchsorted(_SEGMENT_ENDS, magnitude, side="left").astype(np.int32)

    mantissa = (magnitude >> (segment + 1)) & 0x0F
    code = (segment << 4) | mantissa
    # Beyond the last segment the reference saturates rather than wrapping.
    code = np.where(segment >= 8, 0x7F, code)
    mask = np.where(negative, 0x7F, 0xFF)
    return (code ^ mask).astype(np.uint8)


def _build_encode_table() -> np.ndarray:
    return _encode_values(np.arange(-32768, 32768, dtype=np.int32))


_ENCODE_TABLE = _build_encode_table()


def encode(pcm: bytes) -> bytes:
    """16-bit linear PCM to mu-law."""
    if not pcm:
        return b""
    samples = np.frombuffer(pcm, dtype=np.int16)
    # +32768 shifts the signed domain onto the table's index space.
    return _ENCODE_TABLE[samples.astype(np.int32) + 32768].tobytes()


def resample(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    """Linear-interpolated rate conversion between 16-bit PCM streams.

    Linear, not windowed-sinc. Upsampling 8 kHz telephony audio invents nothing above
    4 kHz whatever the filter, and the downsample the other way is throwing away
    material a phone line cannot carry. Measured against a proper resampler on live
    transcription, the recognised text was identical; the cost is a fraction of the
    20 ms frame budget rather than most of it.
    """
    if source_rate == target_rate or not pcm:
        return pcm

    samples = np.frombuffer(pcm, dtype=np.int16)
    if samples.size == 0:
        return b""

    count = int(round(samples.size * target_rate / source_rate))
    if count <= 0:
        return b""

    # endpoint=False keeps successive frames butt-joined: sampling right up to the last
    # input sample would repeat it at the head of the next frame, and fifty of those a
    # second is an audible buzz.
    position = np.linspace(0, samples.size, num=count, endpoint=False)
    converted = np.interp(position, np.arange(samples.size), samples.astype(np.float32))
    return np.rint(converted).astype(np.int16).tobytes()


def to_telephony(pcm: bytes, source_rate: int, *, target_rate: int = 8000) -> bytes:
    """Agent audio out: PCM at our rate to mu-law at the line's."""
    return encode(resample(pcm, source_rate, target_rate))


def from_telephony(mulaw: bytes, target_rate: int, *, source_rate: int = 8000) -> bytes:
    """Caller audio in: mu-law off the line to PCM at the rate the ear wants."""
    return resample(decode(mulaw), source_rate, target_rate)
