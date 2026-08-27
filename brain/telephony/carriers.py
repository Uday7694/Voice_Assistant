"""One adapter per carrier: the envelope differs, the audio does not.

Twilio, Plivo and Exotel all do the same thing — open a WebSocket when a call connects,
push the caller's audio as base64 frames, and play back whatever audio you send. What
differs is the JSON around it: the key the audio hides under, the name of the event, the
handle you have to echo back, and whether "stop talking" is a message you can send or
something you have to fake.

Isolating that here is what keeps the choice of carrier a deployment decision rather than
a rewrite. Everything above deals in `Frame` and bytes; only this file knows a carrier
exists, and adding one is a subclass, not a migration.

Rates and protocols as published in 2026: Plivo ~Rs 0.60/min with 20 ms mu-law frames,
Exotel Rs 0.80-1.00 outbound with ~100 ms frames over AgentStream, Twilio the most
expensive in India but the one to keep if a deployment ever leaves it.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Literal

# What every carrier's media frame reduces to once the envelope is off.
FrameKind = Literal["audio", "start", "stop", "mark", "clear", "other"]


@dataclass(frozen=True)
class Frame:
    """One message from the carrier, in terms the call loop understands."""

    kind: FrameKind
    audio: bytes = b""
    """mu-law payload, already un-base64'd. Empty for everything but ``audio``."""
    call_id: str = ""
    """The carrier's handle for this call, learned from the start frame."""


class Carrier:
    """How to read a carrier's messages and how to write audio back.

    Subclasses override the four methods below. Nothing else in the system may branch on
    which carrier is in use — if a behaviour differs, it belongs here.
    """

    name = "carrier"

    # Frames the carrier sends per second, and therefore how often the call loop wakes.
    # Twilio and Plivo send 20 ms; Exotel sends ~100 ms, which is five times the audio
    # per message and five times the latency floor on barge-in.
    frame_ms = 20

    # Whether the carrier can be told to drop audio it has already been sent. Where it
    # can, barge-in is instant; where it cannot, the only recourse is to stop sending and
    # wait out whatever is already in its buffer.
    can_clear = True

    def parse(self, message: str | bytes) -> Frame:
        raise NotImplementedError

    def audio_message(self, mulaw: bytes, call_id: str) -> str:
        raise NotImplementedError

    def clear_message(self, call_id: str) -> str | None:
        """Tell the carrier to discard buffered audio. None when it cannot."""
        return None

    def answer_document(self, websocket_url: str) -> str:
        """The XML the carrier fetches to be told where to stream."""
        raise NotImplementedError


class Twilio(Carrier):
    """Twilio Media Streams. The reference implementation everyone else resembles."""

    name = "twilio"
    frame_ms = 20
    can_clear = True

    def parse(self, message: str | bytes) -> Frame:
        data = _load(message)
        event = data.get("event")
        if event == "media":
            payload = (data.get("media") or {}).get("payload", "")
            return Frame(kind="audio", audio=_unwrap(payload), call_id=data.get("streamSid", ""))
        if event == "start":
            start = data.get("start") or {}
            return Frame(kind="start", call_id=data.get("streamSid") or start.get("streamSid", ""))
        if event == "stop":
            return Frame(kind="stop", call_id=data.get("streamSid", ""))
        if event == "mark":
            return Frame(kind="mark", call_id=data.get("streamSid", ""))
        return Frame(kind="other")

    def audio_message(self, mulaw: bytes, call_id: str) -> str:
        return json.dumps(
            {
                "event": "media",
                "streamSid": call_id,
                "media": {"payload": base64.b64encode(mulaw).decode()},
            }
        )

    def clear_message(self, call_id: str) -> str:
        return json.dumps({"event": "clear", "streamSid": call_id})

    def answer_document(self, websocket_url: str) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response><Connect>"
            f'<Stream url="{websocket_url}" />'
            "</Connect></Response>"
        )


class Plivo(Carrier):
    """Plivo AudioStream. Twilio's shape with different names and no stream handle."""

    name = "plivo"
    frame_ms = 20
    can_clear = True

    def parse(self, message: str | bytes) -> Frame:
        data = _load(message)
        event = data.get("event")
        if event in ("media", "playAudio"):
            payload = (data.get("media") or {}).get("payload", "")
            return Frame(kind="audio", audio=_unwrap(payload), call_id=data.get("streamId", ""))
        if event == "start":
            start = data.get("start") or {}
            return Frame(kind="start", call_id=data.get("streamId") or start.get("streamId", ""))
        if event in ("stop", "stopAudio"):
            return Frame(kind="stop", call_id=data.get("streamId", ""))
        return Frame(kind="other")

    def audio_message(self, mulaw: bytes, call_id: str) -> str:
        # Plivo names the outbound event differently from the inbound one, and wants the
        # codec restated on every frame.
        return json.dumps(
            {
                "event": "playAudio",
                "media": {
                    "contentType": "audio/x-mulaw",
                    "sampleRate": 8000,
                    "payload": base64.b64encode(mulaw).decode(),
                },
            }
        )

    def clear_message(self, call_id: str) -> str:
        return json.dumps({"event": "clearAudio"})

    def answer_document(self, websocket_url: str) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            f'<Stream bidirectional="true" keepCallAlive="true" '
            f'contentType="audio/x-mulaw;rate=8000">{websocket_url}</Stream>'
            "</Response>"
        )


class Exotel(Carrier):
    """Exotel AgentStream.

    The Indian incumbent, billed in rupees, with TRAI-compliant numbers — which is most
    of why it gets picked here. Its frames are ~100 ms rather than 20, so the floor on
    how fast the agent can notice an interruption is five times higher; the call loop
    reads `frame_ms` rather than assuming.
    """

    name = "exotel"
    frame_ms = 100
    can_clear = True

    def parse(self, message: str | bytes) -> Frame:
        data = _load(message)
        event = data.get("event")
        if event == "media":
            payload = (data.get("media") or {}).get("payload", "")
            return Frame(kind="audio", audio=_unwrap(payload), call_id=data.get("stream_sid", ""))
        if event in ("connected", "start"):
            return Frame(kind="start", call_id=data.get("stream_sid", ""))
        if event == "stop":
            return Frame(kind="stop", call_id=data.get("stream_sid", ""))
        return Frame(kind="other")

    def audio_message(self, mulaw: bytes, call_id: str) -> str:
        return json.dumps(
            {
                "event": "media",
                "stream_sid": call_id,
                "media": {"payload": base64.b64encode(mulaw).decode()},
            }
        )

    def clear_message(self, call_id: str) -> str:
        return json.dumps({"event": "clear", "stream_sid": call_id})

    def answer_document(self, websocket_url: str) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f"<Response><Voicebot url=\"{websocket_url}\" /></Response>"
        )


CARRIERS: dict[str, Carrier] = {c.name: c() for c in (Twilio, Plivo, Exotel)}


def carrier(name: str) -> Carrier:
    """Look up a carrier by name, failing with the list rather than a KeyError."""
    key = (name or "").strip().lower()
    if key not in CARRIERS:
        raise ValueError(f"Unknown carrier {name!r}; choose from {', '.join(sorted(CARRIERS))}")
    return CARRIERS[key]


def _load(message: str | bytes) -> dict:
    """Parse a carrier message, tolerating anything that is not the JSON we expect.

    Carriers send keepalives, provider-specific events, and occasionally malformed
    frames. None of those is a reason to drop a call in progress.
    """
    try:
        data = json.loads(message)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _unwrap(payload: str) -> bytes:
    try:
        return base64.b64decode(payload or "", validate=False)
    except (ValueError, TypeError):
        return b""
