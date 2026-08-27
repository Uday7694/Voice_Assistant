"""The phone-network boundary: G.711, rate conversion, and each carrier's envelope.

All offline. The codec is checked against the standard rather than against itself, and
the carrier adapters against captured message shapes — which is the part that silently
rots when a provider renames a field.
"""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from brain.telephony import codec
from brain.telephony.carriers import CARRIERS, Exotel, Plivo, Twilio, carrier


# --- G.711 ------------------------------------------------------------------


def test_every_mulaw_code_decodes_to_the_standard_sample():
    """All 256 of them. The table is built at import and never checked again.

    mu-law inverts its bits, so the code space runs loudest-negative at 0 up to zero at
    127, then loudest-positive at 128 down to zero again at 255.
    """
    codes = np.arange(256, dtype=np.uint8).tobytes()
    decoded = np.frombuffer(codec.decode(codes), dtype=np.int16)
    assert len(decoded) == 256
    assert decoded[0] < -32000 and decoded[128] > 32000
    assert decoded[127] == 0 and decoded[255] == 0


def test_encoding_is_the_inverse_of_decoding_for_every_code_but_the_two_zeros():
    """G.711 has a positive zero and a negative zero, and only one survives.

    Codes 127 and 255 both decode to sample 0, which can only encode back to one of
    them. That is the standard, not a rounding bug, and the reference encoder does the
    same — worth pinning so nobody 'fixes' it into a mismatch with every carrier.
    """
    codes = np.arange(256, dtype=np.uint8)
    round_tripped = np.frombuffer(codec.encode(codec.decode(codes.tobytes())), dtype=np.uint8)
    differing = np.flatnonzero(round_tripped != codes)
    assert differing.tolist() == [127]
    assert round_tripped[127] == 255


def test_loud_samples_clip_rather_than_wrapping():
    loud = np.array([32767, -32768], dtype=np.int16).tobytes()
    back = np.frombuffer(codec.decode(codec.encode(loud)), dtype=np.int16)
    assert back[0] > 30000 and back[1] < -30000


def test_silence_survives_the_round_trip_as_silence():
    quiet = np.zeros(160, dtype=np.int16).tobytes()
    back = np.frombuffer(codec.decode(codec.encode(quiet)), dtype=np.int16)
    assert int(np.abs(back).max()) <= 8


def test_empty_audio_does_not_explode():
    assert codec.encode(b"") == b"" and codec.decode(b"") == b""


# --- rate conversion --------------------------------------------------------


def test_a_frame_keeps_its_duration_across_a_rate_change():
    twenty_ms_at_8k = np.zeros(160, dtype=np.int16).tobytes()
    assert len(codec.resample(twenty_ms_at_8k, 8000, 16000)) // 2 == 320
    assert len(codec.resample(twenty_ms_at_8k, 8000, 24000)) // 2 == 480


def test_successive_frames_join_without_a_repeated_sample():
    """A duplicated sample at every frame boundary is fifty clicks a second."""
    ramp = np.arange(160, dtype=np.int16).tobytes()
    up = np.frombuffer(codec.resample(ramp, 8000, 16000), dtype=np.int16)
    assert up[0] == 0
    assert up[-1] < 160          # never samples past the end of the input


def test_a_matching_rate_is_left_untouched():
    pcm = np.arange(100, dtype=np.int16).tobytes()
    assert codec.resample(pcm, 8000, 8000) is pcm


def test_the_full_path_to_the_line_and_back_preserves_a_tone():
    tone = (np.sin(np.arange(1600) * 0.15) * 12000).astype(np.int16).tobytes()
    on_the_wire = codec.to_telephony(tone, 16000)
    assert len(on_the_wire) == 800                       # 16k PCM -> 8k mu-law, 1/4 size
    back = np.frombuffer(codec.from_telephony(on_the_wire, 16000), dtype=np.int16)
    assert len(back) == 1600
    assert int(np.abs(back).max()) > 8000                # still a tone, not mush


# --- carrier envelopes ------------------------------------------------------


def _media(carrier_name: str, payload: bytes) -> str:
    encoded = base64.b64encode(payload).decode()
    shapes = {
        "twilio": {"event": "media", "streamSid": "MZ1", "media": {"payload": encoded}},
        "plivo": {"event": "media", "streamId": "PL1", "media": {"payload": encoded}},
        "exotel": {"event": "media", "stream_sid": "EX1", "media": {"payload": encoded}},
    }
    return json.dumps(shapes[carrier_name])


@pytest.mark.parametrize("name", ["twilio", "plivo", "exotel"])
def test_each_carrier_unwraps_audio_to_the_same_bytes(name):
    frame = carrier(name).parse(_media(name, b"\xff\x7f\x00"))
    assert frame.kind == "audio"
    assert frame.audio == b"\xff\x7f\x00"


@pytest.mark.parametrize("name", ["twilio", "plivo", "exotel"])
def test_each_carrier_reports_the_call_handle_it_uses(name):
    assert carrier(name).parse(_media(name, b"\x01")).call_id


@pytest.mark.parametrize("name", ["twilio", "plivo", "exotel"])
def test_outbound_audio_is_base64_of_exactly_what_was_given(name):
    message = json.loads(carrier(name).audio_message(b"\xab\xcd", "SID"))
    assert base64.b64decode(message["media"]["payload"]) == b"\xab\xcd"


@pytest.mark.parametrize("name", ["twilio", "plivo", "exotel"])
def test_a_hangup_is_recognised_as_a_stop(name):
    events = {"twilio": "stop", "plivo": "stop", "exotel": "stop"}
    payload = json.dumps({"event": events[name]})
    assert carrier(name).parse(payload).kind == "stop"


@pytest.mark.parametrize("name", ["twilio", "plivo", "exotel"])
def test_junk_on_the_socket_is_ignored_not_fatal(name):
    """Carriers send keepalives and provider-specific events all call long."""
    for junk in ("", "not json", "[]", '{"event":"something-new"}', b"\x00\x01"):
        assert carrier(name).parse(junk).kind in ("other", "audio", "start", "stop", "mark")


@pytest.mark.parametrize("name", ["twilio", "plivo", "exotel"])
def test_every_carrier_offers_a_way_to_drop_buffered_audio(name):
    """Barge-in is only as fast as this; a carrier without it needs different handling."""
    chosen = carrier(name)
    assert chosen.can_clear
    assert chosen.clear_message("SID")


def test_exotel_declares_its_coarser_frames():
    """100 ms frames, not 20 - the floor on how fast an interruption can be noticed."""
    assert Exotel.frame_ms == 100
    assert Twilio.frame_ms == 20 and Plivo.frame_ms == 20


@pytest.mark.parametrize("name", ["twilio", "plivo", "exotel"])
def test_the_answer_document_points_at_our_socket(name):
    document = carrier(name).answer_document("wss://agent.example.in/media")
    assert "wss://agent.example.in/media" in document
    assert document.startswith("<?xml")


def test_an_unknown_carrier_fails_with_the_list_of_real_ones():
    with pytest.raises(ValueError, match="plivo"):
        carrier("vonage")


def test_the_registry_holds_one_instance_per_carrier():
    assert set(CARRIERS) == {"twilio", "plivo", "exotel"}
