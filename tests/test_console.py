"""The console's pure parts: audio conversion, and what the trace panel says.

Nothing here starts Gradio or touches a network. The turn loop is the brain's, already
covered; what is worth pinning is the arithmetic between a browser and the ear, because
it is silent when wrong — a bad conversion does not raise, it transcribes as nonsense.
"""

from __future__ import annotations

import numpy as np
import pytest

from ui import player, trace
from ui.transcribe import resample, to_pcm


# --- browser audio to the ear's PCM ----------------------------------------


def test_float_samples_become_sixteen_bit():
    rate, pcm = to_pcm(48000, np.array([0.0, 1.0, -1.0], dtype=np.float32))
    assert rate == 48000
    assert np.frombuffer(pcm, dtype=np.int16).tolist() == [0, 32767, -32767]


def test_samples_over_full_scale_are_clamped_not_wrapped():
    """A loud syllable must clip, not fold round to the opposite sign."""
    _, pcm = to_pcm(48000, np.array([1.4, -1.4], dtype=np.float32))
    assert np.frombuffer(pcm, dtype=np.int16).tolist() == [32767, -32767]


def test_a_stereo_headset_is_mixed_down_not_half_dropped():
    stereo = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    values = np.frombuffer(to_pcm(48000, stereo)[1], dtype=np.int16)
    assert values.tolist() == [16383, 16383]


def test_integer_input_survives_the_trip():
    _, pcm = to_pcm(16000, np.array([0, 1000, -1000], dtype=np.int16))
    assert np.frombuffer(pcm, dtype=np.int16).tolist() == [0, 1000, -1000]


# --- rate conversion --------------------------------------------------------


def test_downsampling_keeps_the_clip_the_same_length_in_seconds():
    one_second = np.zeros(48000, dtype=np.int16).tobytes()
    out = resample(one_second, 48000, 16000)
    assert len(out) // 2 == 16000


def test_a_matching_rate_is_left_alone():
    pcm = np.arange(100, dtype=np.int16).tobytes()
    assert resample(pcm, 16000, 16000) is pcm


def test_empty_audio_does_not_explode():
    assert resample(b"", 48000, 16000) == b""


# --- the trace panel --------------------------------------------------------


def test_an_empty_turn_says_so_rather_than_rendering_blank():
    assert "No turn yet" in trace.render({})


def test_a_confident_intent_reads_as_live_and_an_unconfident_one_as_a_warning():
    confident = trace.render({"intent": "book_appointment", "confidence": 0.9})
    unsure = trace.render({"intent": "book_appointment", "confidence": 0.3})
    assert 'class="v live"' in confident
    assert 'class="v warn"' in unsure


def test_the_total_is_kept_out_of_the_stage_breakdown():
    """It is the sum of the bars; charted with them it flattens every one."""
    rendered = trace.render(
        {"stage_ms": {"turn_total": 5000.0, "intent": 300.0, "planner_total": 900.0}}
    )
    assert "5000 ms" not in rendered
    assert "300 ms" in rendered and "900 ms" in rendered


def test_a_slow_stage_is_marked_slow():
    assert "step slow" in trace.render({"stage_ms": {"planner_total": 1500.0}})
    assert "step slow" not in trace.render({"stage_ms": {"planner_total": 120.0}})


def test_captured_values_are_escaped_not_injected():
    rendered = trace.render({"slots": {"patient_name": "<script>alert(1)</script>"}})
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_a_failed_tool_is_flagged_apart_from_a_successful_one():
    rendered = trace.render({"tools": [{"name": "book", "ok": False, "ms": 12}]})
    assert 'class="v alert"' in rendered


# --- the agent's own audio element ------------------------------------------


def test_a_clip_becomes_an_autoplaying_element_carrying_its_own_audio():
    tone = np.zeros(2400, dtype=np.int16)
    html = player.element(24000, tone, token="abc123")
    assert "autoplay" in html
    assert "data:audio/wav;base64," in html
    assert 'data-turn="abc123"' in html


def test_two_identical_replies_produce_two_different_elements():
    """Same words twice running must still play twice, not sit silent the second time."""
    tone = np.zeros(2400, dtype=np.int16)
    assert player.element(24000, tone, token="one") != player.element(24000, tone, token="two")


def test_no_audio_renders_the_quiet_placeholder_rather_than_a_broken_player():
    assert player.element(24000, np.array([], dtype=np.int16)) == player.SILENT
    assert player.element(24000, None) == player.SILENT


def test_the_wav_header_declares_the_rate_the_samples_were_made_at():
    import struct

    data = player.wav_bytes(24000, np.zeros(10, dtype=np.int16))
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    assert struct.unpack("<I", data[24:28])[0] == 24000
