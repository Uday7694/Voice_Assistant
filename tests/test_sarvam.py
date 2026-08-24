"""Sarvam streaming adapter tests.


No network. These guard the thing that actually breaks in an adapter written without a

key: the arguments we pass drifting from what the SDK accepts. Every kwarg is checked

against the installed signature, so an SDK upgrade that renames or drops one fails here

rather than at the first spoken word.

"""


from __future__ import annotations


import base64

import inspect

from types import SimpleNamespace


import pytest


from brain.speech import sarvam as sv


def _params(fn):

    return inspect.signature(fn).parameters


def _connect_params(service: str):

    from sarvamai import AsyncSarvamAI


    client = AsyncSarvamAI(api_subscription_key="test")

    return _params(getattr(client, service).connect)


# --- STT wiring ------------------------------------------------------------


def test_stt_connect_kwargs_exist_in_the_sdk():

    params = _connect_params("speech_to_text_streaming")

    for kwarg in (

        "language_code",

        "model",

        "mode",

        "sample_rate",

        "input_audio_codec",

        "vad_signals",

        "high_vad_sensitivity",

    ):

        assert kwarg in params, kwarg


def test_vad_flags_are_strings_not_booleans():

    """The published docs show booleans; the SDK types these as Literal['true','false'].


    Passing a real bool is silently wrong, and VAD is what drives barge-in.

    """

    params = _connect_params("speech_to_text_streaming")

    for flag in ("vad_signals", "high_vad_sensitivity"):

        assert "true" in str(params[flag].annotation), flag


def test_the_declared_input_codec_is_one_the_sdk_accepts():

    annotation = str(_connect_params("speech_to_text_streaming")["input_audio_codec"].annotation)

    assert sv.INPUT_CODEC in annotation


def test_what_feed_sends_matches_the_codec_it_declared():

    """The connect-time codec and the bytes on the wire must agree.


    They live in different places and are easy to change independently; that mismatch

    only shows up against the live service, as an empty transcript.

    """

    wrapped = sv._wav(bytes(320), 16000)

    if sv.INPUT_CODEC == "wav":

        assert wrapped[:4] == b"RIFF"

    else:

        assert sv.INPUT_CODEC.startswith("pcm"), sv.INPUT_CODEC


def test_the_wav_wrapper_declares_the_real_sample_rate():

    import struct


    wrapped = sv._wav(bytes(320), 8000)

    assert wrapped[:4] == b"RIFF" and wrapped[8:12] == b"WAVE"

    assert struct.unpack("<I", wrapped[24:28])[0] == 8000


def test_the_stt_model_is_one_the_sdk_knows():

    assert sv.STT_MODEL in str(_connect_params("speech_to_text_streaming")["model"].annotation)


# --- TTS wiring ------------------------------------------------------------


def test_tts_configure_kwargs_exist_in_the_sdk():

    from sarvamai.text_to_speech_streaming.socket_client import (

        AsyncTextToSpeechStreamingSocketClient as Socket,

    )


    params = _params(Socket.configure)

    for kwarg in (

        "target_language_code",

        "speaker",

        "speech_sample_rate",

        "output_audio_codec",

        "pace",

        "min_buffer_size",

        "max_chunk_length",

    ):

        assert kwarg in params, kwarg


def test_the_tts_model_is_one_the_sdk_knows():

    assert sv.TTS_MODEL in str(_connect_params("text_to_speech_streaming")["model"].annotation)


def test_completion_event_flag_is_a_string():

    annotation = str(_connect_params("text_to_speech_streaming")["send_completion_event"].annotation)

    assert "true" in annotation


def test_buffer_size_respects_the_service_floor():
    """Below the floor the service 422s and no audio is produced at all.

    Measured live: 24 is rejected, 50 is accepted, and the error names no parameter --
    it reads only "Input parameters has to be a valid dictionary". Lowering this to buy
    latency silently breaks every spoken turn, so the floor is asserted here.
    """
    from sarvamai.text_to_speech_streaming.socket_client import (
        AsyncTextToSpeechStreamingSocketClient as Socket,
    )

    sdk_default = _params(Socket.configure)["min_buffer_size"].default
    assert sv.MIN_BUFFER_SIZE >= sdk_default, "values under the SDK default are rejected"


# --- message parsing -------------------------------------------------------


def _message(kind, **data):

    return SimpleNamespace(type=kind, data=SimpleNamespace(**data))


def test_vad_signals_become_speech_events():

    assert sv._parse_stt(_message("events", signal_type="START_SPEECH")).kind == "speech_start"

    assert sv._parse_stt(_message("events", signal_type="END_SPEECH")).kind == "speech_end"


def test_an_unknown_signal_is_ignored_rather_than_guessed():

    assert sv._parse_stt(_message("events", signal_type="SOMETHING_NEW")) is None


def test_transcripts_are_read_from_either_field():

    """STT returns `transcript`; STT-translate returns `translation`."""

    assert sv._parse_stt(_message("data", transcript="hello")).text == "hello"

    assert sv._parse_stt(_message("data", translation="hello")).text == "hello"


def test_a_message_without_data_is_ignored():

    assert sv._parse_stt(SimpleNamespace(type="data", data=None)) is None


def test_tts_audio_is_base64_decoded():

    encoded = base64.b64encode(b"pcm").decode()

    assert sv._parse_tts_audio(_message("audio", audio=encoded)) == b"pcm"


def test_undecodable_audio_does_not_end_the_utterance():

    assert sv._parse_tts_audio(_message("audio", audio="!!!not base64!!!")) is None

    assert sv._parse_tts_audio(SimpleNamespace(data=None)) is None


def test_the_final_event_is_recognised():

    """Without this the utterance ends on a timeout instead of a signal."""

    assert sv._is_final(_message("events", event_type="final")) is True

    assert sv._is_final(_message("events", event_type="chunk")) is False


# --- contract --------------------------------------------------------------


def test_client_refuses_to_build_without_a_key():

    import os


    previous = os.environ.pop("SARVAM_API_KEY", None)

    try:

        with pytest.raises(RuntimeError, match="SARVAM_API_KEY"):

            sv.client()

    finally:

        if previous is not None:

            os.environ["SARVAM_API_KEY"] = previous


def test_streaming_is_the_whole_point_of_this_adapter():

    """Mouth.say takes an iterator so sentences overlap synthesis.


    A signature taking a plain string would mean waiting for the full reply first.

    """

    assert inspect.isasyncgenfunction(sv.Mouth.say)

    assert inspect.isasyncgenfunction(sv.Ear.events)



# --- latency tuning --------------------------------------------------------


def test_planner_params_are_tuned_for_first_token():
    """Measured warm, median of 3: baseline 328 ms, low 299 ms, no-wiki 291 ms."""
    from brain.providers import SARVAM

    assert SARVAM.extra_body.get("reasoning_effort") == "low"
    assert SARVAM.extra_body.get("wiki_grounding") is False


def test_vendor_params_go_through_extra_body_not_kwargs():
    """The OpenAI SDK validates its own signature.

    A vendor parameter it does not know raises TypeError before the request is sent,
    which previously took the whole fallback provider down with it.
    """
    import inspect

    from brain import llm as llm_module

    source = inspect.getsource(llm_module.LLMClient)
    assert "extra_body=provider.extra_body" in source
    assert "**provider.extra_body" not in source


@pytest.mark.asyncio
async def test_warm_is_best_effort_and_never_raises():
    """A failed warm-up is an optimisation missing, not a reason to refuse the call."""
    from brain.llm import LLMClient
    from brain.providers import Provider

    broken = Provider(
        name="broken",
        base_url="http://127.0.0.1:9",  # nothing listening
        api_key_env="SARVAM_API_KEY",
        planner_model="m",
        fast_model="m",
    )
    client = LLMClient(provider=broken, fast_provider=broken, fallback=None)
    await client.warm()  # must not raise
