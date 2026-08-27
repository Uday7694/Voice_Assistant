"""Prosody is pure string work, so it is testable without spending a TTS credit."""

from __future__ import annotations

import random

from brain.speech.fillers import Backchannel
from brain.speech.prosody import (
    beat_after_marker,
    gap_after,
    silence,
    speakable,
    space_numbers,
)
from brain.speech.voice import profile_for


def test_ten_digit_number_splits_five_and_five():
    assert space_numbers("call 9876543210") == "call 98765, 43210"


def test_reference_code_keeps_its_letters_as_a_word():
    # Spelling "A P T" is both slower and against the style rules; the pause belongs
    # at the seam between letters and digits, not between letters.
    assert space_numbers("APT44847") == "APT, 448 47"


def test_digit_grouping_never_strands_a_single_digit():
    # "123 456 7" reads the stray 7 as its own number and lands as a stutter.
    assert space_numbers("APT1234567") == "APT, 123 45 67"


def test_short_numbers_are_left_alone():
    assert space_numbers("room 204 at 10") == "room 204 at 10"


# The markers are whatever the line book wrote as backchannels for this call's
# language. Nothing in the speech layer holds a list of its own.
MARKERS = ("Right", "Okay", "Sure", "ठीक है")


def test_leading_marker_gets_a_beat():
    assert beat_after_marker("Right which department?", MARKERS) == "Right, which department?"


def test_a_marker_in_another_script_works_the_same_way():
    assert beat_after_marker("ठीक है आपका नाम", MARKERS) == "ठीक है, आपका नाम"


def test_marker_alone_is_not_padded():
    assert beat_after_marker("Sure.", MARKERS) == "Sure."


def test_word_starting_with_a_marker_is_not_split():
    assert beat_after_marker("Okayama is a city", MARKERS) == "Okayama is a city"


def test_with_no_markers_nothing_is_inserted():
    """A language whose backchannels have not been written yet still speaks."""
    assert beat_after_marker("Right which department?") == "Right which department?"


def test_a_long_sentence_is_left_exactly_as_written():
    """Inserting a breath by counting words produced "at ten in, the morning".

    Nothing in the speech layer knows where a clause ends in every language the agent
    speaks, and a pause in the wrong place is heard as a stumble. The phrasing is fixed
    where the sentence is written instead.
    """
    line = (
        "I will book the slot at ten in the morning and send you a confirmation "
        "message shortly"
    )
    assert speakable(line) == line


def test_a_question_hands_the_floor_over_with_the_longest_pause():
    assert gap_after("Which department?") > gap_after("Booked.")


def test_a_backchannel_barely_pauses_at_all():
    """It is an aside, not a turn; a gap after it invites the caller to answer it."""
    assert gap_after("okay", "filler") < gap_after("Booked.")


def test_a_line_ending_mid_clause_gets_a_short_gap():
    assert gap_after("cardiology and") == 140


def test_the_gap_is_playable_audio_of_the_right_length():
    # 16-bit mono: two bytes a sample.
    assert len(silence(250, 16000)) == 250 * 16 * 2


def test_an_empty_line_gets_no_gap():
    assert gap_after("") == 0


def test_speakable_leaves_no_stumbling_punctuation():
    out = speakable("Sure, your number is 9876543210.")
    assert ",," not in out and ", ." not in out


def test_speakable_passes_devanagari_through_unharmed():
    line = "किस विभाग में अपॉइंटमेंट चाहिए?"
    assert speakable(line, "hi-IN") == line


def test_readback_is_slower_than_a_plain_ask():
    assert profile_for("meera", "hi-IN", "readback").pace < profile_for("meera", "hi-IN", "ask").pace


def test_filler_is_quicker_and_quieter():
    filler = profile_for("meera", "en-IN", "filler")
    ask = profile_for("meera", "en-IN", "ask")
    assert filler.pace > ask.pace and filler.loudness < ask.loudness


def test_unknown_language_still_yields_a_usable_voice():
    assert profile_for("meera", "ta-IN").speaker


def test_backchannel_never_repeats_itself():
    channel = Backchannel(("Right.", "Okay.", "Sure.", "Mm-hm."), rng=random.Random(1))
    picks = [channel.ack() for _ in range(8)]
    assert all(a != b for a, b in zip(picks, picks[1:]))


def test_one_word_answers_earn_no_acknowledgement():
    channel = Backchannel(("Right.", "Okay.", "Sure.", "Mm-hm."), rng=random.Random(1))
    assert channel.should_ack("yes") is False


def test_acknowledgement_never_lands_twice_running():
    channel = Backchannel(("Right.", "Okay.", "Sure."), rng=random.Random(3))
    decisions = [channel.should_ack("I need a cardiology appointment") for _ in range(12)]
    assert not any(a and b for a, b in zip(decisions, decisions[1:]))
    assert any(decisions)  # and it does fire sometimes


# --- code-mixing, but not a change of language ------------------------------


def test_english_words_inside_an_indian_sentence_are_fine():
    """This is how people speak. An agent that refuses to code-mix sounds wrong."""
    from brain.script import is_wrong_language

    assert not is_wrong_language("రేపు ఉదయం పదికి, మధ్యాహ్నం మూడున్నరకు slots ఉన్నాయి", "te-IN")
    assert not is_wrong_language("Uday जी, कौन सा department चाहिए?", "hi-IN")


def test_a_whole_english_sentence_on_an_indian_call_is_not():
    from brain.script import is_wrong_language

    assert is_wrong_language("Which department do you need today?", "te-IN")
    assert is_wrong_language("Please tell me the patient name.", "hi-IN")


def test_an_english_call_is_never_flagged():
    from brain.script import is_wrong_language

    assert not is_wrong_language("Which department do you need today?", "en-IN")


def test_a_short_reply_is_not_judged():
    """"OK" is not a change of language, and two letters have no ratio worth measuring."""
    from brain.script import is_wrong_language

    assert not is_wrong_language("OK", "te-IN")


def test_the_rule_needs_no_table_of_scripts():
    """A language added tomorrow is covered without touching this module."""
    from brain.script import expects_native_script

    assert expects_native_script("or-IN") and expects_native_script("as-IN")
    assert not expects_native_script("en-IN")
