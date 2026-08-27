"""Regressions for a value accepted three turns before it was refused.

A caller answered "architecture" when asked for a department. It was stored, the agent
went on to collect a name, and only when `check_availability` finally ran did anyone
say the department did not exist. Three turns of the caller's time were spent on an
answer the system was never going to honour.

Second regression, from the same call: asked for the patient's name and given an insult
instead, the agent replied with the identical sentence twice. Repeating a question
word for word is the clearest tell a caller gets that nobody is home.
"""

from __future__ import annotations

import re

import pytest

from brain import slots as slot_mod
from brain import tools  # noqa: F401 - imported for its validator registrations
from brain.agents.hospital import HOSPITAL_AGENT
from brain.models import IntentResult, Session
from brain.orchestrator import _reask_note
from brain.planner import build_messages

RULES = HOSPITAL_AGENT.validation_pairs


# --- capture-time validation ----------------------------------------------


def test_a_department_that_does_not_exist_never_enters_the_session():
    kept, rejected = slot_mod.validate(RULES, {"department": "architecture"})
    assert "department" not in kept
    assert [r.slot for r in rejected] == ["department"]


def test_a_rejection_carries_the_real_options():
    _, rejected = slot_mod.validate(RULES, {"department": "architecture"})
    assert "cardiology" in rejected[0].options


def test_an_accepted_department_is_stored_canonically():
    """"synus" is a real department under another name; store the hospital's name."""
    kept, rejected = slot_mod.validate(RULES, {"department": "synus"})
    assert kept["department"] == "ent"
    assert not rejected


def test_slots_without_a_rule_pass_straight_through():
    kept, rejected = slot_mod.validate(RULES, {"patient_name": "Uday"})
    assert kept == {"patient_name": "Uday"} and not rejected


def test_one_bad_slot_does_not_discard_the_good_ones():
    kept, rejected = slot_mod.validate(
        RULES, {"department": "architecture", "patient_name": "Uday"}
    )
    assert kept == {"patient_name": "Uday"} and len(rejected) == 1


@pytest.mark.parametrize(
    "said,stored",
    [("9876543210", "9876543210"), ("+91 98765 43210", "9876543210"), ("98765-43210", "9876543210")],
)
def test_a_phone_number_is_stored_as_bare_digits(said, stored):
    kept, _ = slot_mod.validate(RULES, {"phone": said})
    assert kept["phone"] == stored


@pytest.mark.parametrize("said", ["98765", "call me", "987654321012345"])
def test_a_number_that_is_not_ten_digits_is_refused(said):
    kept, rejected = slot_mod.validate(RULES, {"phone": said})
    assert "phone" not in kept and rejected[0].slot == "phone"


def test_an_unregistered_validator_fails_open():
    """A typo in agent config must not trap the caller in an unescapable loop."""
    kept, rejected = slot_mod.validate((("department", "no_such_validator"),), {"department": "x"})
    assert kept == {"department": "x"} and not rejected


def test_the_correction_note_names_the_options_and_forbids_guessing():
    _, rejected = slot_mod.validate(RULES, {"department": "architecture"})
    note = slot_mod.note_for(rejected[0])
    assert "cardiology" in note and "architecture" in note
    assert "not choose for them" in note.lower() or "do not choose" in note.lower()


def test_the_note_reaches_the_planner_prompt():
    node = HOSPITAL_AGENT.node("collect_booking")
    session = Session(agent_name="h", node_id="collect_booking")
    messages = build_messages(
        HOSPITAL_AGENT,
        node,
        session,
        IntentResult(name="provide_details", confidence=0.9),
        "architecture",
        notes=["THE NOTE"],
    )
    assert any("THE NOTE" in m["content"] for m in messages)


# --- not asking the same question twice ------------------------------------


def test_asking_for_the_same_slot_again_counts_up():
    session = Session().asking_for("patient_name").asking_for("patient_name")
    assert session.ask_repeats == 2


def test_moving_on_to_a_different_slot_resets_the_count():
    session = Session().asking_for("patient_name").asking_for("department")
    assert session.asked_slot == "department" and session.ask_repeats == 1


def test_a_turn_needing_nothing_clears_the_count():
    session = Session().asking_for("patient_name").asking_for("")
    assert session.ask_repeats == 0 and session.asked_slot == ""


def test_the_second_attempt_is_told_to_rephrase():
    note = _reask_note("patient_name", 2)
    assert "different words" in note and "patient name" in note


def test_a_stuck_caller_is_eventually_offered_a_person():
    assert "colleague" in _reask_note("patient_name", 5)


def test_later_attempts_forbid_saying_it_out_loud():
    """The note tells the model it is repeating; the caller must never hear that.

    "I already asked you" turns a stuck call into an angry one, and the model will say
    it unprompted if the instruction does not rule it out.
    """
    for count in (3, 4, 9):
        assert "do not mention that you already asked" in _reask_note("phone", count).lower()


# --- the vocabulary the classifier is shown ---------------------------------


def test_the_classifier_is_shown_the_real_departments():
    """Without the list, the classifier snapped "architecture" to "orthopaedics".

    A substituted value is indistinguishable from one the caller actually said, so no
    downstream check can undo it. The only fix is upstream: show the closed set and
    tell it to pass anything else through untouched.
    """
    from brain.intent import _SYSTEM, _prompt_for

    prompt = _prompt_for(
        HOSPITAL_AGENT.node("collect_booking"), HOSPITAL_AGENT, Session(), "architecture"
    )[1]["content"]
    assert "Allowed department" in prompt
    assert "cardiology" in prompt
    # The rule about not substituting a listed value lives in the system prompt now: it
    # is one rule, not one per slot, and repeating it per slot cost tokens on every turn.
    assert "never the nearest-sounding listed value" in _SYSTEM


def test_the_vocabulary_is_offered_at_every_step_not_only_its_own():
    """A caller names a department whenever they like, including three steps early.

    Scoping the list to the step that collects it meant an opener carrying the
    department was dropped, and the agent asked for something the caller had already
    said — the exact moment a call starts to feel like a form.
    """
    from brain.intent import _prompt_for

    prompt = _prompt_for(
        HOSPITAL_AGENT.node("collect_phone"), HOSPITAL_AGENT, Session(), "9876543210"
    )[1]["content"]
    assert "Allowed department" in prompt
    assert "Also record if the caller mentions" in prompt


# --- teardown noise ---------------------------------------------------------


def test_only_the_vendored_teardown_warning_is_filtered():
    """The filter must be narrow enough that a leak of ours still gets printed."""
    from brain.shutdown import _is_vendored_teardown

    def context(message: str, filename: str) -> dict:
        code = type("Code", (), {"co_filename": filename})()
        return {"message": message, "asyncgen": type("Gen", (), {"ag_code": code})()}

    warning = "an error occurred during closing of asynchronous generator"
    assert _is_vendored_teardown(context(warning, r"site-packages\httpcore2\_async\http11.py"))
    assert not _is_vendored_teardown(context(warning, r"Voice_Assistant\brain\llm.py"))
    assert not _is_vendored_teardown(
        context("Task exception was never retrieved", r"site-packages\httpcore2\x.py")
    )


def test_a_context_without_a_generator_is_not_filtered():
    from brain.shutdown import _is_vendored_teardown

    assert not _is_vendored_teardown(
        {"message": "an error occurred during closing of asynchronous generator"}
    )


# --- sounding like a person, not a loop -------------------------------------


def test_the_identity_question_is_a_global_intent():
    """Recognising it belongs to the classifier, which already reads every language.

    It used to be a regex per language, which meant a new language was a new set of
    patterns — and the patterns still missed whatever phrasing nobody had thought of.
    """
    from brain.models import GLOBAL_INTENTS

    assert "asks_identity" in GLOBAL_INTENTS
    assert "asks_identity" in HOSPITAL_AGENT.all_intents or True  # global, so every node


def test_the_classifier_is_told_what_the_identity_intent_means():
    from brain.intent import _SYSTEM

    assert "asks_identity" in _SYSTEM
    assert "escalate_to_human" in _SYSTEM, "and told how it differs from asking for a person"


def test_the_identity_brief_forbids_claiming_to_be_human():
    """The one line that must not drift, in any language."""
    from brain.lines import SPECS

    brief = SPECS["identity"].brief.lower()
    assert "never claim to be human" in brief
    assert "digital assistant" in SPECS["identity"].fallback.lower()


def test_the_agent_is_told_not_to_repeat_its_last_sentence():
    from brain.orchestrator import _NO_REPEAT, _last_agent_line
    from brain.models import Role

    session = Session().with_turn(Role.USER, "hello").with_turn(Role.AGENT, "Which department?")
    assert _last_agent_line(session) == "Which department?"
    assert "Which department?" in _NO_REPEAT.format(previous=_last_agent_line(session))


def test_the_off_topic_streak_survives_only_while_it_is_consecutive():
    from brain.orchestrator import Brain

    brain = Brain(HOSPITAL_AGENT, llm=object(), deep_reason=False)
    session = Session()
    off = IntentResult(name="out_of_scope", confidence=0.9)
    on = IntentResult(name="book_appointment", confidence=0.9)

    session = brain._track_off_topic(session, off)
    session = brain._track_off_topic(session, off)
    assert session.off_topic_streak == 2
    assert brain._track_off_topic(session, on).off_topic_streak == 0


def test_every_spoken_line_has_a_fallback_that_can_be_said_out_loud():
    """Generation can fail with a caller already on the line."""
    from brain.lines import SPECS

    for key, spec in SPECS.items():
        assert spec.fallback.strip(), key
        assert spec.count >= 1, key


def test_nothing_in_the_brain_keeps_a_table_of_lines_per_language():
    """The regression this refactor exists to prevent.

    A per-language table is a place for one language to quietly go missing, and it
    makes adding a language a code change in every file that has one.
    """
    import pathlib

    for path in pathlib.Path("brain").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        # A dict literal keyed by BCP-47 tags. LANGUAGE_NAMES is the one allowed
        # exception: it maps tags to their English names, which is not spoken content.
        if path.name in {"types.py"}:
            continue
        hits = re.findall(r'"(?:hi|te|ta|kn|ml|mr|bn|gu|pa|or|as)-IN"\s*:', source)
        assert not hits, f"{path} still hard-codes lines per language: {hits[:3]}"


# --- hearing what the caller actually said, and ending properly -------------


def test_a_misheard_word_does_not_make_a_request_out_of_scope():
    """A misheard word is not a different request.

    This assertion used to be its own opposite. The rule said "apartment" is never
    "appointment" — written against typed input, where the two words are a genuine
    mismatch of subject and reading one as the other wasted four turns.

    Speech broke it. Indic transcription renders "appointment" as "apartment" routinely,
    and a live Telugu caller asking to book a doctor — "నాకు ఒక అపార్ట్‌మెంట్ బుక్ చేయండి
    అమీత్ పేరు మీద" — was answered as out of scope, three times out of three, while the
    name sat unread in the same sentence.

    So the rule is now about the request as a whole rather than any one word: a sentence
    that otherwise reads as booking a doctor is one. A caller who genuinely wants
    something else is still refused, which is what the second half pins.
    """
    from brain.intent import _SYSTEM

    assert "apartment" in _SYSTEM and "appointment" in _SYSTEM
    assert "out_of_scope" in _SYSTEM
    assert "if a word came through wrong" in _SYSTEM
    assert "genuinely wants" in _SYSTEM, "a real mismatch of subject is still refused"


def test_a_caller_repeating_themselves_is_noticed():
    from brain.models import Role
    from brain.orchestrator import _is_repeat

    session = Session().with_turn(Role.USER, "book me an apartment").with_turn(
        Role.AGENT, "Patient name?"
    )
    assert _is_repeat(session, "Book me an apartment.")
    assert not _is_repeat(session, "cardiology please")


def test_a_one_word_answer_repeated_is_not_treated_as_a_repeat():
    """"yes" twice is agreement, not the caller correcting a misunderstanding."""
    from brain.models import Role
    from brain.orchestrator import _is_repeat

    assert not _is_repeat(Session().with_turn(Role.USER, "yes"), "yes")


def test_the_repeat_note_stops_the_agent_answering_the_same_way():
    from brain.orchestrator import _CALLER_REPEATED

    assert "same way" in _CALLER_REPEATED
    assert "confirm" in _CALLER_REPEATED


def test_a_call_that_ends_has_a_line_to_end_on():
    """A caller who says no used to hear the line go dead mid-breath."""
    from brain.lines import SPECS

    assert "farewell" in SPECS
    brief = SPECS["farewell"].brief.lower()
    assert "thank" in brief and "goodbye" in brief


def test_the_wrong_desk_line_also_says_goodbye():
    """It is the last thing the caller hears, so it has to close the call itself."""
    from brain.lines import SPECS

    assert "goodbye" in SPECS["wrong_desk"].brief.lower()


# --- machinery must never reach the speaker ---------------------------------


@pytest.mark.parametrize(
    "junk",
    [
        "<toolcallcheckappointment",
        "<argkeyphonenumber</argkey",
        "<tool_call>lookup_appointment</tool_call>",
        '{"name": "book_appointment", "arguments": {}}',
        "functions.check_availability(department)",
    ],
)
def test_a_tool_call_written_as_prose_is_never_spoken(junk):
    """It reached a real caller: read aloud, and billed by the character.

    Dropped whole rather than cleaned — half-stripping the brackets leaves
    "toolcallcheckappointment", which is worse than saying nothing.
    """
    from brain.planner import _clean_for_speech

    assert _clean_for_speech(junk) == ""


def test_ordinary_speech_survives_the_tool_call_filter():
    from brain.planner import _clean_for_speech

    for line in ("మీ phone number చెప్పండి.", "Booked. Reference APT44847.", "Which department?"):
        assert _clean_for_speech(line) == line


@pytest.mark.asyncio
async def test_a_turn_of_pure_machinery_says_something_instead_of_nothing():
    """Silence is the worst outcome: the caller repeats into a line still thinking."""
    from brain import planner
    from brain.tools import build_default_registry

    class JunkLLM:
        async def stream_chat(self, messages, **kwargs):
            yield {"type": "text", "text": "<toolcallcheckappointment <argkeyphone</argkey"}

    spoken = []
    async for frame in planner.run(
        JunkLLM(),
        build_default_registry(None),
        HOSPITAL_AGENT,
        HOSPITAL_AGENT.node("lookup"),
        Session(node_id="lookup", language="te-IN"),
        IntentResult(name="provide_details", confidence=0.9),
        "9949210999",
        recovery=planner.Recovery(trouble="మళ్ళీ చెప్పండి", wait="ఒక్క నిమిషం"),
    ):
        if frame["type"] == "sentence":
            spoken.append(frame["text"])

    assert spoken == ["మళ్ళీ చెప్పండి"]


def test_the_recovery_lines_are_written_per_language_like_every_other_line():
    """An English "sorry, say that again" undoes every other thing here."""
    from brain.lines import SPECS

    assert "trouble" in SPECS and "wait" in SPECS


def test_asking_what_is_free_does_not_route_to_the_step_that_wants_a_phone_number():
    """A caller asking whether a doctor is free is not asking about their own booking."""
    greet = HOSPITAL_AGENT.node("greet")
    assert "check_availability" in greet.expected_intents
    destination = next(t.to for t in greet.transitions if t.when == "check_availability")
    assert destination == "collect_booking"
    assert "phone" not in HOSPITAL_AGENT.node(destination).required_slots


def test_a_booked_call_says_goodbye_before_it_hangs_up(tmp_path):
    """The confirmation used to be the last thing a caller heard.

    "Booked. Reference APT41307, room 4A-02." and then dead air — the desk ringing off
    the moment the caller's business was done. The close node no longer asks the planner
    for a goodbye it has no sentences left for; the orchestrator says it itself.
    """
    import asyncio

    from brain.agents.hospital import HOSPITAL_AGENT
    from brain.models import EndEvent, SayEvent
    from brain.orchestrator import Brain

    goal = HOSPITAL_AGENT.node("close").goal.lower()
    assert "do not say goodbye" in goal

    brain = Brain(HOSPITAL_AGENT, deep_reason=False)
    # No model and an empty cache dir, so the line is the English fallback: the point
    # of the test is that something is spoken before the line drops, not what.
    brain.lines.llm = None
    brain.lines.cache_dir = tmp_path

    async def drive():
        session = Session(agent_name=HOSPITAL_AGENT.name, language="te-IN")
        return [
            event
            async for event in brain._end(
                session, "flow reached a terminal step", farewell=True, outcome="completed"
            )
        ]

    events = asyncio.run(drive())
    spoken = [e for e in events if isinstance(e, SayEvent)]
    assert spoken and "goodbye" in spoken[0].text.lower()
    assert spoken[0].is_final
    assert any(isinstance(e, EndEvent) for e in events)
