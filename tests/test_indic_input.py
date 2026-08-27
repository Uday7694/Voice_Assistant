"""Regressions for two bugs that only surfaced in Indian-language conversations.

Both produced the same visible symptom — the agent answering "sorry, I didn't catch
that" to a perfectly clear sentence, twice, then escalating to a human — while every
English test passed.
"""

from __future__ import annotations

import pytest

from brain import guardrails
from brain.agents.hospital import HOSPITAL_AGENT
from brain.intent import _prompt_for
from brain.models import Role, Session

TELUGU = "నాకు కార్డియాలజీలో అపాయింట్\u200cమెంట్ కావాలి"  # contains a ZWNJ


# --- invisible characters --------------------------------------------------


def test_zero_width_characters_are_stripped():
    """ZWNJ is invisible, common in typed Indic text, and flips classification.

    Measured: this sentence classifies as book_appointment without it and unknown with
    it. Sarvam STT does not emit ZWNJ, so spoken input worked while typed input failed.
    """
    assert "\u200c" in TELUGU
    assert "\u200c" not in guardrails.normalise_input(TELUGU)


def test_normalisation_covers_the_whole_invisible_range():
    for char in ("\u200b", "\u200c", "\u200d", "\u200e", "\u2060", "\ufeff", "\u00ad"):
        assert guardrails.normalise_input(f"a{char}b") == "ab", repr(char)


def test_visible_indic_text_is_untouched():
    """Stripping must not damage the script itself."""
    clean = "నాకు అపాయింట్మెంట్ కావాలి"
    assert guardrails.normalise_input(clean) == clean
    assert guardrails.normalise_input("नमस्ते") == "नमस्ते"


def test_normalisation_leaves_ordinary_text_alone():
    assert guardrails.normalise_input("book me a slot") == "book me a slot"
    assert guardrails.normalise_input("") == ""


# --- prompt construction ---------------------------------------------------


def _session_with(text: str) -> Session:
    return Session(agent_name="h", node_id="greet", language="te-IN").with_turn(
        Role.USER, text
    )


def test_pre_turn_history_yields_the_utterance_exactly_once():
    """Given the conversation *before* this turn, the utterance appears once.

    ``_prompt_for`` renders whatever history it is handed; keeping the current turn out
    of it is the caller's job. This pins the shape the orchestrator has to produce —
    the duplicated variant is what made the model answer "unknown".
    """
    text = "నా పేరు ఉదయ్"
    node = HOSPITAL_AGENT.node("greet")
    pre_turn = Session(agent_name="h", node_id="greet", language="te-IN")
    body = _prompt_for(node, HOSPITAL_AGENT, pre_turn, text)[-1]["content"]
    assert body.count(text) == 1, "utterance should appear once, as 'User just said'"


def test_duplicated_history_is_what_the_orchestrator_must_avoid():
    """Documents the failure mode: appending first renders the utterance twice."""
    text = "నా పేరు ఉదయ్"
    node = HOSPITAL_AGENT.node("greet")
    body = _prompt_for(node, HOSPITAL_AGENT, _session_with(text), text)[-1]["content"]
    assert body.count(text) == 2, "this is the shape that broke classification"


def test_earlier_turns_are_still_given_to_the_classifier():
    """Excluding the current turn must not throw away the rest of the conversation."""
    node = HOSPITAL_AGENT.node("greet")
    session = _session_with("first thing said")
    prompt = _prompt_for(node, HOSPITAL_AGENT, session, "second thing said")
    assert "first thing said" in prompt[-1]["content"]


@pytest.mark.asyncio
async def test_orchestrator_classifies_against_pre_turn_history():
    """End-to-end guard: the Brain must not hand the classifier its own input twice."""
    from brain.orchestrator import Brain

    seen: list[Session] = []

    class StubLLM:
        provider = type("P", (), {"name": "stub", "planner_model": "m", "fast_model": "m"})()
        fallback = None

    async def spy(llm, agent, session, text):
        seen.append(session)
        from brain.models import IntentResult

        return IntentResult(name="book_appointment", confidence=0.9)

    import brain.orchestrator as orch

    original = orch.intent_mod.classify
    orch.intent_mod.classify = spy
    try:
        brain = Brain(HOSPITAL_AGENT, llm=StubLLM(), deep_reason=False)
        session = brain.start()
        async for _ in brain.handle(session.session_id, "నా పేరు ఉదయ్"):
            break
    finally:
        orch.intent_mod.classify = original

    assert seen, "classify was never called"
    assert all("ఉదయ్" not in t.text for t in seen[0].history), (
        "the utterance under classification was already in history"
    )


# --- agent configuration ---------------------------------------------------


def test_telugu_is_enabled_on_the_agent():
    """The language settler only accepts languages the agent declares."""
    assert "te-IN" in HOSPITAL_AGENT.languages


def test_greet_accepts_a_caller_who_opens_with_details():
    """Nobody opens a call by stating a bare intent."""
    node = HOSPITAL_AGENT.node("greet")
    assert "provide_details" in node.expected_intents
    assert any(t.when == "provide_details" for t in node.transitions)


# --- language stickiness ---------------------------------------------------


def _brain():
    from brain.orchestrator import Brain

    brain = Brain.__new__(Brain)  # the settler is a pure helper
    brain.agent = HOSPITAL_AGENT
    return brain


def _detected(language: str, confidence: float = 0.95):
    from brain.models import IntentResult

    return IntentResult(name="book_appointment", confidence=confidence, language=language)


def test_a_locked_session_never_switches_language():
    """Naming a language is an instruction, not a guess.

    One English-looking line ("book appointment tomorrow") would otherwise flip the
    whole call to English and never flip back.
    """
    session = Session(
        agent_name="h", node_id="greet", language="hi-IN", metadata={"language_locked": True}
    )
    settled = _brain()._settle_language(session, _detected("en-IN"), "book appointment tomorrow")
    assert settled.language == "hi-IN"


def test_language_is_locked_by_default():
    """The call keeps the language it started in.

    One English word from a Hindi caller used to flip the whole conversation, and
    because the agent then answered in English the caller followed it there and it
    never came back.
    """
    session = Session(agent_name="h", node_id="greet", language="hi-IN")
    settled = _brain()._settle_language(session, _detected("en-IN"), "book appointment tomorrow")
    assert settled.language == "hi-IN"


def test_following_the_caller_can_be_opted_back_into():
    """Still available per session, for deployments that want it."""
    session = Session(
        agent_name="h", node_id="greet", language="hi-IN",
        metadata={"language_locked": False},
    )
    settled = _brain()._settle_language(session, _detected("en-IN"), "book appointment tomorrow")
    assert settled.language == "en-IN"


def test_locking_does_not_stop_the_first_language_being_set():
    session = Session(agent_name="h", node_id="greet", metadata={"language_locked": True})
    settled = _brain()._settle_language(session, _detected("en-IN"), "hello there friend")
    assert settled.language == HOSPITAL_AGENT.languages[0]


def test_the_classifier_is_told_that_latin_script_is_not_english():
    """Indian callers type their own language in Latin letters.

    "mera naam amit hai" is Hindi; classifying it as English flips the call.
    """
    from brain.intent import _SYSTEM

    assert "Latin" in _SYSTEM
    assert "mera naam amit hai" in _SYSTEM


# --- brevity and script ----------------------------------------------------


def test_the_turn_is_capped_at_two_sentences():
    """A desk agent needing three sentences to ask for a name is padding."""
    from brain.planner import MAX_SENTENCES_PER_TURN

    assert MAX_SENTENCES_PER_TURN == 2


def test_the_style_prompt_demands_brevity():
    from brain.planner import _STYLE

    lowered = _STYLE.lower()
    assert "one short sentence" in lowered
    assert "no preamble" in lowered


def test_the_planner_token_cap_is_tight_enough_for_two_sentences():
    from brain.planner import PLANNER_MAX_TOKENS

    assert PLANNER_MAX_TOKENS <= 200


def test_the_script_rule_is_stated_once_for_every_language():
    """Told only "Hindi", the model answers in romanised Hindi.

    That reads as Hindi to a person but reaches a TTS voice expecting Devanagari, which
    then mispronounces it. The rule used to be a table naming each script; it is now
    one sentence, because every language here has exactly one script and a model that
    knows the language knows which — including languages nobody listed.
    """
    from brain.planner import _SCRIPT_RULE

    assert "own script" in _SCRIPT_RULE
    assert "Latin letters" in _SCRIPT_RULE


def test_english_words_are_allowed_but_english_replies_are_not():
    """Indian callers code-mix, and an agent that refuses to is the one that sounds odd.

    "slot" and "cardiology" in a Telugu sentence is how people speak. A whole English
    sentence on a Telugu call is the model dropping the language.
    """
    from brain.planner import _SCRIPT_RULE
    from brain.script import is_wrong_language

    assert "appointment" in _SCRIPT_RULE, "the rule names the loanwords it permits"
    assert not is_wrong_language("రేపు ఉదయం పదికి slots ఉన్నాయి", "te-IN")
    assert is_wrong_language("Which department do you need for the appointment?", "te-IN")


def test_the_reply_language_instruction_forbids_transliteration():
    from brain.models import IntentResult
    from brain.planner import build_messages

    session = Session(agent_name="h", node_id="greet", language="hi-IN")
    messages = build_messages(
        HOSPITAL_AGENT,
        HOSPITAL_AGENT.node("greet"),
        session,
        IntentResult(name="book_appointment", confidence=0.9),
        "mujhe appointment chahiye",
    )
    body = " ".join(m["content"] for m in messages)
    assert "Hindi" in body
    assert "own script" in body
    assert "Never write Hindi words in Latin letters" in body


def test_the_classifier_is_told_the_step_does_not_constrain_the_caller():
    """The node goal describes the assistant's job, not the caller's options.

    Presented as "Current step: greet", the classifier answered "unknown" to every
    utterance at that node — in every language — which escalated the call on turn two.
    """
    from brain.intent import _SYSTEM

    assert "not what the caller is" in _SYSTEM
    assert "never answer" in _SYSTEM.lower()


# --- phone collection ------------------------------------------------------


def test_a_phone_number_is_collected_before_confirming():
    """The closing line promises an SMS, so a number has to exist by then."""
    node = HOSPITAL_AGENT.node("collect_phone")
    assert node.required_slots == ("phone",)
    assert any(t.to == "confirm" for t in node.transitions)


def test_the_slot_step_leads_into_phone_collection():
    node = HOSPITAL_AGENT.node("offer_slots")
    assert any(t.when == "slots_filled" and t.to == "collect_phone" for t in node.transitions)


def test_booking_cannot_be_called_without_a_phone_number():
    from brain.tools import build_default_registry

    tool = build_default_registry().get("book_appointment")
    assert "phone" in tool.parameters["required"]


@pytest.mark.asyncio
async def test_the_booking_handler_refuses_without_a_phone_number():
    """Structural, not a prompt instruction: the model cannot talk its way past it."""
    from brain.tools import build_default_registry

    registry = build_default_registry()
    outcome = await registry.invoke(
        "book_appointment", {"patient_name": "Amit", "slot": "tomorrow 10:00 am"}
    )
    assert "error" in outcome.result


# --- character economy -----------------------------------------------------


def test_identifiers_are_not_spelled_out_in_words():
    """Bulbul bills per character; ten digits as words is the priciest line in a call."""
    from brain.planner import _STYLE

    lowered = _STYLE.lower()
    assert "plain digits" in lowered
    assert "leave the phone" in lowered


def test_the_confirm_step_does_not_repeat_the_phone_number():
    goal = HOSPITAL_AGENT.node("confirm").goal.lower()
    assert "do not repeat the phone number" in goal


def test_the_style_prompt_states_that_output_costs_money():
    from brain.planner import _STYLE

    assert "costs money to synthesise" in _STYLE
