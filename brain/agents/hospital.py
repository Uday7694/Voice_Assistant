"""Seed agent: hospital appointment desk.

This is configuration, not code. In production the same shape is a row in Postgres
authored through the dashboard's agent builder.
"""

from __future__ import annotations

from ..flow import Agent, Node, SlotRule, Transition

HOSPITAL_AGENT = Agent(
    name="apollo_front_desk",
    persona=(
        "You are Meera, the appointment desk assistant for a multi-speciality hospital in "
        "India. You help callers book, check, and reschedule doctor appointments. You are "
        "warm, brisk, and respectful. Callers may speak English, Hindi, or a mix of both; "
        "you will be told which language to use on each turn."
    ),
    # A language must be listed here before the agent will switch into it: the language
    # settler only accepts a detected language that the agent declares.
    languages=("en-IN", "hi-IN", "te-IN"),
    entry_node="greet",
    guardrail_topics=(
        "medical advice, diagnosis, symptoms, or medication",
        "test results or clinical reports",
        "billing disputes and insurance claims",
    ),
    refusal_line="I can book you in with a doctor who can answer that properly.",
    purpose_line=(
        "booking, checking and rescheduling doctor appointments at the hospital"
    ),
    # Checked at capture, not at tool time. "architecture" used to be accepted as a
    # department and only refused three turns later, after the caller had given a name
    # and heard slots read out.
    slot_rules=(
        SlotRule(slot="department", validator="department"),
        SlotRule(slot="doctor", validator="doctor"),
        SlotRule(slot="phone", validator="phone"),
        # "Book it tomorrow" is a preference, not a choice of slot. Without this the
        # flow moved on to the phone number with no time chosen, and the caller heard
        # what time their appointment was for the first time in the SMS.
        SlotRule(slot="slot", validator="slot"),
    ),
    nodes=(
        Node(
            id="greet",
            goal="Greet the caller in one short line and ask how you can help.",
            expected_intents=(
                "book_appointment",
                # "Is the cardiology doctor free?" is not "what is my appointment?".
                # Without an intent of its own it classified as check_appointment,
                # which routes to lookup, which opens by asking for a phone number —
                # so a caller with a general question was asked to identify himself
                # before anyone had established he was a patient at all.
                "check_availability",
                "check_appointment",
                "reschedule",
                "provide_details",
            ),
            # Callers open with the real question far more often than with an intent.
            allowed_tools=("deep_reason",),
            transitions=(
                Transition(when="book_appointment", to="collect_booking", reason="wants to book"),
                Transition(
                    when="check_availability",
                    to="collect_booking",
                    reason="asking what is free",
                ),
                Transition(when="check_appointment", to="lookup", reason="wants to check"),
                Transition(when="reschedule", to="lookup", reason="wants to reschedule"),
                # Nobody opens a call by stating a bare intent. "I need a cardiology
                # appointment" carries the department with it and classifies as
                # provide_details, which left the caller stuck at greet repeating
                # themselves until max_turns escalated the call.
                Transition(
                    when="provide_details",
                    to="collect_booking",
                    reason="opened with booking details",
                ),
            ),
            max_turns=3,
            on_max_turns="escalate",
        ),
        Node(
            id="collect_booking",
            goal=(
                "Collect the patient's name and the department they need. Ask for one thing "
                "at a time. Once you have both, offer available slots."
            ),
            required_slots=("patient_name", "department"),
            expected_intents=(
                "provide_details",
                "book_appointment",
                "check_availability",
                "change_department",
            ),
            # deep_reason handles the "which department do I even need?" questions that
            # the fast model should not answer from its own head.
            allowed_tools=("check_availability", "deep_reason"),
            transitions=(Transition(when="slots_filled", to="choose_doctor", reason="details collected"),),
            max_turns=6,
        ),
        Node(
            # A caller who is told a time before being told whose time it is has been
            # booked with a stranger. Naming the doctor is also the moment the hospital
            # sounds like a hospital rather than a booking form: the fee, the
            # experience and the languages they speak all live on this step.
            id="choose_doctor",
            goal=(
                "Name the doctors available in this department, with one detail each - "
                "seniority or languages, whichever helps - and ask which one they want. "
                "If only one is available, say so and move on."
            ),
            required_slots=("doctor",),
            expected_intents=(
                "choose_doctor",
                "provide_details",
                "ask_doctor_details",
                "change_department",
            ),
            allowed_tools=("find_doctors", "check_availability", "deep_reason"),
            transitions=(
                Transition(when="slots_filled", to="offer_slots", reason="doctor chosen"),
                Transition(
                    when="change_department", to="collect_booking", reason="wants another department"
                ),
            ),
            max_turns=5,
        ),
        Node(
            id="offer_slots",
            goal=(
                "Read out this doctor's open slots and ask the caller to pick one. Do not "
                "book anything yet, and never offer a time the tool did not return."
            ),
            required_slots=("slot",),
            expected_intents=("choose_slot", "ask_other_times", "provide_details"),
            allowed_tools=("check_availability", "find_doctors", "deep_reason"),
            transitions=(
                Transition(when="slots_filled", to="collect_phone", reason="slot chosen"),
                Transition(when="ask_other_times", to="offer_slots", reason="wants other options"),
            ),
            max_turns=5,
        ),
        Node(
            id="collect_phone",
            goal=(
                "Ask for a ten digit mobile number for the confirmation message. Nothing "
                "else."
            ),
            required_slots=("phone",),
            expected_intents=("provide_details", "change_slot"),
            transitions=(
                Transition(when="slots_filled", to="confirm", reason="number given"),
                Transition(when="change_slot", to="offer_slots", reason="wants a different slot"),
            ),
            max_turns=3,
        ),
        Node(
            # No deep_reason here: this node exists to read details back and take a
            # yes or no. A 6-25 s detour mid-confirmation strands the caller at the
            # one point in the call where they are ready to commit.
            id="confirm",
            goal=(
                "Say exactly one sentence: the patient name, the doctor, the day and the "
                "time, ending in 'shall I confirm?'. Nothing before it and nothing after "
                "it. Do not repeat the phone number and do not book anything here."
            ),
            # One sentence, enforced. Told to "read the details back, then ask for a yes
            # or no", the model said "shall I say yes?" first and the details second — so
            # the caller was asked to confirm before hearing what they were confirming,
            # and the question they answered was about nothing.
            max_sentences=1,
            expected_intents=("confirm_yes", "confirm_no", "change_slot"),
            transitions=(
                Transition(when="confirm_yes", to="close", reason="caller confirmed"),
                Transition(when="confirm_no", to="offer_slots", reason="caller declined"),
                Transition(when="change_slot", to="offer_slots", reason="wants a different slot"),
            ),
            max_turns=4,
        ),
        Node(
            id="lookup",
            goal=(
                "Ask for the caller's phone number or booking reference, then read back the "
                "appointment you find."
            ),
            required_slots=("phone",),
            expected_intents=("provide_details", "reschedule", "cancel"),
            allowed_tools=("lookup_appointment", "deep_reason"),
            transitions=(
                Transition(when="reschedule", to="offer_slots", reason="wants a new time"),
                Transition(when="slots_filled", to="close", reason="details given"),
            ),
            max_turns=5,
        ),
        Node(
            # No deep_reason here either. Terminal, max_turns=2 — an answer that takes
            # 6-25 s cannot arrive before the call is over, so offering the tool would
            # only promise the caller something that never comes.
            id="close",
            goal=(
                "Call book_appointment now, then read back the exact reference code it "
                "returns and the room number, and say an SMS is on its way. Never state a "
                "reference code the tool did not give you. Do not say goodbye — the desk "
                "closes the call itself, and a goodbye here would crowd out the room "
                "number the caller actually needs."
            ),
            allowed_tools=("book_appointment",),
            terminal=True,
            max_turns=2,
        ),
    ),
)
