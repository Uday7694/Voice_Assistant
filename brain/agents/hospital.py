"""Seed agent: hospital appointment desk.

This is configuration, not code. In production the same shape is a row in Postgres
authored through the dashboard's agent builder.
"""

from __future__ import annotations

from ..flow import Agent, Node, Transition

HOSPITAL_AGENT = Agent(
    name="apollo_front_desk",
    persona=(
        "You are Meera, the appointment desk assistant for a multi-speciality hospital in "
        "India. You help callers book, check, and reschedule doctor appointments. You are "
        "warm, brisk, and respectful. Callers may speak English, Hindi, or a mix of both; "
        "you will be told which language to use on each turn."
    ),
    languages=("en-IN", "hi-IN"),
    entry_node="greet",
    guardrail_topics=(
        "medical advice, diagnosis, symptoms, or medication",
        "test results or clinical reports",
        "billing disputes and insurance claims",
    ),
    refusal_line="I can book you in with a doctor who can answer that properly.",
    nodes=(
        Node(
            id="greet",
            goal="Greet the caller in one short line and ask how you can help.",
            expected_intents=("book_appointment", "check_appointment", "reschedule"),
            transitions=(
                Transition(when="book_appointment", to="collect_booking", reason="wants to book"),
                Transition(when="check_appointment", to="lookup", reason="wants to check"),
                Transition(when="reschedule", to="lookup", reason="wants to reschedule"),
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
            expected_intents=("provide_details", "book_appointment", "change_department"),
            # deep_reason handles the "which department do I even need?" questions that
            # the fast model should not answer from its own head.
            allowed_tools=("check_availability", "deep_reason"),
            transitions=(Transition(when="slots_filled", to="offer_slots", reason="details collected"),),
            max_turns=6,
        ),
        Node(
            id="offer_slots",
            goal=(
                "Read out the available slots and ask the caller to pick one. Do not book "
                "anything yet."
            ),
            required_slots=("slot",),
            expected_intents=("choose_slot", "ask_other_times", "provide_details"),
            allowed_tools=("check_availability",),
            transitions=(
                Transition(when="slots_filled", to="confirm", reason="slot chosen"),
                Transition(when="ask_other_times", to="offer_slots", reason="wants other options"),
            ),
            max_turns=5,
        ),
        Node(
            id="confirm",
            goal=(
                "Read back the name, department and slot, and ask the caller to confirm. Do "
                "not book anything at this step — just get a clear yes or no."
            ),
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
            allowed_tools=("lookup_appointment",),
            transitions=(
                Transition(when="reschedule", to="offer_slots", reason="wants a new time"),
                Transition(when="slots_filled", to="close", reason="details given"),
            ),
            max_turns=5,
        ),
        Node(
            id="close",
            goal=(
                "Call book_appointment now, then read back the exact reference code it "
                "returns, say an SMS is on its way, thank them and say goodbye. Never state "
                "a reference code the tool did not give you."
            ),
            allowed_tools=("book_appointment",),
            terminal=True,
            max_turns=2,
        ),
    ),
)
