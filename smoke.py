"""Scripted end-to-end run against the real Groq API. Non-interactive."""

from __future__ import annotations

import asyncio
import sys

from brain.agents import HOSPITAL_AGENT
from brain.models import EscalateEvent, IntentEvent, SayEvent, TimingEvent, ToolEvent
from brain.orchestrator import Brain

SCRIPT = [
    "<call connected>",
    "hi, I want to book an appointment with a heart doctor",
    "my name is Asha Verma",
    "tomorrow morning works",
    "yes please confirm that",
]


async def main() -> int:
    brain = Brain(HOSPITAL_AGENT)
    session = brain.start(channel="smoke")

    for line in SCRIPT:
        if line != "<call connected>":
            print(f"\nyou    {line}")
        async for event in brain.handle(session.session_id, line):
            if isinstance(event, SayEvent):
                print(f"agent  {event.text}")
            elif isinstance(event, IntentEvent):
                print(f"       [intent {event.intent.name} {event.intent.confidence:.2f} "
                      f"slots={event.intent.slots}]")
            elif isinstance(event, ToolEvent):
                print(f"       [tool {event.name} ok={event.ok} {event.ms:.0f}ms -> {event.result}]")
            elif isinstance(event, TimingEvent):
                print("       [" + " ".join(f"{k}={v:.0f}ms" for k, v in event.stage_ms.items()) + "]")
            elif isinstance(event, EscalateEvent):
                print(f"       [escalated: {event.reason}]")

        state = brain.get(session.session_id)
        if state and state.ended:
            print("\n[session ended]")
            break

    final = brain.get(session.session_id)
    print(f"\nnode={final.node_id} slots={final.slots}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
