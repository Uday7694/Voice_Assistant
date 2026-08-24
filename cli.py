"""Terminal harness for the brain.

Talk to the agent by typing. Every event the brain emits is shown, including the
per-stage timings — this is the loop you iterate the conversation design in, long
before any audio is involved.

    python cli.py
    python cli.py --quiet     # replies only
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from rich.console import Console

from brain.agents import HOSPITAL_AGENT
from brain.models import (
    EndEvent,
    ErrorEvent,
    EscalateEvent,
    IntentEvent,
    SayEvent,
    TimingEvent,
    ToolEvent,
    TransitionEvent,
)
from brain.orchestrator import Brain

console = Console()


def render(event, *, quiet: bool) -> None:
    if isinstance(event, SayEvent):
        console.print(f"[bold green]agent[/] {event.text}")
    elif quiet:
        return
    elif isinstance(event, IntentEvent):
        i = event.intent
        slots = f" slots={i.slots}" if i.slots else ""
        console.print(f"[dim]  intent={i.name} conf={i.confidence:.2f} lang={i.language}{slots}[/]")
    elif isinstance(event, TransitionEvent):
        console.print(f"[dim]  node {event.from_node} -> {event.to_node} ({event.reason})[/]")
    elif isinstance(event, ToolEvent):
        status = "ok" if event.ok else "FAILED"
        console.print(f"[dim]  tool {event.name} {status} in {event.ms:.0f}ms -> {event.result}[/]")
    elif isinstance(event, TimingEvent):
        parts = " ".join(f"{k}={v:.0f}ms" for k, v in event.stage_ms.items())
        console.print(f"[dim]  {parts}[/]")
    elif isinstance(event, EscalateEvent):
        console.print(f"[bold yellow]  -> escalated to human ({event.reason})[/]")
    elif isinstance(event, EndEvent):
        console.print(f"[bold blue]  -> call ended ({event.reason})[/]")
    elif isinstance(event, ErrorEvent):
        console.print(f"[bold red]  error: {event.message}[/]")


async def main(quiet: bool) -> None:
    brain = Brain(HOSPITAL_AGENT)
    session = brain.start(channel="cli")
    console.print(f"[bold]{HOSPITAL_AGENT.name}[/] session {session.session_id} — Ctrl-C to quit\n")

    # The agent speaks first, exactly as it would on an inbound call.
    async for event in brain.handle(session.session_id, "<call connected>"):
        render(event, quiet=quiet)

    while True:
        try:
            text = console.input("[bold cyan]you[/]   ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye")
            return
        if not text:
            continue
        if text in {"/quit", "/exit"}:
            return
        if text == "/state":
            console.print(brain.get(session.session_id))
            continue

        async for event in brain.handle(session.session_id, text):
            render(event, quiet=quiet)

        current = brain.get(session.session_id)
        if current and current.ended:
            console.print("[dim]session closed[/]")
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true", help="show replies only")
    parser.add_argument("--debug", action="store_true", help="verbose logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING)
    asyncio.run(main(quiet=args.quiet))
