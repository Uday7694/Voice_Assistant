"""Flow graph: the deterministic skeleton the LLM reasons inside.

The planner decides *what* to say. This module decides *where the conversation is* and
what it is allowed to do there. Keeping control here is what stops an agent rambling,
looping, or wandering out of its job.
"""

from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel

from .models import IntentResult, Session

# Transition conditions, evaluated in declaration order.
#   "slots_filled" — every required slot for the node has a value
#   "always"       — unconditional fallthrough
#   any other      — matches an intent name


class Transition(BaseModel):
    model_config = {"frozen": True}

    when: str
    to: str
    reason: str = ""


class Node(BaseModel):
    """One step of the conversation."""

    model_config = {"frozen": True}

    id: str
    goal: str
    """One line telling the planner what to accomplish here. Injected into the prompt."""

    required_slots: tuple[str, ...] = ()
    expected_intents: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    transitions: tuple[Transition, ...] = ()
    max_turns: int = 6
    on_max_turns: str = "escalate"
    terminal: bool = False

    def missing_slots(self, slots: dict[str, str]) -> tuple[str, ...]:
        return tuple(s for s in self.required_slots if not slots.get(s))


class Agent(BaseModel):
    """A configured assistant. In production this is a row in Postgres."""

    model_config = {"frozen": True}

    name: str
    persona: str
    languages: tuple[str, ...] = ("en-IN",)
    entry_node: str = "greet"
    nodes: tuple[Node, ...] = ()
    guardrail_topics: tuple[str, ...] = ()
    """Topics this agent must refuse outright, e.g. medical advice."""

    refusal_line: str = "I'm not able to help with that, but I can connect you to someone who can."

    def node(self, node_id: str) -> Node:
        for candidate in self.nodes:
            if candidate.id == node_id:
                return candidate
        raise KeyError(f"Agent {self.name!r} has no node {node_id!r}")

    @property
    def all_intents(self) -> tuple[str, ...]:
        seen: list[str] = []
        for node in self.nodes:
            for intent in node.expected_intents:
                if intent not in seen:
                    seen.append(intent)
        return tuple(seen)


class FlowDecision(BaseModel):
    model_config = {"frozen": True}

    node_id: str
    changed: bool
    reason: str = ""
    force_escalate: bool = False
    force_end: bool = False


def next_node(agent: Agent, session: Session, intent: IntentResult) -> FlowDecision:
    """Pick the node for this turn. Pure function — no I/O, trivially testable."""
    current = agent.node(session.node_id or agent.entry_node)

    if intent.name == "end_conversation":
        return FlowDecision(node_id=current.id, changed=False, reason="user ended", force_end=True)
    if intent.name == "escalate_to_human":
        return FlowDecision(
            node_id=current.id, changed=False, reason="user asked for a human", force_escalate=True
        )

    if session.turns_in_node >= current.max_turns:
        if current.on_max_turns == "escalate":
            return FlowDecision(
                node_id=current.id,
                changed=False,
                reason=f"stuck in {current.id}",
                force_escalate=True,
            )
        return FlowDecision(
            node_id=current.on_max_turns,
            changed=current.on_max_turns != current.id,
            reason="max turns reached",
        )

    target = _match_transition(current, session, intent)
    if target is None:
        return FlowDecision(node_id=current.id, changed=False, reason="stay")
    return FlowDecision(node_id=target.to, changed=target.to != current.id, reason=target.reason)


def _match_transition(node: Node, session: Session, intent: IntentResult) -> Transition | None:
    for transition in node.transitions:
        if transition.when == "always":
            return transition
        if transition.when == "slots_filled" and not node.missing_slots(session.slots):
            return transition
        if transition.when == intent.name and intent.is_confident:
            return transition
    return None


def validate(agent: Agent) -> Sequence[str]:
    """Catch broken graphs at load time rather than mid-call."""
    problems: list[str] = []
    ids = {node.id for node in agent.nodes}
    if agent.entry_node not in ids:
        problems.append(f"entry_node {agent.entry_node!r} is not a node")
    for node in agent.nodes:
        for transition in node.transitions:
            if transition.to not in ids:
                problems.append(f"{node.id} -> unknown node {transition.to!r}")
        if node.on_max_turns not in ids and node.on_max_turns != "escalate":
            problems.append(f"{node.id}.on_max_turns -> unknown node {node.on_max_turns!r}")
        if not node.terminal and not node.transitions:
            problems.append(f"{node.id} is a dead end but not marked terminal")
    return problems
