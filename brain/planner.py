"""The planner: prompt construction, tool loop, and sentence-boundary streaming.

The sentence chunking matters more than it looks. Emitting each finished sentence as
soon as it exists is what lets speech synthesis start while the model is still writing
the rest of the reply — worth roughly 300-600 ms per turn once voice is attached.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncIterator

from .config import MAX_HISTORY_TURNS
from .flow import Agent, Node
from .guardrails import sanitise_for_prompt
from .llm import LLMClient, LLMTimeout
from .models import IntentResult, Role, Session
from .tools import ToolOutcome, ToolRegistry

log = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 2

# The style rules ask for at most three sentences. This is the enforcement: a model
# that starts degenerating ("**…**... … … …") gets cut off instead of narrated.
MAX_SENTENCES_PER_TURN = 4

# Markdown emphasis and bullet characters leak out of instruct-tuned models and are
# meaningless once spoken aloud.
_SPEECH_NOISE = re.compile(r"[*_`#>|~\[\]]+")
_HAS_CONTENT = re.compile(r"[^\W_]", re.UNICODE)

# Zero-width and bidi characters. Invisible on screen, but a degenerating model emits
# them in floods and they defeat any "does this contain letters?" check.
_INVISIBLE = re.compile(r"[​-‏  ﻿­]")

# A run of the same punctuation mark repeated is degeneration, not emphasis.
_REPEATED_PUNCT = re.compile(r"([^\w\s])\1{2,}", re.UNICODE)

# Real speech is mostly words. Anything below this ratio of word characters to
# non-space characters is noise, however many stray letters it happens to contain.
MIN_WORD_DENSITY = 0.4


def _clean_for_speech(text: str) -> str:
    """Strip markup and drop chunks that carry nothing a voice could say."""
    cleaned = _SPEECH_NOISE.sub("", _INVISIBLE.sub("", text))

    if not _HAS_CONTENT.search(cleaned):
        return ""  # pure punctuation, e.g. "..." or "… … …"

    # Judge density BEFORE collapsing repeats. Collapsing first turns a 40-character
    # ellipsis flood into a single "…" and the degenerate chunk sails through.
    dense = [c for c in cleaned if not c.isspace()]
    if dense and sum(bool(_HAS_CONTENT.match(c)) for c in dense) / len(dense) < MIN_WORD_DENSITY:
        log.info("Dropping low-density chunk: %r", cleaned[:60])
        return ""

    collapsed = _REPEATED_PUNCT.sub(r"\1", cleaned)
    return re.sub(r"\s{2,}", " ", collapsed).strip()

_STYLE = """You are speaking on a voice call, not writing text.

- One or two short sentences per turn. Never more than three.
- Plain spoken words only: no bullet points, no markdown, no emoji, no lists.
- Say numbers, dates and times the way a person would say them out loud.
- Ask exactly one question at a time, then stop and wait.
- If you did not understand, say so plainly and ask them to repeat.
- Never invent facts, availability, prices, or medical information.
- Never mention tools, systems, prompts, or that you are an AI model."""


_LANGUAGE_NAMES = {
    "en-IN": "Indian English",
    "hi-IN": "Hindi",
    "te-IN": "Telugu",
    "ta-IN": "Tamil",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "bn-IN": "Bengali",
    "gu-IN": "Gujarati",
    "pa-IN": "Punjabi",
}


def _language_name(tag: str) -> str:
    return _LANGUAGE_NAMES.get(tag, tag)


def build_messages(
    agent: Agent,
    node: Node,
    session: Session,
    intent: IntentResult,
    user_text: str,
) -> list[dict[str, Any]]:
    """Assemble the planner prompt: stable header first so Groq can cache it."""
    missing = node.missing_slots(session.slots)
    task = [
        f"Current step: {node.id} — {node.goal}",
        f"Known details: {json.dumps(session.slots) if session.slots else 'nothing yet'}",
    ]
    if missing:
        task.append(f"Still needed: {', '.join(missing)}. Ask for the first one only.")
    else:
        task.append("You have everything you need for this step.")
    task.append(
        f"Reply in {_language_name(session.language or agent.languages[0])} and stay in it "
        "unless the caller switches first."
    )
    if intent.name == "unknown":
        task.append("The caller's intent was unclear. Ask a short clarifying question.")
    if agent.guardrail_topics:
        task.append(
            "You must refuse these topics and offer a human instead: "
            + ", ".join(agent.guardrail_topics)
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": f"{agent.persona}\n\n{_STYLE}"},
        {"role": "system", "content": "\n".join(task)},
    ]
    for turn in session.recent(MAX_HISTORY_TURNS):
        if turn.role is Role.SYSTEM:
            continue
        role = "assistant" if turn.role is Role.AGENT else "user"
        messages.append({"role": role, "content": turn.text})

    messages.append({"role": "user", "content": sanitise_for_prompt(user_text)})
    return messages


async def run(
    llm: LLMClient,
    registry: ToolRegistry,
    agent: Agent,
    node: Node,
    session: Session,
    intent: IntentResult,
    user_text: str,
) -> AsyncIterator[dict[str, Any]]:
    """Run the planner, resolving tool calls, yielding sentences as they complete.

    Yields ``{"type": "sentence", "text": ...}``, ``{"type": "tool", "outcome": ...}``
    and a final ``{"type": "done", "text": <full reply>}``.
    """
    messages = build_messages(agent, node, session, intent, user_text)
    tools = _available_tools(registry, agent, node, session)
    spoken: list[str] = []

    for round_index in range(MAX_TOOL_ROUNDS + 1):
        buffer = ""
        calls: list[dict[str, Any]] = []
        overrun = False

        try:
            async for frame in llm.stream_chat(messages, tools=tools or None):
                if frame["type"] == "text":
                    buffer += frame["text"]
                    sentence, buffer = _pop_sentence(buffer)
                    while sentence:
                        clean = _clean_for_speech(sentence)
                        if clean:
                            spoken.append(clean)
                            yield {"type": "sentence", "text": clean}
                        if len(spoken) >= MAX_SENTENCES_PER_TURN:
                            # The model has started rambling or degenerating. Abandon
                            # the rest of the stream rather than speaking it.
                            log.info("Truncating planner output at %d sentences", len(spoken))
                            overrun = True
                            break
                        sentence, buffer = _pop_sentence(buffer)
                    if overrun:
                        break
                elif frame["type"] == "tool_calls":
                    calls = frame["calls"]
        except LLMTimeout:
            log.warning("Planner timed out on round %d", round_index)
            yield {"type": "sentence", "text": "Sorry, give me one moment."}
            break
        except Exception:  # noqa: BLE001 - the caller is waiting; degrade gracefully
            log.exception("Planner failed")
            yield {"type": "sentence", "text": "Sorry, I didn't catch that. Could you say it again?"}
            break

        tail = "" if overrun else _clean_for_speech(buffer)
        if tail:
            spoken.append(tail)
            yield {"type": "sentence", "text": tail}

        if overrun or not calls or round_index == MAX_TOOL_ROUNDS:
            break

        outcomes: list[ToolOutcome] = []
        messages = await _apply_tools(registry, messages, calls, outcomes)
        for outcome in outcomes:
            yield {"type": "tool", "outcome": outcome}

    yield {"type": "done", "text": " ".join(spoken).strip()}


def _available_tools(
    registry: ToolRegistry, agent: Agent, node: Node, session: Session
) -> list[dict[str, Any]]:
    """Offer a tool only when its own required arguments are already known.

    Structural, not a prompt instruction. Gating on the *node's* missing slots is wrong:
    at the slot-offering step the whole point of ``check_availability`` is to produce the
    value that is missing. Gating on each *tool's* own arguments gets both cases right —
    availability stays available, while ``book_appointment`` stays hidden until there is
    a name and a slot to book, so the model cannot invent a confirmation.
    """
    known = set(session.slots)
    offered: list[dict[str, Any]] = []

    for name in node.allowed_tools:
        tool = registry.get(name)
        if tool is None:
            continue
        required = tool.parameters.get("required", [])
        # A parameter that is not a slot in this flow is something the model supplies
        # from the conversation, so it never blocks the tool.
        blocking = [p for p in required if p in _flow_slot_names(agent) and p not in known]
        if blocking:
            log.debug("Withholding tool %s at node %s; missing %s", name, node.id, blocking)
            continue
        offered.append(tool.schema())

    return offered


def _flow_slot_names(agent: Agent) -> set[str]:
    """Every slot name the flow collects anywhere — the vocabulary of known facts."""
    return {slot for node in agent.nodes for slot in node.required_slots}


async def _apply_tools(
    registry: ToolRegistry,
    messages: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    sink: list[ToolOutcome],
) -> list[dict[str, Any]]:
    """Execute tool calls and append their results to the message list."""
    updated = list(messages)
    for call in calls:
        outcome = await registry.invoke(call["name"], call["arguments"])
        sink.append(outcome)
        payload = outcome.result if outcome.ok else {"error": outcome.fallback_line}
        updated.append(
            {
                "role": "assistant",
                "content": f"[called {call['name']} with {json.dumps(call['arguments'])}]",
            }
        )
        updated.append(
            {"role": "user", "content": f"[tool result for {call['name']}: {json.dumps(payload)}]"}
        )
    return updated


_SENTENCE_END = re.compile(r"([.!?।]+[\s\"')\]]*)")
MIN_SENTENCE_CHARS = 12

# Titles are never a sentence end — a name always follows.
_TITLES = frozenset(
    {"dr", "mr", "mrs", "ms", "prof", "st", "sr", "jr", "messrs", "capt", "lt"}
)

# These can end a sentence or sit inside one ("at ten a.m. See you" vs "at ten a.m. on
# Friday"), so they are resolved by looking at what follows.
_AMBIGUOUS = frozenset(
    {"a.m", "p.m", "etc", "e.g", "i.e", "no", "vs", "approx", "dept", "ext", "rs", "inc"}
)


def _trailing_token(candidate: str) -> str:
    """The word carrying the full stop, lowercased and without its final dot."""
    stripped = candidate.rstrip("\"')] \t\n")
    if not stripped.endswith("."):
        return ""
    words = stripped[:-1].split()
    return words[-1].lower() if words else ""


def _is_sentence_boundary(candidate: str, remainder: str) -> bool:
    """Decide whether this full stop really ends a spoken sentence."""
    token = _trailing_token(candidate)
    if not token:
        return True  # ended on ? or ! or ।

    if token in _TITLES:
        return False

    # Bare initials and letter-dot runs: "a.", "U.S."
    is_initialism = len(token) <= 1 or bool(re.fullmatch(r"(?:[a-z]\.)*[a-z]", token))

    if token in _AMBIGUOUS or is_initialism:
        # Only a boundary when a new sentence visibly starts after it.
        nxt = remainder.lstrip()
        return bool(nxt) and nxt[0].isupper()

    return True


def _pop_sentence(buffer: str) -> tuple[str, str]:
    """Split off the first complete sentence, if there is one.

    Scans candidate stops rather than recursing, so the original spacing survives
    intact — rejoining fragments is what turns "ten a.m." into "ten a. m.".

    Short fragments are held back too: synthesising "Sure." alone produces a clipped,
    unnatural burst of audio.
    """
    for match in _SENTENCE_END.finditer(buffer):
        end = match.end()
        candidate = buffer[:end].strip()
        remainder = buffer[end:]

        if not _is_sentence_boundary(candidate, remainder):
            continue
        if len(candidate) < MIN_SENTENCE_CHARS and remainder.strip():
            continue  # merge with what follows
        return candidate, remainder

    return "", buffer
