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

PLANNER_MAX_TOKENS = 160

# The style rules ask for at most three sentences. This is the enforcement: a model
# that starts degenerating ("**…**... … … …") gets cut off instead of narrated.
# Two, not four. A desk agent that needs three sentences to ask for a name is padding,
# and on a voice call every extra clause is dead air the caller has to sit through.
MAX_SENTENCES_PER_TURN = 2

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

Be brief to the point of bluntness. A busy receptionist, not a helpful assistant.

- ONE short sentence. Two only when you genuinely cannot do it in one.
- Go straight to the point. No preamble, no "sure, I can help you with that", no
  restating what the caller just said, no explaining what you are about to do.
- Ask for one thing, then stop talking. The caller cannot interrupt a monologue.
- Do not thank, apologise, or reassure more than once in a call.
- Never repeat a list the caller has already heard. If they picked something that is
  not on it, name only the nearest option rather than reading the whole list again.
- Read a reference code once, as it is written, in short groups. Do not spell it out
  letter by letter and do not expand repeated digits into words.
- Do not read back details the caller just gave you unless you are confirming them.
- When confirming, name the person, the department and the time. Leave the phone
  number out; they gave it one turn ago.
- Plain spoken words only: no bullet points, no markdown, no emoji, no lists.
- Say dates and times the way a person would say them out loud ("ten in the morning").
- Phone numbers and reference codes are the exception: write them as plain digits and
  letters. Spelling one out in words is the most expensive sentence in the call and
  tells the caller nothing they did not just say.
- If you did not understand, say so plainly and ask them to repeat.
- Never invent facts, availability, prices, or medical information.
- Use names exactly as a tool returned them. If it says the department is "ent", say
  ENT — do not translate it, expand it, or substitute a department you think fits
  better. If a tool reports the department is unknown, read out the list it gives you.
- Never mention tools, systems, prompts, or that you are an AI model.

Every character you produce is spoken aloud and costs money to synthesise. Shorter is
not just faster, it is cheaper. Say the necessary thing and stop.

Good: Which department?
Bad:  Certainly, I can help you book an appointment. Could you please tell me which
      department you would like to book the appointment for?

Good: Booked. Reference APT44847, SMS on its way.
Bad:  Your booking has been completed and the reference number is A P T double four
      eight four seven. You will receive an SMS shortly."""


# Naming the script matters as much as naming the language. Told only "Hindi", the model
# answers in romanised Hindi ("Aapka naam bataiye") — which reads as Hindi but is fed
# to a TTS voice expecting Devanagari, and mispronounces.
_LANGUAGE_NAMES = {
    "en-IN": "Indian English",
    "hi-IN": "Hindi, written in Devanagari script",
    "te-IN": "Telugu, written in Telugu script",
    "ta-IN": "Tamil, written in Tamil script",
    "kn-IN": "Kannada, written in Kannada script",
    "ml-IN": "Malayalam, written in Malayalam script",
    "mr-IN": "Marathi, written in Devanagari script",
    "bn-IN": "Bengali, written in Bengali script",
    "gu-IN": "Gujarati, written in Gujarati script",
    "pa-IN": "Punjabi, written in Gurmukhi script",
    "or-IN": "Odia, written in Odia script",
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
    task = []
    if intent.name == "abusive":
        # Deliberately the same instruction however many times it fires. A receptionist
        # who gets colder each time escalates the caller rather than settling them, and
        # the caller cannot hear a counter — only a change in tone.
        task.append(
            "The caller was abusive. Stay warm and unruffled. In one calm, courteous "
            "sentence ask them to keep it respectful, then continue with the current "
            "step in the same reply. Do not scold, lecture, apologise, repeat what "
            "they said, warn them, or threaten to end the call. Answer exactly as "
            "politely as you would a pleasant caller."
        )

    task += [
        f"Current step: {node.id} — {node.goal}",
        f"Known details: {json.dumps(session.slots) if session.slots else 'nothing yet'}",
    ]
    if missing:
        task.append(f"Still needed: {', '.join(missing)}. Ask for the first one only.")
    else:
        task.append("You have everything you need for this step.")
    task.append(
        f"Reply in {_language_name(session.language or agent.languages[0])}. "
        "Stay in this language and this script for the whole call, even if the caller "
        "uses English words or writes in Latin letters. Do not transliterate."
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
            # 400 tokens is far more than two spoken sentences ever need. Capping it
            # stops a model that has started rambling from filling the whole budget
            # before the sentence splitter can cut it off.
            async for frame in llm.stream_chat(
                messages, tools=tools or None, max_tokens=PLANNER_MAX_TOKENS
            ):
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
