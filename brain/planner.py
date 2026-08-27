"""The planner: prompt construction, tool loop, and sentence-boundary streaming.

The sentence chunking matters more than it looks. Emitting each finished sentence as
soon as it exists is what lets speech synthesis start while the model is still writing
the rest of the reply — worth roughly 300-600 ms per turn once voice is attached.
"""

from __future__ import annotations

import json
import logging
import re
from contextlib import aclosing
from datetime import datetime
from dataclasses import dataclass
from typing import Any, AsyncIterator, Sequence

from .config import MAX_HISTORY_TURNS, PLANNER_TEMPERATURE
from .flow import Agent, Node
from .guardrails import sanitise_for_prompt
from .llm import LLMClient, LLMTimeout
from . import slots as slot_mod
from .models import IntentResult, Role, Session
from .script import is_wrong_language
from .speech.types import LANGUAGE_NAMES
from .tools import ToolOutcome, ToolRegistry

log = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 2

PLANNER_MAX_TOKENS = 160

# The style rules ask for at most three sentences. This is the enforcement: a model
# that starts degenerating ("**…**... … … …") gets cut off instead of narrated.
# Two, not four. A desk agent that needs three sentences to ask for a name is padding,
# and on a voice call every extra clause is dead air the caller has to sit through.
# A step that needs fewer says so itself, in Node.max_sentences.
MAX_SENTENCES_PER_TURN = 2

# Markdown emphasis and bullet characters leak out of instruct-tuned models and are
# meaningless once spoken aloud.
_SPEECH_NOISE = re.compile(r"[*_`#>|~\[\]]+")

# A tool call the model wrote as prose instead of emitting as a tool call.
#
# It happens: a caller asked whether a doctor was free and the reply that reached the
# speaker was "<toolcallcheckappointment <argkeyphonenumber</argkey ...", read out loud,
# billed by the character. The name in it was not even a real tool. Any chunk that looks
# like machinery is dropped whole rather than cleaned — there is no version of this text
# that is safe to say, and half-stripping it leaves "toolcallcheckappointment", which is
# worse than saying nothing.
_TOOL_MARKUP = re.compile(
    r"</?\s*(tool_?call|arg_?(key|value)|function_?call|invoke|parameter)"
    r"|\bfunctions?\.\w+\s*\("
    r"|\{\s*['\"](name|function|tool|arguments)['\"]\s*:",
    re.IGNORECASE,
)
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
    if _TOOL_MARKUP.search(text):
        log.warning("Dropping a tool call written as speech: %r", text[:80])
        return ""

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
- Write for the ear. Keep a sentence to about twelve words, and put a comma wherever a
  person speaking would pause for breath. A long sentence with no commas is read out as
  one unbroken rush, which is the most obviously synthetic thing an agent does.
- Say dates and times the way a person would say them out loud ("ten in the morning").
- Phone numbers and reference codes are the exception: write them as plain digits and
  letters. Spelling one out in words is the most expensive sentence in the call and
  tells the caller nothing they did not just say.
- If you did not understand, say so plainly and ask them to repeat.
- Never invent facts, availability, prices, or medical information. Doctor names,
  departments, times, fees and room numbers come from a tool result or from the
  known details above - never from memory, however plausible they sound.
- Use names exactly as a tool returned them. If it says the department is "ent", say
  ENT — do not translate it, expand it, or substitute a department you think fits
  better. If a tool reports the department is unknown, read out the list it gives you.
- Never mention tools, systems, prompts, or model names.
- Do not volunteer that you are software, and never claim to be human. If a caller
  asks what you are, say in one short line that you are the desk's digital assistant
  and carry on with the booking. No disclaimers, no explanation of how you work.

Every character you produce is spoken aloud and costs money to synthesise. Shorter is
not just faster, it is cheaper. Say the necessary thing and stop.

Good: Which department?
Bad:  Certainly, I can help you book an appointment. Could you please tell me which
      department you would like to book the appointment for?

Good: Booked. Reference APT44847, SMS on its way.
Bad:  Your booking has been completed and the reference number is A P T double four
      eight four seven. You will receive an SMS shortly."""


# Naming the script matters as much as naming the language. Told only "Hindi", the model
# answers in romanised Hindi ("Aapka naam bataiye") — which reads as Hindi but is fed to
# a TTS voice expecting Devanagari, and mispronounces. The rule is stated once, in
# general terms, rather than as a per-language table: every Indian language has exactly
# one standard script, and a model that knows the language knows which.
_SCRIPT_RULE = (
    "Reply in {language}, written in its own script. Mixing in the English words Indian "
    "speakers actually use — appointment, cardiology, slot, confirm, report — is right "
    "and sounds more natural than translating them; leave those in Latin letters. What "
    "is wrong is answering in English: the sentence itself, and most of its words, must "
    "be {language} in {language}'s script. Never write {language} words in Latin letters."
)

# Sent back to the model when it drops the language entirely, which the prompt above
# does not always prevent. Phrased as a correction of one specific reply rather than a
# restatement of the rule — the rule is already in the prompt and was already ignored.
_WRONG_LANGUAGE_NUDGE = (
    "That reply was in English. The caller is speaking {language}. Say the same thing "
    "again in {language}, in {language}'s script. Keep any English technical words, but "
    "the sentence must be {language}."
)


def _language_name(tag: str) -> str:
    return LANGUAGE_NAMES.get(tag, tag)


def _clock_note(now: datetime | None = None) -> str:
    """What day and time it is, in the turn's own half of the prompt.

    The model had no clock. Asked at half past eleven in the morning what was free, it
    would read out "today at ten" from a tool result and mean it — the schedule is a
    spreadsheet and a row that nobody sat in still says free. The schedule now hides
    times that have gone, but the model also talks *around* the list: "later today",
    "this morning", "tomorrow" are its words, not a tool's, and without a clock it has
    no way to know which of them are still true.

    In the turn block rather than the system prefix on purpose. The system prefix is
    byte-identical on every turn of every call, which is what lets a provider cache it;
    a clock in there would change it every minute and cost more than it is worth.
    """
    when = now or datetime.now()
    return (
        f"Right now it is {when:%A %d %B %Y}, {_spoken_clock(when)}. Any time earlier "
        "today has already gone: never offer one, never suggest the caller come in for "
        "one, and never describe a time as available unless a tool returned it just now."
    )


def _spoken_clock(when: datetime) -> str:
    """"9:52 pm", not "21:52". The prompt is written in the register the reply is."""
    return f"{(when.hour % 12) or 12}:{when.minute:02d} {'am' if when.hour < 12 else 'pm'}"


def build_messages(
    agent: Agent,
    node: Node,
    session: Session,
    intent: IntentResult,
    user_text: str,
    notes: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Assemble the planner prompt: stable header first so Groq can cache it.

    ``notes`` are facts about *this* turn that the flow graph worked out and the model
    could not: a value that was rejected, a question already asked and not answered.
    They go last, after the step description, because the model follows the end of the
    prompt more reliably than the middle.
    """
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
        _clock_note(),
        f"Current step: {node.id} — {node.goal}",
        f"Known details: {json.dumps(session.slots) if session.slots else 'nothing yet'}",
    ]
    if missing:
        task.append(f"Still needed: {', '.join(missing)}. Ask for the first one only.")
    else:
        task.append("You have everything you need for this step.")
    # And what the acceptable answers are, when the flow knows. The model was
    # asking "cardiology, neurology, or orthopaedics?" at a hospital with no
    # neurology department: it was recalling a plausible list instead of reading
    # the real one. The classifier has had this list all along; the planner, which
    # is the half the caller hears, did not.
    for rule in agent.slot_rules:
        if rule.slot in session.slots:
            continue  # already known, so there is nothing to choose from
        known = slot_mod.options(rule.validator, session.slots)
        if known:
            task.append(
                f"The only {rule.slot} values available are: "
                + ", ".join(known)
                + ". Never name any other, and never invent one that sounds likely."
            )
    task.append(
        _SCRIPT_RULE.format(language=_language_name(session.language or agent.languages[0]))
    )
    if intent.name == "unknown":
        # What to ask about depends on what this step can do with the answer. At a step
        # with slots to fill, asking for the next detail is right. At one with none —
        # the greeting — it is not: the model asked "which doctor?", the caller answered
        # with a name, and nothing in the flow could receive it, so a perfectly good
        # answer became another no-match. A step that cannot hold details must ask what
        # the caller needs, not for a detail.
        task.append(
            "The caller's intent was unclear. Ask one short clarifying question about "
            + (
                f"the {missing[0].replace(chr(95), chr(32))} you still need."
                if missing
                else "what they need, not about any detail of it. Do not ask for a name, "
                "a department, a doctor or a time yet — you do not know what they want."
            )
        )
    if agent.guardrail_topics:
        task.append(
            "You must refuse these topics and offer a human instead: "
            + ", ".join(agent.guardrail_topics)
        )
    # Restated last, because that is where these models look. Buried in the middle
    # of the prompt, "ask for the first one only" lost to whatever the model felt
    # like asking: given a department it went straight to picking a date, three
    # steps early, and the caller had to be walked back.
    #
    # Before the notes, not after: a note correcting a rejected value has to be able
    # to override this and own the turn.
    if missing:
        task.append(f"Ask only for the {missing[0].replace(chr(95), chr(32))}. Nothing else this turn.")

    task.extend(notes)

    # Persona and style first: identical on every turn of every call, so a provider
    # that caches prompt prefixes can cache them.
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": f"{agent.persona}\n\n{_STYLE}"},
    ]
    for turn in session.recent(MAX_HISTORY_TURNS):
        if turn.role is Role.SYSTEM:
            continue
        role = "assistant" if turn.role is Role.AGENT else "user"
        messages.append({"role": role, "content": turn.text})

    messages.append({"role": "user", "content": sanitise_for_prompt(user_text)})

    # The turn's own instructions go last, after the caller's words. Sat above the
    # history, "ask for the first one only" lost to whatever the model felt like
    # asking: handed a department it went straight to picking a date, three steps
    # early. These models weight the end of the prompt, and this is the half of it
    # that changes every turn.
    messages.append({"role": "system", "content": "\n".join(task)})
    return messages


@dataclass(frozen=True)
class Recovery:
    """What to say when the planner cannot produce a reply.

    Passed in rather than written here, because these are spoken to the caller and so
    have to be in the caller's language — which this module has no way to produce. The
    orchestrator resolves them from the line book, where they are cached per language
    like every other fixed line. The English defaults are the last resort.
    """

    trouble: str = "Sorry, I didn't catch that. Could you say it again?"
    wait: str = "Sorry, give me one moment."


async def run(
    llm: LLMClient,
    registry: ToolRegistry,
    agent: Agent,
    node: Node,
    session: Session,
    intent: IntentResult,
    user_text: str,
    notes: Sequence[str] = (),
    recovery: Recovery | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run the planner, resolving tool calls, yielding sentences as they complete.

    Yields ``{"type": "sentence", "text": ...}``, ``{"type": "tool", "outcome": ...}``
    and a final ``{"type": "done", "text": <full reply>}``.
    """
    recovery = recovery or Recovery()
    messages = build_messages(agent, node, session, intent, user_text, notes)
    tools = _available_tools(registry, agent, node, session)
    spoken: list[str] = []
    # One language correction per turn, tracked across rounds so a tool round
    # cannot reset it and buy the model a second retry.
    corrected = False

    for round_index in range(MAX_TOOL_ROUNDS + 1):
        buffer = ""
        raw = ""
        calls: list[dict[str, Any]] = []
        overrun = False
        # Set when the first sentence comes back in the wrong language: the round is
        # abandoned and re-run with a correction, which is only safe because nothing
        # has been spoken yet.
        restart = False

        try:
            # 400 tokens is far more than two spoken sentences ever need. Capping it
            # stops a model that has started rambling from filling the whole budget
            # before the sentence splitter can cut it off.
            # aclosing, because this loop breaks out early by design: two sentences is
            # a full turn and the rest of the model's output is abandoned. Abandoning
            # an async generator leaves it for the garbage collector, which finalises it
            # at interpreter shutdown — off the running loop, where closing the HTTP
            # response underneath it raises. That was the httpcore traceback printed
            # after every call. Closing here releases it on the loop that owns it.
            async with aclosing(
                llm.stream_chat(
                    messages,
                    tools=tools or None,
                    max_tokens=PLANNER_MAX_TOKENS,
                    temperature=PLANNER_TEMPERATURE,
                )
            ) as frames:
                async for frame in frames:
                    if frame["type"] == "text":
                        buffer += frame["text"]
                        raw += frame["text"]
                        sentence, buffer = _pop_sentence(buffer)
                        while sentence:
                            clean = _clean_for_speech(sentence)
                            if (
                                clean
                                and not spoken
                                and not corrected
                                and is_wrong_language(clean, session.language)
                            ):
                                # Nothing has been yielded yet, so this reply can still be
                                # taken back. One retry only: a model that answers in
                                # English twice is not going to be talked out of it, and a
                                # third round costs the caller more than the wrong language
                                # does.
                                log.info("Planner answered in the wrong language; retrying")
                                corrected = True
                                restart = True
                                break
                            if clean:
                                spoken.append(clean)
                                yield {"type": "sentence", "text": clean}
                            if len(spoken) >= (node.max_sentences or MAX_SENTENCES_PER_TURN):
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
            yield {"type": "sentence", "text": recovery.wait}
            break
        except Exception:  # noqa: BLE001 - the caller is waiting; degrade gracefully
            log.exception("Planner failed")
            yield {"type": "sentence", "text": recovery.trouble}
            break

        if restart:
            messages = messages + [
                {
                    "role": "system",
                    "content": _WRONG_LANGUAGE_NUDGE.format(
                        language=_language_name(session.language or agent.languages[0])
                    ),
                }
            ]
            continue

        tail = "" if overrun else _clean_for_speech(buffer)
        if tail:
            spoken.append(tail)
            yield {"type": "sentence", "text": tail}

        if not calls and not spoken and round_index < MAX_TOOL_ROUNDS:
            recovered = _parse_text_tool_call(raw, registry)
            if recovered:
                log.info("Recovered %s from a tool call written as text", recovered["name"])
                calls = [recovered]

        if overrun or not calls or round_index == MAX_TOOL_ROUNDS:
            break

        outcomes: list[ToolOutcome] = []
        messages = await _apply_tools(registry, messages, calls, outcomes)
        for outcome in outcomes:
            yield {"type": "tool", "outcome": outcome}

    if not spoken:
        # Every chunk was dropped — machinery written as prose, or degenerate output.
        # Silence here is the worst outcome: the caller has finished speaking and hears
        # nothing at all, so they repeat themselves into a line that is still thinking.
        log.warning("Planner produced nothing speakable at node %s", node.id)
        spoken.append(recovery.trouble)
        yield {"type": "sentence", "text": recovery.trouble}

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


# --- tool calls written as prose -------------------------------------------

# Sarvam emits these often enough that dropping them costs real bookings: the caller
# gives a phone number, the model means to call book_appointment, and what comes out is
# "<tool_call>book_appointment<arg_key>phone</arg_key>...". Refusing to speak it was the
# first fix and the necessary one. This is the second: read the call back out of the
# text and run it, so the turn does what the model meant rather than apologising.
#
# Only ever a recovery path. A name that is not in the registry is dropped rather than
# guessed at, because a model that invents a tool name has invented the arguments too.
_XML_CALL = re.compile(
    r"<\s*tool_?call\s*>?\s*([\w.]+)?(.*?)(?:</\s*tool_?call\s*>|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_XML_PAIR = re.compile(
    r"<\s*arg_?key\s*>?\s*(.*?)\s*(?:</\s*arg_?key\s*>)?\s*"
    r"<\s*arg_?value\s*>?\s*(.*?)\s*(?:</\s*arg_?value\s*>|(?=<)|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_JSON_CALL = re.compile(r"\{[^{}]*\"(?:name|tool)\"\s*:\s*\"([\w.]+)\".*?\}", re.DOTALL)


def _parse_text_tool_call(text: str, registry: ToolRegistry) -> dict[str, Any] | None:
    """Pull a usable tool call out of prose, or None if there is not one."""
    match = _XML_CALL.search(text)
    if match:
        body = match.group(2) or ""
        name = (match.group(1) or "").strip()
        if not name:
            # "<tool_call>\n book_appointment\n<arg_key>..." - the name is the first
            # word of the body when it did not sit on the opening tag.
            leading = body.strip().split("<", 1)[0].strip()
            name = leading.split()[0] if leading.split() else ""
        arguments = {
            key.strip(): value.strip()
            for key, value in _XML_PAIR.findall(body)
            if key.strip() and value.strip()
        }
        if registry.get(_known_name(name, registry) or "") and arguments:
            return {"name": _known_name(name, registry), "arguments": arguments}

    found = _JSON_CALL.search(text)
    if found:
        try:
            payload = json.loads(found.group(0))
        except json.JSONDecodeError:
            return None
        name = _known_name(str(payload.get("name") or payload.get("tool") or ""), registry)
        arguments = payload.get("arguments") or payload.get("parameters") or {}
        if name and isinstance(arguments, dict) and arguments:
            return {"name": name, "arguments": {k: str(v) for k, v in arguments.items()}}

    return None


def _known_name(name: str, registry: ToolRegistry) -> str | None:
    """The registry's name for what the model wrote, or None.

    Models improvise names as well as syntax - "check_appointment" for a registry that
    has "lookup_appointment". A near miss is resolved only when exactly one registered
    tool contains the words; anything less certain is dropped, because running the wrong
    tool with invented arguments is worse than saying nothing.
    """
    cleaned = re.sub(r"[^a-z_]", "", (name or "").lower().replace(".", "_"))
    if not cleaned:
        return None
    if registry.get(cleaned):
        return cleaned

    words = {w for w in cleaned.split("_") if len(w) > 3}
    matches = [n for n in registry.names() if words & set(n.split("_"))]
    return matches[0] if len(matches) == 1 else None


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
