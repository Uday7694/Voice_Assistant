"""The Brain: one turn of conversation, start to finish.

Deliberately transport-agnostic. It takes text and yields events. A CLI consumes those
events today; a LiveKit agent will consume the same events tomorrow, sending each
SayEvent straight to speech synthesis as it arrives.

Order of operations per turn:
    guardrails (in) -> intent -> slots -> flow decision -> planner (+tools)
    -> guardrails (out) -> persist
"""

from __future__ import annotations

import logging
import re
import time
from typing import AsyncIterator

from . import guardrails, intent as intent_mod, planner, slots as slot_mod, transcripts
from .config import (
    DEEP_REASON_ENABLED,
    LOCK_LANGUAGE,
    MAX_ABUSIVE_TURNS,
    MAX_CONSECUTIVE_NO_MATCH,
    MAX_OFF_TOPIC_TURNS,
    MAX_TURNS_PER_SESSION,
)
from .flow import Agent, next_node, validate
from .lines import LineBook
from .llm import LLMClient
from .memory import InMemorySessionStore, SessionStore
from .models import (
    EndEvent,
    ErrorEvent,
    EscalateEvent,
    Event,
    IntentEvent,
    IntentResult,
    Role,
    SayEvent,
    Session,
    TimingEvent,
    ToolEvent,
    TransitionEvent,
)
from .subagent import DeepSubagent, current_session
from .tools import ToolOutcome, ToolRegistry, build_default_registry

log = logging.getLogger(__name__)


# What the planner is told when it is about to ask the same thing again. The wording
# escalates, because a caller who did not answer twice did not mishear twice — the
# question itself is the problem, so the second attempt rephrases and the third offers
# a way out. The agent never says "I already asked you": blaming the caller is how a
# stuck call becomes an angry one.
_REASK = {
    2: (
        "You already asked for {slot} last turn and did not get it. Ask again in "
        "different words, shorter than before. Do not repeat your previous sentence."
    ),
    3: (
        "You have asked for {slot} twice with no answer. Ask a third time in plainly "
        "different words, and make it answerable in one word — give an example of what "
        "you need if that helps. Do not mention that you already asked."
    ),
}

_REASK_LAST = (
    "You have asked for {slot} {count} times without an answer. Ask once more, very "
    "simply, and offer to put them through to a colleague if it is easier. Do not "
    "mention that you already asked."
)


# Written as a brief, not as a sentence to translate. The old version handed the model
# the purpose line verbatim — "booking, checking and rescheduling doctor appointments" —
# and a model given a three-verb list translates the list: callers heard "यह desk doctor
# appointments book, check और reschedule करने के लिए है", which is a brochure read aloud,
# not a person talking. One verb is what a receptionist actually says.
_OUT_OF_SCOPE = (
    "The caller wants something this desk does not do. This desk is for {purpose}. Tell "
    "them that in one short sentence, in your own words, the way a receptionist would "
    "say it to someone who has come to the wrong counter — name the one thing you do "
    "and offer it. Do not list every variation of it, do not repeat the phrasing above "
    "word for word, and do not say you did not understand: you did, and it is not "
    "something you can help with. Do not ask them to repeat themselves."
)

_NO_REPEAT = (
    "Your last reply was: {previous!r}. Do not say that sentence again, or any close "
    "rewording of it. If the caller did not answer it, the words were the problem."
)


_CALLER_REPEATED = (
    "The caller has just repeated themselves almost word for word. That means your "
    "last reply missed what they asked for. Do not answer the same way again: say "
    "briefly what you understood and ask them to confirm it, or ask what it is they "
    "need. Do not carry on with the current step as though nothing happened."
)


def needs_scope_line(detected: IntentResult, no_match_streak: int) -> bool:
    """Whether this turn should be told what the desk is for.

    Only about *this* turn. The streak is a fallback for a caller the agent keeps
    failing to read, and it used to fire on its own — so a caller whose third attempt
    finally classified as a confident booking request was still told what the desk is
    for, because the two before it had missed. Answering the turn in front of you is the
    whole difference between a desk and a recording.
    """
    if detected.name == "out_of_scope":
        return True
    in_scope_now = detected.name not in ("unknown", "out_of_scope") and detected.is_confident
    return no_match_streak >= 2 and not in_scope_now


def _normalise(text: str) -> str:
    """Lowercased words only, so 'apartment book cheyandi.' matches itself typed twice."""
    return " ".join(re.sub(r"[^\w\s]", " ", text or "").lower().split())


def _is_repeat(session: Session, user_text: str) -> bool:
    """Whether this utterance is the caller saying their last one over again."""
    said = _normalise(user_text)
    if len(said.split()) < 2:
        return False  # "yes" twice is agreement, not a repeat
    for turn in reversed(session.history):
        if turn.role is Role.USER:
            return _normalise(turn.text) == said
    return False


_STUCK_ON = (
    "This is the second unusable {slot} in a row, so asking the same way a third time "
    "will not work either. Give them the options as a numbered choice - first, second - "
    "and ask them to say a number. Do not read the list out again in the same words."
)


def _last_agent_line(session: Session) -> str:
    """The previous thing the agent said, if there is one."""
    for turn in reversed(session.history):
        if turn.role is Role.AGENT:
            return turn.text
    return ""


def _reask_note(slot: str, count: int) -> str:
    """Tell the planner it is repeating itself, and how to not sound like it."""
    spoken = slot.replace("_", " ")
    template = _REASK.get(count, _REASK_LAST)
    return template.format(slot=spoken, count=count)


class Brain:
    def __init__(
        self,
        agent: Agent,
        *,
        llm: LLMClient | None = None,
        registry: ToolRegistry | None = None,
        store: SessionStore | None = None,
        subagent: DeepSubagent | None = None,
        deep_reason: bool | None = None,
    ) -> None:
        problems = validate(agent)
        if problems:
            raise ValueError(f"Invalid agent {agent.name!r}: {'; '.join(problems)}")

        self.agent = agent
        self.llm = llm or LLMClient()

        # Deep reasoning is opt-in: explicit argument first, then DEEP_REASON_ENABLED.
        # Passing a subagent counts as opting in, since constructing one is deliberate.
        #
        # Off is a real off. self.subagent stays None, so no HTTP client is built, no
        # deep_reason tool is registered, and no prompt ever mentions it — the turn is
        # exactly as fast as before this tier existed.
        wants_deep = deep_reason if deep_reason is not None else DEEP_REASON_ENABLED
        if subagent is not None:
            self.subagent: DeepSubagent | None = subagent
        elif wants_deep:
            self.subagent = DeepSubagent()
            if not self.subagent.configured:
                log.warning(
                    "Deep reasoning requested but NVIDIA_API_KEY is unset; "
                    "staying on the fast path."
                )
        else:
            self.subagent = None

        self.registry = registry or build_default_registry(self.subagent)
        # Lines the planner does not write — greeting, handoff, identity — in whatever
        # language the call turns out to be in. Written once per language, then read
        # from disk forever after.
        self.lines = LineBook(agent, self.llm)
        log.info("Deep reasoning %s", "enabled" if self.subagent else "disabled")
        self.store = store or InMemorySessionStore()

    async def aclose(self) -> None:
        """Release everything the brain holds open. Call when the call is over.

        The subagent already had this; the LLM pools did not, and an un-closed pool is
        what put an httpcore traceback after every run's summary.
        """
        if self.subagent is not None:
            await self.subagent.aclose()
        await self.llm.aclose()

    # --- session lifecycle -------------------------------------------------

    def start(self, **metadata) -> Session:
        session = Session(
            agent_name=self.agent.name,
            node_id=self.agent.entry_node,
            metadata=metadata,
        )
        self.store.put(session)
        transcripts.call_started(session)
        return session

    def get(self, session_id: str) -> Session | None:
        return self.store.get(session_id)

    # --- the turn ----------------------------------------------------------

    async def handle(self, session_id: str, user_text: str) -> AsyncIterator[Event]:
        session = self.store.get(session_id)
        if session is None:
            yield ErrorEvent(message=f"Unknown session {session_id!r}", recoverable=False)
            return
        if session.ended:
            yield ErrorEvent(message="Session already ended", recoverable=False)
            return

        # Tag every subagent job started during this turn with the session that owns
        # it. Set before the planner runs, since that is what invokes the tool.
        current_session.set(session_id)

        # Before anything reads the text: invisible characters change how the classifier
        # answers, and they arrive from real keyboards.
        user_text = guardrails.normalise_input(user_text)

        stage_ms: dict[str, float] = {}
        started = time.perf_counter()

        # The classifier needs the conversation *before* this utterance. Appending it
        # first puts it in the prompt twice — once under "Recent conversation" and again
        # as "User just said" — and the model reads the repetition as something it has
        # failed to parse. Measured: the same Telugu sentence classifies as
        # book_appointment against clean history and unknown against duplicated history.
        history_before = session
        session = session.with_turn(Role.USER, user_text)

        # 0. Harvest any subagent answer that landed while the caller was talking. It
        # goes into slots so the planner prompt carries it like any other known fact.
        session = self._absorb_subagent(session)

        # 1. Inbound guardrails, before anything expensive runs.
        verdict = guardrails.check_inbound(user_text, medical_domain=self._is_medical)
        if verdict.escalate:
            async for event in self._escalate(session, verdict.reason, verdict.spoken_response):
                yield event
            return

        # 2. Intent + slot extraction on the fast model.
        mark = time.perf_counter()
        detected = await intent_mod.classify(self.llm, self.agent, history_before, user_text)
        stage_ms["intent"] = (time.perf_counter() - mark) * 1000
        yield IntentEvent(intent=detected)

        # Validate before storing. A slot with a closed vocabulary is checked in the
        # turn it is spoken; a rejected value never enters the session, and the planner
        # is told to correct the caller in this same reply rather than three turns
        # later when a tool finally refuses it.
        kept, rejected = slot_mod.validate(
            self.agent.validation_pairs, detected.slots, known=session.slots
        )
        notes = [slot_mod.note_for(r) for r in rejected]

        # Twice on the same slot means the wording is the problem, not the caller.
        # A caller offered two doctors typed "meeraa" and got the same list read back
        # at them, twice - re-reading a list somebody has already failed to pick from
        # is the least useful thing available.
        if any(bad.slot == session.asked_slot for bad in rejected) and session.ask_repeats >= 2:
            notes.append(_STUCK_ON.format(slot=session.asked_slot.replace(chr(95), chr(32))))
        for bad in rejected:
            log.info("Rejected %s=%r: %s", bad.slot, bad.said, bad.reason)
        session = session.with_slots(kept)

        # The classifier is not deterministic, and the demo cannot depend on it being
        # lucky. Anything with a closed vocabulary that it missed is read straight out
        # of the utterance - the word is either in the sheet or it is not.
        session = session.with_slots(
            slot_mod.recover(self.agent.validation_pairs, user_text, session.slots)
        )
        # "Am I talking to a machine?" — answered from the line book, not the planner.
        # The answer is the same every time by design: it is the one line that must not
        # vary with whatever a caller who keeps pushing talks the model into.
        if detected.name == "asks_identity":
            line = await self.lines.line("identity", session.language or self.agent.languages[0])
            session = session.with_turn(Role.AGENT, line)
            self.store.put(session)
            yield SayEvent(text=line)
            stage_ms["turn_total"] = (time.perf_counter() - started) * 1000
            transcripts.turn(
                session, user_text=user_text, agent_text=line, intent=detected, stage_ms=stage_ms
            )
            yield TimingEvent(stage_ms=stage_ms)
            return

        session = self._track_no_match(session, detected)
        session = self._track_off_topic(session, detected)
        session = self._track_abuse(session, detected)
        session = self._settle_language(session, detected, user_text)

        # An abusive caller is answered politely, not transferred. Only a sustained
        # pattern reaches a person, and that call is flagged so it can be reviewed
        # later rather than disappearing into the ordinary escalation queue.
        if session.abuse_streak > MAX_ABUSIVE_TURNS:
            reason = f"{session.abuse_streak} abusive turns"
            session = session.model_copy(
                update={"flagged": True, "flag_reason": reason}
            )
            self.store.put(session)
            log.warning("Session %s flagged: %s", session.session_id, reason)
            async for event in self._escalate(session, reason):
                yield event
            return

        # A caller who wanted an apartment has the wrong number. They were told what
        # this desk does on the previous turn; a second rewording of the same refusal
        # helps nobody, and it is the shape a caller reads as a machine looping. Say so
        # plainly and hang up — deterministically, with no model in the loop, because
        # this is exactly the point at which a model keeps trying.
        if session.off_topic_streak >= MAX_OFF_TOPIC_TURNS:
            line = await self.lines.line("wrong_desk", session.language or self.agent.languages[0])
            session = session.with_turn(Role.AGENT, line)
            self.store.put(session)
            yield SayEvent(text=line, is_final=True)
            async for event in self._end(session, "off topic", outcome="wrong_desk"):
                yield event
            return

        # 3. Flow control — deterministic, no model involved.
        decision = next_node(self.agent, session, detected)
        if decision.force_end:
            async for event in self._end(
                session, decision.reason, farewell=True, outcome="ended_by_caller"
            ):
                yield event
            return
        if decision.force_escalate or session.no_match_streak >= MAX_CONSECUTIVE_NO_MATCH:
            # The flow's reason only describes an escalation the flow decided on. When
            # the streak is what fired, decision.reason is whatever the flow was doing
            # anyway — transcripts recorded "escalated: stay", which tells whoever reads
            # them nothing about why the call left the agent.
            reason = decision.reason if decision.force_escalate else "repeated no-match"
            async for event in self._escalate(session, reason):
                yield event
            return

        previous_node = session.node_id
        session = session.at_node(decision.node_id)
        if decision.changed:
            yield TransitionEvent(
                from_node=previous_node, to_node=decision.node_id, reason=decision.reason
            )

        if len(session.history) >= MAX_TURNS_PER_SESSION:
            async for event in self._escalate(session, "session length limit"):
                yield event
            return

        # 4. Planner, streaming sentences out as they complete.
        node = self.agent.node(session.node_id)

        # Which slot this turn is chasing, and whether we have chased it before. Asking
        # the same question in the same words twice running is what makes a caller give
        # up; the flow already knows it is happening, so the planner is told.
        missing = node.missing_slots(session.slots)
        target = missing[0] if missing else ""
        session = session.asking_for(target)
        # Never alongside a rejection. A correction and a rephrased re-ask are two
        # instructions pulling the same sentence in different directions, and the model
        # resolves that by doing neither properly. The correction owns the turn; the
        # question it displaces is still unanswered next turn and will be asked then.
        if target and session.ask_repeats > 1 and not rejected:
            notes.append(_reask_note(target, session.ask_repeats))

        # The caller is somewhere this agent cannot follow. Saying what the desk is for
        # is the only thing that gets them unstuck; "I did not understand" is true and
        # useless, and repeating it is what makes a caller give up on a machine.
        if needs_scope_line(detected, session.no_match_streak):
            notes.append(_OUT_OF_SCOPE.format(purpose=self.agent.purpose))

        # The caller said the same thing again. A person hearing that assumes they
        # misheard the first time; a model assumes the caller wants the same answer
        # again, and hands it over. This is the one signal that reliably means an
        # interpretation was wrong, and it needs no language to read.
        if _is_repeat(history_before, user_text):
            notes.append(_CALLER_REPEATED)

        # Never say the same sentence twice running. Identical repetition is the single
        # most machine-like thing in a transcript — more than any wording, more than any
        # voice — and the model will do it unprompted, because from its side the prompt
        # has barely changed.
        previous = _last_agent_line(session)
        if previous:
            notes.append(_NO_REPEAT.format(previous=previous))

        mark = time.perf_counter()
        first_sentence_seen = False
        reply = ""
        tools_used: list[dict[str, Any]] = []

        async for frame in planner.run(
            self.llm, self.registry, self.agent, node, session, detected, user_text, notes
        ):
            if frame["type"] == "sentence":
                if not first_sentence_seen:
                    stage_ms["planner_first_sentence"] = (time.perf_counter() - mark) * 1000
                    first_sentence_seen = True
                yield SayEvent(text=frame["text"])
            elif frame["type"] == "tool":
                outcome = frame["outcome"]
                session = self._adopt_canonical_slots(session, outcome)
                tools_used.append(
                    {"name": outcome.name, "ok": outcome.ok, "ms": round(outcome.ms)}
                )
                yield ToolEvent(
                    name=outcome.name, result=outcome.result, ok=outcome.ok, ms=outcome.ms
                )
            elif frame["type"] == "done":
                reply = frame["text"]

        stage_ms["planner_total"] = (time.perf_counter() - mark) * 1000

        # 5. Outbound guardrails on the assembled reply.
        checked = guardrails.check_outbound(
            reply, medical_domain=self._is_medical, refusal_line=self.agent.refusal_line
        )
        if checked.modified:
            log.info("Outbound guardrail fired: %s", checked.reason)
            reply = checked.text
            yield SayEvent(text=reply, is_final=True)

        session = session.with_turn(Role.AGENT, reply)
        session = self._clear_expert_answer(session)
        self.store.put(session)

        stage_ms["turn_total"] = (time.perf_counter() - started) * 1000
        transcripts.turn(
            session,
            user_text=user_text,
            agent_text=reply,
            intent=detected,
            tools=tools_used,
            stage_ms=stage_ms,
        )
        yield TimingEvent(stage_ms=stage_ms)

        if node.terminal:
            # A booking confirmation is not a goodbye. The caller has just been given a
            # reference code and then, on the old path, the line went dead — which reads
            # as the desk hanging up on them the second their business was useful to it.
            # The farewell is said here, deterministically, rather than left to the
            # planner: the two-sentence cap eats a goodbye tacked onto a confirmation.
            async for event in self._end(
                session, "flow reached a terminal step", farewell=True, outcome="completed"
            ):
                yield event

    # --- helpers -----------------------------------------------------------

    @property
    def _is_medical(self) -> bool:
        return any("medical" in topic or "clinical" in topic for topic in self.agent.guardrail_topics)

    # A short utterance is terrible evidence of language: "yes", "ok" and "haan" look
    # like almost anything. Switching the whole call on one of those makes the agent
    # flip languages mid-conversation, which is far worse than answering in the
    # language it started in.
    MIN_CHARS_TO_SWITCH_LANGUAGE = 12
    MIN_CONFIDENCE_TO_SWITCH_LANGUAGE = 0.8

    def _settle_language(
        self, session: Session, detected: IntentResult, user_text: str
    ) -> Session:
        default = self.agent.languages[0]
        if not session.language:
            return session.model_copy(update={"language": default})

        # Once the call has a language, keep it. A single English word from a Hindi
        # caller would otherwise flip the whole conversation, and because the agent
        # then answers in English the caller follows it there and it never flips back.
        # Per-session metadata wins over the global default either way.
        locked = session.metadata.get("language_locked", LOCK_LANGUAGE)
        if locked:
            return session

        switching = (
            detected.language
            and detected.language != session.language
            and detected.language in self.agent.languages
            and len(user_text.strip()) >= self.MIN_CHARS_TO_SWITCH_LANGUAGE
            and detected.confidence >= self.MIN_CONFIDENCE_TO_SWITCH_LANGUAGE
        )
        if switching:
            log.info("Caller switched language: %s -> %s", session.language, detected.language)
            return session.model_copy(update={"language": detected.language})
        return session

    def _track_abuse(self, session: Session, detected: IntentResult) -> Session:
        """Count consecutive abusive turns, resetting on any ordinary one.

        Resetting matters. People swear out of frustration and then carry on normally;
        counting cumulatively would hand the call to a person twenty turns after the
        caller had already apologised.
        """
        streak = session.abuse_streak + 1 if detected.name == "abusive" else 0
        if streak:
            log.info("Abusive turn %d in session %s", streak, session.session_id)
        return session.model_copy(update={"abuse_streak": streak})

    def _track_off_topic(self, session: Session, detected: IntentResult) -> Session:
        """Count consecutive turns about something this agent does not do."""
        off = detected.name == "out_of_scope"
        streak = session.off_topic_streak + 1 if off else 0
        return session.model_copy(update={"off_topic_streak": streak})

    def _track_no_match(self, session: Session, detected: IntentResult) -> Session:
        missed = detected.name == "unknown" or not detected.is_confident
        streak = session.no_match_streak + 1 if missed else 0
        return session.model_copy(update={"no_match_streak": streak})

    # The slot the planner reads a finished subagent answer from. Consumed after one
    # turn — see _clear_expert_answer.
    EXPERT_SLOT = "expert_answer"

    # Used when a job the caller was promised an answer to comes back empty or errored.
    # Silence would be worse: the agent already said it was checking.
    #
    # Phrased as a fact, not as an instruction. Slots are rendered into the prompt under
    # "Known details", so anything written here is presented to the model as an
    # established fact about the call. An instruction placed there ("Apologise briefly
    # and offer a human") is liable to be read out verbatim — the caller hears the
    # agent's own stage directions. A plain fact degrades safely: spoken as-is it is
    # still a true, coherent sentence.
    EXPERT_FAILED = "The information could not be retrieved."

    # Slot values a tool is authoritative about. When a tool normalises one of these,
    # its answer replaces whatever the caller's phrasing put there.
    CANONICAL_SLOTS = ("department",)

    def _adopt_canonical_slots(self, session: Session, outcome: ToolOutcome) -> Session:
        """Replace caller phrasing with the value the backend actually recognised.

        The caller says "synus"; the tool resolves that to "ent" and returns real ENT
        slots. Without writing it back, the session keeps "synus" and every later prompt
        carries it as an established fact — so the agent reads out ENT times under a
        department name the hospital does not have, and a model asked to say something
        sensible about "synus" invents "surgery".

        Only successful results are trusted: an unknown_department error carries the
        caller's rejected wording, not a correction.
        """
        if not outcome.ok:
            return session

        corrections = {
            key: str(outcome.result[key])
            for key in self.CANONICAL_SLOTS
            if outcome.result.get(key) and str(outcome.result[key]) != session.slots.get(key)
        }
        if not corrections:
            return session

        log.info("Tool %s corrected slots: %s", outcome.name, corrections)
        return session.with_slots(corrections)

    def _absorb_subagent(self, session: Session) -> Session:
        """Fold finished subagent results into the session for exactly one turn.

        Stored as a slot rather than history so the planner sees an established fact
        rather than something a previous speaker said.

        A failed job is reported, not dropped. The planner already told the caller it
        was checking, so staying quiet strands them waiting for an answer that will
        never arrive; a short apology and a human is the honest outcome.
        """
        if self.subagent is None:
            return session
        results = self.subagent.collect(session.session_id)
        if not results:
            return session

        answers = [r.text for r in results if r.ok and r.text]
        for result in results:
            log.info(
                "Subagent %s ok=%s %.0f ms session=%s",
                result.id,
                result.ok,
                result.ms,
                session.session_id,
            )

        value = " ".join(answers) if answers else self.EXPERT_FAILED
        return session.with_slots({self.EXPERT_SLOT: value})

    def _clear_expert_answer(self, session: Session) -> Session:
        """Drop the subagent answer once the planner has used it.

        ``with_slots`` merges and skips empty values, so a slot set once would otherwise
        live for the rest of the call and keep being offered to the planner as a current
        fact. Answers are tied to the question that produced them and go stale
        immediately, so this removes the key outright.
        """
        if self.EXPERT_SLOT not in session.slots:
            return session
        remaining = {k: v for k, v in session.slots.items() if k != self.EXPERT_SLOT}
        return session.model_copy(update={"slots": remaining})

    def _release_subagent(self, session: Session) -> None:
        """Abandon this call's outstanding subagent work.

        A call can end while a job is still running. Nothing polls a dead session, so
        the job would sit in the pool forever holding a queue slot; enough abandoned
        calls and deep_reason is permanently busy for every future caller.
        """
        if self.subagent is None:
            return
        abandoned = self.subagent.discard_session(session.session_id)
        if abandoned:
            log.info("Abandoned %d subagent job(s) for ended session %s", abandoned, session.session_id)

    async def _escalate(
        self, session: Session, reason: str, spoken: str = ""
    ) -> AsyncIterator[Event]:
        self._release_subagent(session)
        # In the caller's language. An English sentence at the end of a Hindi call is
        # the most jarring moment in the conversation, and it lands exactly when the
        # caller is already unhappy.
        line = spoken or await self.lines.line(
            "handoff", session.language or self.agent.languages[0]
        )
        yield SayEvent(text=line, is_final=True)
        yield EscalateEvent(reason=reason)
        session = session.with_turn(Role.AGENT, line).model_copy(
            update={"escalated": True, "ended": True}
        )
        self.store.put(session)
        transcripts.call_ended(session, reason=reason, outcome="escalated")

    async def _end(
        self, session: Session, reason: str, *, farewell: bool = False, outcome: str = "ended"
    ) -> AsyncIterator[Event]:
        """Close the call.

        ``farewell`` for every path that has not already said goodbye: a caller who
        says no, and the caller whose booking just completed. Only the wrong-desk close
        leaves it off, because that line says goodbye itself and two in a row is its own
        kind of strange.
        """
        self._release_subagent(session)
        if farewell:
            line = await self.lines.line(
                "farewell", session.language or self.agent.languages[0]
            )
            session = session.with_turn(Role.AGENT, line)
            yield SayEvent(text=line, is_final=True)
        yield EndEvent(reason=reason)
        session = session.model_copy(update={"ended": True})
        self.store.put(session)
        # What kind of ending, in the flow's own terms rather than the hospital's.
        # Whether an appointment was booked is a domain question, and the tool calls and
        # slots in the same file already answer it.
        transcripts.call_ended(session, reason=reason, outcome=outcome)
