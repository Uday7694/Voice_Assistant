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
import time
from typing import AsyncIterator

from . import guardrails, intent as intent_mod, planner
from .config import (
    DEEP_REASON_ENABLED,
    LOCK_LANGUAGE,
    MAX_CONSECUTIVE_NO_MATCH,
    MAX_TURNS_PER_SESSION,
)
from .flow import Agent, next_node, validate
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
from .tools import ToolRegistry, build_default_registry

log = logging.getLogger(__name__)


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
        log.info("Deep reasoning %s", "enabled" if self.subagent else "disabled")
        self.store = store or InMemorySessionStore()

    # --- session lifecycle -------------------------------------------------

    def start(self, **metadata) -> Session:
        session = Session(
            agent_name=self.agent.name,
            node_id=self.agent.entry_node,
            metadata=metadata,
        )
        self.store.put(session)
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

        session = session.with_slots(detected.slots)
        session = self._track_no_match(session, detected)
        session = self._settle_language(session, detected, user_text)

        # 3. Flow control — deterministic, no model involved.
        decision = next_node(self.agent, session, detected)
        if decision.force_end:
            async for event in self._end(session, decision.reason):
                yield event
            return
        if decision.force_escalate or session.no_match_streak >= MAX_CONSECUTIVE_NO_MATCH:
            reason = decision.reason or "repeated no-match"
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
        mark = time.perf_counter()
        first_sentence_seen = False
        reply = ""

        async for frame in planner.run(
            self.llm, self.registry, self.agent, node, session, detected, user_text
        ):
            if frame["type"] == "sentence":
                if not first_sentence_seen:
                    stage_ms["planner_first_sentence"] = (time.perf_counter() - mark) * 1000
                    first_sentence_seen = True
                yield SayEvent(text=frame["text"])
            elif frame["type"] == "tool":
                outcome = frame["outcome"]
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
        yield TimingEvent(stage_ms=stage_ms)

        if node.terminal:
            async for event in self._end(session, "flow reached a terminal step"):
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
        line = spoken or "Let me put you through to a colleague who can help with this."
        yield SayEvent(text=line, is_final=True)
        yield EscalateEvent(reason=reason)
        self.store.put(
            session.with_turn(Role.AGENT, line).model_copy(
                update={"escalated": True, "ended": True}
            )
        )

    async def _end(self, session: Session, reason: str) -> AsyncIterator[Event]:
        self._release_subagent(session)
        yield EndEvent(reason=reason)
        self.store.put(session.model_copy(update={"ended": True}))
