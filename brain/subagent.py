"""Deep-reasoning subagent on NVIDIA NIM: the slow tier behind the fast voice loop.

Groq answers the caller. This answers the questions Groq should not try to.

The split is latency, not capability. A voice turn has roughly a 1.5 s tool budget
(``TOOL_TIMEOUT``); these are frontier-scale MoE models on NVIDIA's shared free
endpoints. Measured end-to-end on a two-sentence brief:

    moonshotai/kimi-k3                  17641 ms
    minimaxai/minimax-m3                 5903 ms
    deepseek-ai/deepseek-v4-flash-0731   timed out at 45 s
    moonshotai/kimi-k2.6                 404 — not served

So it never runs inside a turn. ``start()`` spawns the job and returns immediately, the
planner says a holding line, and ``collect()`` picks the answer up on a later turn —
the caller hears "let me check that" instead of 17 seconds of silence.

This was written for GLM-5.2, which NVIDIA retired on 2026-08-21; the endpoint now
returns 410 Gone and NIM serves no GLM model at all. MiniMax M3 is the default in its
place — measurably the only reliable one of the four. Kimi K3 is the fallback despite
answering once in 17.6 s and then timing out twice at 40 s: an unreliable second
opinion still beats none when the primary is rate-limited. Every candidate speaks the
OpenAI wire protocol, so changing your mind is a change to ``MODEL`` and nothing else.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import time
from dataclasses import dataclass, field
from openai import OpenAI

log = logging.getLogger(__name__)

BASE_URL = "https://integrate.api.nvidia.com/v1"
API_KEY_ENV = "NVIDIA_API_KEY"

MODEL = os.getenv("SUBAGENT_MODEL", "minimaxai/minimax-m3")

# Tried when the primary is retired (410) or unserved (404) — the exact failure that
# killed GLM-5.2 here, with no warning in any response before the day it happened.
FALLBACK_MODEL = os.getenv("SUBAGENT_FALLBACK_MODEL", "moonshotai/kimi-k3")

# NIM free tiers cap context well below what discovery advertises and expose the real
# limit nowhere, so clamp rather than discover it as a 400 mid-call.
MAX_CONTEXT_CHARS = int(os.getenv("SUBAGENT_MAX_CONTEXT_CHARS", "400000"))

# Wall-clock ceiling for one background job. Generous because nothing is waiting on it
# in real time, but bounded so a hung request cannot leak a task for the whole call.
JOB_TIMEOUT = float(os.getenv("SUBAGENT_JOB_TIMEOUT", "90"))

MAX_TOKENS = int(os.getenv("SUBAGENT_MAX_TOKENS", "2048"))

# Concurrent jobs per process. The free tier is a shared pool; more parallelism buys
# nothing and burns quota faster.
MAX_CONCURRENT_JOBS = int(os.getenv("SUBAGENT_MAX_CONCURRENT_JOBS", "1"))

# Per-attempt ceiling. Separate from JOB_TIMEOUT so one wedged model cannot consume the
# whole budget and leave nothing for the fallback — measured behaviour, not theory.
ATTEMPT_TIMEOUT = float(os.getenv("SUBAGENT_ATTEMPT_TIMEOUT", "40"))

# The free tier 429s after roughly two calls in quick succession. Nothing is waiting on
# a background job, so backing off and retrying costs nothing a caller can perceive.
RETRY_BACKOFF = (4.0, 12.0)

# Hard ceiling on queued work. The planner cannot see that a job is already running, so
# left alone it re-asks on every turn while it waits — 12 identical jobs in a 12-turn
# call, serialised one at a time, each a real request against a tier that 429s after
# two. The cap is the backstop; deduplication below is what actually prevents it.
MAX_QUEUED_JOBS = int(os.getenv("SUBAGENT_MAX_QUEUED_JOBS", "4"))

_SYSTEM = """You are a research and reasoning subagent supporting a live voice call.

You are not talking to the caller. You are reporting to another model that is.

- Answer the brief directly. No preamble, no restatement of the question.
- Be complete but compact: the agent has to say this out loud in a sentence or two.
- Plain prose. No markdown, no bullets, no headings.
- State facts you are confident in. If you do not know, say so in one line.
- Never invent availability, prices, medical facts, or policy."""


# The session a job belongs to. A ContextVar rather than a parameter because tool
# handlers are invoked with model-supplied arguments only — there is nowhere to thread
# a session id through — while the orchestrator sets this once per turn and asyncio
# propagates it into the tool call and the job it starts.
current_session: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_session", default=""
)


class SubagentBusy(RuntimeError):
    """Too much queued work already. Not an error condition — backpressure."""


def _dedupe_key(brief: str, context: str) -> str:
    return f"{brief.strip().casefold()}|{context.strip().casefold()}"


@dataclass
class Job:
    """One in-flight or finished subagent request."""

    id: str
    brief: str
    key: str
    session_id: str
    task: asyncio.Task
    started: float = field(default_factory=time.perf_counter)
    finished: float | None = None
    """Set the moment the task completes.

    Without it ``ms`` measures start-to-collection, and collection happens on whatever
    later turn the caller happens to speak on — inflating every number by seconds of
    unrelated conversation. That is the one metric used to judge whether a subagent is
    too slow, so it has to mean what it says.
    """

    @property
    def done(self) -> bool:
        return self.task.done()

    @property
    def ms(self) -> float:
        end = self.finished if self.finished is not None else time.perf_counter()
        return (end - self.started) * 1000


@dataclass(frozen=True)
class Result:
    id: str
    brief: str
    ok: bool
    text: str = ""
    error: str = ""
    ms: float = 0.0


class DeepSubagent:
    """Runs deep-reasoning briefs on a frontier model, off the conversational hot path."""

    def __init__(
        self,
        *,
        model: str = MODEL,
        fallback_model: str | None = FALLBACK_MODEL,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.fallback_model = fallback_model
        self._api_key = (api_key if api_key is not None else os.getenv(API_KEY_ENV, "")).strip()
        self._client: OpenAI | None = None
        self._jobs: dict[str, Job] = {}
        self._counter = 0
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
        # Build the HTTP client now, at startup, not on the first job. asyncio.create_task
        # runs the new coroutine synchronously until its first real suspension, so a lazy
        # client puts ~100 ms of httpx setup inside the caller's turn — which is exactly
        # the blocking that this whole background design exists to avoid.
        if self.configured:
            self._client_or_raise()

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def _client_or_raise(self) -> OpenAI:
        if not self.configured:
            raise RuntimeError(f"{API_KEY_ENV} is not set; subagent unavailable")
        if self._client is None:
            self._client = OpenAI(base_url=BASE_URL, api_key=self._api_key)
        return self._client

    # --- blocking call ------------------------------------------------------

    async def run(self, brief: str, *, context: str = "", timeout: float = JOB_TIMEOUT) -> str:
        """Await a full answer. For offline/batch use — never call this inside a turn."""
        prompt = brief if not context else f"{brief}\n\nContext:\n{context}"
        if len(prompt) > MAX_CONTEXT_CHARS:
            log.warning("Subagent brief truncated from %d chars", len(prompt))
            prompt = prompt[:MAX_CONTEXT_CHARS]

        models = [self.model] + ([self.fallback_model] if self.fallback_model else [])
        last: Exception | None = None

        attempt_timeout = min(ATTEMPT_TIMEOUT, timeout)
        deadline = time.perf_counter() + timeout

        for model in models:
            for pause in (0.0,) + RETRY_BACKOFF:
                if pause:
                    if time.perf_counter() + pause >= deadline:
                        break
                    await asyncio.sleep(pause)
                try:
                    return await asyncio.to_thread(
                        self._blocking_call, model, prompt, attempt_timeout
                    )
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    log.warning("Subagent model %s failed: %s", model, str(exc)[:160])
                    # 404/410 mean the model is retired or unserved; retrying that is
                    # pure quota burn. Anything else may well succeed on a second try.
                    if _is_permanent(exc):
                        break
                if time.perf_counter() >= deadline:
                    break

        raise last or RuntimeError("no subagent model available")

    def _blocking_call(self, model: str, prompt: str, timeout: float) -> str:
        """The actual HTTP call, run in a worker thread.

        Deliberately the synchronous client. An async client shares the conversation's
        event loop, and the sync portion of issuing a request — SSL context setup,
        connection establishment — is hundreds of milliseconds that block whatever the
        loop touches next, which in a voice turn is the planner stream the caller is
        listening to. A thread keeps all of it off the hot path.
        """
        response = self._client_or_raise().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=MAX_TOKENS,
            timeout=timeout,
        )
        return (response.choices[0].message.content or "").strip()

    # --- background jobs ----------------------------------------------------

    def start(self, brief: str, *, context: str = "") -> str:
        """Spawn a job and return its id immediately. Never blocks.

        Repeating a question that is already in flight returns the running job rather
        than starting a second one. A planner waiting on an answer re-asks every turn
        because nothing in its prompt says a job exists, and each duplicate costs real
        quota for an answer already on its way.

        Raises ``SubagentBusy`` once the queue is full, so the caller can say something
        honest instead of silently deepening a backlog.
        """
        session_id = current_session.get()
        key = _dedupe_key(brief, context)
        for existing in self._jobs.values():
            if existing.key == key and existing.session_id == session_id and not existing.done:
                log.info("Subagent reusing in-flight job %s for repeated brief", existing.id)
                return existing.id

        if len(self._jobs) >= MAX_QUEUED_JOBS:
            raise SubagentBusy(f"{len(self._jobs)} jobs already queued")

        self._counter += 1
        job_id = f"job{self._counter:03d}"

        async def _guarded() -> str:
            async with self._semaphore:
                return await self.run(brief, context=context)

        job = Job(
            id=job_id,
            brief=brief,
            key=key,
            session_id=session_id,
            task=asyncio.create_task(_guarded()),
        )
        job.task.add_done_callback(lambda _, j=job: setattr(j, "finished", time.perf_counter()))
        self._jobs[job_id] = job
        log.info("Subagent job %s started: %.60s", job_id, brief)
        return job_id

    @property
    def pending(self) -> int:
        return sum(1 for j in self._jobs.values() if not j.done)

    def collect(self, session_id: str | None = None) -> list[Result]:
        """Drain finished jobs belonging to one session. Safe to call on every turn.

        The session filter is not optional in practice. One Brain serves every
        concurrent caller from a single job pool, so an unfiltered drain hands whoever
        speaks next whatever finished most recently — one caller hearing the answer to
        another caller's question. Passing ``None`` drains everything and exists only
        for tests and shutdown.
        """
        wanted = [
            job_id
            for job_id, job in self._jobs.items()
            if job.done and (session_id is None or job.session_id == session_id)
        ]
        results: list[Result] = []
        for job_id in wanted:
            results.append(_harvest(self._jobs.pop(job_id)))
        return results

    def discard_session(self, session_id: str) -> int:
        """Drop every job for a finished call. Returns how many were abandoned.

        Without this, a job still in flight when the caller hangs up is never collected
        — nothing polls a dead session — so it occupies a MAX_QUEUED_JOBS slot forever.
        After enough abandoned calls the queue is permanently full and deep_reason stops
        working for everyone.
        """
        doomed = [j for j, job in self._jobs.items() if job.session_id == session_id]
        for job_id in doomed:
            job = self._jobs.pop(job_id)
            if not job.done:
                job.task.cancel()
        return len(doomed)

    async def drain(self, timeout: float = JOB_TIMEOUT) -> list[Result]:
        """Wait for all outstanding jobs, then collect. For tests and batch runs."""
        tasks = [j.task for j in self._jobs.values() if not j.done]
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)
        return self.collect()

    async def aclose(self) -> None:
        """Abandon outstanding work. Call when the conversation ends.

        Returns immediately and never blocks call teardown, but be precise about what
        it does: it cancels the *awaiting* tasks, not the worker threads. Python cannot
        kill a thread, so an in-flight HTTP request runs to completion and its result is
        discarded. That is bounded by ATTEMPT_TIMEOUT rather than unbounded, so a thread
        lingers for at most one attempt after the caller hangs up — acceptable, but it
        does mean quota is spent on answers nobody will hear.
        """
        for job in self._jobs.values():
            if not job.done:
                job.task.cancel()
        tasks = [j.task for j in self._jobs.values()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._jobs.clear()


def _is_permanent(exc: Exception) -> bool:
    """True when retrying this model cannot possibly help."""
    return getattr(exc, "status_code", None) in {400, 401, 403, 404, 410}


def _harvest(job: Job) -> Result:
    """Turn a finished task into a Result. A failed subagent must never raise upward."""
    if job.task.cancelled():
        return Result(id=job.id, brief=job.brief, ok=False, error="cancelled", ms=job.ms)

    error = job.task.exception()
    if error is not None:
        detail = "timed out" if isinstance(error, asyncio.TimeoutError) else str(error)[:200]
        log.warning("Subagent job %s failed: %s", job.id, detail)
        return Result(id=job.id, brief=job.brief, ok=False, error=detail, ms=job.ms)

    text = (job.task.result() or "").strip()
    log.info("Subagent job %s finished in %.0f ms", job.id, job.ms)
    return Result(id=job.id, brief=job.brief, ok=bool(text), text=text, ms=job.ms)
