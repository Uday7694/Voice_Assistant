"""Provider-agnostic LLM client (Groq / NVIDIA NIM — both OpenAI-compatible).

Two entry points: a JSON call for classification, and a streaming chat call with tool
support for the planner. Everything is timeout-bounded, and a failed primary provider
falls back to the other configured one rather than killing the turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import suppress
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from .config import (
    FAST_TIMEOUT,
    FAST_TOTAL_TIMEOUT,
    PLANNER_TIMEOUT,
    PLANNER_TOTAL_TIMEOUT,
)
from .providers import Provider, fallback_chain, resolve, resolve_fast

# Most of what is left of a ceiling that one attempt may take while others are untried.
FIRST_ATTEMPT_SHARE = 0.6

# No attempt is worth making with less than this much time; below it the request cannot
# even finish connecting, and the timeout is the only thing it can produce.
MIN_ATTEMPT_SECONDS = 0.5

# How long teardown waits for abandoned streams to finish reading. Generous enough for
# the tail of a capped reply, short enough that a dead socket cannot hold up a hangup.
DRAIN_TIMEOUT = 2.0

log = logging.getLogger(__name__)

Message = dict[str, Any]


class LLMEmpty(RuntimeError):
    """A provider answered, but with nothing usable in it."""


class LLMTimeout(Exception):
    """The model did not answer inside its budget."""


class LLMClient:
    def __init__(
        self,
        provider: Provider | None = None,
        *,
        fallback: Provider | None = None,
        fast_provider: Provider | None = None,
    ) -> None:
        self.provider = provider or resolve()
        # Intent classification can run somewhere else. Its output is a JSON label the
        # caller never hears, so it does not need the planner's language quality — and
        # on a metered provider it should not spend the same budget.
        self.fast = fast_provider or resolve_fast(self.provider)
        # An ordered chain, fastest first, rather than one nominated stand-in. Which
        # provider is the right fallback depends on which one just failed, and that is
        # not known until it does.
        self.fallbacks: tuple[Provider, ...] = (
            (fallback,) if fallback is not None else fallback_chain()
        )
        self._clients: dict[str, AsyncOpenAI] = {}
        self._drains: set[asyncio.Task] = set()
        log.info(
            "LLM planner=%s/%s fast=%s/%s fallback=%s",
            self.provider.name,
            self.provider.planner_model,
            self.fast.name,
            self.fast.fast_model,
            ", ".join(p.name for p in self.fallbacks) or "none",
        )

    async def warm(self) -> None:
        """Open the TLS connection before the caller says anything.

        A cold first call costs about a second more than a warm one — measured 1315 ms
        to first token cold against 328 ms warm. Most of that is DNS, TLS and pool
        setup, and none of it needs to happen while someone is waiting.

        Uses a plain GET rather than a throwaway completion, so it consumes no tokens
        and costs nothing on a metered provider. That recovers roughly 400 ms of the
        gap; the remainder is server-side warm-up that only a real request settles.

        Best effort by design: a failure here is not a reason to refuse the call.
        """
        for provider in {p.name: p for p in (self.provider, self.fast) if p}.values():
            try:
                # Close the response explicitly. The body is never read here, and an
                # unread body leaves httpx's and httpcore's byte-stream generators
                # suspended on this connection — finalised later by the garbage
                # collector, off the loop, which is where the "generator didn't stop
                # after athrow()" traceback came from. One warm-up per provider per
                # process was enough to produce it on every run.
                response = await self._client(provider)._client.get(f"{provider.base_url}/models")
                await response.aclose()
            except Exception as exc:  # noqa: BLE001 - warming is an optimisation
                log.debug("Warm-up failed for %s: %s", provider.name, str(exc)[:120])

    def _drain_later(self, stream) -> None:
        """Finish an abandoned stream off the critical path.

        The task is held in a set because asyncio keeps only a weak reference to a
        running task; without one the drain can be collected halfway through, which is
        the leak it exists to prevent.
        """
        task = asyncio.create_task(_drain(stream))
        self._drains.add(task)
        task.add_done_callback(self._drains.discard)

    async def aclose(self) -> None:
        """Release every provider connection. Call when the conversation ends.

        Waits briefly for outstanding drains first: closing the clients underneath them
        puts the teardown back where it started. Bounded, because a hung socket must
        not hold up the end of a call.
        """
        if self._drains:
            await asyncio.wait(set(self._drains), timeout=DRAIN_TIMEOUT)
        for client in self._clients.values():
            with suppress(Exception):
                await client.close()
        self._clients.clear()

    @property
    def fallback(self) -> Provider | None:
        """The first stand-in for the planner's provider. For logging and tests."""
        return next((p for p in self.fallbacks if p.name != self.provider.name), None)

    def _chain(self, primary: Provider) -> list[Provider]:
        """Who to try, in order, when ``primary`` is the one that should answer.

        The provider that just failed never appears twice. That was the whole bug: with
        a Sarvam planner and a Groq classifier, the classifier's fallback was Groq, so a
        Groq timeout was retried on Groq.
        """
        return [primary] + [p for p in self.fallbacks if p.name != primary.name]

    def _client(self, provider: Provider) -> AsyncOpenAI:
        if provider.name not in self._clients:
            self._clients[provider.name] = AsyncOpenAI(
                base_url=provider.base_url, api_key=provider.api_key
            )
        return self._clients[provider.name]

    @property
    def planner_model(self) -> str:
        return self.provider.planner_model

    @property
    def fast_model(self) -> str:
        return self.fast.fast_model

    # --- classification ----------------------------------------------------

    async def json_call(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        timeout: float = FAST_TIMEOUT,
        ceiling: float = FAST_TOTAL_TIMEOUT,
        accept=None,
    ) -> dict[str, Any]:
        """Call a model that must reply with a single JSON object.

        ``timeout`` and ``ceiling`` are separate arguments because not every JSON call
        is on the critical path. Classification is, and fails fast into "unknown";
        writing a language's fixed lines is not, and would rather take ten seconds once
        than cache the English fallback forever.
        """
        raw = await self._with_fallback(
            lambda provider, budget: self._json_once(provider, messages, temperature, budget),
            what="json_call",
            primary=self.fast,
            timeout=timeout,
            ceiling=ceiling,
            accept=(lambda raw: accept(_parse_json(raw))) if accept else None,
        )
        return _parse_json(raw)

    async def _json_once(
        self, provider: Provider, messages: list[Message], temperature: float, timeout: float
    ) -> str:
        response = await asyncio.wait_for(
            self._client(provider).chat.completions.create(
                model=provider.fast_model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
                max_tokens=512,
                extra_body=provider.extra_body,
            ),
            # Already scaled and clamped by _with_fallback, which is the only caller
            # that knows how much of the ceiling is left.
            timeout=timeout,
        )
        return response.choices[0].message.content or "{}"

    # --- planning ----------------------------------------------------------

    async def stream_chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 400,
        timeout: float = PLANNER_TIMEOUT,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream planner output.

        Yields ``{"type": "text", "text": ...}`` deltas and, once the stream closes, a
        single ``{"type": "tool_calls", "calls": [...]}`` frame if tools were invoked.

        Fallback applies only to opening the stream. Once tokens are flowing they may
        already have been spoken, so a mid-stream failure is handled by the planner's
        recovery line instead — restarting on another provider would repeat audio the
        caller has already heard.
        """
        stream = None
        last_error: Exception | None = None
        started = time.monotonic()

        chain = self._chain(self.provider)
        for index, provider in enumerate(chain):
            remaining = PLANNER_TOTAL_TIMEOUT - (time.monotonic() - started)
            if remaining <= 0:
                log.warning("Planner gave up: %.1fs ceiling spent", PLANNER_TOTAL_TIMEOUT)
                break

            budget = _attempt_budget(timeout, provider, remaining, index == len(chain) - 1)
            if index:
                log.info("Planner falling back to %s (%.1fs left)", provider.name, remaining)
            try:
                stream = await asyncio.wait_for(
                    self._client(provider).chat.completions.create(
                        model=provider.planner_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                        **({"tools": tools, "tool_choice": "auto"} if tools else {}),
                        extra_body=provider.extra_body,
                    ),
                    timeout=budget,
                )
                break
            except asyncio.TimeoutError:
                last_error = LLMTimeout(f"stream_chat exceeded {budget:.1f}s on {provider.name}")
                log.warning("Planner timed out on %s after %.1fs", provider.name, budget)
            except Exception as exc:  # noqa: BLE001 - try the next provider
                last_error = exc
                log.warning("Planner failed on %s: %s", provider.name, str(exc)[:160])

        if stream is None:
            raise last_error or LLMTimeout("no provider produced a stream")

        pending: dict[int, dict[str, Any]] = {}
        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    yield {"type": "text", "text": delta.content}
                for call in getattr(delta, "tool_calls", None) or []:
                    _accumulate_tool_call(pending, call)

            if pending:
                yield {"type": "tool_calls", "calls": _finalise_tool_calls(pending)}
        finally:
            # The planner abandons this stream routinely — it stops reading the moment
            # it has two sentences, which is the whole point of the sentence cap.
            #
            # Abandoning it cannot be cleaned up here. Awaiting anything in the finally
            # of an async generator that is being closed risks "async generator ignored
            # GeneratorExit", and closing the SDK's iterator does not close the chain
            # underneath it: `AsyncStream.__stream__` closes the HTTP response but
            # leaves its own `_iter_events` generator, and httpx's and httpcore's below
            # that, suspended. The garbage collector then finalises those layers in
            # whatever order it likes, off the loop — which is exactly the
            # "generator didn't stop after athrow()" traceback that appeared after
            # every call.
            #
            # So hand it to a background task instead: read the remainder and let every
            # generator in the chain finish normally, which is the one teardown path
            # the SDK actually supports. It costs nothing extra — the server has
            # already generated those tokens, and max_tokens caps how many there can be.
            self._drain_later(stream)

    # --- shared ------------------------------------------------------------

    async def _with_fallback(
        self,
        call,
        *,
        what: str,
        primary: Provider | None = None,
        timeout: float = FAST_TIMEOUT,
        ceiling: float = FAST_TOTAL_TIMEOUT,
        accept=None,
    ):
        """Try each provider in turn until one answers usefully, or time runs out.

        ``timeout`` is the per-attempt budget before the provider's own scale is applied;
        ``ceiling`` bounds the whole sequence. Without the ceiling a single bad turn
        could spend its first budget, then a scaled retry on a slower provider, and only
        then give up — long past the point where the caller has started talking again.

        ``accept`` decides whether a *successful* response is worth having. A provider
        that answers with nothing is not a provider that answered: NVIDIA returns a
        well-formed {"lines": []} for a writing task it cannot do, and without this
        that empty success ends the sequence — the language quietly gets the English
        fallback while a provider that could have written the line goes untried.
        """
        started = time.monotonic()
        last_error: Exception = LLMTimeout(f"{what}: no provider was tried")

        chain = self._chain(primary or self.provider)
        for index, provider in enumerate(chain):
            remaining = ceiling - (time.monotonic() - started)
            if remaining <= 0:
                log.warning("%s gave up: %.1fs ceiling spent", what, ceiling)
                break

            budget = _attempt_budget(timeout, provider, remaining, index == len(chain) - 1)
            if index:
                log.info("%s falling back to %s (%.1fs left)", what, provider.name, remaining)
            try:
                # Enforced here as well as inside the call. The budget is already
                # clamped to what is left of the ceiling, so bounding the attempt
                # bounds the sequence — and a call that ever forgets to apply the
                # budget it was handed cannot run past it.
                result = await asyncio.wait_for(call(provider, budget), timeout=budget)
                if accept is not None and not accept(result):
                    log.warning("%s answered with nothing on %s", what, provider.name)
                    last_error = LLMEmpty(f"{what} answered with nothing on {provider.name}")
                    continue
                return result
            except asyncio.TimeoutError:
                log.warning("%s timed out on %s after %.1fs", what, provider.name, budget)
                last_error = LLMTimeout(f"{what} timed out on {provider.name}")
            except Exception as exc:  # noqa: BLE001 - try the next provider
                log.warning("%s failed on %s: %s", what, provider.name, str(exc)[:160])
                last_error = exc

        raise last_error


def _attempt_budget(
    timeout: float, provider: Provider, remaining: float, is_last: bool
) -> float:
    """How long this attempt may take.

    The provider's own scale, clamped to what is left of the ceiling — and, while other
    providers are still untried, to a share of it. Without that share the first attempt
    can eat the whole ceiling: a Sarvam planner on a 2.5x scale would spend every second
    of the budget before the fallback it exists for ever got a request.
    """
    room = remaining if is_last else remaining * FIRST_ATTEMPT_SHARE
    return max(min(timeout * provider.timeout_scale, room), MIN_ATTEMPT_SECONDS)


def _parse_json(raw: str | None) -> dict[str, Any]:
    """The object a JSON call was supposed to return, or an empty one."""
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        log.warning("Model returned non-JSON payload: %r", (raw or "")[:200])
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def _drain(stream) -> None:
    """Read what is left of an abandoned stream so it can close itself."""
    with suppress(Exception):
        async for _ in stream:
            pass
    with suppress(Exception):
        await stream.close()


def _accumulate_tool_call(pending: dict[int, dict[str, Any]], call: Any) -> None:
    """Tool calls arrive in fragments across chunks; stitch them by index."""
    slot = pending.setdefault(call.index, {"name": "", "arguments": ""})
    if call.function and call.function.name:
        slot["name"] = call.function.name
    if call.function and call.function.arguments:
        slot["arguments"] += call.function.arguments


def _finalise_tool_calls(pending: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for _, slot in sorted(pending.items()):
        if not slot["name"]:
            continue
        try:
            arguments = json.loads(slot["arguments"] or "{}")
        except json.JSONDecodeError:
            log.warning("Bad tool arguments for %s: %r", slot["name"], slot["arguments"])
            arguments = {}
        calls.append({"name": slot["name"], "arguments": arguments})
    return calls


# Backwards-compatible alias: the brain was written against GroqClient.
GroqClient = LLMClient
