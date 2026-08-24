"""Provider-agnostic LLM client (Groq / NVIDIA NIM — both OpenAI-compatible).

Two entry points: a JSON call for classification, and a streaming chat call with tool
support for the planner. Everything is timeout-bounded, and a failed primary provider
falls back to the other configured one rather than killing the turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from .config import FAST_TIMEOUT, PLANNER_TIMEOUT
from .providers import Provider, resolve, resolve_fallback, resolve_fast

log = logging.getLogger(__name__)

Message = dict[str, Any]


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
        self.fallback = fallback if fallback is not None else resolve_fallback(self.provider)
        self._clients: dict[str, AsyncOpenAI] = {}
        log.info(
            "LLM planner=%s/%s fast=%s/%s fallback=%s",
            self.provider.name,
            self.provider.planner_model,
            self.fast.name,
            self.fast.fast_model,
            self.fallback.name if self.fallback else "none",
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
                await self._client(provider)._client.get(f"{provider.base_url}/models")
            except Exception as exc:  # noqa: BLE001 - warming is an optimisation
                log.debug("Warm-up failed for %s: %s", provider.name, str(exc)[:120])

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
    ) -> dict[str, Any]:
        """Call a model that must reply with a single JSON object."""
        raw = await self._with_fallback(
            lambda provider: self._json_once(provider, messages, temperature, timeout),
            what="json_call",
            primary=self.fast,
        )
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError:
            log.warning("Model returned non-JSON payload: %r", (raw or "")[:200])
            return {}
        return parsed if isinstance(parsed, dict) else {}

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
            timeout=timeout * provider.timeout_scale,
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
        providers = [self.provider] + ([self.fallback] if self.fallback else [])
        stream = None
        last_error: Exception | None = None

        for provider in providers:
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
                    timeout=timeout * provider.timeout_scale,
                )
                break
            except asyncio.TimeoutError:
                last_error = LLMTimeout(f"stream_chat exceeded {timeout}s on {provider.name}")
                log.warning("Planner timed out on %s", provider.name)
            except Exception as exc:  # noqa: BLE001 - try the other provider
                last_error = exc
                log.warning("Planner failed on %s: %s", provider.name, str(exc)[:160])

        if stream is None:
            raise last_error or LLMTimeout("no provider produced a stream")

        pending: dict[int, dict[str, Any]] = {}
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

    # --- shared ------------------------------------------------------------

    async def _with_fallback(self, call, *, what: str, primary: Provider | None = None):
        """Run ``call`` on a provider, retrying once on the fallback.

        ``primary`` defaults to the planner's provider; intent classification passes its
        own, which may be a different one.
        """
        primary = primary or self.provider
        try:
            return await call(primary)
        except asyncio.TimeoutError:
            log.warning("%s timed out on %s", what, primary.name)
            primary_error: Exception = LLMTimeout(f"{what} timed out on {primary.name}")
        except Exception as exc:  # noqa: BLE001
            log.warning("%s failed on %s: %s", what, primary.name, str(exc)[:160])
            primary_error = exc

        if self.fallback is None:
            raise primary_error

        log.info("%s falling back to %s", what, self.fallback.name)
        try:
            return await call(self.fallback)
        except asyncio.TimeoutError as exc:
            raise LLMTimeout(f"{what} timed out on {self.fallback.name}") from exc


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
