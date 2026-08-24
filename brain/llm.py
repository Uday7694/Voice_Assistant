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
from .providers import Provider, resolve, resolve_fallback

log = logging.getLogger(__name__)

Message = dict[str, Any]


class LLMTimeout(Exception):
    """The model did not answer inside its budget."""


class LLMClient:
    def __init__(self, provider: Provider | None = None, *, fallback: Provider | None = None) -> None:
        self.provider = provider or resolve()
        self.fallback = fallback if fallback is not None else resolve_fallback(self.provider)
        self._clients: dict[str, AsyncOpenAI] = {}
        log.info(
            "LLM provider=%s planner=%s fast=%s fallback=%s",
            self.provider.name,
            self.provider.planner_model,
            self.provider.fast_model,
            self.fallback.name if self.fallback else "none",
        )

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
        return self.provider.fast_model

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
                **provider.extra_body,
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
                        **provider.extra_body,
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

    async def _with_fallback(self, call, *, what: str):
        """Run ``call`` on the primary provider, retrying once on the fallback."""
        try:
            return await call(self.provider)
        except asyncio.TimeoutError:
            log.warning("%s timed out on %s", what, self.provider.name)
            primary_error: Exception = LLMTimeout(f"{what} timed out on {self.provider.name}")
        except Exception as exc:  # noqa: BLE001
            log.warning("%s failed on %s: %s", what, self.provider.name, str(exc)[:160])
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
