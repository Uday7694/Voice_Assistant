"""LLM provider registry.

Groq and NVIDIA NIM both speak the OpenAI wire protocol, so one client serves both and
switching is configuration rather than code.

Measured time-to-first-token, three samples each (from Windows, not Mumbai):

    Groq   gpt-oss-20b                323 /  448 /  527 ms
    Groq   gpt-oss-120b               512 /  536 /  558 ms
    NVIDIA nemotron-3-nano-30b-a3b    631 /  714 /  835 ms
    NVIDIA nemotron-nano-9b-v2       1477 / 1477 / 1477 ms
    NVIDIA nemotron-3.5-lightning    1154 / 4029 / 9412 ms

NVIDIA's free endpoints are shared, and the spread is the problem rather than the median
— a 9-second turn is a dead call. Hence: Groq drives live conversation, NVIDIA is the
free development provider and the failover when Groq rate-limits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api_key_env: str
    planner_model: str
    fast_model: str
    extra_body: dict[str, Any] = field(default_factory=dict)
    """Provider-specific parameters, forwarded as the request's ``extra_body``.

    They must go through extra_body rather than as keyword arguments. The OpenAI SDK
    validates its own signature, so a vendor parameter it does not know — NVIDIA's
    chat_template_kwargs, for instance — raises TypeError before the request is ever
    sent, taking the fallback provider down with it.
    """

    rank: int = 50
    """Fallback order, lowest first. Measured time-to-first-token, not preference: a
    provider is chosen as primary for the quality of what it says, but a *fallback* is
    chosen because the primary is already failing and the caller is already waiting."""

    timeout_scale: float = 1.0
    """Multiplier on every timeout budget.

    Budgets are set for production latency on Groq. A slower development provider needs
    room or it times out on every turn, degrades to intent 'unknown', and escalates the
    call — which is exactly what NVIDIA-only runs did at scale 1.0.
    """

    @property
    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "").strip()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


GROQ = Provider(
    name="groq",
    base_url="https://api.groq.com/openai/v1",
    api_key_env="GROQ_API_KEY",
    planner_model=os.getenv("GROQ_PLANNER_MODEL", "openai/gpt-oss-120b"),
    fast_model=os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b"),
    # gpt-oss deliberates unless told not to; a voice turn cannot wait for it.
    extra_body={"reasoning_effort": os.getenv("GROQ_REASONING_EFFORT", "low")},
    rank=10,
)

NVIDIA = Provider(
    name="nvidia",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key_env="NVIDIA_API_KEY",
    # nemotron-3-nano-30b-a3b is the only NVIDIA model measured fast enough to hold a
    # conversation, and it handles tool calls correctly.
    planner_model=os.getenv("NVIDIA_PLANNER_MODEL", "nvidia/nemotron-3-nano-30b-a3b"),
    # Also the nano-30b: nemotron-nano-9b-v2 measured slower (1477 ms TTFT), and the
    # classifier runs on every single turn.
    fast_model=os.getenv("NVIDIA_FAST_MODEL", "nvidia/nemotron-3-nano-30b-a3b"),
    # Nemotron is a reasoning model and thinking is ON by default, which puts the
    # chain-of-thought into `content` itself — the agent literally said "We need to
    # follow instructions: greet in one short line" out loud. This turns it off.
    extra_body={"chat_template_kwargs": {"thinking": False}},
    rank=20,
    timeout_scale=float(os.getenv("NVIDIA_TIMEOUT_SCALE", "3.0")),
)

SARVAM = Provider(
    name="sarvam",
    # /v2 hosts sarvam-105b and the open-weight models, but it is beta and returns 400
    # on accounts without access. /v1 is the one that serves conversations today.
    base_url="https://api.sarvam.ai/v1",
    api_key_env="SARVAM_API_KEY",
    # sarvam-105b-conversations is built for real-time dialogue rather than general
    # reasoning, and it is trained on the 10 Indian languages this agent speaks. In
    # Telugu it produces flow-appropriate replies where the gpt-oss models produce
    # translated-sounding English. Verified live: streaming and tool calling both work.
    # sarvam-m and sarvam-30b are deprecated; sarvam-105b needs the beta /v2 endpoint.
    planner_model=os.getenv("SARVAM_PLANNER_MODEL", "sarvam-105b-conversations"),
    fast_model=os.getenv("SARVAM_FAST_MODEL", "sarvam-105b-conversations"),
    extra_body={
        # Measured on a warm connection, median of 3: baseline 328 ms to first token,
        # reasoning_effort=low 299 ms, wiki_grounding off 291 ms. Each is worth ~10%,
        # which is small but free.
        #
        # wiki_grounding is off for a second reason: this agent answers from its flow
        # and its tools, never from an encyclopedia, and grounding it against one only
        # widens the surface for inventing facts the hospital never said.
        "reasoning_effort": os.getenv("SARVAM_REASONING_EFFORT", "low"),
        "wiki_grounding": False,
    },
    # Measured 1.1-2.3 s per call against Groq's 0.3-0.6 s. The budgets are set for Groq,
    # so without headroom every turn times out and escalates.
    rank=30,
    timeout_scale=float(os.getenv("SARVAM_TIMEOUT_SCALE", "2.5")),
)

PROVIDERS = {p.name: p for p in (GROQ, NVIDIA, SARVAM)}


def resolve(name: str | None = None) -> Provider:
    """Pick the primary provider: explicit argument, then env, then whatever has a key."""
    wanted = (name or os.getenv("LLM_PROVIDER", "")).strip().lower()
    if wanted:
        if wanted not in PROVIDERS:
            raise ValueError(f"Unknown provider {wanted!r}; choose from {sorted(PROVIDERS)}")
        provider = PROVIDERS[wanted]
        if not provider.configured:
            raise RuntimeError(f"{provider.api_key_env} is not set for provider {wanted!r}")
        return provider

    for provider in (GROQ, NVIDIA, SARVAM):
        if provider.configured:
            return provider

    raise RuntimeError("No LLM provider configured. Set GROQ_API_KEY or NVIDIA_API_KEY.")


def resolve_fast(primary: Provider) -> Provider:
    """Provider for intent classification, which can differ from the planner's.

    Worth splitting because the two roles are not alike. The planner's words are spoken
    to the caller, so language quality decides whether the agent sounds native. Intent
    classification returns a JSON label nobody ever hears, so it should be whatever is
    cheapest and fastest.

    That matters here beyond latency: Sarvam bills LLM calls against the same credit
    balance as speech, so running classification there spends the budget twice per turn
    on the half of it the caller cannot hear.
    """
    wanted = os.getenv("FAST_LLM_PROVIDER", "").strip().lower()
    if not wanted:
        return primary
    if wanted not in PROVIDERS:
        raise ValueError(f"Unknown FAST_LLM_PROVIDER {wanted!r}; choose from {sorted(PROVIDERS)}")
    chosen = PROVIDERS[wanted]
    return chosen if chosen.configured else primary


def fallback_chain() -> tuple[Provider, ...]:
    """Every configured provider, fastest first.

    A chain rather than a single fallback, and ordered by measured speed rather than by
    declaration order. The old version returned the first configured provider that was
    not the *planner's* — which, with a Sarvam planner and a Groq classifier, was Groq
    itself. A classifier timeout on Groq therefore retried on Groq, with the same budget
    and the same overloaded endpoint, and the turn degraded to intent "unknown" twice
    as slowly as it needed to.
    """
    if os.getenv("LLM_FALLBACK", "1").strip() in {"0", "false", "no"}:
        return ()
    return tuple(sorted((p for p in PROVIDERS.values() if p.configured), key=lambda p: p.rank))


def resolve_fallback(primary: Provider) -> Provider | None:
    """The fastest configured provider that is not ``primary``."""
    return next((p for p in fallback_chain() if p.name != primary.name), None)
