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
    """Provider-specific parameters. Sending Groq's reasoning_effort to NVIDIA errors."""

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
    timeout_scale=float(os.getenv("NVIDIA_TIMEOUT_SCALE", "3.0")),
)

PROVIDERS = {p.name: p for p in (GROQ, NVIDIA)}


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

    for provider in (GROQ, NVIDIA):
        if provider.configured:
            return provider

    raise RuntimeError("No LLM provider configured. Set GROQ_API_KEY or NVIDIA_API_KEY.")


def resolve_fallback(primary: Provider) -> Provider | None:
    """The other configured provider, used when the primary fails mid-conversation."""
    if os.getenv("LLM_FALLBACK", "1").strip() in {"0", "false", "no"}:
        return None
    for provider in PROVIDERS.values():
        if provider.name != primary.name and provider.configured:
            return provider
    return None
