"""Runtime configuration. Fails fast when required secrets are missing."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Model selection and provider endpoints live in brain/providers.py.

# Budgets in seconds. The planner budget is what protects the turn latency target.
PLANNER_TIMEOUT = 8.0
FAST_TIMEOUT = 3.0
TOOL_TIMEOUT = 1.5

# The deep-reasoning subagent is opt-in and off by default.
#
# Default off because the two tiers have opposite failure modes. Groq answers in
# hundreds of milliseconds and is what a voice call needs; the subagent answers in
# 6-25 s, which the caller experiences as the agent going quiet and coming back turns
# later. That trade is worth making deliberately, per deployment, not by default.
#
# When off nothing is constructed: no HTTP client, no tool in the registry, no mention
# in any prompt. The fast path is bit-for-bit what it was before the tier existed.
DEEP_REASON_ENABLED = os.getenv("DEEP_REASON_ENABLED", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Once a call has a language, keep it.
#
# The alternative — following whoever is speaking — sounds right and behaves badly. A
# Hindi caller who says one English word ("appointment", "booking", "OK") flips the
# whole call to English, and because the agent is then speaking English the caller
# often answers in English too, so it never flips back. Romanised Hindi makes it worse:
# "mera naam amit hai" is ASCII and reads as English to a classifier.
#
# Set LANGUAGE_FOLLOWS_CALLER=1 for the old behaviour, per deployment.
LOCK_LANGUAGE = os.getenv("LANGUAGE_FOLLOWS_CALLER", "").strip() not in {"1", "true", "yes", "on"}

# Conversation limits enforced in code, never in the prompt.
MAX_TURNS_PER_SESSION = 40
MAX_CONSECUTIVE_NO_MATCH = 2
MAX_HISTORY_TURNS = 12
