"""Runtime configuration. Fails fast when required secrets are missing."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Model selection and provider endpoints live in brain/providers.py.

# Budgets in seconds. The planner budget is what protects the turn latency target.
# This budget covers opening the stream — time to first token, not the whole reply.
# Eight seconds was a whole-response budget applied to a first byte, and multiplied by
# Sarvam's 2.5x scale it came to twenty, which is not a phone call any more.
PLANNER_TIMEOUT = float(os.getenv("PLANNER_TIMEOUT", "5.0"))
# Four seconds. Two was set against measured English latency of 0.3-0.9 s on Groq and is
# right for that case and wrong for two others. Telugu and Devanagari cost several times
# the tokens of the same sentence in Latin script, so the same turn takes longer; and a
# provider at its rate limit answers a 429 that the SDK retries internally, which alone
# can exceed two seconds. Both were fires on a provider that was about to answer
# correctly — the budget expired, the turn degraded to "unknown", and a caller who had
# just given his name was asked for it again.
#
# Headroom on a genuinely failing provider is dead air, so this is not free; it is bought
# back by FAST_TOTAL_TIMEOUT, which still caps the whole turn including fallbacks.
FAST_TIMEOUT = float(os.getenv("FAST_TIMEOUT", "4.0"))

# Ceilings across *all* attempts, retries included.
#
# The per-attempt budget alone does not bound a turn: a classifier timeout used to be
# followed by a scaled retry on a slower provider, so one bad turn could spend ten
# seconds before giving up and answering "unknown" anyway. The caller hears that as the
# line going dead.
#
# Classification degrades gracefully — an unclassified turn becomes "unknown" and the
# agent asks a clarifying question — so its ceiling would be tight if the classifier
# were the fastest thing available. It is not: running classification on Sarvam costs
# 1.3-2.7 s measured, against Groq's 0.3-0.9 s, and the ceiling has to cover the slow
# case or every hesitation from the provider becomes a misunderstood caller. Seven
# seconds gives the first attempt roughly 4 s and still leaves room for two fallbacks.
#
# The planner's words are the reply itself, so it is given room to finish somewhere
# rather than fail fast.
FAST_TOTAL_TIMEOUT = float(os.getenv("FAST_TOTAL_TIMEOUT", "7.0"))
PLANNER_TOTAL_TIMEOUT = float(os.getenv("PLANNER_TOTAL_TIMEOUT", "12.0"))

LINE_TIMEOUT = float(os.getenv("LINE_TIMEOUT", "8.0"))
LINE_TOTAL_TIMEOUT = float(os.getenv("LINE_TOTAL_TIMEOUT", "25.0"))
PLANNER_TOTAL_TIMEOUT = float(os.getenv("PLANNER_TOTAL_TIMEOUT", "12.0"))
TOOL_TIMEOUT = float(os.getenv("TOOL_TIMEOUT", "1.5"))

# How much the planner is allowed to vary. Low, because this agent is not writing prose:
# it is asking for a name, reading back a time, and confirming a booking, and the caller
# does not benefit from a fresh turn of phrase each time. At 0.3 the same six-turn call
# went a different way on every run - asking for a date before a name, skipping the
# doctor - which is not creativity, it is an unreliable demo. Not zero: identical
# phrasing on a re-ask is its own tell, and the no-repeat note needs somewhere to move.
PLANNER_TEMPERATURE = float(os.getenv("PLANNER_TEMPERATURE", "0.1"))

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

# Conversation limits, enforced in code and never in the prompt. Every one is
# overridable per deployment: a hospital desk and an outbound campaign do not want the
# same patience, and changing that should not mean editing source.
MAX_TURNS_PER_SESSION = int(os.getenv("MAX_TURNS_PER_SESSION", "40"))
# Three, not two. On the second no-match the planner is now told to say what this desk
# is actually for, which is the thing most likely to unstick the caller; escalating on
# that same turn would hand them to a person before they ever heard it.
MAX_CONSECUTIVE_NO_MATCH = int(os.getenv("MAX_CONSECUTIVE_NO_MATCH", "3"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "12"))

# Abusive turns answered politely before the call is flagged and handed to a person.
#
# Three, not one. People swear out of frustration and usually carry on normally once
# asked to stop, and hanging up on a first outburst is worse service than absorbing it.
# The count resets after any ordinary turn, so this measures a sustained pattern rather
# than a bad moment. Every one of those turns still gets a polite answer — the agent's
# tone does not harden as the count rises.
MAX_ABUSIVE_TURNS = int(os.getenv("MAX_ABUSIVE_TURNS", "3"))

# Consecutive turns about something this agent does not do, before the call is ended.
# Two, because the first one is answered with what the desk *is* for; if that does not
# land, a third rewording of the same refusal will not land either. A caller who wanted
# an apartment has the wrong number, and saying so is kinder than another round.
MAX_OFF_TOPIC_TURNS = int(os.getenv("MAX_OFF_TOPIC_TURNS", "2"))
