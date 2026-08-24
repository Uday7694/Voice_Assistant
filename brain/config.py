"""Runtime configuration. Fails fast when required secrets are missing."""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

# Model selection and provider endpoints live in brain/providers.py.

# Budgets in seconds. The planner budget is what protects the turn latency target.
PLANNER_TIMEOUT = 8.0
FAST_TIMEOUT = 3.0
TOOL_TIMEOUT = 1.5

# Conversation limits enforced in code, never in the prompt.
MAX_TURNS_PER_SESSION = 40
MAX_CONSECUTIVE_NO_MATCH = 2
MAX_HISTORY_TURNS = 12
