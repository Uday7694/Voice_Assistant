"""Fixed spoken lines, per language.

A handful of things the agent says are not written by the planner: the handoff to a
person, and anything else that has to be said even when the model is unavailable or has
just been cut off. They still have to be said in the caller's language — an English
sentence at the end of a Hindi call is the most jarring moment in the conversation, and
it lands exactly when the caller is already unhappy.

Kept as data rather than in the orchestrator so adding a language is a table entry, and
so an agent can override any of them without touching control flow.
"""

from __future__ import annotations

DEFAULT_LANGUAGE = "en-IN"

# Handing the call to a person. Deliberately short: it is spoken on a path the caller
# did not choose, and Bulbul bills per character.
HANDOFF = {
    "en-IN": "Let me put you through to a colleague who can help.",
    "hi-IN": "मैं आपकी बात एक सहकर्मी से करवाती हूँ।",
    "te-IN": "మిమ్మల్ని ఒక సహోద్యోగికి కలుపుతున్నాను.",
    "ta-IN": "உங்களை ஒரு சகஊழியரிடம் இணைக்கிறேன்.",
    "kn-IN": "ನಿಮ್ಮನ್ನು ಸಹೋದ್ಯೋಗಿಗೆ ಸಂಪರ್ಕಿಸುತ್ತೇನೆ.",
    "ml-IN": "ഞാൻ നിങ്ങളെ ഒരു സഹപ്രവർത്തകനുമായി ബന്ധിപ്പിക്കാം.",
    "mr-IN": "मी तुमची एका सहकाऱ्याशी बोलणी करून देते.",
    "bn-IN": "আমি আপনাকে একজন সহকর্মীর সঙ্গে যুক্ত করছি।",
    "gu-IN": "હું તમને એક સહકર્મી સાથે જોડું છું.",
    "pa-IN": "ਮੈਂ ਤੁਹਾਨੂੰ ਇੱਕ ਸਹਿਕਰਮੀ ਨਾਲ ਜੋੜਦੀ ਹਾਂ।",
}


def handoff(language: str) -> str:
    """The handoff line for a language, falling back to English."""
    return HANDOFF.get(language) or HANDOFF[DEFAULT_LANGUAGE]
