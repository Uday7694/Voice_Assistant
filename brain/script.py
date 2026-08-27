"""Is this reply actually in the caller's language?

Indian callers do not speak unmixed languages, and an agent that tries to is the one
that sounds wrong. "अपॉइंटमेंट", "slot", "cardiology", "confirm" — these are the words
people use in Hindi and Telugu sentences, and translating them into pure Sanskritised
Hindi or literary Telugu makes a hospital desk sound like a textbook. Some English in
the sentence is right.

A whole sentence of English is not. That is not code-mixing, it is the model dropping
the language, and on a Telugu call it is the moment the caller stops understanding.

The line between the two is drawn by counting scripts rather than words, and without a
table of which script belongs to which language: every language this agent speaks is
written in a non-Latin script, so "does this reply contain a real proportion of
non-Latin letters?" answers the question for all of them, including any added later.
"""

from __future__ import annotations

import re

# Below this share of native letters, the reply has changed language rather than
# borrowed from another one. Set low deliberately. Heavy mixing is normal speech —
# "Uday जी, कौन सा department चाहिए?" is one third Devanagari and completely fine —
# so this is not a measure of how much English is acceptable. It only catches the
# reply that has no meaningful Hindi or Telugu in it at all.
MIN_NATIVE_RATIO = 0.2

# Enough letters to judge. "OK" and "Hmm" are not a change of language, and a two-letter
# reply has no ratio worth measuring.
MIN_LETTERS = 12

_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def expects_native_script(language: str) -> bool:
    """Whether replies in this language should be mostly non-Latin.

    True for everything except English. No list of scripts: the agent's languages are
    all written in their own, and a language added tomorrow will be too.
    """
    return not (language or "").lower().startswith("en")


def native_ratio(text: str) -> float:
    """Share of the letters that are not Latin. 1.0 is pure Devanagari or Telugu."""
    letters = _LETTER.findall(text or "")
    if not letters:
        return 0.0
    native = sum(1 for c in letters if ord(c) > 0x024F)
    return native / len(letters)


def is_wrong_language(text: str, language: str) -> bool:
    """Whether this reply has abandoned the caller's language rather than mixed into it."""
    if not expects_native_script(language):
        return False
    if len(_LETTER.findall(text or "")) < MIN_LETTERS:
        return False
    return native_ratio(text) < MIN_NATIVE_RATIO
