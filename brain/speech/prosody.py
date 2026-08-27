"""Turn planner text into something a voice can say like a person.

The planner writes sentences. A person speaking those sentences does things the text
does not contain: they break a ten digit number into two halves, they put a beat after
"right" before getting to the point, they do not say a reference code as one unbroken
word. Bulbul has no SSML, so the only prosody channel is the text itself — punctuation
and spacing are the pause controls, and this module is where they get inserted.

Pure string in, string out. No network, no state, so the whole layer is testable
without spending a credit.
"""

from __future__ import annotations

import re
from typing import Sequence

# A comma is the short pause; an em dash reads longer in Bulbul than a comma but
# shorter than a full stop. Named so the punctuation choice can be retuned in one place
# after a listen test rather than hunted through regexes.
BEAT = ","
LONG_BEAT = " —"

# A leading discourse marker — "right", "okay", the local equivalent — spoken without a
# beat after it fuses into the next word, and the whole line lands as one flat burst.
# That is the single most machine-like thing a synthesiser does.
#
# Which words those are is language-specific, and this module refuses to hold a list per
# language: the caller passes them in, and the caller gets them from the backchannel
# vocabulary the line book already wrote for that language. The filler words and the
# words that need a beat after them are the same words.


def _marker_pattern(markers: Sequence[str]) -> re.Pattern | None:
    usable = [m.strip() for m in markers if m and m.strip()]
    if not usable:
        return None
    # Longest first, so "one moment" wins over "one".
    usable.sort(key=len, reverse=True)
    return re.compile(
        r"^(" + "|".join(re.escape(m) for m in usable) + r")(?![\w])[,\s]*",
        re.IGNORECASE | re.UNICODE,
    )


# A run of digits long enough that nobody says it in one breath.
_DIGIT_RUN = re.compile(r"(?<!\d)(\d{5,})(?!\d)")

# Reference codes: letters welded to digits, e.g. APT44847. The letters stay a word —
# spelling them out is both slower and, per the style rules, forbidden — but the seam
# between letters and digits is where a person pauses.
_CODE = re.compile(r"\b([A-Z]{2,5})[- ]?(\d{3,})\b")

# Indian mobile numbers are said as five and five. Anything else splits into threes
# from the left, which is how a person reads an unfamiliar number off a screen.
PHONE_LENGTH = 10


def _chunks(run: str, size: int = 3) -> list[str]:
    """Split into groups, never leaving a single digit stranded at the end.

    A trailing group of one is read as a number of its own, which the caller hears as
    a stutter. When it happens the last two groups are merged and re-split evenly:
    "1234567" becomes 123 45 67, not 123 456 7.
    """
    groups = [run[i : i + size] for i in range(0, len(run), size)]
    if len(groups) > 1 and len(groups[-1]) == 1:
        merged = groups[-2] + groups[-1]
        half = len(merged) // 2
        groups[-2:] = [merged[:half], merged[half:]]
    return groups


def _group_digits(run: str, sep: str = f"{BEAT} ") -> str:
    """Break a digit run the way a person reads one off a screen."""
    if len(run) == PHONE_LENGTH:
        # Indian mobiles are said as five and five, always.
        return f"{run[:5]}{sep}{run[5:]}"
    return sep.join(_chunks(run))


def space_numbers(text: str) -> str:
    """Break codes and long digit runs so they are heard, not just pronounced."""
    # Inside a reference code the groups get a space, not a comma: the caller is
    # holding one item in their head, and a comma between every pair turns a five
    # digit code into five separate announcements.
    text = _CODE.sub(lambda m: f"{m.group(1)}{BEAT} {_group_digits(m.group(2), ' ')}", text)
    return _DIGIT_RUN.sub(lambda m: _group_digits(m.group(1)), text)


def beat_after_marker(text: str, markers: Sequence[str] = ()) -> str:
    """Give a leading discourse marker its pause: 'Right which department'."""
    pattern = _marker_pattern(markers)
    match = pattern.match(text) if pattern else None
    if not match:
        return text
    rest = text[match.end() :].lstrip()
    # "Sure." is the whole line: there is nothing after the marker to pause before, and
    # inserting one produces "Sure, ." which the synthesiser reads as a stumble.
    if not re.search(r"[^\W_]", rest, re.UNICODE):
        return text
    return f"{match.group(1)}{BEAT} {rest}"


# There is deliberately no "insert a breath into a long sentence" step here.
#
# It was tried. Splitting an unpunctuated run at the middle word is language-neutral and
# wrong: it produced "at ten in, the morning" in English and cut a Telugu verb phrase in
# half. Nothing in this layer knows where a clause ends in every language the agent
# speaks, and a pause in the wrong place is heard as a stumble — worse than the run it
# was meant to fix.
#
# The phrasing is fixed where it is written instead: the planner and the line book are
# both told to write short sentences and to put commas where a speaker would pause.
# Punctuation that arrives here is honoured; punctuation that is missing is not invented.


def speakable(text: str, language: str = "en-IN", markers: Sequence[str] = ()) -> str:
    """Full pass: what actually goes to the synthesiser.

    Order matters: the marker beat goes in before the numbers are grouped, so a line
    that opens with an acknowledgement and then reads out a code gets both.
    """
    if not text or not text.strip():
        return ""
    out = beat_after_marker(text.strip(), markers)
    out = space_numbers(out)
    # Collapse anything the substitutions doubled up: ",," and ", ." are audible as a
    # stumble, not as a pause.
    out = re.sub(r"\s*,\s*,+", ",", out)
    out = re.sub(r",\s*([.?!।])", r"\1", out)
    return re.sub(r"\s{2,}", " ", out).strip()


# --- silence between lines -------------------------------------------------

# What a person leaves after finishing a sentence, in milliseconds. Without it the
# synthesiser starts the next line the instant the last one ends, and two sentences
# delivered with no seam between them is the tell that survives every other fix.
#
# A question gets the longest gap because it hands the floor over: the caller needs a
# beat to realise it is their turn, and an agent that ploughs on is one that talks over
# people. Keyed on punctuation rather than on language, which is why "?" and "।" and
# "?" all appear — the marks are shared across scripts even when nothing else is.
GAP_MS = {
    "question": 420,
    "statement": 260,
    "clause": 140,
    "filler": 90,
}

_QUESTION_MARKS = "?؟？"
_STOP_MARKS = ".।۔。!"


def gap_after(text: str, kind: str = "ask") -> int:
    """Milliseconds of silence to leave after speaking this line."""
    if kind == "filler":
        return GAP_MS["filler"]

    stripped = (text or "").rstrip(" \"')]”’")
    if not stripped:
        return 0
    if stripped[-1] in _QUESTION_MARKS:
        return GAP_MS["question"]
    if stripped[-1] in _STOP_MARKS:
        return GAP_MS["statement"]
    return GAP_MS["clause"]


def silence(milliseconds: int, sample_rate: int, *, width: int = 2) -> bytes:
    """That gap as playable 16-bit PCM."""
    return bytes(int(sample_rate * milliseconds / 1000) * width)
