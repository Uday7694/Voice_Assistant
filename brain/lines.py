"""Fixed spoken lines, written once per language by the model instead of by hand.

Some lines are not written by the planner: the greeting, the handoff to a person, the
line that ends a call the agent cannot help with, the backchannel that fills the gap
while the planner thinks. They still have to be said in the caller's language, and the
obvious way to do that — a dictionary keyed by BCP-47 tag — is the wrong shape twice
over. Adding a language means editing every table in the codebase, and every table is a
place for one language to quietly go missing. Worse, the translations are only as good
as whoever typed them, and nobody on the team speaks all twelve.

So a line is declared as a *brief* in English — what it has to accomplish, not what it
says — and the model writes it in whatever language the call is in. That happens once
ever: the result is cached on disk, so the second call in Telugu reads a file, and the
audio cache downstream then hits too, because the text is byte-identical.

An English fallback ships with each brief. It is what a call gets when generation fails
and there is a caller waiting; a slightly wrong language beats silence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .config import LINE_TIMEOUT, LINE_TOTAL_TIMEOUT
from .llm import LLMClient
from .speech.types import LANGUAGE_NAMES

log = logging.getLogger(__name__)

CACHE_DIR = Path(os.getenv("LINE_CACHE_DIR", ".cache/lines"))


@dataclass(frozen=True)
class LineSpec:
    """What a line has to do, described rather than written."""

    key: str
    brief: str
    fallback: str
    count: int = 1
    """How many interchangeable variants to write. More than one only for lines said
    repeatedly in a single call — a backchannel that never varies is worse than none."""


SPECS: dict[str, LineSpec] = {
    "opener": LineSpec(
        key="opener",
        brief=(
            "The first thing the caller hears. Give your own first name and ask how you "
            "can help. Warm, unhurried, under ten words."
        ),
        fallback="Hello, this is the appointment desk. How can I help?",
    ),
    "handoff": LineSpec(
        key="handoff",
        brief=(
            "You are passing the call to a human colleague. Say so in one short "
            "sentence. Do not apologise at length and do not explain why."
        ),
        fallback="Let me put you through to a colleague who can help.",
    ),
    "wrong_desk": LineSpec(
        key="wrong_desk",
        brief=(
            "The caller wants something this desk does not do and has already been told "
            "what it does. In one line: say kindly that you cannot help with it, then "
            "thank them for calling and say goodbye. The goodbye is not optional — this "
            "line is the last thing they hear, and a call that stops mid-breath is the "
            "rudest thing the agent can do. No offer to transfer, no long apology."
        ),
        fallback="I'm afraid that's not something this desk can help with. Do take care.",
    ),
    "farewell": LineSpec(
        key="farewell",
        brief=(
            "The caller is done and the call is about to end. Thank them for calling, "
            "wish them well, and say goodbye — warmly, the way a receptionist who "
            "means it would, in one short line. It has to work both for someone who "
            "just booked and for someone who decided not to, so say nothing about what "
            "happened on the call. No summary, no offer of further help, no invitation "
            "to call back."
        ),
        fallback="Thank you for calling, do take care. Goodbye.",
    ),
    "trouble": LineSpec(
        key="trouble",
        brief=(
            "Something went wrong on your side and you have nothing to say yet. Ask "
            "them to say it once more, in one short line, without explaining why and "
            "without apologising twice."
        ),
        fallback="Sorry, I didn't catch that. Could you say it again?",
    ),
    "wait": LineSpec(
        key="wait",
        brief=(
            "You need a moment before you can answer. Three or four words that hold "
            "the line without promising anything."
        ),
        fallback="Sorry, give me one moment.",
    ),
    "identity": LineSpec(
        key="identity",
        brief=(
            "The caller has asked whether they are speaking to a person. In at most "
            "fifteen words: give your name, say you are this desk's digital assistant, "
            "and end by offering to get on with what they called about. Never claim to "
            "be human. No disclaimer, no apology, no explanation of how you work."
        ),
        fallback="I'm the digital assistant at this desk. Shall we get on with it?",
    ),
    "ack": LineSpec(
        key="ack",
        brief=(
            "A single word that a person says while they are still listening — the "
            "spoken equivalent of a nod. One or two words, never more. It must be "
            "neutral: not a greeting, not an exclamation, not surprise, sympathy or "
            "dismay, and not answerable as a question. If in doubt, the plainest word "
            "for yes or I see."
        ),
        fallback="Right.",
        count=4,
    ),
    "hold": LineSpec(
        key="hold",
        brief=(
            "Said while looking something up, so the caller knows the line is still "
            "live. At most four words. Not a sentence, not an apology, and nothing "
            "about what you are doing beyond that you are checking."
        ),
        fallback="One moment.",
        count=3,
    ),
}

_SYSTEM = """You write single spoken lines for a telephone agent.

Return ONLY a JSON object of the form {"lines": ["...", "..."]}, nothing else.

- Write in the language named below, in that language's own script. The English
  words Indian speakers actually use in this setting — appointment, slot, report,
  confirm — are welcome and sound more natural than translating them. Answering in
  English is not: the sentence and most of its words must be in the named language.
  Never write that language's own words in Latin letters.
- Every line is spoken aloud by a synthetic voice. Plain words only: no markdown, no
  emoji, no stage directions.
- Say it the way a busy, warm receptionist would say it out loud, not the way it would
  be written down. Short beats polite.
- Each string must be a complete, natural line on its own.
- Write for the ear: about twelve words at most, with a comma wherever a person
  speaking would pause for breath. A line with no commas is read out in one rush."""


def _language_name(tag: str) -> str:
    return LANGUAGE_NAMES.get(tag, tag)


class LineBook:
    """Per-agent, per-language store of the lines the planner does not write.

    Reads come from memory, then disk, then the model. Only the last of those costs
    anything, and only once per agent, language and line.
    """

    def __init__(
        self, agent, llm: LLMClient | None = None, *, cache_dir: Path | None = None
    ) -> None:
        self.agent = agent
        self.llm = llm
        self.cache_dir = cache_dir or CACHE_DIR
        self._memory: dict[tuple[str, str], tuple[str, ...]] = {}

    # --- reading -----------------------------------------------------------

    async def line(self, key: str, language: str) -> str:
        """One line. The first variant when a spec declares several."""
        return (await self.variants(key, language))[0]

    async def variants(self, key: str, language: str) -> tuple[str, ...]:
        spec = SPECS[key]
        cached = self._memory.get((key, language)) or self._from_disk(key, language)
        if cached:
            self._memory[(key, language)] = cached
            return cached

        written = await self._write(spec, language)
        if not written:
            return (spec.fallback,)

        self._memory[(key, language)] = written
        self._to_disk(key, language, written)
        return written

    async def warm(self, language: str) -> None:
        """Write everything this language needs, before the caller needs it.

        Called while the greeting is being spoken. After the first call in a language
        this is pure disk reads, and the line layer costs nothing for the rest of the
        deployment's life.
        """
        for key in SPECS:
            try:
                await self.variants(key, language)
            except Exception:  # noqa: BLE001 - warming is an optimisation, never a blocker
                log.debug("Could not warm %s for %s", key, language)

    # --- writing -----------------------------------------------------------

    async def _write(self, spec: LineSpec, language: str) -> tuple[str, ...]:
        if self.llm is None:
            return ()
        prompt = (
            f"Language: {_language_name(language)} ({language})\n"
            f"Who you are: {self.agent.persona}\n"
            f"What this desk does: {self.agent.purpose}\n"
            f"Write {spec.count} version(s) of this line:\n{spec.brief}"
        )
        try:
            data = await self.llm.json_call(
                [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
                # Warm, not deterministic. At zero the four backchannels come back as
                # four copies of the same word, which is the one thing they must not be.
                temperature=0.6,
                timeout=LINE_TIMEOUT,
                ceiling=LINE_TOTAL_TIMEOUT,
                # A provider that returns an empty list has not written the line.
                # NVIDIA answers {"lines": []} for this task; without this the sequence
                # stops there and the language quietly gets the English fallback.
                accept=lambda payload: bool(_usable(payload, spec.count)),
            )
        except Exception as exc:  # noqa: BLE001 - fall back to English rather than fail a call
            log.warning("Could not write %s in %s (%s); using the English fallback", spec.key, language, exc)
            return ()
        return _usable(data, spec.count)

    # --- disk --------------------------------------------------------------

    def _path(self, language: str) -> Path:
        safe = re.sub(r"[^\w.-]", "_", f"{self.agent.name}-{language}")
        return self.cache_dir / f"{safe}.json"

    def _from_disk(self, key: str, language: str) -> tuple[str, ...]:
        entry = _load(self._path(language)).get(key)
        # Entries are stamped with the brief that produced them. Rewriting a brief and
        # getting the old line back forever is the failure this cache invites: it is
        # never wrong, never expires, and nobody remembers it is there. A stamp that
        # does not match the brief in this file is not a cache hit.
        if not isinstance(entry, dict) or entry.get("brief") != _stamp(SPECS[key].brief):
            return ()
        stored = entry.get("lines")
        if not isinstance(stored, list) or not stored:
            return ()
        # Through the same cleaner as a fresh write: a cache written by an older
        # version is repaired on read rather than replayed forever.
        return _usable(stored, len(stored))

    def _to_disk(self, key: str, language: str, lines: tuple[str, ...]) -> None:
        """Best effort. A cache that cannot be written must not end a call."""
        path = self._path(language)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            contents = _load(path)
            contents[key] = {"brief": _stamp(SPECS[key].brief), "lines": list(lines)}
            path.write_text(json.dumps(contents, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            log.debug("Could not cache %s for %s", key, language)


def _stamp(brief: str) -> str:
    """Short fingerprint of a brief, so an edited brief invalidates its own cache."""
    return hashlib.sha1(brief.encode("utf-8")).hexdigest()[:12]


def _usable(data, count: int) -> tuple[str, ...]:
    """Pull clean strings out of whatever shape the model returned.

    Defensive because this runs once and the result is then cached forever: a stray
    markdown fence written into the cache would be spoken on every call in that
    language until someone deleted the file by hand.
    """
    if isinstance(data, dict):
        # The asked-for shape is {"lines": [...]}, but "return only JSON" is advice,
        # not a guarantee: models answer with {"lines": "..."} for a single line, or
        # with one key per line. Take the first list if there is one, otherwise every
        # string value in order. Falling back to English because the wrapper was the
        # wrong shape is the worst outcome available.
        listed = next((v for v in data.values() if isinstance(v, list)), None)
        data = listed if listed is not None else [v for v in data.values() if isinstance(v, str)]
    if isinstance(data, str):
        data = [data]
    if not isinstance(data, list):
        return ()

    lines = []
    for item in data:
        if not isinstance(item, str):
            continue
        clean = re.sub(r"[*_`#>|~\[\]]+", "", item).strip()
        # A line that ends on a comma trails off when it is spoken. The brief asks
        # for commas where a speaker pauses, and a one-word backchannel takes that
        # literally: "అవునండి," is a nod that never lands.
        clean = clean.rstrip(" ,;،؛")
        if clean:
            lines.append(clean)
    return tuple(lines[:count])


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
