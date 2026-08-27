"""The hospital's own data: departments, doctors, schedules, bookings.

Read from ``data/hospital.xlsx`` — see ``scripts/build_hospital_workbook.py``, which
generates it. A spreadsheet because that is what a small hospital actually hands you,
and because a demo whose data can be edited by the people who know the domain is worth
more than one whose data is a dict in a Python file.

Everything the agent says about a doctor — the name, the fee, the room, the times —
comes from a row here. That is the point: the model is not asked to remember a schedule,
it is asked to read one out. Pointing this at a real hospital information system means
replacing this module and nothing else, because the tools depend on these functions and
never on the workbook.

Loaded once and cached. The file does not change under a running process, and re-reading
1500 rows on every turn would put disk I/O inside the latency budget for no benefit.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

WORKBOOK = Path(os.getenv("HOSPITAL_WORKBOOK", "data/hospital.xlsx"))

# How many free slots to hand the planner at once. A caller cannot hold more than about
# three times in their head, and the style rules cap the reply at two sentences anyway;
# sending twenty invites the model to read twenty.
MAX_SLOTS_OFFERED = 3
MAX_DOCTORS_OFFERED = 3

# How soon from now a slot may be offered. The schedule is a spreadsheet and has no
# opinion about what time it is, so at half past eleven the desk cheerfully offered
# "today at ten" — a time that had already gone — and then booked it. Nothing
# downstream would have caught it: the row says free, because nobody sat in it.
#
# Not zero. A caller told at 09:58 that 10:00 is open cannot be anywhere by 10:00, and
# a desk that offers it is setting them up to miss it.
BOOKING_LEAD_MINUTES = int(os.getenv("BOOKING_LEAD_MINUTES", "30"))

# How close a misheard or mistyped name has to be before it counts as that doctor,
# and how far clear of the next-best candidate. Both matter: the floor stops "Sharma"
# becoming "Verma", and the margin stops a name that sits between two doctors being
# resolved to whichever happened to sort first.
#
# 0.72 measured against this roster: the closest two real names are Ramesh/Suresh
# and Krishnan/Khan, both at 0.67, so the floor sits above the worst genuine
# collision while still catching "meeraa", "rameshh" and "iyar". A roster where two
# doctors are closer than that needs the margin to do the work instead - which it
# does, because a real collision scores nearly the same against both.
NAME_MATCH_FLOOR = 0.8
NAME_MATCH_MARGIN = 0.15

# Words that are not part of a name.
_TITLES = {"dr", "doctor"}


@dataclass(frozen=True)
class Doctor:
    doctor_id: str
    name: str
    department: str
    gender: str
    qualification: str
    designation: str
    experience_years: int
    consultation_fee: int
    languages: str
    room: str
    opd_days: str
    opd_start: str
    opd_end: str
    slot_minutes: int
    status: str
    notes: str = ""

    @property
    def available(self) -> bool:
        return self.status.lower() == "available"

    def spoken(self) -> dict[str, Any]:
        """The subset worth saying out loud, in the words a receptionist would use."""
        return {
            "name": self.name,
            "department": self.department,
            "designation": self.designation,
            "qualification": self.qualification,
            "experience_years": self.experience_years,
            "consultation_fee": self.consultation_fee,
            "languages": self.languages,
            "room": self.room,
            "opd_days": self.opd_days,
            "opd_hours": f"{self.opd_start} to {self.opd_end}",
            "status": self.status,
            **({"notes": self.notes} if self.notes else {}),
        }


@dataclass(frozen=True)
class Slot:
    slot_id: str
    doctor_id: str
    doctor: str
    department: str
    day: str
    at: str
    status: str

    @property
    def free(self) -> bool:
        return self.status.lower() == "free"

    def spoken(self) -> str:
        """"tomorrow 10:00 am" — a time a person would say, not a timestamp."""
        return f"{_spoken_day(self.day)} {_spoken_time(self.at)}"

    def upcoming(self, now: datetime | None = None) -> bool:
        """Whether this slot is still far enough ahead to offer someone."""
        try:
            when = datetime.fromisoformat(f"{self.day}T{self.at}")
        except ValueError:
            # A row the workbook wrote in a shape we cannot read. Let it through: the
            # schedule is the hospital's, and hiding its rows because of a formatting
            # problem is a worse failure than offering one odd time.
            log.warning("Slot %s has an unreadable day/time: %r %r", self.slot_id, self.day, self.at)
            return True
        return when >= (now or datetime.now()) + timedelta(minutes=BOOKING_LEAD_MINUTES)

    def bookable(self, now: datetime | None = None) -> bool:
        return self.free and self.upcoming(now)


@dataclass
class Hospital:
    facts: dict[str, str] = field(default_factory=dict)
    departments: dict[str, int] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    doctors: tuple[Doctor, ...] = ()
    slots: tuple[Slot, ...] = ()
    bookings: list[dict[str, Any]] = field(default_factory=list)

    # --- departments ---------------------------------------------------

    def resolve_department(self, said: str) -> str | None:
        """Map what the caller said onto a real department, or None.

        None is the important half. Handing back one department's slots under whatever
        name the caller used is how "synus" once produced three real-looking times for a
        department that does not exist.

        Matched on the whole phrase first, then on the words inside it. Callers say
        "heart doctor", "bone specialist", "ENT department" - the alias sheet cannot
        list every wrapper around the word that matters, and the classifier passes the
        caller's phrasing through untouched precisely so this layer can decide.
        """
        key = _normalise(said)
        if not key:
            return None

        direct = self._department_by_name(key)
        if direct:
            return direct

        words = [w for w in key.split() if w not in _NOT_A_DEPARTMENT]
        for size in (2, 1):  # "general medicine" before "general"
            for start in range(len(words) - size + 1):
                found = self._department_by_name(" ".join(words[start : start + size]))
                if found:
                    return found
        return None

    def _department_by_name(self, key: str) -> str | None:
        if key in self.departments:
            return key
        return self.aliases.get(key)

    # --- doctors ---------------------------------------------------------

    def resolve_doctor(self, said: str, department: str = "") -> Doctor | None:
        """Find a doctor by however the caller said the name.

        Callers say "Ramesh", "Dr Iyer", "Ramesh garu" - rarely the full name as it is
        written, and often not quite right. A caller offered Dr. Meera Krishnan typed
        "meeraa" and was asked to choose again, twice, from a list of two: an exact
        match is the wrong bar for a name a person is repeating back over a phone line.

        Three passes, loosest last: the whole name, then any word of it, then the
        closest word within a tolerance. Narrowed by department when one is known,
        because two departments can each have a Rao - and because a near match against
        two candidates is safe in a way that a near match against thirteen is not.
        """
        key = _normalise(said).replace("dr ", "").replace("doctor ", "")
        if not key:
            return None

        pool = [d for d in self.doctors if not department or d.department == department]
        for doctor in pool:
            if _normalise(doctor.name) == key:
                return doctor

        spoken_words = {w for w in key.split() if len(w) > 2}
        for doctor in pool:
            name_words = {w for w in _normalise(doctor.name).split() if w not in _TITLES}
            if spoken_words & name_words:
                return doctor

        return self._closest_doctor(spoken_words, pool)

    def _closest_doctor(self, spoken: set[str], pool: list[Doctor]) -> Doctor | None:
        """The doctor whose name a misspelling was reaching for, if it is unambiguous.

        Two guards, because accepting the wrong doctor is worse than asking again: the
        match has to be close in absolute terms, and it has to beat the runner-up
        clearly. "meeraa" against Meera Krishnan and Zoya Sheikh is 0.91 to 0.0 and
        obvious; a name that sits between two doctors is refused and asked about.
        """
        if not spoken or not pool:
            return None

        scored: list[tuple[float, Doctor]] = []
        for doctor in pool:
            name_words = [w for w in _normalise(doctor.name).split() if w not in _TITLES]
            best = max(
                (SequenceMatcher(None, said, word).ratio() for said in spoken for word in name_words),
                default=0.0,
            )
            scored.append((best, doctor))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        best_score, best_doctor = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0

        if best_score >= NAME_MATCH_FLOOR and best_score - runner_up >= NAME_MATCH_MARGIN:
            return best_doctor
        return None

    def doctors_in(self, department: str) -> tuple[Doctor, ...]:
        return tuple(d for d in self.doctors if d.department == department and d.available)

    # --- schedule --------------------------------------------------------

    def bookable_slots(
        self, *, department: str = "", doctor: str = "", on: str = ""
    ) -> tuple[Slot, ...]:
        """Every slot that could still be booked, soonest first, unlimited.

        The one place that decides what "available" means, so that what the caller is
        offered and what they are allowed to pick can never disagree. They did: a time
        that had already passed was filtered out of the list read to the caller and
        still matched when they asked for it by name.
        """
        found = [
            slot
            for slot in self.slots
            if slot.bookable()
            and (not department or slot.department == department)
            and (not doctor or slot.doctor == doctor)
            and (not on or slot.day == on)
        ]
        found.sort(key=lambda s: (s.day, s.at))
        return tuple(found)

    def free_slots(
        self, *, department: str = "", doctor: str = "", on: str = "", limit: int = MAX_SLOTS_OFFERED
    ) -> tuple[Slot, ...]:
        """The handful of those actually read out to the caller."""
        return self.bookable_slots(department=department, doctor=doctor, on=on)[:limit]

    def hold(self, slot_id: str) -> None:
        """Mark a slot taken for the rest of this process.

        In memory, not written back to the workbook. The file is a fixture: a demo that
        edits it degrades every time someone runs through the flow, and by the tenth
        call there is nothing free left to offer.
        """
        self.slots = tuple(
            Slot(**{**s.__dict__, "status": "booked"}) if s.slot_id == slot_id else s
            for s in self.slots
        )

    def match_slot(self, said: str, *, department: str = "", doctor: str = "") -> Slot | None:
        """The slot a caller means when they say "tomorrow ten" or "Friday morning"."""
        key = _normalise(said)
        if not key:
            return None

        candidates = self.bookable_slots(department=department, doctor=doctor)

        wanted_day = _day_from_words(key)
        hour, minute, half = _time_from_words(key)

        for slot in candidates:
            if wanted_day and slot.day != wanted_day:
                continue
            if hour is None:
                return slot  # a day was named and nothing else: the first free one
            slot_hour, slot_minute = (int(part) for part in slot.at.split(":"))
            # Only the twelve-hour clock, because callers say "ten" and mean
            # whichever ten this doctor sits. But when they did say which half of
            # the day, honour it: "three in the morning" is not answered with 3 pm.
            if slot_hour % 12 != hour % 12:
                continue
            if half == "am" and slot_hour >= 12:
                continue
            if half == "pm" and slot_hour < 12:
                continue
            if minute is None or slot_minute == minute:
                return slot
        return None

    # --- bookings --------------------------------------------------------

    def book(self, **row: Any) -> dict[str, Any]:
        self.bookings.append(row)
        return row

    def find_booking(self, *, phone: str = "", reference: str = "") -> dict[str, Any] | None:
        digits = re.sub(r"\D", "", phone or "")
        for row in reversed(self.bookings):
            if reference and str(row.get("reference", "")).upper() == reference.upper():
                return row
            if digits and re.sub(r"\D", "", str(row.get("phone", ""))) == digits:
                return row
        return None


# --- loading ---------------------------------------------------------------


@lru_cache(maxsize=1)
def load(path: str | None = None) -> Hospital:
    """Read the workbook once. Cached: the file does not change under a running call."""
    workbook_path = Path(path or WORKBOOK)
    from openpyxl import load_workbook

    book = load_workbook(workbook_path, read_only=True, data_only=True)
    hospital = Hospital()

    hospital.facts = {
        str(key): str(value) for key, value in _rows(book, "Hospital") if key is not None
    }
    for name, floor, aliases in _rows(book, "Departments"):
        if not name:
            continue
        hospital.departments[str(name)] = int(floor or 0)
        for alias in str(aliases or "").split(","):
            if alias.strip():
                hospital.aliases[_normalise(alias)] = str(name)

    hospital.doctors = tuple(
        Doctor(
            doctor_id=str(row[0]), name=str(row[1]), department=str(row[2]), gender=str(row[3]),
            qualification=str(row[4]), designation=str(row[5]), experience_years=int(row[6] or 0),
            consultation_fee=int(row[7] or 0), languages=str(row[8]), room=str(row[9]),
            opd_days=str(row[10]), opd_start=str(row[11]), opd_end=str(row[12]),
            slot_minutes=int(row[13] or 0), status=str(row[14]), notes=str(row[15] or ""),
        )
        for row in _rows(book, "Doctors")
        if row and row[0]
    )

    hospital.slots = tuple(
        Slot(
            slot_id=str(row[0]), doctor_id=str(row[1]), doctor=str(row[2]),
            department=str(row[3]), day=str(row[4]), at=str(row[5]), status=str(row[6]),
        )
        for row in _rows(book, "Slots")
        if row and row[0]
    )

    hospital.bookings = [
        {
            "reference": str(row[0]), "patient_name": str(row[1]), "phone": str(row[2]),
            "doctor": str(row[3]), "department": str(row[4]), "date": str(row[5]),
            "time": str(row[6]), "status": str(row[7]),
        }
        for row in _rows(book, "Bookings")
        if row and row[0]
    ]

    book.close()
    log.info(
        "Loaded %s: %d departments, %d doctors, %d slots",
        workbook_path,
        len(hospital.departments),
        len(hospital.doctors),
        len(hospital.slots),
    )
    return hospital


def _rows(book, sheet_name: str) -> list[tuple]:
    if sheet_name not in book.sheetnames:
        return []
    return [row for row in book[sheet_name].iter_rows(min_row=2, values_only=True) if any(row)]


# --- saying dates and times the way a person does --------------------------


# Words that wrap a department without being one. Stripped before matching, so "heart
# doctor" and "bone specialist" reach the alias sheet as "heart" and "bone".
_NOT_A_DEPARTMENT = {
    "doctor", "doctors", "dr", "department", "dept", "specialist", "consultant",
    "appointment", "booking", "for", "the", "a", "an", "please", "garu", "ji", "sir",
    "madam", "need", "want", "with", "some", "any",
}


# Everything a caller says is matched through _normalise, so what it throws away is
# what the hospital cannot hear. `[^\w\s]` threw away the vowel signs: Python's \w is
# str.isalnum(), which is False for a combining mark, so "రేపు" normalised to "ర ప" and
# "మధ్యాహ్నం" to "మధ య హ న". Every Telugu and Hindi word reaching the schedule arrived
# in pieces, and none of them matched anything.
#
# So it strips by Unicode category instead: punctuation and symbols go, letters, marks
# and digits stay. A Latin sentence normalises exactly as it did before.
_PUNCTUATION = {"P", "S"}


def _normalise(text: str) -> str:
    stripped = "".join(
        " " if unicodedata.category(c)[0] in _PUNCTUATION else c for c in str(text or "")
    )
    return " ".join(stripped.lower().split())


def _spoken_day(iso: str) -> str:
    """"tomorrow" beats "the twenty-sixth" on a phone call."""
    try:
        day = date.fromisoformat(iso)
    except ValueError:
        return iso
    delta = (day - date.today()).days
    if delta == 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    if delta < 7:
        return day.strftime("%A")
    return day.strftime("%A %d %B")


def _spoken_time(at: str) -> str:
    try:
        moment = datetime.strptime(at, "%H:%M")
    except ValueError:
        return at
    hour = moment.hour % 12 or 12
    suffix = "am" if moment.hour < 12 else "pm"
    return f"{hour}:{moment.minute:02d} {suffix}"


# A caller on a Telugu call says "రేపు", not "tomorrow", and a Hindi caller says "कल".
# Read in English only, both landed as no day at all: "రేపు book చేయండి" matched the
# first free slot in the sheet, whatever day it sat on, and the caller found out when
# their appointment actually was from the SMS. Romanised spellings are here too, because
# speech recognition returns them as often as it returns the script.
_RELATIVE_DAYS = {
    "today": 0, "aaj": 0, "आज": 0, "eeroju": 0, "iroju": 0, "ఈరోజు": 0,
    "tomorrow": 1, "tmrw": 1, "kal": 1, "कल": 1, "repu": 1, "repe": 1, "రేపు": 1,
    "day after": 2, "day after tomorrow": 2,
    "parson": 2, "परसों": 2, "ellundi": 2, "ఎల్లుండి": 2,
}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _day_from_words(key: str) -> str:
    """The ISO date a caller means by "tomorrow", "రేపు" or "Friday", or "" for neither.

    Longest phrase first, and on word boundaries. Both matter: read shortest-first,
    "day after tomorrow" matched "tomorrow" and booked the wrong day; read as a bare
    substring, the Hindi word for tomorrow, "kal", matches inside the name Kalyan.
    """
    for phrase in sorted(_RELATIVE_DAYS, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", key):
            return (date.today() + timedelta(days=_RELATIVE_DAYS[phrase])).isoformat()
    for index, name in enumerate(_WEEKDAYS):
        if name in key or name[:3] in key.split():
            ahead = (index - date.today().weekday()) % 7 or 7
            return (date.today() + timedelta(days=ahead)).isoformat()
    return ""


# "The first one" is how people pick from a list they were just read, in every language
# on this desk. It is not a time and never resolves to one on its own — it is an index
# into what the caller was actually offered, which is why it is read here and resolved
# by whoever holds that list.
_ORDINALS = {
    # No bare "one": on a booking desk "one" is one o'clock far more often than it is
    # the first of three. "first" carries the meaning that matters.
    1: ("first", "1st", "pehla", "pehli", "पहला", "पहली", "modati", "modatidi", "మొదటి", "మొదటిది"),
    2: ("second", "2nd", "dusra", "dusri", "दूसरा", "दूसरी", "rendo", "rendavadi", "రెండో", "రెండవది"),
    3: ("third", "3rd", "teesra", "teesri", "तीसरा", "तीसरी", "mudo", "mudavadi", "మూడో", "మూడవది"),
}

# "The last one" and "the earliest" are ordinals too, just counted from the other end.
# No "later". "Three, or later?" is a question about times, not a pick of the last one.
_LAST = ("last", "final", "aakhri", "आखिरी", "chivari", "చివరి", "చివరిది")
_EARLIEST = ("earliest", "soonest", "asap", "anytime", "any time", "whenever", "jaldi", "जल्दी")


def _ordinal_from_words(key: str) -> int | None:
    """Which of the offered slots the caller means, 1-based, or -1 for the last.

    None when they did not pick by position at all. "Anytime" counts as the first: a
    caller who says they do not mind has still chosen, and reading the list back at
    somebody who just said "whenever" is the most machine-like thing on offer.
    """
    for phrase in _LAST:
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", key):
            return -1
    for index, phrases in _ORDINALS.items():
        for phrase in phrases:
            if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", key):
                return index
    for phrase in _EARLIEST:
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", key):
            return 1
    return None


_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}


# Same reason as the day words: "ఉదయం 10" and "सुबह 10" are how the agent itself reads a
# morning slot out, so they are how the caller says it back.
_MORNING = re.compile(r"(?<!\w)(am|a m|morning|सुबह|subah|ఉదయం|udayam)(?!\w)")
_AFTERNOON = re.compile(
    r"(?<!\w)(pm|p m|afternoon|evening|night|tonight"
    r"|दोपहर|dopahar|शाम|shaam|रात|raat"
    r"|మధ్యాహ్నం|madhyahnam|సాయంత్రం|sayantram|రాత్రి|ratri)(?!\w)"
)


def _time_from_words(key: str) -> tuple[int | None, int | None, str]:
    """"ten thirty", "10:30", "4 pm" -> (hour, minute, half of the day).

    Minute is None when the caller did not say one; the half is "" when they did
    not say which, which is most of the time - "ten" means whichever ten that
    doctor sits.
    """
    half = "am" if _MORNING.search(key) else ("pm" if _AFTERNOON.search(key) else "")
    # The separator is a space by the time this runs: _normalise strips punctuation, so
    # "10:20" arrives as "10 20". Read without that, "10 20" fell through to the bare
    # hour and lost its minutes — a caller who asked for twenty past ten was booked at
    # ten o'clock, and the confirmation read the wrong time convincingly back at them.
    stamp = re.search(r"(?<!\d)(\d{1,2})[:. ]([0-5]\d)(?!\d)", key)
    if stamp:
        return int(stamp.group(1)), int(stamp.group(2)), half

    # "Ten to eleven" names two numbers and only the second one is the hour. Read left
    # to right it came out as nine fifty — an hour and ten minutes from what the caller
    # said, and the read-back agreed with itself, so nothing sounded wrong.
    to_hour = _TO_THE_HOUR.search(key)
    spoken_hour = key[to_hour.end() :] if to_hour else key

    hour: int | None = None
    bare = re.search(r"\b(\d{1,2})\b", spoken_hour)
    if bare and 1 <= int(bare.group(1)) <= 12:
        hour = int(bare.group(1))
    else:
        for word, value in _WORD_NUMBERS.items():
            if re.search(rf"\b{word}\b", spoken_hour):
                hour = value
                break
    if hour is None:
        return None, None, half

    minute = _minutes_from_words(key)
    if to_hour and minute is not None:
        # Said as minutes before the hour, stored as a clock time: quarter to eleven
        # is 10:45, not 11:45.
        hour, minute = (hour - 1) or 12, 60 - minute
    return hour, minute, half


# Only the minutes a schedule is ever cut into. "Ten forty" is a real slot on this
# roster and came back as ten o'clock, because the only minute words read were
# "thirty", "half" and "quarter".
_WORD_MINUTES = {
    "quarter past": 15, "quarter": 15, "half past": 30, "half": 30,
    "five": 5, "ten": 10, "fifteen": 15, "twenty": 20, "twenty five": 25,
    "thirty": 30, "thirty five": 35, "forty": 40, "forty five": 45,
    "fifty": 50, "fifty five": 55,
}


# "Ten to eleven", not "I want to book at eleven": the minute word has to sit directly
# in front of the "to", or every sentence with "to" in it loses an hour.
_TO_THE_HOUR = re.compile(
    r"(?<!\w)(quarter|five|ten|fifteen|twenty|twenty five)\s+(?:to|before)(?!\w)"
)


def _minutes_from_words(key: str) -> int | None:
    """The minutes in "ten forty" or "half past two", or None for a whole hour.

    Read from the words *after* the hour, because the same words are both: in "ten
    forty" the ten is the hour and the forty is the minutes, and in "twenty past ten"
    it is the other way round.
    """
    to_hour = _TO_THE_HOUR.search(key)
    if to_hour:
        # Returned as minutes-before, and turned into a clock time by the caller above,
        # which is the only place that also holds the hour.
        return _WORD_MINUTES[to_hour.group(1)]
    before_past = re.split(r"\b(?:past|bajkar)\b", key)
    if len(before_past) > 1:  # "twenty past ten" — the minutes are said first
        found = _first_minute_word(before_past[0])
        if found is not None:
            return found
    words = key.split()
    for index, word in enumerate(words):
        if word.isdigit() or word in _WORD_NUMBERS:
            return _first_minute_word(" ".join(words[index + 1 :]))
    return None


def _first_minute_word(text: str) -> int | None:
    for phrase in sorted(_WORD_MINUTES, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text):
            return _WORD_MINUTES[phrase]
    return None
