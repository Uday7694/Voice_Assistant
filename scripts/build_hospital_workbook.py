"""Generate the demo hospital workbook: departments, doctors, schedules, bookings.

    python scripts/build_hospital_workbook.py

Writes ``data/hospital.xlsx``. Deterministic — same seed, same file — so the fixture can
be regenerated in review and diffed by regenerating rather than by reading binary.

This stands in for a hospital information system. The point is not the data, it is the
shape: the agent reads a schedule it did not invent, so every doctor name, fee and slot
it says out loud came from a row somewhere. That is the difference between a demo and a
system that can be pointed at a real HIS by changing one loader.

Dates are generated relative to the day it is run, which is why it is a script and not a
checked-in constant: a schedule full of last month's slots demos nothing.
"""

from __future__ import annotations

import random
from datetime import date, datetime, time, timedelta
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "hospital.xlsx"
SEED = 20260825
DAYS_OF_SCHEDULE = 14

HOSPITAL = {
    "name": "Apollo Multi-Speciality Hospital",
    "branch": "Jubilee Hills",
    "city": "Hyderabad",
    "address": "Road No. 72, Jubilee Hills, Hyderabad 500033",
    "phone": "04023607777",
    "emergency_phone": "1066",
    "opd_hours": "8:00 am to 8:00 pm",
    "closed_on": "Sunday afternoons",
}

DEPARTMENTS = [
    # (name, floor, spoken aliases — what callers actually say)
    ("cardiology", 3, "heart, cardiac, cardio, chest pain, bp, blood pressure"),
    ("orthopaedics", 2, "bone, bones, joint, knee, ortho, orthopedics, fracture, back pain"),
    ("ent", 1, "ear, nose, throat, sinus, synus, e n t, hearing, tonsils"),
    ("general medicine", 1, "general, physician, gp, fever, cold, viral, body pain"),
    ("paediatrics", 2, "child, children, kids, baby, paediatric, pediatrics"),
    ("dermatology", 4, "skin, hair, derma, rash, acne"),
    ("gynaecology", 4, "gynae, obstetrics, pregnancy, women, ladies doctor"),
]

# (name, department, gender, qualification, designation, experience, fee, languages,
#  room, opd days, opd start, opd end, slot minutes)
DOCTORS = [
    ("Dr. Ramesh Iyer", "cardiology", "M", "MBBS, MD, DM Cardiology", "Senior Consultant",
     22, 800, "English, Telugu, Tamil, Hindi", "3A-12", "Mon,Tue,Wed,Thu,Fri", "9:00", "13:00", 20),
    ("Dr. Sneha Reddy", "cardiology", "F", "MBBS, MD, DNB Cardiology", "Consultant",
     11, 600, "English, Telugu, Hindi", "3A-14", "Mon,Wed,Fri,Sat", "15:00", "19:00", 20),
    ("Dr. Arun Prasad", "orthopaedics", "M", "MBBS, MS Orthopaedics", "Head of Department",
     26, 900, "English, Telugu, Hindi", "2B-04", "Mon,Tue,Thu,Sat", "10:00", "14:00", 15),
    ("Dr. Kavya Menon", "orthopaedics", "F", "MBBS, MS, Fellowship Sports Medicine", "Consultant",
     9, 650, "English, Malayalam, Hindi", "2B-06", "Tue,Wed,Fri", "16:00", "19:00", 15),
    ("Dr. Faisal Khan", "ent", "M", "MBBS, MS ENT", "Senior Consultant",
     17, 700, "English, Hindi, Urdu, Telugu", "1C-09", "Mon,Wed,Thu,Fri", "9:30", "13:30", 15),
    ("Dr. Lakshmi Rao", "ent", "F", "MBBS, DLO, DNB ENT", "Consultant",
     13, 600, "English, Telugu, Kannada", "1C-11", "Tue,Thu,Sat", "15:30", "18:30", 15),
    ("Dr. Vivek Sharma", "general medicine", "M", "MBBS, MD General Medicine", "Consultant",
     14, 500, "English, Hindi, Telugu", "1A-02", "Mon,Tue,Wed,Thu,Fri,Sat", "8:00", "12:00", 10),
    ("Dr. Anita Desai", "general medicine", "F", "MBBS, MD, Diabetology", "Senior Consultant",
     19, 600, "English, Hindi, Marathi, Telugu", "1A-05", "Mon,Tue,Wed,Thu,Fri", "17:00", "20:00", 10),
    ("Dr. Suresh Babu", "paediatrics", "M", "MBBS, MD Paediatrics", "Senior Consultant",
     21, 650, "English, Telugu, Hindi", "2A-01", "Mon,Tue,Thu,Fri,Sat", "9:00", "13:00", 15),
    ("Dr. Priya Nair", "paediatrics", "F", "MBBS, DCH, DNB Paediatrics", "Consultant",
     10, 550, "English, Malayalam, Telugu, Hindi", "2A-03", "Mon,Wed,Fri", "16:00", "19:00", 15),
    ("Dr. Nikhil Verma", "dermatology", "M", "MBBS, MD Dermatology", "Consultant",
     8, 700, "English, Hindi", "4B-07", "Tue,Wed,Thu,Sat", "11:00", "15:00", 15),
    ("Dr. Meera Krishnan", "gynaecology", "F", "MBBS, MS Obstetrics and Gynaecology",
     "Head of Department", 24, 850, "English, Telugu, Tamil, Hindi", "4A-02",
     "Mon,Tue,Wed,Fri,Sat", "10:00", "14:00", 20),
    ("Dr. Zoya Sheikh", "gynaecology", "F", "MBBS, DGO, DNB", "Consultant",
     12, 650, "English, Hindi, Urdu", "4A-04", "Mon,Thu,Sat", "15:00", "19:00", 20),
]

# A doctor who is away. The agent has to say so rather than offering slots that a caller
# would turn up for — a schedule with no exceptions in it never exercises that path.
ON_LEAVE = {"Dr. Nikhil Verma": "on leave until next month"}

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

HEADER = Font(bold=True)


def doctor_id(index: int) -> str:
    return f"DOC{index + 101}"


def _sheet(book: Workbook, title: str, headers: list[str]):
    sheet = book.create_sheet(title)
    sheet.append(headers)
    for column in range(1, len(headers) + 1):
        sheet.cell(row=1, column=column).font = HEADER
        sheet.column_dimensions[get_column_letter(column)].width = max(14, len(headers[column - 1]) + 4)
    sheet.freeze_panes = "A2"
    return sheet


def build() -> Path:
    rng = random.Random(SEED)
    book = Workbook()
    book.remove(book.active)

    hospital = _sheet(book, "Hospital", ["field", "value"])
    for key, value in HOSPITAL.items():
        hospital.append([key, value])

    departments = _sheet(book, "Departments", ["department", "floor", "aliases"])
    for name, floor, aliases in DEPARTMENTS:
        departments.append([name, floor, aliases])

    doctors = _sheet(
        book,
        "Doctors",
        [
            "doctor_id", "name", "department", "gender", "qualification", "designation",
            "experience_years", "consultation_fee", "languages", "room",
            "opd_days", "opd_start", "opd_end", "slot_minutes", "status", "notes",
        ],
    )
    for index, row in enumerate(DOCTORS):
        (name, department, gender, qualification, designation, experience, fee,
         languages, room, opd_days, start, end, slot_minutes) = row
        note = ON_LEAVE.get(name, "")
        doctors.append([
            doctor_id(index), name, department, gender, qualification, designation,
            experience, fee, languages, room, opd_days, start, end, slot_minutes,
            "on leave" if note else "available", note,
        ])

    slots = _sheet(book, "Slots", ["slot_id", "doctor_id", "doctor", "department", "date", "time", "status"])
    today = date.today()
    slot_number = 1
    for index, row in enumerate(DOCTORS):
        name, department = row[0], row[1]
        opd_days, start, end, slot_minutes = row[9], row[10], row[11], row[12]
        if name in ON_LEAVE:
            continue
        working = set(opd_days.split(","))
        for offset in range(DAYS_OF_SCHEDULE):
            day = today + timedelta(days=offset)
            if WEEKDAYS[day.weekday()] not in working:
                continue
            cursor = datetime.combine(day, _parse_time(start))
            closing = datetime.combine(day, _parse_time(end))
            while cursor < closing:
                # Roughly a third of the book is already taken. A schedule where
                # everything is free never makes the agent say "that one has gone".
                status = "booked" if rng.random() < 0.35 else "free"
                slots.append([
                    f"SLT{slot_number:05d}", doctor_id(index), name, department,
                    day.isoformat(), cursor.strftime("%H:%M"), status,
                ])
                slot_number += 1
                cursor += timedelta(minutes=slot_minutes)

    bookings = _sheet(
        book,
        "Bookings",
        ["reference", "patient_name", "phone", "doctor", "department", "date", "time", "status"],
    )
    # Two existing bookings so "check my appointment" has something real to find.
    tomorrow = (today + timedelta(days=1)).isoformat()
    bookings.append(["APT41822", "Ravi Kumar", "9848012345", "Dr. Ramesh Iyer",
                     "cardiology", tomorrow, "10:00", "confirmed"])
    bookings.append(["APT41855", "Sunitha Rao", "9949210999", "Dr. Lakshmi Rao",
                     "ent", (today + timedelta(days=3)).isoformat(), "16:00", "confirmed"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    book.save(OUTPUT)
    return OUTPUT


def _parse_time(text: str) -> time:
    hour, minute = text.split(":")
    return time(int(hour), int(minute))


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
