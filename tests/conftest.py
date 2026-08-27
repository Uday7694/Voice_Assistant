"""Test-wide fixtures.

The transcript log is the one part of this system that writes outside the process by
design, so the suite points it somewhere disposable. Without this every test run appends
call records to the real log, which is both noise in the training data and a set of
files nobody meant to create.
"""

from __future__ import annotations

import pytest

from brain import transcripts


@pytest.fixture(autouse=True)
def transcripts_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(transcripts, "DIRECTORY", tmp_path / "conversations")
    return tmp_path / "conversations"
