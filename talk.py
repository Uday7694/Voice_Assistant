"""Hold a spoken conversation with the agent. You type, it talks back.

    python talk.py                 # Telugu
    python talk.py hi-IN           # Hindi
    python talk.py en-IN --silent  # no audio, no TTS credits
    python talk.py hi-IN --auto    # let the caller's language win, as a real call would

Typing rather than speaking is deliberate. Sarvam credits are finite and speech-to-text
would spend one on every turn just to read back something you already know you said;
this way the only credit a turn costs is the reply you actually hear. `--silent` spends
none at all and still exercises the whole brain.

Playback uses winsound, which is part of Windows and needs nothing installed. It plays a
clip at a time and blocks until each finishes, so this cannot demonstrate barge-in —
interrupting the agent needs a real audio pipeline. Everything above the media layer is
the same code a phone call runs.

Ctrl-C or an empty line ends the call.
"""

from __future__ import annotations

import asyncio
import struct
import sys
import tempfile
from pathlib import Path

# The Windows console is cp1252, which cannot carry Telugu or Devanagari in either
# direction: printing a reply raises UnicodeEncodeError, and typed input arrives as
# unpaired surrogates that later blow up inside the HTTP client. stdin matters as much
# as stdout here.
for _stream in (sys.stdin, sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

from brain.agents.hospital import HOSPITAL_AGENT  # noqa: E402  - after load_dotenv
from brain.models import EndEvent, EscalateEvent, SayEvent  # noqa: E402
from brain.orchestrator import Brain  # noqa: E402
from brain.speech import WEB_SAMPLE_RATE, Mouth, SarvamError  # noqa: E402

OPENERS = {
    "te-IN": "నమస్కారం! నేను మీరా. మీకు ఎలా సహాయం చేయగలను?",
    "hi-IN": "नमस्ते! मैं मीरा बोल रही हूँ। मैं आपकी क्या मदद कर सकती हूँ?",
    "en-IN": "Hello! This is Meera. How can I help you today?",
}


def wav(pcm: bytes, sample_rate: int = WEB_SAMPLE_RATE) -> bytes:
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    return header + b"data" + struct.pack("<I", len(pcm)) + pcm


class Speaker:
    """Sarvam TTS to the sound card, counting what it spends."""

    def __init__(self, language: str, silent: bool = False) -> None:
        self.language = language
        self.silent = silent
        self.calls = 0
        self._mouth = None if silent else Mouth(
            language=language, codec="linear16", sample_rate=WEB_SAMPLE_RATE
        )

    async def say(self, text: str) -> None:
        if self.silent or not text.strip():
            return

        pcm = bytearray()
        try:
            async for frame in self._mouth.say_once(text):
                pcm.extend(frame)
        except SarvamError as exc:
            # Out of credits, or a bad speaker/model pairing. Keep the conversation
            # going in text rather than ending the call over it.
            print(f"    [speech unavailable: {exc}]")
            self.silent = True
            return

        self.calls += 1
        if pcm:
            _play(bytes(pcm))


def _play(pcm: bytes) -> None:
    import winsound

    # PlaySound needs a real file; SND_MEMORY exists but is unreliable for WAV here.
    path = Path(tempfile.gettempdir()) / "_agent_turn.wav"
    path.write_bytes(wav(pcm))
    winsound.PlaySound(str(path), winsound.SND_FILENAME)


async def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    language = args[0] if args else "te-IN"
    silent = "--silent" in sys.argv

    if language not in HOSPITAL_AGENT.languages:
        print(f"{language} is not enabled on this agent. Available: {', '.join(HOSPITAL_AGENT.languages)}")
        return 1

    # Naming a language here locks it. In a real call the agent follows whoever is
    # speaking, but on the command line the language is an instruction, not a guess —
    # and one English-looking line ("book appointment tomorrow") would otherwise flip
    # the call and never flip back.
    locked = "--auto" not in sys.argv
    brain = Brain(HOSPITAL_AGENT)
    session = brain.start(channel="talk", language_locked=locked)
    # Pin the language up front so the first turn is already in it. Otherwise the agent
    # opens in its default language and only switches once it has heard enough to be
    # confident, which wastes the first exchange.
    session = session.model_copy(update={"language": language})
    brain.store.put(session)

    # Open the connections while the greeting is being read, not on the first reply.
    await brain.llm.warm()

    speaker = Speaker(language, silent)
    opener = OPENERS.get(language, OPENERS["en-IN"])

    print(f"\n  {HOSPITAL_AGENT.name} · {language} · {'text only' if silent else 'speaking'}")
    print("  empty line or Ctrl-C to hang up\n")
    print(f"  agent> {opener}")
    await speaker.say(opener)

    while True:
        try:
            said = input("\n  you  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not said:
            break

        reply: list[str] = []
        finished = False
        try:
            async for event in brain.handle(session.session_id, said):
                if isinstance(event, SayEvent):
                    reply.append(event.text)
                elif isinstance(event, EscalateEvent):
                    print(f"    [escalated: {event.reason}]")
                    finished = True
                elif isinstance(event, EndEvent):
                    finished = True
        except Exception as exc:  # noqa: BLE001 - report and let the caller decide
            print(f"    [turn failed: {type(exc).__name__}: {exc}]")
            continue

        spoken = " ".join(reply).strip()
        if spoken:
            print(f"  agent> {spoken}")
            await speaker.say(spoken)

        if finished:
            break

    print(f"\n  call ended · {speaker.calls} TTS call(s) spent\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
