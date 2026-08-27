"""Hold a conversation with the agent. Speak or type; it talks back either way.

    python talk.py                 # Telugu, typed
    python talk.py hi-IN           # Hindi, typed
    python talk.py hi-IN --voice   # Hindi, spoken — microphone open, barge-in live
    python talk.py en-IN --silent  # no audio at all, no TTS credits
    python talk.py hi-IN --auto    # let the caller's language win, as a real call would

Both inputs are live at once under `--voice`: the microphone is open for the whole call
and the keyboard still works, so a turn can be spoken or typed and whichever arrives
first is the turn. That is not a convenience — it is what makes the thing testable. A
tricky utterance can be typed exactly, over and over, while everything around it runs on
real speech.

Typing stays the default because speech costs money in both directions. Under `--voice`
every second of audio is transcribed whether anyone is talking or not; typed turns spend
credits only on the reply. `--silent` spends none and still exercises the whole brain.

Output streams. Frames reach the sound card as they are synthesised, so the reply starts
while the rest of it is still being written, and the caller can talk over it: voice
activity stops playback mid-word and the unspoken remainder is never synthesised.

Ctrl-C or an empty line ends the call.
"""

from __future__ import annotations

import asyncio
import re
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
from brain.speech import Mouth, SarvamError  # noqa: E402
from brain.speech.sarvam import close_tts_warmer, warm_tts  # noqa: E402
from brain.speech.audio import Playback  # noqa: E402
from brain.speech.listen import Listener  # noqa: E402
from brain.speech.types import TTS_SAMPLE_RATE, WEB_SAMPLE_RATE  # noqa: E402
from brain.shutdown import quiet_vendored_asyncgen_teardown  # noqa: E402
from brain.speech.fillers import Backchannel  # noqa: E402
from brain.speech.prosody import gap_after, silence  # noqa: E402

def wav(pcm: bytes, sample_rate: int = TTS_SAMPLE_RATE) -> bytes:
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    return header + b"data" + struct.pack("<I", len(pcm)) + pcm


class Speaker:
    """Sarvam TTS to the sound card, counting what it spends.

    A Mouth per utterance rather than one for the call. Pace, pitch and loudness are
    per-connection settings and Mouth already opens a connection per utterance, so
    letting each line declare what it is ("greeting", "readback", "filler") costs
    nothing and is most of what stops the voice sounding flat.

    Frames go to the sound card as they are synthesised, not after. Measured on one
    sentence: audio started at 1.65 s while generation ran to 2.49 s, so the caller
    hears the first words 0.8 s before the last ones exist. Collecting the whole clip
    first spends that 0.8 s on silence, every sentence, all call.
    """

    def __init__(
        self,
        language: str,
        silent: bool = False,
        markers: tuple[str, ...] = (),
        playback: Playback | None = None,
    ) -> None:
        self.language = language
        self.silent = silent
        self.markers = markers
        self.playback = playback
        self.calls = 0
        # Bulbul bills per character, so characters are the number that matters, not
        # requests. Tracked here so the cost of a change is visible while making it.
        self.chars = 0
        # Set by barge-in. Checked between frames so the rest of a sentence the caller
        # has already talked over is never synthesised, let alone played.
        self._interrupted = False

    def interrupt(self) -> None:
        """The caller started talking. Stop, mid-word if necessary."""
        self._interrupted = True
        if self.playback is not None:
            self.playback.stop()

    def resume(self) -> None:
        self._interrupted = False

    async def say(self, text: str, *, kind: str = "ask", cache: bool = False) -> None:
        if not text.strip():
            return

        # Count even when silent. Bulbul bills per character, so the cost of a prompt
        # change has to be measurable without paying for it — otherwise every attempt
        # to make the agent terser costs credits to evaluate.
        self.chars += len(text)
        if self.silent or self._interrupted:
            return

        mouth = Mouth(
            language=self.language,
            kind=kind,
            markers=self.markers,
            codec="linear16",
            sample_rate=TTS_SAMPLE_RATE,
        )
        pcm = bytearray()
        try:
            async for frame in mouth.say_once(text, cache=cache):
                if self._interrupted:
                    # Closing the socket is the only way to cancel Bulbul mid-utterance;
                    # breaking out of the iterator is what closes it.
                    break
                if self.playback is not None:
                    self.playback.write(frame)
                else:
                    pcm.extend(frame)
        except SarvamError as exc:
            # Out of credits, or a bad speaker/model pairing. Keep the conversation
            # going in text rather than ending the call over it.
            print(f"    [speech unavailable: {exc}]")
            self.silent = True
            return

        self.calls += 1
        if self.playback is not None:
            # Hold the turn until the words are actually out. Returning early would let
            # the next sentence queue behind this one correctly, but would also let the
            # call hang up or start listening while audio is still in flight.
            await self.playback.drain()
        elif pcm:
            _play(bytes(pcm))


def _play(pcm: bytes) -> None:
    """Fallback for a machine with no working output stream: one clip, blocking."""
    import winsound

    # PlaySound needs a real file; SND_MEMORY exists but is unreliable for WAV here.
    path = Path(tempfile.gettempdir()) / "_agent_turn.wav"
    path.write_bytes(wav(pcm))
    winsound.PlaySound(str(path), winsound.SND_FILENAME)


# A line carrying a reference code or a phone number is a read-back, and read-backs
# are the lines callers ask to hear again. Slowing those down and leaving everything
# else alone is a one-regex change that removes a whole class of repeat turns.
_HAS_NUMBER = re.compile(r"\d{4,}")
_SORRY = re.compile(r"(sorry|apolog)|माफ़|क्षమ", re.IGNORECASE | re.UNICODE)


def _kind_of(text: str) -> str:
    """Classify a spoken line so the voice can deliver it the way a person would."""
    if _SORRY.search(text):
        return "apology"
    if _HAS_NUMBER.search(text):
        return "readback"
    return "ask"


async def _typed() -> str:
    """One line from the keyboard, off the event loop.

    In a thread because `input` blocks, and blocking the loop would stop the microphone
    pump and the transcriber along with it — the keyboard would silently deafen the
    agent for as long as nobody typed.
    """
    return (await asyncio.to_thread(input)).strip()


async def next_turn(listener: Listener | None) -> str:
    """Whatever the caller does first: says something, or types it.

    Both sources are waited on together and the loser is cancelled, which is the only
    arrangement where neither one has to finish before the other is heard. Waiting on
    them in sequence — read a line, then listen — is the walkie-talkie the microphone
    was opened to avoid.
    """
    if listener is None:
        return await _typed()

    typed = asyncio.create_task(_typed(), name="typed")
    spoken = asyncio.create_task(listener.next_turn(), name="spoken")
    done, pending = await asyncio.wait(
        (typed, spoken), return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
    # The keyboard thread cannot actually be cancelled — `input` holds it until the user
    # presses return — but the task is detached, so the line typed after a spoken turn
    # is simply discarded rather than arriving three turns late.
    first = done.pop()
    return first.result()


async def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    language = args[0] if args else "te-IN"
    silent = "--silent" in sys.argv
    voice_in = "--voice" in sys.argv

    if language not in HOSPITAL_AGENT.languages:
        print(f"{language} is not enabled on this agent. Available: {', '.join(HOSPITAL_AGENT.languages)}")
        return 1

    # Naming a language here locks it. In a real call the agent follows whoever is
    # speaking, but on the command line the language is an instruction, not a guess —
    # and one English-looking line ("book appointment tomorrow") would otherwise flip
    # the call and never flip back.
    locked = "--auto" not in sys.argv
    quiet_vendored_asyncgen_teardown()
    brain = Brain(HOSPITAL_AGENT)
    session = brain.start(channel="talk", language_locked=locked)
    # Pin the language up front so the first turn is already in it. Otherwise the agent
    # opens in its default language and only switches once it has heard enough to be
    # confident, which wastes the first exchange.
    session = session.model_copy(update={"language": language})
    brain.store.put(session)

    # The output device is opened once for the call, not once per clip. Opening it per
    # utterance costs a device round trip at the front of every reply, which is audible
    # as a clipped first syllable.
    playback = None
    if not silent:
        try:
            playback = Playback(TTS_SAMPLE_RATE)
            playback.start()
        except Exception as exc:  # noqa: BLE001 - a missing device is not a crash
            print(f"  [no streaming output device ({exc}); falling back to clip playback]")
            playback = None

    listener = None
    try:
        # Open the connections while the greeting is being read, not on the first reply.
        await brain.llm.warm()
        # Same idea for speech: the first utterance otherwise pays for the TLS and
        # WebSocket handshake, which is 850 ms the caller hears as a pause before hello.
        if not silent:
            warm_tts()

        # Written by the model the first time this agent speaks a language, then read
        # from disk. No table of greetings per language to keep in step with anything.
        await brain.lines.warm(language)
        backchannel = await Backchannel.for_call(brain.lines, language)
        speaker = Speaker(language, silent, markers=backchannel.markers, playback=playback)
        opener = await brain.lines.line("opener", language)

        # Barge-in. The callback fires on voice activity, well before any transcript
        # exists, and empties the buffer PortAudio is reading from — so the agent stops
        # within one 20 ms block of the caller opening their mouth, which is about what
        # a person manages.
        if voice_in:
            listener = await Listener(
                language,
                sample_rate=WEB_SAMPLE_RATE,
                on_speech_start=speaker.interrupt,
            ).__aenter__()

        mode = "text only" if silent else "speaking"
        if listener is not None:
            mode += " · listening"
        print(f"\n  {HOSPITAL_AGENT.name} · {language} · {mode}")
        print(
            "  speak or type; empty line or Ctrl-C to hang up\n"
            if listener is not None
            else "  empty line or Ctrl-C to hang up\n"
        )
        print(f"  agent> {opener}")
        # The opener is byte-identical on every call, so it is synthesised once and
        # replayed from disk after that: no credits, and first audio at the speed of a
        # file read rather than a round trip.
        await speaker.say(opener, kind="greeting", cache=True)

        while True:
            print("\n  you  > ", end="", flush=True)
            try:
                said = await next_turn(listener)
            except (EOFError, KeyboardInterrupt):
                break
            except SarvamError as exc:
                print(f"\n    [listening failed: {exc}]")
                break
            if not said:
                break
            if listener is not None:
                # Echo it: a spoken turn is the one thing in the transcript the caller
                # cannot check by scrolling up, and a misheard word explains a wrong
                # answer better than any log line.
                print(said)
            # The caller has finished talking, so the agent may speak again.
            speaker.resume()

            # The backchannel goes out before the planner has written a word. It is a
            # cached clip, so it starts effectively instantly, and it covers the gap that
            # is the clearest tell that something synthetic is on the line — a person does
            # not fall silent for a second and then deliver a finished sentence.
            if backchannel.should_ack(said):
                await speaker.say(backchannel.ack(), kind="filler", cache=True)

            finished = False
            try:
                async for event in brain.handle(session.session_id, said):
                    if isinstance(event, SayEvent):
                        # Speak each sentence as it lands rather than waiting for the whole
                        # reply. The planner is still writing sentence two while sentence
                        # one is in the air, which is where the perceived latency goes.
                        print(f"  agent> {event.text}")
                        await speaker.say(event.text, kind=_kind_of(event.text))
                    elif isinstance(event, EscalateEvent):
                        print(f"    [escalated: {event.reason}]")
                        finished = True
                    elif isinstance(event, EndEvent):
                        finished = True
            except Exception as exc:  # noqa: BLE001 - report and let the caller decide
                print(f"    [turn failed: {type(exc).__name__}: {exc}]")
                continue

            if finished:
                break

        # Bulbul bills 3 credits per 1000 characters, so characters are the unit that
        # matters here, not turns and not requests.
        credits = speaker.chars * 3 / 1000
        label = "would cost" if silent else "spoken"
        print()
        print(
            "  call ended: "
            f"{speaker.chars} characters {label} "
            f"(~{credits:.2f} credits, {speaker.calls} TTS calls)"
        )
        print()
    finally:
        # Devices first, then sockets, then the provider pools — the reverse of the
        # order they were opened in. A microphone left running keeps feeding a
        # transcriber that is being torn down underneath it.
        await close_tts_warmer()
        if listener is not None:
            await listener.__aexit__(None, None, None)
        if playback is not None:
            playback.close()
        # Release the provider connection pools on the loop that owns them. Left open,
        # they are torn down at interpreter shutdown instead, which throws into
        # generators no longer attached to a running loop and prints an httpcore
        # traceback after the call summary.
        await brain.aclose()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
