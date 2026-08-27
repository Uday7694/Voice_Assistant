"""Verify the Sarvam speech layer end to end. No microphone needed.

Speaks a line with Bulbul, feeds the audio straight back into Saaras, and prints what
comes out. If the transcript resembles the input line, both directions work.

    python probe_sarvam.py
    python probe_sarvam.py hi-IN
"""

from __future__ import annotations

import asyncio
import sys
import time

# The Windows console defaults to cp1252, which cannot encode Devanagari, Tamil or
# Telugu — printing a test line crashes before a single API call is made.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

from brain.speech.sarvam import TTS_SAMPLE_RATE, WEB_SAMPLE_RATE, Ear, Mouth  # noqa: E402

LINES = {
    "en-IN": "Your appointment with the cardiology department is confirmed for tomorrow morning.",
    "hi-IN": "आपका अपॉइंटमेंट कल सुबह दस बजे कन्फर्म हो गया है।",
    "ta-IN": "உங்கள் சந்திப்பு நாளை காலை பத்து மணிக்கு உறுதி செய்யப்பட்டது.",
    "te-IN": "మీ అపాయింట్‌మెంట్ రేపు ఉదయం పది గంటలకు నిర్ధారించబడింది.",
    "kn-IN": "ನಿಮ್ಮ ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ ನಾಳೆ ಬೆಳಿಗ್ಗೆ ಹತ್ತು ಗಂಟೆಗೆ ದೃಢಪಡಿಸಲಾಗಿದೆ."
}


async def synthesise(language: str, text: str) -> bytes:
    """Bulbul: text -> raw PCM, measuring time to the first audio frame."""
    mouth = Mouth(language=language, codec="linear16", sample_rate=TTS_SAMPLE_RATE)
    started = time.perf_counter()
    first_ms = None
    pcm = bytearray()

    async for frame in mouth.say_once(text):
        if first_ms is None:
            first_ms = (time.perf_counter() - started) * 1000
        pcm.extend(frame)

    total_ms = (time.perf_counter() - started) * 1000
    seconds = len(pcm) / (TTS_SAMPLE_RATE * 2)
    print(f"  TTS  first frame {first_ms or -1:.0f} ms · total {total_ms:.0f} ms "
          f"· {len(pcm)} bytes ({seconds:.1f}s audio)")
    return bytes(pcm)


async def transcribe(language: str, pcm: bytes) -> None:
    """Saaras: PCM back to text, in 100 ms frames as a live call would send it."""
    frame_bytes = WEB_SAMPLE_RATE // 10 * 2  # 100 ms of 16-bit mono
    started = time.perf_counter()

    async with Ear(language=language, sample_rate=WEB_SAMPLE_RATE) as ear:
        async def push() -> None:
            for offset in range(0, len(pcm), frame_bytes):
                await ear.feed(pcm[offset : offset + frame_bytes])  # Ear wraps it
                await asyncio.sleep(0.01)  # faster than real time, still ordered
            await ear.flush()

        pusher = asyncio.create_task(push())
        try:
            async with asyncio.timeout(30):
                async for event in ear.events():
                    stamp = (time.perf_counter() - started) * 1000
                    if event.kind == "transcript":
                        print(f"  STT  transcript @{stamp:.0f} ms: {event.text!r}")
                        return
                    print(f"  STT  {event.kind} @{stamp:.0f} ms")
        except asyncio.TimeoutError:
            print("  STT  timed out with no transcript")
        finally:
            pusher.cancel()


async def main() -> int:
    languages = [sys.argv[1]] if len(sys.argv) > 1 else list(LINES)

    for language in languages:
        text = LINES.get(language)
        if text is None:
            print(f"{language}: no sample line")
            continue

        print(f"\n{language}: {text}")
        try:
            pcm = await synthesise(language, text)
            if pcm:
                await transcribe(language, pcm)
            else:
                print("  TTS returned no audio")
        except Exception as exc:  # noqa: BLE001 - this is a diagnostic script
            print(f"  FAILED: {type(exc).__name__}: {str(exc)[:200]}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
