"""One open microphone, for the whole call.

The shape here is the difference between a voice agent and a walkie-talkie. The obvious
design opens the microphone when it is the caller's turn and closes it while the agent
talks, which is simple and wrong: a microphone that is shut during the reply cannot hear
the caller interrupt, and interrupting is most of what people do to a machine that has
started saying the wrong thing.

So capture runs from the moment the call connects until it ends, and the two things that
come out of it are handled separately. Final transcripts queue up as caller turns.
Voice-activity starts fire a callback immediately — before any transcript exists, which
is the whole point, because by the time a transcript is ready the caller has been talking
over the agent for a second or more.

Everything above this deals in strings and one callback; nothing above it knows there is
a socket, a device, or a vendor.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from .audio import Microphone
from .sarvam import Ear
from .types import WEB_SAMPLE_RATE

log = logging.getLogger(__name__)


class Listener:
    """The caller's side of the call: a microphone, a transcriber, and a queue of turns.

    ``on_speech_start`` is called from the event loop the moment voice activity begins.
    Keep it cheap — stopping playback is what it is for.
    """

    def __init__(
        self,
        language: str = "en-IN",
        *,
        sample_rate: int = WEB_SAMPLE_RATE,
        on_speech_start: Callable[[], None] | None = None,
    ) -> None:
        self.language = language
        self.sample_rate = sample_rate
        self.on_speech_start = on_speech_start
        self._turns: asyncio.Queue[str] = asyncio.Queue()
        self._mic: Microphone | None = None
        self._ear: Ear | None = None
        self._tasks: list[asyncio.Task] = []
        self._failure: BaseException | None = None

    async def __aenter__(self) -> "Listener":
        self._mic = Microphone(self.sample_rate)
        self._mic.start()
        self._ear = await Ear(language=self.language, sample_rate=self.sample_rate).__aenter__()
        self._tasks = [
            asyncio.create_task(self._pump(), name="mic-pump"),
            asyncio.create_task(self._read(), name="stt-read"),
        ]
        return self

    async def __aexit__(self, *exc) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - teardown
                pass
        self._tasks = []
        if self._mic is not None:
            self._mic.close()
            self._mic = None
        if self._ear is not None:
            await self._ear.__aexit__(*exc)
            self._ear = None

    async def _pump(self) -> None:
        """Microphone frames to the transcriber, continuously."""
        assert self._mic is not None and self._ear is not None
        try:
            async for frame in self._mic.frames():
                await self._ear.feed(frame)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced through `next_turn`
            log.exception("Microphone pump stopped")
            self._fail(exc)

    async def _read(self) -> None:
        """Transcriber events to caller turns, and voice activity to the callback."""
        assert self._ear is not None
        # Segments of the current utterance, and whether the caller has stopped talking.
        #
        # Both orderings happen and neither can be assumed. Measured against the live
        # service on a 1.7 s utterance, the events arrived speech_start, speech_end,
        # transcript — the text lands *after* the end signal, so flushing only on
        # speech_end would have queued nothing at all and the agent would have sat there
        # while the caller waited. On a longer utterance the transcript for an earlier
        # segment arrives while the caller is still going.
        #
        # So a turn is complete when both facts are in, whichever order they arrive:
        # the caller has stopped, and there is text.
        buffer: list[str] = []
        stopped = False

        def flush() -> None:
            nonlocal buffer, stopped
            if buffer:
                # Joined rather than queued separately, so a caller who pauses mid
                # sentence gets one answer to the whole thought instead of two answers
                # to its halves.
                self._turns.put_nowait(" ".join(buffer))
            buffer = []
            stopped = False

        try:
            async for event in self._ear.events():
                if event.kind == "speech_start":
                    stopped = False
                    if self.on_speech_start is not None:
                        self.on_speech_start()
                elif event.kind == "transcript" and event.text.strip():
                    buffer.append(event.text.strip())
                    if stopped:
                        flush()
                elif event.kind == "speech_end":
                    stopped = True
                    if buffer:
                        flush()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced through `next_turn`
            log.exception("Transcription stopped")
            self._fail(exc)

    def _fail(self, exc: BaseException) -> None:
        """Record a background failure and unblock whoever is waiting for a turn."""
        if self._failure is None:
            self._failure = exc
            self._turns.put_nowait("")

    async def next_turn(self) -> str:
        """Wait for the caller to say something and stop. Empty string on failure."""
        text = await self._turns.get()
        if self._failure is not None:
            raise self._failure
        return text
