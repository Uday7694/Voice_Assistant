"""Pre-opened TTS connections, because opening one is most of the wait.

Measured against the live service, on a warm process: getting the socket up costs 591 ms
and the actual synthesis of a short line costs 318 ms. So roughly two thirds of the
silence between a caller finishing and the agent starting is spent on a handshake that
has nothing to do with what is being said.

It is paid on every utterance because there is no way not to. Bulbul's TTS socket has no
cancel message, so the only way to stop the agent mid-sentence is to close the
connection — which means a long-lived shared socket and barge-in are mutually exclusive,
and barge-in is not negotiable on a phone call.

The way out is to pay it early instead of never. A connection is opened in the background
while the classifier and the planner are still working, and by the time there is a
sentence to speak it is sitting ready. The utterance still gets a socket of its very own,
still closes it to interrupt — it just did not have to wait for it.

Idle survival was measured before this was built: a connection held open and unused for
30 seconds configured and synthesised normally, at no penalty. `MAX_IDLE` keeps a wide
margin under that anyway, because a socket the server has quietly dropped costs a
failed turn, and a socket discarded early costs nothing but a reconnect nobody waits for.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# How many connections to keep ready. One covers the common case — the agent says one
# thing per turn — and two covers a reply that arrives as two sentences without the
# second waiting on a reconnect.
POOL_SIZE = 2

# How long a pooled connection may sit unused before it is thrown away rather than
# risked. Measured good at 30 s; a third of that is margin against the service changing
# its mind, and the cost of being wrong is a reconnect in the background.
MAX_IDLE = 10.0


@dataclass
class _Ready:
    """An opened connection and the context manager that owns its teardown."""

    context: Any
    socket: Any
    opened_at: float

    @property
    def stale(self) -> bool:
        return time.monotonic() - self.opened_at > MAX_IDLE


class SocketWarmer:
    """Keeps a few TTS connections open and ready to hand out.

    Not a connection pool in the usual sense: nothing is ever returned. A caller takes a
    connection, uses it for exactly one utterance, and closes it. The warmer's only job
    is that there is one waiting next time.
    """

    def __init__(self, connect, *, size: int = POOL_SIZE) -> None:
        self._connect = connect
        self._size = size
        self._ready: list[_Ready] = []
        self._filling: set[asyncio.Task] = set()
        self._enabled = True

    def start(self) -> None:
        """Begin filling. Safe to call repeatedly; tops up to the target."""
        if not self._enabled:
            return
        while len(self._ready) + len(self._filling) < self._size:
            task = asyncio.create_task(self._open_one(), name="tts-prewarm")
            self._filling.add(task)
            task.add_done_callback(self._filling.discard)

    async def _open_one(self) -> None:
        try:
            context = self._connect()
            socket = await context.__aenter__()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a failed pre-warm is not a failed turn
            # Deliberately quiet. Nobody is waiting on this; the utterance that needs a
            # socket will open its own and report the failure properly if it persists.
            log.debug("TTS pre-warm failed: %s", exc)
            return
        self._ready.append(_Ready(context=context, socket=socket, opened_at=time.monotonic()))

    def take(self) -> tuple[Any, Any] | None:
        """A ready connection, or None if the caller should open its own.

        Synchronous and non-blocking on purpose: an utterance must never wait on the
        warmer. Not having one ready is the ordinary case on the first turn of a call.
        """
        while self._ready:
            candidate = self._ready.pop(0)
            if candidate.stale:
                self._discard(candidate)
                continue
            self.start()  # replace what was just taken, in the background
            return candidate.context, candidate.socket
        self.start()
        return None

    def _discard(self, ready: _Ready) -> None:
        """Close a connection nobody is going to use."""
        task = asyncio.create_task(_close(ready.context), name="tts-discard")
        self._filling.add(task)
        task.add_done_callback(self._filling.discard)

    async def aclose(self) -> None:
        """Drop everything. Called when the process is shutting down."""
        self._enabled = False
        for task in list(self._filling):
            task.cancel()
        self._filling.clear()
        ready, self._ready = self._ready, []
        for item in ready:
            await _close(item.context)


async def _close(context: Any) -> None:
    try:
        await context.__aexit__(None, None, None)
    except Exception:  # noqa: BLE001 - teardown of a socket nobody is listening to
        pass
