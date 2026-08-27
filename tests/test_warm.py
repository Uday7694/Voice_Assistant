"""Pre-opened synthesis connections.

Every test drives a fake connector, so nothing here opens a socket. What is worth pinning
is the behaviour around the happy path: that an utterance never waits on the pool, that a
connection left sitting too long is thrown away rather than risked, and that a pre-warm
that fails is invisible to the turn that needed it.
"""

from __future__ import annotations

import asyncio

import pytest

from brain.speech import warm
from brain.speech.warm import SocketWarmer


class FakeContext:
    """Stands in for the SDK's connection context manager."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.entered = False
        self.closed = False

    async def __aenter__(self):
        if self.fail:
            raise ConnectionError("no route to host")
        self.entered = True
        return f"socket-{id(self)}"

    async def __aexit__(self, *exc):
        self.closed = True


class Connector:
    """Hands out FakeContexts and counts how often it was asked."""

    def __init__(self, fail: bool = False) -> None:
        self.made: list[FakeContext] = []
        self.fail = fail

    def __call__(self) -> FakeContext:
        context = FakeContext(fail=self.fail)
        self.made.append(context)
        return context


async def _settle() -> None:
    """Let the warmer's background tasks run."""
    for _ in range(6):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_a_started_warmer_opens_up_to_its_target():
    connector = Connector()
    warmer = SocketWarmer(connector, size=2)
    warmer.start()
    await _settle()
    assert len(connector.made) == 2
    await warmer.aclose()


@pytest.mark.asyncio
async def test_taking_a_connection_hands_back_an_already_opened_one():
    connector = Connector()
    warmer = SocketWarmer(connector, size=1)
    warmer.start()
    await _settle()

    taken = warmer.take()
    assert taken is not None
    context, socket = taken
    assert context.entered and socket.startswith("socket-")
    await warmer.aclose()


@pytest.mark.asyncio
async def test_taking_one_immediately_starts_replacing_it():
    """The next utterance must not pay for the one before it."""
    connector = Connector()
    warmer = SocketWarmer(connector, size=1)
    warmer.start()
    await _settle()

    warmer.take()
    await _settle()
    assert len(connector.made) == 2, "took one, did not open a replacement"
    await warmer.aclose()


@pytest.mark.asyncio
async def test_an_empty_pool_returns_nothing_rather_than_waiting():
    """An utterance opens its own connection; it never blocks on the warmer."""
    warmer = SocketWarmer(Connector(), size=1)
    assert warmer.take() is None
    await warmer.aclose()


@pytest.mark.asyncio
async def test_a_connection_left_too_long_is_discarded_not_handed_out():
    """A socket the service has quietly dropped costs a failed turn."""
    connector = Connector()
    warmer = SocketWarmer(connector, size=1)
    warmer.start()
    await _settle()

    stale = warmer._ready[0]
    stale.opened_at -= warm.MAX_IDLE + 1
    assert warmer.take() is None
    await _settle()
    assert stale.context.closed, "the stale connection was leaked rather than closed"
    await warmer.aclose()


@pytest.mark.asyncio
async def test_a_failed_prewarm_is_silent_and_leaves_the_pool_empty():
    """Nobody is waiting on it; the turn that needs a socket will open its own."""
    warmer = SocketWarmer(Connector(fail=True), size=2)
    warmer.start()
    await _settle()
    assert warmer.take() is None
    await warmer.aclose()


@pytest.mark.asyncio
async def test_closing_the_warmer_closes_what_it_was_holding():
    connector = Connector()
    warmer = SocketWarmer(connector, size=2)
    warmer.start()
    await _settle()
    held = list(warmer._ready)

    await warmer.aclose()
    assert all(item.context.closed for item in held)
    assert warmer.take() is None, "a closed warmer must not start opening again"


@pytest.mark.asyncio
async def test_a_closed_warmer_stays_closed():
    warmer = SocketWarmer(Connector(), size=1)
    await warmer.aclose()
    warmer.start()
    await _settle()
    assert warmer.take() is None
