"""Silence one third-party teardown warning, and nothing else.

The OpenAI SDK vendors its own httpx and httpcore (`httpx2`, `httpcore2`). When a
streaming response finishes, the SDK breaks out of its own event loop on the `[DONE]`
marker and closes the HTTP response, but leaves the byte-stream generators underneath
it suspended. The garbage collector finalises those later, and httpcore2's teardown
raises `RuntimeError: generator didn't stop after athrow()`, which asyncio prints as a
traceback — after the call has ended, after the summary, on a process that is exiting.

Nothing is leaking by then: the response is closed, the socket is released, the process
is on its way out. The traceback is noise, and it is noise from code we do not own.

Everything we *do* own was fixed rather than filtered — abandoned planner streams are
drained on the loop, the warm-up response is closed, and the client pools are closed
when the call ends. This filter is deliberately narrow enough that a leak of ours would
still be printed: it matches only asyncgen-finalisation failures raised from inside the
vendored packages.
"""

from __future__ import annotations

import asyncio
import logging

log = logging.getLogger(__name__)

_MESSAGE = "an error occurred during closing of asynchronous generator"
_VENDORED = ("httpcore2", "httpx2")


def _is_vendored_teardown(context: dict) -> bool:
    if not str(context.get("message", "")).startswith(_MESSAGE):
        return False
    generator = context.get("asyncgen")
    filename = getattr(getattr(generator, "ag_code", None), "co_filename", "")
    return any(package in filename for package in _VENDORED)


def quiet_vendored_asyncgen_teardown(loop: asyncio.AbstractEventLoop | None = None) -> None:
    """Route that one warning to the debug log instead of the terminal."""
    loop = loop or asyncio.get_running_loop()
    previous = loop.get_exception_handler()

    def handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
        if _is_vendored_teardown(context):
            log.debug("Vendored asyncgen teardown: %s", context.get("message"))
            return
        (previous or loop.default_exception_handler)(loop, context)

    loop.set_exception_handler(handler)
