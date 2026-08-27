"""The endpoint a carrier calls: answer URL in, media socket out.

Three routes and no more. The carrier fetches `/answer` when a call arrives and is told
to stream to `/media`; `/media` is the WebSocket that becomes a `Call`; `/dial` places an
outbound call so the same agent can ring a customer as easily as answer one.

Deliberately thin. Everything that decides how the call goes is in the brain, and
everything that knows what a carrier's frames look like is in `carriers.py`; this file
only introduces them.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, Response

from ..agents.hospital import HOSPITAL_AGENT
from ..flow import Agent
from ..orchestrator import Brain
from .call import Call
from .carriers import Carrier, carrier

log = logging.getLogger(__name__)

# Where the carrier should stream to. It has to be a public wss:// URL — the carrier
# dials it from its own network, so localhost is never right outside a tunnel.
PUBLIC_URL = os.getenv("TELEPHONY_PUBLIC_URL", "wss://localhost:8080/media")

# Which carrier's dialect to speak. One deployment, one carrier; the adapters exist so
# this is an environment variable rather than a migration.
CARRIER_NAME = os.getenv("TELEPHONY_CARRIER", "plivo")

DEFAULT_LANGUAGE = os.getenv("TELEPHONY_LANGUAGE", "en-IN")


def build(agent: Agent | None = None, *, brain: Brain | None = None) -> FastAPI:
    """The app, with one brain shared across every call it handles."""
    chosen: Carrier = carrier(CARRIER_NAME)
    shared = brain or Brain(agent or HOSPITAL_AGENT)
    app = FastAPI(title="Voice agent telephony bridge")

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "carrier": chosen.name,
                "agent": shared.agent.name,
                "languages": list(shared.agent.languages),
                "stream_url": PUBLIC_URL,
            }
        )

    @app.api_route("/answer", methods=["GET", "POST"])
    async def answer() -> Response:
        """What the carrier fetches when a call connects, inbound or outbound."""
        return Response(
            content=chosen.answer_document(PUBLIC_URL),
            media_type="application/xml",
        )

    @app.websocket("/media")
    async def media(websocket: WebSocket) -> None:
        """One connection, one call."""
        await websocket.accept()
        language = websocket.query_params.get("language", DEFAULT_LANGUAGE)
        call = Call(_Adapter(websocket), chosen, shared, language=language)
        try:
            await call.run()
        except Exception:  # noqa: BLE001 - a failed call must not take the server down
            log.exception("Call failed")
        finally:
            with_suppressed(websocket)

    @app.post("/dial")
    async def dial(request: Request) -> JSONResponse:
        """Place an outbound call.

        Left as an explicit not-implemented rather than a plausible stub. Every carrier's
        outbound API differs in auth, in the field names, and in what regulatory
        paperwork has to be attached — in India an outbound campaign needs DLT
        registration and consent records, and a stub that looks like it works is how that
        gets skipped. Wire it per carrier with real credentials.
        """
        body = await request.json()
        log.info("Outbound call requested to %s", body.get("to"))
        return JSONResponse(
            {
                "error": "outbound dialling is not wired up",
                "carrier": chosen.name,
                "next": (
                    f"Call {chosen.name}'s outbound API with answer_url pointing at "
                    "this server's /answer. Check DLT registration and consent first."
                ),
            },
            status_code=501,
        )

    return app


class _Adapter:
    """FastAPI's WebSocket, in the two-method shape `Call` asks for."""

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket

    async def send(self, message: str) -> None:
        await self._websocket.send_text(message)

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        try:
            while True:
                yield await self._websocket.receive_text()
        except Exception:  # noqa: BLE001 - a closed socket is a hangup, not an error
            return


def with_suppressed(websocket: WebSocket) -> None:
    """Best-effort close. The caller is already gone by the time this matters."""
    try:
        import asyncio

        asyncio.create_task(websocket.close())
    except Exception:  # noqa: BLE001
        pass
