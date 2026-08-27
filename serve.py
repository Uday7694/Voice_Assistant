"""Run the telephony bridge, so a carrier can put calls through the agent.

    python serve.py                      # Plivo dialect, port 8080
    python serve.py --carrier exotel
    python serve.py --carrier twilio --port 9000

The carrier dials the media socket from its own network, so it needs a public wss:// URL.
Point TELEPHONY_PUBLIC_URL at whatever fronts this — a tunnel while developing, a load
balancer in production — and give the carrier's answer URL as https://<that host>/answer.
"""

from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

import os  # noqa: E402 - after load_dotenv


def main() -> int:
    parser = argparse.ArgumentParser(description="Voice agent telephony bridge")
    parser.add_argument("--carrier", default=os.getenv("TELEPHONY_CARRIER", "plivo"),
                        choices=["plivo", "exotel", "twilio"])
    parser.add_argument("--port", type=int, default=int(os.getenv("TELEPHONY_PORT", "8080")))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--public-url", default=os.getenv("TELEPHONY_PUBLIC_URL", ""))
    args = parser.parse_args()

    # Set before importing the server, which reads them at module scope.
    os.environ["TELEPHONY_CARRIER"] = args.carrier
    if args.public_url:
        os.environ["TELEPHONY_PUBLIC_URL"] = args.public_url

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "websockets", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    import uvicorn  # noqa: E402 - after the environment is set

    from brain.telephony.server import PUBLIC_URL, build  # noqa: E402

    logging.getLogger(__name__).info(
        "Bridge up for %s; carrier should stream to %s", args.carrier, PUBLIC_URL
    )
    uvicorn.run(build(), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
