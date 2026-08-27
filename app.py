"""Run the Voice Desk console.

    python app.py                # http://127.0.0.1:7860
    python app.py --share        # a public link, for testing from a phone
    python app.py --port 8080

A microphone needs a secure context. localhost counts as one, so the default URL works;
reaching the console over plain http from another machine does not, and the browser will
refuse to record without saying why. Use --share for that.
"""

from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

from ui.layout import launch  # noqa: E402  - after load_dotenv


def main() -> int:
    parser = argparse.ArgumentParser(description="Voice Desk agent console")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--share", action="store_true", help="public link via Gradio")
    parser.add_argument("--debug", action="store_true", help="log every turn's internals")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Their per-request lines drown the brain's own logging, which is the part worth
    # reading while a call is in progress.
    for noisy in ("httpx", "httpcore", "urllib3", "gradio", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    sys.exit(main())
