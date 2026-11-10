#!/usr/bin/env python3
"""Run the Basir server: phone ingest, Prometheus /metrics and the web UI.

Usage:
    python scripts/serve.py                  # http://127.0.0.1:8000
    python scripts/serve.py --port 8001
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind (0.0.0.0 to accept phones on the LAN).")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    uvicorn.run("basir.api.app:create_app", factory=True, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
