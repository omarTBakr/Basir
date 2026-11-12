#!/usr/bin/env python3
"""Run the Basir Console: the web UI and APIs on loopback, the phone stream alone on the LAN.

Listeners (one process, one shared state):
    127.0.0.1:8000   Console: pages, /metrics, live view and map sockets, /ws/stream without a token
    0.0.0.0:8001     phones only, HTTPS (self-signed): the phone page at /, /pair and
                     /ws/stream/{device_id} with the token as the first message or a Bearer
                     header (INGEST_HOST / INGEST_PORT / INGEST_TLS)
    127.0.0.1:8002   MLflow UI, frameable copy for the Experiments tab (MLFLOW_EMBED_PORT)
    127.0.0.1:8003   Temporal UI, frameable copy for the Pipelines tab (TEMPORAL_EMBED_PORT)

Usage:
    python scripts/serve.py               # all of the above
    python scripts/serve.py --no-lan      # no phone listener on the LAN
    python scripts/serve.py --port 8100   # Console on another port
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
import uvicorn

from basir.api.app import create_app
from basir.api.embed import create_proxy_app
from basir.api.ingest import create_ingest_app
from basir.api.lan import lan_addresses
from basir.api.middleware import redact_uvicorn_logs
from basir.api.tls import ensure_certificate
from utilities.config import get_settings

logger = logging.getLogger("serve")


async def serve(servers: list[uvicorn.Server]) -> None:
    """Run every server until one stops; Ctrl+C stops them all."""
    tasks = [asyncio.create_task(s.serve()) for s in servers]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for s in servers:
            s.should_exit = True
        await asyncio.gather(*tasks, return_exceptions=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000, help="Console port (always on 127.0.0.1).")
    parser.add_argument("--no-lan", action="store_true", help="Do not start the phone listener.")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    redact_uvicorn_logs()  # no device token in any uvicorn log line, even from an old `?token=` client
    settings = get_settings()
    console = create_app(settings)

    def server(app: FastAPI, host: str, port: int, name: str, tls: tuple[Path, Path] | None = None) -> uvicorn.Server:
        logger.info("%-11s %s://%s:%d", name, "https" if tls else "http", host, port)
        cert, key = tls if tls else (None, None)
        config = uvicorn.Config(
            app, host=host, port=port, log_level="info", ws_max_size=32 * 2**20, ssl_certfile=cert, ssl_keyfile=key
        )
        return uvicorn.Server(config)

    servers = [server(console, "127.0.0.1", args.port, "Console")]
    if not args.no_lan:
        addresses = [ip for _, ip in lan_addresses()] or ["127.0.0.1"]
        tls = ensure_certificate(settings.certs_dir, addresses) if settings.ingest_tls else None
        phones = server(create_ingest_app(console), settings.ingest_host, settings.ingest_port, "Phones", tls)
        servers.append(phones)
        for address in addresses:
            logger.info("Phones open %s://%s:%d", "https" if tls else "http", address, settings.ingest_port)
    if settings.mlflow_ui_url:
        servers.append(
            server(
                create_proxy_app(settings.mlflow_ui_url, console_port=args.port),
                "127.0.0.1",
                settings.mlflow_embed_port,
                "MLflow UI",
            )
        )
    servers.append(
        server(
            create_proxy_app(settings.temporal_ui_url, console_port=args.port),
            "127.0.0.1",
            settings.temporal_embed_port,
            "Temporal UI",
        )
    )
    # Ctrl+C reaches one server's signal handler; `serve` then stops the others.
    with suppress(KeyboardInterrupt):
        asyncio.run(serve(servers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
