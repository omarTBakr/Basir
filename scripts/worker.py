#!/usr/bin/env python3
"""Run a Temporal worker for Basir's pipelines (task queue `basir-pipelines`) until Ctrl+C.

python scripts/worker.py                 # connects to TEMPORAL_ADDRESS (default 127.0.0.1:7233)
python scripts/worker.py --concurrency 2 # at most 2 activities at once
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from basir.pipelines.runtime import TASK_QUEUE, connect, create_worker
from utilities.config import get_settings

logger = logging.getLogger("worker")


async def serve(concurrency: int) -> None:
    client = await connect()
    logger.info("Worker on task queue '%s' at %s", TASK_QUEUE, get_settings().temporal_address)
    await create_worker(client, concurrency).run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--concurrency", type=int, default=4, help="Activities that may run at once.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    try:
        asyncio.run(serve(args.concurrency))
    except KeyboardInterrupt:
        logger.info("Worker stopped")
    except RuntimeError as e:  # temporalio raises RuntimeError when the server is unreachable
        logger.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
