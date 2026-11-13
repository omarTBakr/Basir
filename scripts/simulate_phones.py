#!/usr/bin/env python3
"""Replay a TUM sequence as N phones streaming to the Basir server at once, then print per-phone stats.

Usage:
    python scripts/simulate_phones.py 2                          # 2 phones, 15 fps, fr1/desk, 30 s
    python scripts/simulate_phones.py 4 --fps 30 --duration 60
    python scripts/simulate_phones.py 2 --dataset datasets/toy_room --fps 0   # as fast as possible
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from basir.api.phone_simulator import record, simulate
from basir.ingest.datasets.prepare import dataset_dir
from basir.ingest.sources.tum import TUMDatasetSource

logger = logging.getLogger("simulate_phones")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phones", type=int, help="Number of simulated phones.")
    parser.add_argument("--fps", type=float, default=15.0, help="Frames per second per phone; 0 = as fast as possible.")
    parser.add_argument("--dataset", default="fr1/desk", help="Preset key or dataset folder.")
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds to stream.")
    parser.add_argument("--max-frames", type=int, default=300, help="Frames loaded and looped.")
    parser.add_argument("--url", default="ws://127.0.0.1:8000", help="Server WebSocket base URL.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    if args.phones < 1:
        parser.error("phones must be >= 1")

    try:
        root = dataset_dir(args.dataset)
        frames = record(TUMDatasetSource(root, require_pose=False), max_frames=args.max_frames)
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        return 1
    logger.info("Streaming %d frames of %s as %d phones at %s fps", len(frames), root.name, args.phones, args.fps)

    try:
        results = asyncio.run(simulate(args.url, args.phones, frames, args.fps, args.duration))
    except OSError as e:
        logger.error("Could not reach the server at %s: %s (start it with scripts/serve.py)", args.url, e)
        return 1
    for stats in results:
        print(stats.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
