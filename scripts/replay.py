#!/usr/bin/env python3
"""Replay a recorded RGB-D sequence at real-time speed.

Shows RGB beside colourised depth with the frame index, time and camera position.
Keys: space = pause / resume, q or Esc = quit. Each run is logged to MLflow
(experiment `phase0-replay`).

    python scripts/replay.py fr1/desk                 # preset (must be downloaded)
    python scripts/replay.py datasets/my_recording    # any TUM-layout folder
    python scripts/replay.py fr1/desk --speed 2       # twice as fast; 0 = as fast as possible
    python scripts/replay.py fr1/desk --no-window     # headless: statistics only
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from basir.core.types import Frame  # noqa: E402
from basir.ingest.sources.tum import TUMDatasetSource  # noqa: E402
from scripts.download_dataset import dataset_dir  # noqa: E402
from utilities.experiment import log_metrics, log_params, start_run  # noqa: E402

logger = logging.getLogger("replay")
WINDOW = "Basir replay"


@dataclass
class ReplayStats:
    frames: int = 0
    wall_seconds: float = 0.0
    valid_depth_sum: float = 0.0

    @property
    def fps(self) -> float:
        return self.frames / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def depth_valid_pct(self) -> float:
        return 100.0 * self.valid_depth_sum / self.frames if self.frames else 0.0


def render(frame: Frame, max_depth: float = 5.0) -> np.ndarray:
    """RGB | colourised depth side by side, with an info overlay (BGR for OpenCV)."""
    rgb = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    if frame.depth is not None:
        scaled = np.clip(frame.depth / max_depth * 255, 0, 255).astype(np.uint8)
        depth = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        depth[frame.depth <= 0] = 0
        canvas = np.hstack([rgb, depth])
    else:
        canvas = rgb
    pos = frame.pose.position if frame.pose is not None else None
    text = f"#{frame.id}  t={frame.timestamp:.2f}"
    if pos is not None:
        text += f"  cam=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}) m"
    cv2.putText(canvas, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def replay(source: TUMDatasetSource, speed: float = 1.0, show: bool = True) -> ReplayStats:
    """Play every frame, sleeping so frames follow their timestamps (scaled by `speed`)."""
    stats = ReplayStats()
    start_wall = time.perf_counter()
    first_t: float | None = None
    paused_for = 0.0

    for frame in source:
        first_t = frame.timestamp if first_t is None else first_t
        if speed > 0:
            due = (frame.timestamp - first_t) / speed
            delay = due - (time.perf_counter() - start_wall - paused_for)
            if delay > 0:
                time.sleep(delay)

        stats.frames += 1
        if frame.depth is not None:
            stats.valid_depth_sum += float(np.mean(frame.depth > 0))

        if show:
            cv2.imshow(WINDOW, render(frame))
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                pause_start = time.perf_counter()
                while (key := cv2.waitKey(50) & 0xFF) not in (ord(" "), ord("q"), 27):
                    pass
                paused_for += time.perf_counter() - pause_start
            if key in (ord("q"), 27):
                break

    stats.wall_seconds = time.perf_counter() - start_wall - paused_for
    if show:
        cv2.destroyAllWindows()
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="Preset key (e.g. fr1/desk) or dataset folder.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed; 0 = as fast as possible.")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-window", action="store_true", help="Headless: print statistics only.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    try:
        root = dataset_dir(args.dataset)
        source = TUMDatasetSource(root, stride=args.stride, max_frames=args.max_frames)
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        return 1

    with start_run("phase0-replay", run_name=root.name, tags={"dataset": root.name}):
        log_params({"dataset": root.name, "speed": args.speed, "stride": args.stride,
                    "frames_available": len(source), "intrinsics_source": source.intrinsics_source})
        stats = replay(source, speed=args.speed, show=not args.no_window)
        log_metrics({"frames": stats.frames, "fps": stats.fps, "depth_valid_pct": stats.depth_valid_pct,
                     "wall_seconds": stats.wall_seconds})

    print(f"Replayed {stats.frames} frames from {root.name} in {stats.wall_seconds:.1f} s "
          f"({stats.fps:.1f} FPS), valid depth {stats.depth_valid_pct:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
