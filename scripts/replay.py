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
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from basir.core.types import Frame
from basir.ingest.datasets.prepare import dataset_dir
from basir.ingest.playback import Pacer, PlaybackStats, play
from basir.ingest.sources.tum import TUMDatasetSource
from utilities.experiment import log_metrics, log_params, start_run

logger = logging.getLogger("replay")
WINDOW = "Basir replay"


DEPTH_DISPLAY_MAX_M = 5.0  # depth colour scale: 0 m to this


def render(frame: Frame) -> np.ndarray:
    """RGB | colourised depth side by side, with an info overlay (BGR for OpenCV)."""
    rgb = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    if frame.depth is not None:
        scaled = np.clip(frame.depth / DEPTH_DISPLAY_MAX_M * 255, 0, 255).astype(np.uint8)
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


def replay(source: TUMDatasetSource, speed: float = 1.0, show: bool = True) -> PlaybackStats:
    """Play every frame at its recorded pace (scaled by `speed`), optionally in a window."""
    pacer = Pacer(speed)
    stats = play(source, pacer, on_frame=(lambda frame: _show(frame, pacer)) if show else None)
    if show:
        cv2.destroyAllWindows()
    return stats


def _show(frame: Frame, pacer: Pacer) -> bool:
    """Draw the frame; space pauses / resumes, q or Esc quits. Returns False to stop."""
    cv2.imshow(WINDOW, render(frame))
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        pacer.pause()
        while (key := cv2.waitKey(50) & 0xFF) not in (ord(" "), ord("q"), 27):
            pass
        pacer.resume()
    return key not in (ord("q"), 27)


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
        log_params(
            {
                "dataset": root.name,
                "speed": args.speed,
                "stride": args.stride,
                "frames_available": len(source),
                "intrinsics_source": source.intrinsics_source,
            }
        )
        stats = replay(source, speed=args.speed, show=not args.no_window)
        log_metrics(stats.metrics())

    print(
        f"Replayed {stats.frames} frames from {root.name} in {stats.wall_seconds:.1f} s "
        f"({stats.fps:.1f} FPS), valid depth {stats.depth_valid_pct:.1f}%"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
