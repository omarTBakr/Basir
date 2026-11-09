"""TUM RGB-D file formats: timestamped text files and per-sequence calibration.

Standard library only, so lightweight tools (the dataset downloader) can use it.
"""

from __future__ import annotations

import json
from pathlib import Path

# TUM depth PNGs store depth in metres multiplied by this factor.
TUM_DEPTH_FACTOR = 5000.0

# Calibrated colour-camera intrinsics per Freiburg sensor (640x480), from the TUM
# RGB-D benchmark's camera calibration page. fr1 / fr2 also have mild lens distortion,
# which Basir ignores for now.
TUM_CALIBRATION: dict[str, dict[str, float]] = {
    "freiburg1": {"fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3},
    "freiburg2": {"fx": 520.9, "fy": 521.0, "cx": 325.1, "cy": 249.7},
    "freiburg3": {"fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6},
}
TUM_IMAGE_SIZE = (640, 480)


def parse_tum_file(path: Path) -> dict[float, str]:
    """Parse a TUM timestamped file (rgb.txt, depth.txt, groundtruth.txt): t -> rest of line."""
    data: dict[float, str] = {}
    if not Path(path).is_file():
        return data
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and not parts[0].startswith("#"):
            try:
                data[float(parts[0])] = parts[1]
            except ValueError:
                continue
    return data


def calibration_for(dataset_dir: Path) -> tuple[dict[str, float] | None, str]:
    """Intrinsics for a dataset folder and where they came from.

    Order: the dataset's own intrinsics.json, then the TUM calibration matching
    'freiburgN' in the folder name. Returns (None, "none") if neither applies.
    """
    dataset_dir = Path(dataset_dir)
    own = dataset_dir / "intrinsics.json"
    if own.is_file():
        return json.loads(own.read_text()), "intrinsics.json"
    name = dataset_dir.name.lower()
    for sensor, calib in TUM_CALIBRATION.items():
        if sensor in name:
            width, height = TUM_IMAGE_SIZE
            return {**calib, "width": width, "height": height, "depth_scale": TUM_DEPTH_FACTOR}, sensor
    return None, "none"
