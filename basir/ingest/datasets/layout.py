"""The index files of a TUM-layout dataset folder: associations.txt and dataset_meta.json."""

from __future__ import annotations

import bisect
import datetime
import json
import logging
from pathlib import Path
from typing import Any

from basir.ingest.datasets.tum_format import parse_tum_file, write_tum_file

logger = logging.getLogger("basir.datasets")


def is_prepared(dataset_dir: Path) -> bool:
    """True once a folder holds a TUM index (rgb.txt) or Basir's dataset_meta.json."""
    return (dataset_dir / "rgb.txt").exists() or (dataset_dir / "dataset_meta.json").exists()


def associate_tum_dataset(dataset_dir: Path, max_difference: float = 0.02) -> tuple[Path | None, Path | None]:
    """Write associations.txt (and associations_with_gt.txt if groundtruth.txt exists).

    Standard TUM association: each RGB frame is paired greedily with the closest unused
    depth frame within `max_difference` seconds; ground truth uses the nearest pose.
    """
    dataset_dir = Path(dataset_dir)
    rgb = parse_tum_file(dataset_dir / "rgb.txt")
    depth = parse_tum_file(dataset_dir / "depth.txt")
    if not rgb or not depth:
        return None, None

    depth_keys = sorted(depth)
    candidates = sorted(
        (abs(t_rgb - t_d), t_rgb, t_d)
        for t_rgb in rgb
        for t_d in depth_keys[
            bisect.bisect_left(depth_keys, t_rgb - max_difference) : bisect.bisect_right(
                depth_keys, t_rgb + max_difference
            )
        ]
    )
    used_rgb: set[float] = set()
    used_depth: set[float] = set()
    matches: list[tuple[float, float]] = []
    for _, t_rgb, t_d in candidates:
        if t_rgb not in used_rgb and t_d not in used_depth:
            used_rgb.add(t_rgb)
            used_depth.add(t_d)
            matches.append((t_rgb, t_d))
    matches.sort()

    assoc_file = dataset_dir / "associations.txt"
    write_tum_file(
        assoc_file,
        "# timestamp_rgb rgb_file timestamp_depth depth_file",
        [f"{t_rgb:.6f} {rgb[t_rgb]} {t_d:.6f} {depth[t_d]}" for t_rgb, t_d in matches],
    )
    logger.info("associations.txt: %d RGB-D pairs (max diff %.3f s)", len(matches), max_difference)

    gt = parse_tum_file(dataset_dir / "groundtruth.txt")
    if not gt:
        return assoc_file, None

    gt_keys = sorted(gt)
    gt_rows = []
    for t_rgb, t_d in matches:
        t_gt = _nearest(gt_keys, t_rgb)
        if t_gt is not None and abs(t_gt - t_rgb) <= max_difference:
            gt_rows.append(f"{t_rgb:.6f} {rgb[t_rgb]} {t_d:.6f} {depth[t_d]} {t_gt:.6f} {gt[t_gt]}")
    gt_file = dataset_dir / "associations_with_gt.txt"
    write_tum_file(
        gt_file,
        "# timestamp_rgb rgb_file timestamp_depth depth_file timestamp_gt tx ty tz qx qy qz qw",
        gt_rows,
    )
    return assoc_file, gt_file


def _nearest(sorted_keys: list[float], t: float) -> float | None:
    i = bisect.bisect_left(sorted_keys, t)
    candidates = sorted_keys[max(i - 1, 0) : i + 1]
    return min(candidates, key=lambda k: abs(k - t)) if candidates else None


def write_dataset_metadata(
    dataset_dir: Path,
    dataset_name: str,
    source: str,
    archive_path: Path | None = None,
    description: str | None = None,
) -> Path:
    """Write dataset_meta.json describing where the dataset came from and what it holds."""
    dataset_dir = Path(dataset_dir)
    is_tum = (dataset_dir / "rgb.txt").is_file() and (dataset_dir / "depth.txt").is_file()
    meta: dict[str, Any] = {
        "name": dataset_name,
        "source": source,
        "format": "tum_rgbd" if is_tum else "generic",
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "directory": str(dataset_dir),
        "has_groundtruth": (dataset_dir / "groundtruth.txt").is_file(),
    }
    if is_tum:
        meta["frames"] = {
            key: len(parse_tum_file(dataset_dir / filename))
            for key, filename in (("rgb", "rgb.txt"), ("depth", "depth.txt"), ("associated", "associations.txt"))
        }
    if description:
        meta["description"] = description
    if archive_path and archive_path.is_file():
        meta["archive_filename"] = archive_path.name
        meta["archive_size_bytes"] = archive_path.stat().st_size

    meta_path = dataset_dir / "dataset_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta_path
