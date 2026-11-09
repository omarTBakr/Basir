#!/usr/bin/env python3
"""Download and prepare RGB-D datasets for Basir.

Gets a TUM RGB-D preset, a URL or a local archive / folder into the datasets directory,
extracts it, pairs RGB and depth frames (associations.txt) and writes dataset_meta.json.
It can also create a tiny synthetic TUM-format dataset for offline tests.

Examples:
    python scripts/download_dataset.py fr1/room                      # preset
    python scripts/download_dataset.py fr1/room --name room          # preset, custom folder
    python scripts/download_dataset.py https://host/scene.tgz        # URL
    python scripts/download_dataset.py ~/recordings/kitchen.tar.gz   # local archive
    python scripts/download_dataset.py --create-toy-dataset --name toy_room
    python scripts/download_dataset.py --list-presets
"""

from __future__ import annotations

import argparse
import bisect
import datetime
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Any
import urllib.parse

# Allow `python scripts/download_dataset.py` from anywhere (the project root is not on
# sys.path when a script is run directly).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from basir.ingest.sources.tum_format import TUM_DEPTH_FACTOR, parse_tum_file  # noqa: E402
from utilities.config import get_settings  # noqa: E402
from utilities.utility import (  # noqa: E402
    archive_stem,
    download_file,
    extract_archive,
    is_archive,
    is_url,
    verify_checksum,
    write_png_gray16,
    write_png_rgb,
)

logger = logging.getLogger("download_dataset")

# ---------------------------------------------------------------------------
# TUM RGB-D presets
# ---------------------------------------------------------------------------

# TUM moved the benchmark to cvg.cit.tum.de; vision.in.tum.de only redirects there.
TUM_BASE_URL = "https://cvg.cit.tum.de/rgbd/dataset"

# key -> (sequence name, description). Sizes measured from the server.
_TUM_SEQUENCES: dict[str, tuple[str, str]] = {
    "fr1/room": ("rgbd_dataset_freiburg1_room", "~746 MB, handheld traversal of a whole office room"),
    "fr1/desk": ("rgbd_dataset_freiburg1_desk", "~328 MB, desk with monitor, mugs, books"),
    "fr1/desk2": ("rgbd_dataset_freiburg1_desk2", "~333 MB, second desk sequence"),
    "fr1/xyz": ("rgbd_dataset_freiburg1_xyz", "~427 MB, translation along 3 principal axes"),
    "fr1/rpy": ("rgbd_dataset_freiburg1_rpy", "~391 MB, rotation around roll, pitch, yaw"),
    "fr1/plant": ("rgbd_dataset_freiburg1_plant", "~663 MB, house plant on a table"),
    "fr1/teddy": ("rgbd_dataset_freiburg1_teddy", "~838 MB, teddy bear on an office chair"),
    "fr1/360": ("rgbd_dataset_freiburg1_360", "~399 MB, 360 degree turn in the room"),
    "fr1/floor": ("rgbd_dataset_freiburg1_floor", "~701 MB, floor loop trajectory"),
    "fr2/desk": ("rgbd_dataset_freiburg2_desk", "~1.8 GB, office desk, calibrated camera"),
    "fr2/xyz": ("rgbd_dataset_freiburg2_xyz", "~2.1 GB, translation along 3 axes"),
    "fr2/rpy": ("rgbd_dataset_freiburg2_rpy", "~1.9 GB, rotation along 3 axes"),
    "fr2/large_with_loop": ("rgbd_dataset_freiburg2_large_with_loop", "~2.7 GB, industrial hall with loop closure"),
    "fr3/long_office": ("rgbd_dataset_freiburg3_long_office_household", "~1.4 GB, long office household loop"),
    "fr3/nostructure_texture_near": ("rgbd_dataset_freiburg3_nostructure_texture_near_withloop", "~745 MB"),
    "fr3/structure_texture_far": ("rgbd_dataset_freiburg3_structure_texture_far", "~497 MB"),
    "fr3/sitting_xyz": ("rgbd_dataset_freiburg3_sitting_xyz", "~739 MB, dynamic scene, people sitting"),
    "fr3/walking_xyz": ("rgbd_dataset_freiburg3_walking_xyz", "~503 MB, dynamic scene, people walking"),
}


def _preset(name: str, description: str) -> dict[str, str]:
    group = name.split("_")[2]  # rgbd_dataset_freiburg1_room -> freiburg1
    return {"url": f"{TUM_BASE_URL}/{group}/{name}.tgz", "name": name, "description": description}


PRESETS: dict[str, dict[str, str]] = {
    key: _preset(name, desc) for key, (name, desc) in _TUM_SEQUENCES.items()
}


def _aliases(key: str, name: str) -> set[str]:
    """fr1/room -> fr1_room, tum_fr1_room, freiburg1_room, rgbd_dataset_freiburg1_room."""
    flat = key.replace("/", "_")
    return {flat, f"tum_{flat}", name, name.removeprefix("rgbd_dataset_")}


PRESET_ALIASES: dict[str, str] = {
    alias: key for key, info in PRESETS.items() for alias in _aliases(key, info["name"])
}


def resolve_preset_key(identifier: str | None) -> str | None:
    """Canonical preset key for a key or alias (case, '-', ' ' and '/' are normalised)."""
    if not identifier:
        return None
    raw = identifier.strip().lower()
    if raw in PRESETS:
        return raw
    return PRESET_ALIASES.get(raw.replace("-", "_").replace(" ", "_").replace("/", "_"))


def get_preset_info(identifier: str | None) -> dict[str, str] | None:
    key = resolve_preset_key(identifier)
    return PRESETS[key] if key else None


def list_presets() -> None:
    print(f"{'PRESET':<30} DATASET NAME")
    print("-" * 80)
    for key, info in sorted(PRESETS.items()):
        print(f"{key:<30} {info['name']}\n  └─ {info['description']}")


# ---------------------------------------------------------------------------
# TUM frame association
# ---------------------------------------------------------------------------

def _nearest(sorted_keys: list[float], t: float) -> float | None:
    i = bisect.bisect_left(sorted_keys, t)
    candidates = sorted_keys[max(i - 1, 0) : i + 1]
    return min(candidates, key=lambda k: abs(k - t)) if candidates else None


def associate_tum_dataset(
    dataset_dir: Path, max_difference: float = 0.02, write_groundtruth: bool = True
) -> tuple[Path | None, Path | None]:
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
    _write_lines(
        assoc_file,
        "# timestamp_rgb rgb_file timestamp_depth depth_file",
        [f"{t_rgb:.6f} {rgb[t_rgb]} {t_d:.6f} {depth[t_d]}" for t_rgb, t_d in matches],
    )
    logger.info("associations.txt: %d RGB-D pairs (max diff %.3f s)", len(matches), max_difference)

    gt = parse_tum_file(dataset_dir / "groundtruth.txt") if write_groundtruth else {}
    if not gt:
        return assoc_file, None

    gt_keys = sorted(gt)
    gt_rows = []
    for t_rgb, t_d in matches:
        t_gt = _nearest(gt_keys, t_rgb)
        if t_gt is not None and abs(t_gt - t_rgb) <= max_difference:
            gt_rows.append(f"{t_rgb:.6f} {rgb[t_rgb]} {t_d:.6f} {depth[t_d]} {t_gt:.6f} {gt[t_gt]}")
    gt_file = dataset_dir / "associations_with_gt.txt"
    _write_lines(
        gt_file,
        "# timestamp_rgb rgb_file timestamp_depth depth_file timestamp_gt tx ty tz qx qy qz qw",
        gt_rows,
    )
    return assoc_file, gt_file


def _write_lines(path: Path, header: str, lines: list[str]) -> None:
    path.write_text("\n".join([header, *lines]) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Synthetic toy dataset (offline tests)
# ---------------------------------------------------------------------------

TOY_WIDTH, TOY_HEIGHT = 64, 48
TOY_BASE_DEPTH_M = 1.5
TOY_DEPTH_STEP_M = 0.05
_TOY_COLORS = [(100, 150, 200), (120, 170, 220), (140, 190, 240), (160, 210, 240), (180, 220, 255)]


def toy_intrinsics(width: int = TOY_WIDTH, height: int = TOY_HEIGHT) -> dict[str, float]:
    """TUM fr1 / Kinect intrinsics (640x480) scaled to the toy image size."""
    intr = get_settings().default_camera_intrinsics
    sx, sy = width / intr["width"], height / intr["height"]
    return {
        "fx": intr["fx"] * sx,
        "fy": intr["fy"] * sy,
        "cx": (intr["cx"] + 0.5) * sx - 0.5,  # keep the principal point on the pixel-centre grid
        "cy": (intr["cy"] + 0.5) * sy - 0.5,
        "width": float(width),
        "height": float(height),
        "depth_scale": TUM_DEPTH_FACTOR,
    }


def create_toy_dataset(target_dir: Path, num_frames: int = 5, overwrite: bool = False) -> Path:
    """Create a tiny synthetic TUM RGB-D dataset.

    64x48 frames at ~30 FPS; frame i has constant depth 1.5 m + 0.05 m * i (stored with
    the TUM factor 5000); intrinsics.json holds the matching scaled camera matrix.
    """
    target_dir = Path(target_dir)
    if target_dir.exists() and any(target_dir.iterdir()):
        if not overwrite:
            logger.info("Toy dataset already exists at %s.", target_dir)
            return target_dir
        shutil.rmtree(target_dir)
    (target_dir / "rgb").mkdir(parents=True)
    (target_dir / "depth").mkdir()

    rgb_lines, depth_lines, gt_lines = [], [], []
    for i in range(num_frames):
        t_rgb = 1305031453.0 + i * 0.033333
        t_depth = t_rgb + 0.0021  # real sensors are a few ms apart
        rgb_rel, depth_rel = f"rgb/{t_rgb:.6f}.png", f"depth/{t_depth:.6f}.png"
        depth_m = TOY_BASE_DEPTH_M + i * TOY_DEPTH_STEP_M
        write_png_rgb(target_dir / rgb_rel, TOY_WIDTH, TOY_HEIGHT, _TOY_COLORS[i % len(_TOY_COLORS)])
        write_png_gray16(target_dir / depth_rel, TOY_WIDTH, TOY_HEIGHT, round(depth_m * TUM_DEPTH_FACTOR))
        rgb_lines.append(f"{t_rgb:.6f} {rgb_rel}")
        depth_lines.append(f"{t_depth:.6f} {depth_rel}")
        gt_lines.append(f"{t_rgb:.6f} {0.05 * i:.4f} {0.01 * i:.4f} {0.5 + 0.02 * i:.4f} 0 0 0 1")

    _write_lines(target_dir / "rgb.txt", "# timestamp filename", rgb_lines)
    _write_lines(target_dir / "depth.txt", "# timestamp filename", depth_lines)
    _write_lines(target_dir / "groundtruth.txt", "# timestamp tx ty tz qx qy qz qw", gt_lines)
    (target_dir / "intrinsics.json").write_text(json.dumps(toy_intrinsics(), indent=2))

    associate_tum_dataset(target_dir)
    write_dataset_metadata(
        target_dir,
        target_dir.name,
        source="synthetic_toy_generator",
        description=f"Synthetic {num_frames}-frame TUM-compatible dataset for offline unit tests",
    )
    return target_dir


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

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
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
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


# ---------------------------------------------------------------------------
# Download pipeline
# ---------------------------------------------------------------------------

def resolve_source_and_dest(
    source: str, name: str | None = None, output_dir: Path | str | None = None
) -> tuple[str, str, Path]:
    """Turn a preset / URL / local path into (download source, folder name, destination).

    The destination is `output_dir / name`; an absolute `name` is used as-is.
    `output_dir` defaults to Settings.datasets_dir (which follows DATA_DIR).
    """
    output_dir = Path(output_dir) if output_dir else get_settings().datasets_dir
    preset = get_preset_info(source)
    if preset:
        resolved_source, default_name = preset["url"], preset["name"]
    elif is_url(source) or Path(source).expanduser().exists():
        resolved_source, default_name = source.strip(), archive_stem(source.strip())
    else:
        raise ValueError(
            f"Unknown dataset source '{source}': not a preset, URL or existing path. "
            "Run with --list-presets to see the presets."
        )
    folder = (name or default_name).strip()
    dest = Path(folder) if Path(folder).is_absolute() else output_dir / folder
    return resolved_source, folder, dest.resolve()


def _is_prepared(dest: Path) -> bool:
    return (dest / "rgb.txt").exists() or (dest / "dataset_meta.json").exists()


def dataset_dir(identifier: str) -> Path:
    """Folder of an already prepared dataset: an existing path, or a preset under datasets_dir.

    Raises FileNotFoundError with the download command if a preset is not downloaded yet.
    """
    path = Path(identifier).expanduser()
    if path.is_dir():
        return path.resolve()
    preset = get_preset_info(identifier)
    if preset is None:
        raise FileNotFoundError(f"'{identifier}' is neither a folder nor a known preset")
    folder = get_settings().datasets_dir / preset["name"]
    if not _is_prepared(folder):
        raise FileNotFoundError(
            f"{identifier} is not downloaded yet. Run: python scripts/download_dataset.py {identifier}"
        )
    return folder


def download_dataset(
    source: str,
    name: str | None = None,
    output_dir: Path | str | None = None,
    extract: bool = True,
    remove_archive: bool = False,
    overwrite: bool = False,
    flatten: bool = True,
    associate: bool = True,
    max_difference: float = 0.02,
    checksum: str | None = None,
    checksum_type: str = "sha256",
    resume: bool = True,
    dry_run: bool = False,
    quiet: bool = False,
) -> Path:
    """Download (or copy), extract, associate and describe a dataset. Returns its folder.

    With `extract=False` the downloaded archive path is returned instead.
    """
    resolved_source, dataset_name, dest = resolve_source_and_dest(source, name, output_dir)

    incomplete = False
    if dest.exists() and any(dest.iterdir()) and not overwrite:
        if _is_prepared(dest):
            logger.info("'%s' is already prepared at %s (use --overwrite to redo).", dataset_name, dest)
            return dest
        # Non-empty without rgb.txt / dataset_meta.json: a previous extraction failed.
        logger.warning("%s looks incomplete; it will be re-extracted.", dest)
        incomplete = True

    if dry_run:
        logger.info("[DRY RUN] Source:      %s", resolved_source)
        logger.info("[DRY RUN] Destination: %s", dest)
        return dest

    archive: Path | None = None
    if is_url(resolved_source):
        filename = Path(urllib.parse.urlparse(resolved_source).path).name or f"{dataset_name}.tar.gz"
        archive = dest.parent / "_archives" / filename
        download_file(resolved_source, archive, checksum, checksum_type, resume=resume, quiet=quiet)
    else:
        local = Path(resolved_source).expanduser().resolve()
        if local.is_dir():
            if local != dest:
                shutil.rmtree(dest, ignore_errors=True)
                shutil.copytree(local, dest)
        else:
            archive = local
            if checksum and not verify_checksum(archive, checksum, checksum_type):
                raise ValueError(f"Checksum mismatch for local file {archive}")

    if archive is not None:
        if not extract:
            logger.info("Archive saved to %s (extraction skipped).", archive)
            return archive
        if is_archive(archive):
            extract_archive(archive, dest, flatten_single_dir=flatten, overwrite=overwrite or incomplete)
            if remove_archive:
                archive.unlink()

    if associate:
        associate_tum_dataset(dest, max_difference=max_difference)
    write_dataset_metadata(dest, dataset_name, resolved_source, archive_path=archive)
    logger.info("Dataset '%s' ready at %s", dataset_name, dest)
    return dest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download_dataset.py",
        description="Download, extract and prepare RGB-D datasets for Basir.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:" + __doc__.split("Examples:", 1)[1],
    )
    parser.add_argument("source", nargs="?", help="Preset (e.g. fr1/room), URL, local archive or folder.")
    parser.add_argument("-n", "--name", help="Folder name under the output directory (default: from the source).")
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Datasets root (default: Settings.datasets_dir, i.e. $DATA_DIR/datasets or <project>/datasets).",
    )
    parser.add_argument("--no-extract", action="store_true", help="Keep the archive without extracting it.")
    parser.add_argument("--remove-archive", action="store_true", help="Delete the archive after extraction.")
    parser.add_argument("-f", "--overwrite", action="store_true", help="Redo an already prepared dataset.")
    parser.add_argument("--no-flatten", action="store_true", help="Keep the archive's single top-level folder.")
    parser.add_argument("--no-associate", action="store_true", help="Skip writing associations.txt.")
    parser.add_argument("--max-difference", type=float, default=0.02, help="Max RGB-depth time gap in s (default 0.02).")
    parser.add_argument("--checksum", help="Expected digest of the archive.")
    parser.add_argument("--checksum-type", choices=["sha256", "md5"], default="sha256")
    parser.add_argument("--no-resume", action="store_true", help="Restart instead of resuming a partial download.")
    parser.add_argument("-l", "--list-presets", action="store_true", help="List the presets and exit.")
    parser.add_argument("--create-toy-dataset", action="store_true", help="Create a synthetic 5-frame dataset.")
    parser.add_argument("-d", "--dry-run", action="store_true", help="Show what would happen; write nothing.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only warnings and errors; no progress bar.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging and tracebacks.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_presets:
        list_presets()
        return 0
    if not (args.source or args.create_toy_dataset):
        parser.error("give a SOURCE, --create-toy-dataset or --list-presets")

    try:
        if args.create_toy_dataset:
            root = Path(args.output_dir) if args.output_dir else get_settings().datasets_dir
            dest = create_toy_dataset(root / (args.name or "toy_dataset"), overwrite=args.overwrite)
        else:
            dest = download_dataset(
                args.source,
                name=args.name,
                output_dir=args.output_dir,
                extract=not args.no_extract,
                remove_archive=args.remove_archive,
                overwrite=args.overwrite,
                flatten=not args.no_flatten,
                associate=not args.no_associate,
                max_difference=args.max_difference,
                checksum=args.checksum,
                checksum_type=args.checksum_type,
                resume=not args.no_resume,
                dry_run=args.dry_run,
                quiet=args.quiet,
            )
    except Exception as e:
        logger.error("Failed: %s", e, exc_info=args.verbose)
        return 1
    if not args.quiet:
        print(f"Dataset ready at: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
