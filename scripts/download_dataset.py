#!/usr/bin/env python3
"""Download and prepare RGB-D datasets for Basir.

Gets a TUM RGB-D preset, a URL or a local archive / folder into the datasets directory,
extracts it, pairs RGB and depth frames (associations.txt) and writes dataset_meta.json.
It can also create a tiny synthetic TUM-format dataset for offline tests. The work is
done by basir.ingest.datasets; this script is the command line around it.

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
import logging
from pathlib import Path
import sys

# Allow `python scripts/download_dataset.py` from anywhere (the project root is not on
# sys.path when a script is run directly).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from basir.ingest.datasets.prepare import download_dataset
from basir.ingest.datasets.presets import PRESETS
from basir.ingest.datasets.toy import create_toy_dataset
from utilities.config import get_settings

logger = logging.getLogger("download_dataset")


def list_presets() -> None:
    print(f"{'PRESET':<30} DATASET NAME")
    print("-" * 80)
    for key, info in sorted(PRESETS.items()):
        print(f"{key:<30} {info['name']}\n  └─ {info['description']}")


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
        "-o",
        "--output-dir",
        default=None,
        help="Datasets root (default: Settings.datasets_dir, i.e. $DATA_DIR/datasets or <project>/datasets).",
    )
    parser.add_argument("--no-extract", action="store_true", help="Keep the archive without extracting it.")
    parser.add_argument("--remove-archive", action="store_true", help="Delete the archive after extraction.")
    parser.add_argument("-f", "--overwrite", action="store_true", help="Redo an already prepared dataset.")
    parser.add_argument("--no-flatten", action="store_true", help="Keep the archive's single top-level folder.")
    parser.add_argument("--no-associate", action="store_true", help="Skip writing associations.txt.")
    parser.add_argument(
        "--max-difference", type=float, default=0.02, help="Max RGB-depth time gap in s (default 0.02)."
    )
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
