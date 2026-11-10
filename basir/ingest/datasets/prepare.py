"""Get a dataset ready on disk: resolve a preset / URL / path, download, extract, index, describe."""

from __future__ import annotations

import logging
from pathlib import Path
import shutil
import urllib.parse

from basir.ingest.datasets.layout import associate_tum_dataset, is_prepared, write_dataset_metadata
from basir.ingest.datasets.presets import get_preset_info
from utilities.config import get_settings
from utilities.utility import archive_stem, download_file, extract_archive, is_archive, is_url, verify_checksum

logger = logging.getLogger("basir.datasets")


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
            "Run download_dataset.py --list-presets to see the presets."
        )
    folder = (name or default_name).strip()
    dest = Path(folder) if Path(folder).is_absolute() else output_dir / folder
    return resolved_source, folder, dest.resolve()


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
    if not is_prepared(folder):
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
        if is_prepared(dest):
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
