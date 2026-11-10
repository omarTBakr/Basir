"""General-purpose helpers shared across Basir.

Nothing here knows about Basir itself: no settings, no datasets, no models. Each
section groups one kind of job, so callers import only what they need.

Sections:
    1. URLs and paths
    2. Hashing
    3. Archives
    4. HTTP downloads
    5. PNG writing
    6. .env files
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import re
import shutil
import struct
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib

from tqdm import tqdm

logger = logging.getLogger("basir.utility")

# ---------------------------------------------------------------------------
# 1. URLs and paths
# ---------------------------------------------------------------------------

ARCHIVE_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".tar", ".zip")


def is_url(text: str) -> bool:
    """True for http, https and ftp URLs."""
    return isinstance(text, str) and urllib.parse.urlparse(text).scheme in ("http", "https", "ftp")


def is_archive(path: str | Path) -> bool:
    """True if the file name ends with a supported archive suffix."""
    return Path(path).name.lower().endswith(ARCHIVE_SUFFIXES)


def archive_stem(url_or_path: str) -> str:
    """File name of a URL or path without its archive suffix ('a/b/scene.tar.gz' -> 'scene')."""
    name = Path(urllib.parse.urlparse(url_or_path).path or url_or_path).name
    for suffix in ARCHIVE_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)] or "dataset"
    return name or "dataset"


def is_within_directory(path: str | Path, base: str | Path) -> bool:
    """True if `path` resolves to a location inside `base` (guards against '../' escapes)."""
    return Path(base, path).resolve().is_relative_to(Path(base).resolve())


# ---------------------------------------------------------------------------
# 2. Hashing
# ---------------------------------------------------------------------------


def file_digest(path: str | Path, algorithm: str = "sha256") -> str:
    """Hex digest of a file, read in chunks."""
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while chunk := f.read(1 << 16):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checksum(path: str | Path, expected: str, algorithm: str = "sha256") -> bool:
    """Compare a file's digest with `expected` (case-insensitive)."""
    actual = file_digest(path, algorithm)
    if actual != expected.strip().lower():
        logger.error("Checksum mismatch for %s: expected %s, got %s", Path(path).name, expected, actual)
        return False
    return True


# ---------------------------------------------------------------------------
# 3. Archives
# ---------------------------------------------------------------------------

_IGNORED_ARCHIVE_ENTRIES = {"__MACOSX", ".DS_Store"}


def extract_archive(
    archive_path: str | Path,
    dest_dir: str | Path,
    flatten_single_dir: bool = True,
    overwrite: bool = False,
) -> Path:
    """Safely extract a tar or zip archive into `dest_dir`.

    A non-empty `dest_dir` is left alone unless `overwrite` is set. If the archive holds
    one top-level folder and `flatten_single_dir` is set, its contents go straight into
    `dest_dir`. Members that would land outside the destination are rejected.
    """
    archive_path, dest_dir = Path(archive_path), Path(dest_dir)
    if dest_dir.exists() and any(dest_dir.iterdir()):
        if not overwrite:
            logger.info("%s already exists; skipping extraction.", dest_dir)
            return dest_dir
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    staging = dest_dir.parent / f".tmp_{dest_dir.name}_{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        _extract_into(archive_path, staging)
        entries = [p for p in staging.iterdir() if p.name not in _IGNORED_ARCHIVE_ENTRIES]
        if flatten_single_dir and len(entries) == 1 and entries[0].is_dir():
            entries = list(entries[0].iterdir())
        for entry in entries:
            target = dest_dir / entry.name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            shutil.move(str(entry), str(target))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return dest_dir


def _extract_into(archive_path: Path, target: Path) -> None:
    name = archive_path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            for member in zf.namelist():
                if not is_within_directory(member, target):
                    raise RuntimeError(f"Unsafe path in zip: {member}")
            zf.extractall(target)
    elif is_archive(archive_path):
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(target, filter="data")  # rejects absolute and '../' members
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.name}")


# ---------------------------------------------------------------------------
# 4. HTTP downloads
# ---------------------------------------------------------------------------

USER_AGENT = "Basir-Downloader/1.0"


class IncompleteDownloadError(OSError):
    """The connection closed before the whole file arrived (the .part file is kept)."""


def download_file(
    url: str,
    target_file: str | Path,
    checksum: str | None = None,
    checksum_type: str = "sha256",
    resume: bool = True,
    quiet: bool = False,
) -> Path:
    """Download `url` to `target_file`, resuming a previous `.part` file if present.

    The file is only renamed to its final name once every byte has arrived (and the
    checksum, if given, matches), so an interrupted download is never mistaken for a
    finished one.
    """
    target_file = Path(target_file)
    if target_file.is_file() and target_file.stat().st_size > 0:
        if not checksum or verify_checksum(target_file, checksum, checksum_type):
            logger.info("Already downloaded: %s", target_file)
            return target_file
        logger.warning("Existing file failed its checksum; downloading again.")
        target_file.unlink()

    target_file.parent.mkdir(parents=True, exist_ok=True)
    part_file = target_file.with_name(target_file.name + ".part")
    offset = part_file.stat().st_size if resume and part_file.exists() else 0

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if offset:
        request.add_header("Range", f"bytes={offset}-")

    try:
        response = urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as e:
        if e.code == 416 and offset:  # the .part already holds the whole file
            total = _total_from_content_range(e.headers.get("Content-Range")) or offset
            return _finish_download(part_file, target_file, total, checksum, checksum_type)
        raise

    with response:
        resumed = response.getcode() == 206 and offset > 0
        length = int(response.headers.get("Content-Length") or 0)
        total = length + offset if resumed else length
        with (
            tqdm(
                total=total or None,
                initial=offset if resumed else 0,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=target_file.name,
                disable=quiet,
            ) as progress,
            open(part_file, "ab" if resumed else "wb") as f,
        ):
            while chunk := response.read(1 << 16):
                f.write(chunk)
                progress.update(len(chunk))

    return _finish_download(part_file, target_file, total, checksum, checksum_type)


def _total_from_content_range(value: str | None) -> int | None:
    """Total size from 'bytes 0-9/100' or 'bytes */100'."""
    total = (value or "").rpartition("/")[2].strip()
    return int(total) if total.isdigit() else None


def _finish_download(
    part_file: Path, target_file: Path, expected_size: int, checksum: str | None, checksum_type: str
) -> Path:
    size = part_file.stat().st_size if part_file.exists() else 0
    if expected_size and size != expected_size:
        raise IncompleteDownloadError(
            f"Got {size} of {expected_size} bytes; run again to resume from {part_file.name}."
        )
    if checksum and not verify_checksum(part_file, checksum, checksum_type):
        part_file.unlink()
        raise ValueError(f"Checksum verification failed for {target_file.name}")
    part_file.replace(target_file)
    return target_file


# ---------------------------------------------------------------------------
# 5. PNG writing (no imaging library needed)
# ---------------------------------------------------------------------------


def write_png_rgb(path: str | Path, width: int, height: int, color: tuple[int, int, int]) -> None:
    """Write a solid-colour 8-bit RGB PNG."""
    _write_png(path, width, height, bit_depth=8, color_type=2, pixel=bytes(color))


def write_png_gray16(path: str | Path, width: int, height: int, value: int) -> None:
    """Write a constant 16-bit greyscale PNG (the format of TUM and Kinect depth maps)."""
    _write_png(path, width, height, bit_depth=16, color_type=0, pixel=struct.pack(">H", value))


def _write_png(path: str | Path, width: int, height: int, bit_depth: int, color_type: int, pixel: bytes) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    row = b"\x00" + pixel * width  # filter byte 0 ("none") + pixels
    header = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(row * height))
        + chunk(b"IEND", b"")
    )
    Path(path).write_bytes(png)


# ---------------------------------------------------------------------------
# 6. .env files
# ---------------------------------------------------------------------------

_VAR_PATTERN = re.compile(r"\$\{([A-Za-z0-9_]+)\}|\$([A-Za-z0-9_]+)")


def parse_env_file(path: str | Path) -> dict[str, str]:
    """Parse a .env file into a dict.

    Supports comments, `export` prefixes, quoted values (text after the closing quote
    is ignored) and ${VAR} / $VAR expansion. Single-quoted values are taken literally.
    """
    path = Path(path)
    values: dict[str, str] = {}
    if not path.is_file():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, raw = (part.strip() for part in line.split("=", 1))

        quote = raw[:1] if raw[:1] in ("'", '"') and raw[:1] in raw[1:] else ""
        value = raw[1 : raw.index(quote, 1)] if quote else raw.split(" #", 1)[0].strip()

        if quote != "'":
            value = _VAR_PATTERN.sub(lambda m: values.get(m[1] or m[2], os.getenv(m[1] or m[2], "")), value)
        values[key] = value
    return values
