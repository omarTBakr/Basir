"""Tests for utilities/utility.py."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import struct
import tarfile
import zipfile
import zlib

import pytest

from utilities.utility import (
    IncompleteDownloadError,
    archive_stem,
    download_file,
    extract_archive,
    file_digest,
    is_archive,
    is_url,
    is_within_directory,
    parse_env_file,
    verify_checksum,
    write_png_gray16,
    write_png_rgb,
)

# ---------------------------------------------------------------------------
# 1. URLs and paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("https://cvg.cit.tum.de/data.tgz", True),
        ("http://example.com/dataset.zip", True),
        ("ftp://host/file", True),
        ("/local/path/data.tgz", False),
        ("fr1_room", False),
    ],
)
def test_is_url(text: str, expected: bool) -> None:
    assert is_url(text) is expected


@pytest.mark.parametrize(
    "name, expected",
    [("a.tar.gz", True), ("a.tgz", True), ("a.zip", True), ("a.tar.bz2", True), ("A.TAR.XZ", True),
     ("a.txt", False), ("a.png", False)],
)
def test_is_archive(name: str, expected: bool) -> None:
    assert is_archive(name) is expected


@pytest.mark.parametrize(
    "source, expected",
    [
        ("https://example.com/data/scene1.tar.gz", "scene1"),
        ("/tmp/recordings/office.tgz", "office"),
        ("https://example.com/dataset.zip", "dataset"),
        ("plain_folder", "plain_folder"),
        ("https://example.com/", "dataset"),
    ],
)
def test_archive_stem(source: str, expected: str) -> None:
    assert archive_stem(source) == expected


def test_is_within_directory(tmp_path: Path) -> None:
    assert is_within_directory("inner/file.txt", tmp_path)
    assert not is_within_directory("../escape.txt", tmp_path)
    assert not is_within_directory("/etc/passwd", tmp_path)


# ---------------------------------------------------------------------------
# 2. Hashing
# ---------------------------------------------------------------------------


def test_file_digest_and_verify_checksum(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_bytes(b"basir spatial intelligence")
    sha = hashlib.sha256(b"basir spatial intelligence").hexdigest()
    md5 = hashlib.md5(b"basir spatial intelligence").hexdigest()

    assert file_digest(sample) == sha
    assert verify_checksum(sample, sha.upper())
    assert verify_checksum(sample, md5, "md5")
    assert not verify_checksum(sample, "wrong")


# ---------------------------------------------------------------------------
# 3. Archives
# ---------------------------------------------------------------------------


def test_extract_and_flatten_tar(tmp_path: Path) -> None:
    src = tmp_path / "raw" / "nested_root"
    src.mkdir(parents=True)
    (src / "file1.txt").write_text("hello 1")
    archive = tmp_path / "test.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(src, arcname="nested_root")

    dest = extract_archive(archive, tmp_path / "out")
    assert (dest / "file1.txt").read_text() == "hello 1"


def test_extract_without_flatten_keeps_top_folder(tmp_path: Path) -> None:
    archive = tmp_path / "test.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("inner/doc.txt", "zip test content")
    dest = extract_archive(archive, tmp_path / "out", flatten_single_dir=False)
    assert (dest / "inner" / "doc.txt").read_text() == "zip test content"


def test_extract_and_flatten_zip(tmp_path: Path) -> None:
    archive = tmp_path / "test.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("inner_folder/doc.txt", "zip test content")
    dest = extract_archive(archive, tmp_path / "out")
    assert (dest / "doc.txt").read_text() == "zip test content"


def test_extract_skips_non_empty_destination_unless_overwrite(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("new.txt", "new")
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "old.txt").write_text("old")

    extract_archive(archive, dest)
    assert (dest / "old.txt").exists() and not (dest / "new.txt").exists()
    extract_archive(archive, dest, overwrite=True)
    assert (dest / "new.txt").exists() and not (dest / "old.txt").exists()


def test_tar_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 5
        tar.addfile(info, io.BytesIO(b"pwned"))
    with pytest.raises(tarfile.TarError):
        extract_archive(archive, tmp_path / "out" / "dest")
    assert not (tmp_path / "out" / "escape.txt").exists()


def test_zip_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "pwned")
    with pytest.raises(RuntimeError):
        extract_archive(archive, tmp_path / "out" / "dest")
    assert not (tmp_path / "out" / "escape.txt").exists()


def test_unsupported_archive_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "file.rar"
    bogus.write_bytes(b"x")
    with pytest.raises(ValueError):
        extract_archive(bogus, tmp_path / "out")


# ---------------------------------------------------------------------------
# 4. HTTP downloads (local server, no internet)
# ---------------------------------------------------------------------------

PAYLOAD = bytes(range(256)) * 400  # 102_400 bytes


def test_full_download(tmp_path: Path, file_server) -> None:
    file_server.files["data.tgz"] = PAYLOAD
    out = download_file(file_server.url("data.tgz"), tmp_path / "data.tgz", quiet=True)
    assert out.read_bytes() == PAYLOAD
    assert not (tmp_path / "data.tgz.part").exists()


def test_resume_from_partial_file(tmp_path: Path, file_server) -> None:
    file_server.files["data.tgz"] = PAYLOAD
    (tmp_path / "data.tgz.part").write_bytes(PAYLOAD[:40_000])
    out = download_file(file_server.url("data.tgz"), tmp_path / "data.tgz", quiet=True)
    assert out.read_bytes() == PAYLOAD
    assert file_server.requests[-1]["range"] == "bytes=40000-"


def test_no_resume_starts_over(tmp_path: Path, file_server) -> None:
    file_server.files["data.tgz"] = PAYLOAD
    (tmp_path / "data.tgz.part").write_bytes(b"garbage")
    out = download_file(file_server.url("data.tgz"), tmp_path / "data.tgz", resume=False, quiet=True)
    assert out.read_bytes() == PAYLOAD
    assert file_server.requests[-1]["range"] is None


def test_complete_part_file_answered_with_416(tmp_path: Path, file_server) -> None:
    file_server.files["data.tgz"] = PAYLOAD
    (tmp_path / "data.tgz.part").write_bytes(PAYLOAD)
    out = download_file(file_server.url("data.tgz"), tmp_path / "data.tgz", quiet=True)
    assert out.read_bytes() == PAYLOAD


def test_truncated_download_raises_keeps_part_then_resumes(tmp_path: Path, file_server) -> None:
    file_server.files["data.tgz"] = PAYLOAD
    file_server.truncate["data.tgz"] = 10_000
    with pytest.raises(OSError):
        download_file(file_server.url("data.tgz"), tmp_path / "data.tgz", quiet=True)
    assert not (tmp_path / "data.tgz").exists()
    assert (tmp_path / "data.tgz.part").exists()

    del file_server.truncate["data.tgz"]
    out = download_file(file_server.url("data.tgz"), tmp_path / "data.tgz", quiet=True)
    assert out.read_bytes() == PAYLOAD


def test_incomplete_download_error_is_an_oserror() -> None:
    assert issubclass(IncompleteDownloadError, OSError)


def test_checksum_mismatch_raises_and_discards(tmp_path: Path, file_server) -> None:
    file_server.files["data.tgz"] = PAYLOAD
    with pytest.raises(ValueError):
        download_file(file_server.url("data.tgz"), tmp_path / "data.tgz", checksum="0" * 64, quiet=True)
    assert not (tmp_path / "data.tgz").exists()
    assert not (tmp_path / "data.tgz.part").exists()


def test_existing_file_with_matching_checksum_is_not_downloaded(tmp_path: Path, file_server) -> None:
    file_server.files["data.tgz"] = PAYLOAD
    digest = hashlib.sha256(PAYLOAD).hexdigest()
    download_file(file_server.url("data.tgz"), tmp_path / "data.tgz", checksum=digest, quiet=True)
    calls = len(file_server.requests)
    download_file(file_server.url("data.tgz"), tmp_path / "data.tgz", checksum=digest, quiet=True)
    assert len(file_server.requests) == calls


def test_404_raises(tmp_path: Path, file_server) -> None:
    with pytest.raises(OSError):
        download_file(file_server.url("missing.tgz"), tmp_path / "missing.tgz", quiet=True)
    assert not (tmp_path / "missing.tgz").exists()


# ---------------------------------------------------------------------------
# 5. PNG writing
# ---------------------------------------------------------------------------


def _decode_png(path: Path) -> tuple[int, int, int, int, bytes]:
    """(width, height, bit_depth, color_type, raw scanlines) of a single-IDAT PNG."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, header = 8, b"", None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag, chunk = data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])
        assert crc == zlib.crc32(tag + chunk)
        if tag == b"IHDR":
            header = struct.unpack(">IIBB", chunk[:10])
        elif tag == b"IDAT":
            idat += chunk
        pos += 12 + length
    assert header is not None
    return (*header, zlib.decompress(idat))


def test_write_png_rgb(tmp_path: Path) -> None:
    path = tmp_path / "rgb.png"
    write_png_rgb(path, 4, 3, (10, 20, 30))
    width, height, depth, color_type, raw = _decode_png(path)
    assert (width, height, depth, color_type) == (4, 3, 8, 2)
    assert raw == (b"\x00" + bytes((10, 20, 30)) * 4) * 3


def test_write_png_gray16(tmp_path: Path) -> None:
    path = tmp_path / "depth.png"
    write_png_gray16(path, 5, 2, 7500)
    width, height, depth, color_type, raw = _decode_png(path)
    assert (width, height, depth, color_type) == (5, 2, 16, 0)
    assert struct.unpack(">H", raw[1:3])[0] == 7500


# ---------------------------------------------------------------------------
# 6. .env files
# ---------------------------------------------------------------------------


def test_parse_comments_quotes_and_exports(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
        # Comment line
        APP_NAME="Basir Vision"
        APP_ENV='testing'
        DEBUG=False # inline comment
        export API_PORT=9001
        EMPTY_VAL=
        not a variable
        """
    )
    assert parse_env_file(env_file) == {
        "APP_NAME": "Basir Vision",
        "APP_ENV": "testing",
        "DEBUG": "False",
        "API_PORT": "9001",
        "EMPTY_VAL": "",
    }


def test_quoted_value_followed_by_comment(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('APP_NAME="Basir Vision" # the name\nHASH="a # not a comment"\n')
    parsed = parse_env_file(env_file)
    assert parsed["APP_NAME"] == "Basir Vision"
    assert parsed["HASH"] == "a # not a comment"


def test_variable_expansion_except_single_quotes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BASIR_TEST_HOME", "/data")
    env_file = tmp_path / ".env"
    env_file.write_text("ROOT=/srv\nA=${ROOT}/datasets\nB=\"$BASIR_TEST_HOME/x\"\nC='$ROOT/literal'\n")
    parsed = parse_env_file(env_file)
    assert parsed["A"] == "/srv/datasets"
    assert parsed["B"] == "/data/x"
    assert parsed["C"] == "$ROOT/literal"


def test_missing_env_file_is_empty(tmp_path: Path) -> None:
    assert parse_env_file(tmp_path / "nope.env") == {}
