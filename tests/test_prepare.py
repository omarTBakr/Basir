"""Tests for basir/ingest/datasets/prepare.py (resolution and the download pipeline)."""

from __future__ import annotations

import json
from pathlib import Path
import tarfile

import pytest

from basir.ingest.datasets.prepare import dataset_dir, download_dataset, resolve_source_and_dest
from basir.ingest.datasets.presets import PRESETS
from basir.ingest.datasets.toy import create_toy_dataset

# ---------------------------------------------------------------------------
# Source / destination resolution
# ---------------------------------------------------------------------------


def test_resolve_preset(tmp_path: Path) -> None:
    src, name, dest = resolve_source_and_dest("fr1/room", output_dir=tmp_path)
    assert (src, name, dest) == (
        PRESETS["fr1/room"]["url"],
        "rgbd_dataset_freiburg1_room",
        (tmp_path / "rgbd_dataset_freiburg1_room").resolve(),
    )


def test_resolve_preset_with_custom_name(tmp_path: Path) -> None:
    _, name, dest = resolve_source_and_dest("fr1/room", name="my_room", output_dir=tmp_path)
    assert (name, dest) == ("my_room", (tmp_path / "my_room").resolve())


def test_resolve_url(tmp_path: Path) -> None:
    url = "https://example.com/scene_7.tgz"
    assert resolve_source_and_dest(url, output_dir=tmp_path) == (url, "scene_7", (tmp_path / "scene_7").resolve())


def test_resolve_absolute_name(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere" / "desk"
    assert resolve_source_and_dest("fr1/desk", name=str(target), output_dir=tmp_path)[2] == target.resolve()


def test_resolve_unknown_source_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown dataset source"):
        resolve_source_and_dest("not_a_preset_or_path", output_dir=tmp_path)


def test_default_output_dir_follows_data_dir(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("BASIR_DATA_DIR", str(tmp_path / "hdd"))
    dest = resolve_source_and_dest("fr1/desk")[2]
    assert dest == (tmp_path / "hdd" / "datasets" / "rgbd_dataset_freiburg1_desk").resolve()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _toy_archive(tmp_path: Path, folder: str = "seq", frames: int = 3) -> Path:
    raw = create_toy_dataset(tmp_path / f"raw_{folder}", num_frames=frames)
    archive = tmp_path / f"{folder}.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(raw, arcname=folder)
    return archive


def test_local_archive_end_to_end(tmp_path: Path) -> None:
    archive = _toy_archive(tmp_path, "rgbd_dataset_freiburg1_sample")
    dest = download_dataset(str(archive), name="my_sample", output_dir=tmp_path / "datasets")
    assert (dest / "associations.txt").is_file()
    meta = json.loads((dest / "dataset_meta.json").read_text())
    assert (meta["name"], meta["format"], meta["frames"]["associated"]) == ("my_sample", "tum_rgbd", 3)


def test_url_end_to_end(tmp_path: Path, file_server) -> None:
    file_server.files["seq.tgz"] = _toy_archive(tmp_path).read_bytes()
    dest = download_dataset(file_server.url("seq.tgz"), output_dir=tmp_path / "datasets", quiet=True)
    assert dest == (tmp_path / "datasets" / "seq").resolve()
    assert (tmp_path / "datasets" / "_archives" / "seq.tgz").is_file()
    assert json.loads((dest / "dataset_meta.json").read_text())["frames"]["associated"] == 3


def test_remove_archive_and_no_extract(tmp_path: Path, file_server) -> None:
    file_server.files["seq.tgz"] = _toy_archive(tmp_path).read_bytes()
    kept = download_dataset(file_server.url("seq.tgz"), name="a", output_dir=tmp_path / "d", extract=False, quiet=True)
    assert kept == tmp_path / "d" / "_archives" / "seq.tgz"
    download_dataset(file_server.url("seq.tgz"), name="b", output_dir=tmp_path / "d", remove_archive=True, quiet=True)
    assert not kept.exists()


def test_local_folder_is_copied(tmp_path: Path) -> None:
    raw = create_toy_dataset(tmp_path / "recording", num_frames=2)
    (raw / "dataset_meta.json").unlink()
    dest = download_dataset(str(raw), output_dir=tmp_path / "datasets")
    assert dest == (tmp_path / "datasets" / "recording").resolve()
    assert (dest / "rgb.txt").is_file() and (dest / "dataset_meta.json").is_file()


def test_half_extracted_destination_is_re_extracted(tmp_path: Path) -> None:
    archive = _toy_archive(tmp_path)
    leftover = tmp_path / "datasets" / "seq" / "rgb" / "partial.png"
    leftover.parent.mkdir(parents=True)
    leftover.write_bytes(b"half")
    dest = download_dataset(str(archive), output_dir=tmp_path / "datasets")
    assert (dest / "rgb.txt").is_file() and not leftover.exists()


def test_prepared_destination_is_kept(tmp_path: Path) -> None:
    archive = _toy_archive(tmp_path)
    dest = download_dataset(str(archive), output_dir=tmp_path / "datasets")
    (dest / "marker").write_text("keep")
    download_dataset(str(archive), output_dir=tmp_path / "datasets")
    assert (dest / "marker").exists()


def test_local_checksum_mismatch_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Checksum"):
        download_dataset(str(_toy_archive(tmp_path)), output_dir=tmp_path / "d", checksum="0" * 64)


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    dest = download_dataset("fr1/desk", output_dir=tmp_path, dry_run=True)
    assert dest == (tmp_path / "rgbd_dataset_freiburg1_desk").resolve()
    assert list(tmp_path.iterdir()) == []


def test_dataset_dir_resolution(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("BASIR_DATA_DIR", str(tmp_path))  # fr1/desk is not downloaded here
    toy = create_toy_dataset(tmp_path / "toy", num_frames=2)
    assert dataset_dir(str(toy)) == toy.resolve()
    with pytest.raises(FileNotFoundError, match="not downloaded yet"):
        dataset_dir("fr1/desk")
    with pytest.raises(FileNotFoundError, match="neither a folder"):
        dataset_dir("no_such_thing")
