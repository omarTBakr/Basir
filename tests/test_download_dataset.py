"""Tests for scripts/download_dataset.py (TUM presets, association, toy data, pipeline, CLI)."""

from __future__ import annotations

import io
import json
from pathlib import Path
import struct
import tarfile
import zlib

import pytest

import scripts.download_dataset as dd
from scripts.download_dataset import (
    PRESET_ALIASES,
    PRESETS,
    associate_tum_dataset,
    build_parser,
    create_toy_dataset,
    download_dataset,
    get_preset_info,
    main,
    resolve_preset_key,
    resolve_source_and_dest,
)

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "identifier, key",
    [
        ("fr1/room", "fr1/room"),
        ("FR1/Room", "fr1/room"),
        ("fr1_room", "fr1/room"),
        ("fr1-room", "fr1/room"),
        ("tum_fr1_room", "fr1/room"),
        ("freiburg1_room", "fr1/room"),
        ("rgbd_dataset_freiburg1_room", "fr1/room"),
        ("tum_fr2_desk", "fr2/desk"),
        ("fr3_long_office", "fr3/long_office"),
        ("rgbd_dataset_freiburg3_long_office_household", "fr3/long_office"),
        ("rgbd_dataset_freiburg1_360", "fr1/360"),
    ],
)
def test_preset_keys_and_aliases(identifier: str, key: str) -> None:
    assert resolve_preset_key(identifier) == key


@pytest.mark.parametrize("identifier", ["unknown_dataset_xyz", "", None])
def test_unknown_preset(identifier: str | None) -> None:
    assert resolve_preset_key(identifier) is None
    assert get_preset_info(identifier) is None


def test_presets_use_current_tum_host_and_match_names() -> None:
    for key, info in PRESETS.items():
        assert info["url"].startswith(dd.TUM_BASE_URL + "/freiburg"), key
        assert info["url"].endswith(f"/{info['name']}.tgz"), key


def test_fr1_360_points_at_the_real_file() -> None:
    assert get_preset_info("fr1/360")["url"].endswith("/freiburg1/rgbd_dataset_freiburg1_360.tgz")


def test_every_alias_resolves_to_a_preset() -> None:
    assert set(PRESET_ALIASES.values()) == set(PRESETS)


@pytest.mark.network
@pytest.mark.parametrize("key", sorted(PRESETS))
def test_preset_url_is_live(key: str) -> None:
    import urllib.request

    request = urllib.request.Request(PRESETS[key]["url"], method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as res:
        assert res.status == 200


# ---------------------------------------------------------------------------
# Association
# ---------------------------------------------------------------------------


def test_associate_frames(tmp_path: Path) -> None:
    (tmp_path / "rgb.txt").write_text("# c\n1.000000 rgb/1.png\n1.033000 rgb/2.png\n1.066000 rgb/3.png\n")
    (tmp_path / "depth.txt").write_text("# c\n1.002000 d/1.png\n1.035000 d/2.png\n1.068000 d/3.png\n")
    (tmp_path / "groundtruth.txt").write_text(
        "# c\n1.000000 0.1 0.2 0.3 0 0 0 1\n1.033000 0.2 0.2 0.3 0 0 0 1\n1.066000 0.3 0.2 0.3 0 0 0 1\n"
    )
    assoc, assoc_gt = associate_tum_dataset(tmp_path, max_difference=0.02)

    rows = [line.split() for line in assoc.read_text().splitlines() if not line.startswith("#")]
    assert rows[0] == ["1.000000", "rgb/1.png", "1.002000", "d/1.png"]
    assert len(rows) == 3
    gt_rows = [line.split() for line in assoc_gt.read_text().splitlines() if not line.startswith("#")]
    assert len(gt_rows) == 3 and gt_rows[1][5] == "0.2"


def test_association_respects_max_difference(tmp_path: Path) -> None:
    (tmp_path / "rgb.txt").write_text("1.000 rgb/1.png\n2.000 rgb/2.png\n")
    (tmp_path / "depth.txt").write_text("1.010 d/1.png\n2.500 d/2.png\n")  # second is 0.5 s away
    assoc, assoc_gt = associate_tum_dataset(tmp_path, max_difference=0.02)
    rows = [line for line in assoc.read_text().splitlines() if not line.startswith("#")]
    assert len(rows) == 1
    assert assoc_gt is None  # no groundtruth.txt


def test_association_needs_rgb_and_depth(tmp_path: Path) -> None:
    assert associate_tum_dataset(tmp_path) == (None, None)


# ---------------------------------------------------------------------------
# Toy dataset
# ---------------------------------------------------------------------------


def _first_depth_value(path: Path) -> int:
    data = path.read_bytes()
    pos, idat = 8, b""
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        if data[pos + 4 : pos + 8] == b"IDAT":
            idat += data[pos + 8 : pos + 8 + length]
        pos += 12 + length
    return struct.unpack(">H", zlib.decompress(idat)[1:3])[0]


def test_toy_dataset_files(tmp_path: Path) -> None:
    toy = create_toy_dataset(tmp_path / "toy", num_frames=4)
    for name in ("rgb.txt", "depth.txt", "groundtruth.txt", "associations.txt",
                 "associations_with_gt.txt", "intrinsics.json", "dataset_meta.json"):
        assert (toy / name).is_file(), name
    assert len(list((toy / "rgb").glob("*.png"))) == 4


def test_toy_depth_uses_tum_factor(tmp_path: Path) -> None:
    toy = create_toy_dataset(tmp_path / "toy", num_frames=3)
    metres = [_first_depth_value(p) / dd.TUM_DEPTH_FACTOR for p in sorted((toy / "depth").glob("*.png"))]
    assert metres == pytest.approx([1.5, 1.55, 1.6])


def test_toy_intrinsics_match_image_size(tmp_path: Path) -> None:
    intr = json.loads((create_toy_dataset(tmp_path / "toy", num_frames=2) / "intrinsics.json").read_text())
    assert (intr["width"], intr["height"]) == (dd.TOY_WIDTH, dd.TOY_HEIGHT)
    assert intr["cx"] == pytest.approx((dd.TOY_WIDTH - 1) / 2)
    assert intr["cy"] == pytest.approx((dd.TOY_HEIGHT - 1) / 2)
    assert intr["fx"] == pytest.approx(525.0 * dd.TOY_WIDTH / 640)


def test_toy_metadata(tmp_path: Path) -> None:
    meta = json.loads((create_toy_dataset(tmp_path / "toy", num_frames=7) / "dataset_meta.json").read_text())
    assert meta["format"] == "tum_rgbd"
    assert meta["frames"] == {"rgb": 7, "depth": 7, "associated": 7}
    assert meta["has_groundtruth"] is True
    assert "7-frame" in meta["description"]


def test_toy_dataset_kept_unless_overwrite(tmp_path: Path) -> None:
    toy = create_toy_dataset(tmp_path / "toy", num_frames=2)
    (toy / "marker").write_text("keep")
    create_toy_dataset(toy, num_frames=4)
    assert (toy / "marker").exists()
    create_toy_dataset(toy, num_frames=4, overwrite=True)
    assert not (toy / "marker").exists()


# ---------------------------------------------------------------------------
# Source / destination resolution
# ---------------------------------------------------------------------------


def test_resolve_preset(tmp_path: Path) -> None:
    src, name, dest = resolve_source_and_dest("fr1/room", output_dir=tmp_path)
    assert (src, name, dest) == (PRESETS["fr1/room"]["url"], "rgbd_dataset_freiburg1_room",
                                 (tmp_path / "rgbd_dataset_freiburg1_room").resolve())


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_parser() -> None:
    args = build_parser().parse_args(["fr1/room", "--name", "room", "--dry-run"])
    assert (args.source, args.name, args.dry_run, args.output_dir) == ("fr1/room", "room", True, None)


def test_main_creates_toy_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--create-toy-dataset", "--name", "toy_room", "-o", str(tmp_path)]) == 0
    assert (tmp_path / "toy_room" / "intrinsics.json").is_file()
    assert "Dataset ready at" in capsys.readouterr().out


def test_main_list_presets(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list-presets"]) == 0
    assert "fr1/room" in capsys.readouterr().out


def test_main_reports_failure(tmp_path: Path) -> None:
    assert main(["not_a_preset_or_path", "-o", str(tmp_path), "-q"]) == 1


def test_main_without_source_is_a_usage_error() -> None:
    with pytest.raises(SystemExit):
        main([])
