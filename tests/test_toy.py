"""Tests for basir/ingest/datasets/toy.py."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import zlib

import pytest

from basir.ingest.datasets import toy
from basir.ingest.datasets.toy import create_toy_dataset
from basir.ingest.datasets.tum_format import TUM_DEPTH_FACTOR

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
    for name in (
        "rgb.txt",
        "depth.txt",
        "groundtruth.txt",
        "associations.txt",
        "associations_with_gt.txt",
        "intrinsics.json",
        "dataset_meta.json",
    ):
        assert (toy / name).is_file(), name
    assert len(list((toy / "rgb").glob("*.png"))) == 4


def test_toy_depth_uses_tum_factor(tmp_path: Path) -> None:
    toy = create_toy_dataset(tmp_path / "toy", num_frames=3)
    metres = [_first_depth_value(p) / TUM_DEPTH_FACTOR for p in sorted((toy / "depth").glob("*.png"))]
    assert metres == pytest.approx([1.5, 1.55, 1.6])


def test_toy_intrinsics_match_image_size(tmp_path: Path) -> None:
    intr = json.loads((create_toy_dataset(tmp_path / "toy", num_frames=2) / "intrinsics.json").read_text())
    assert (intr["width"], intr["height"]) == (toy.TOY_WIDTH, toy.TOY_HEIGHT)
    assert intr["cx"] == pytest.approx((toy.TOY_WIDTH - 1) / 2)
    assert intr["cy"] == pytest.approx((toy.TOY_HEIGHT - 1) / 2)
    assert intr["fx"] == pytest.approx(525.0 * toy.TOY_WIDTH / 640)


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
