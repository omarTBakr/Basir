"""Tests for basir/ingest/sources (TUM frame source and file formats)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from basir.ingest.datasets.toy import create_toy_dataset
from basir.ingest.datasets.tum_format import TUM_CALIBRATION, calibration_for, parse_tum_file
from basir.ingest.sources.base import FrameSource
from basir.ingest.sources.tum import TUMDatasetSource


@pytest.fixture
def toy(tmp_path: Path) -> Path:
    return create_toy_dataset(tmp_path / "toy", num_frames=6)


def test_is_a_frame_source(toy: Path) -> None:
    assert isinstance(TUMDatasetSource(toy), FrameSource)


def test_frames_depth_rgb_and_poses(toy: Path) -> None:
    source = TUMDatasetSource(toy)
    frames = list(source)
    assert len(source) == len(frames) == 6
    assert [f.id for f in frames] == list(range(6))
    assert np.allclose([f.depth[0, 0] for f in frames], [1.5, 1.55, 1.6, 1.65, 1.7, 1.75])
    first = frames[0]
    assert first.rgb.shape == (48, 64, 3) and first.rgb.dtype == np.uint8
    assert tuple(first.rgb[0, 0]) == (100, 150, 200)  # RGB order, not OpenCV's BGR
    assert first.depth.dtype == np.float32
    # Poses match groundtruth.txt (tx = 0.05 * i, tz = 0.5 + 0.02 * i)
    assert np.allclose(frames[2].pose.position, [0.10, 0.02, 0.54])
    assert all(np.diff([f.timestamp for f in frames]) > 0)


def test_stride_and_max_frames(toy: Path) -> None:
    assert len(TUMDatasetSource(toy, stride=2)) == 3
    assert len(TUMDatasetSource(toy, max_frames=4)) == 4
    assert np.isclose(list(TUMDatasetSource(toy, stride=2))[1].depth[0, 0], 1.6)
    with pytest.raises(ValueError):
        TUMDatasetSource(toy, stride=0)


def test_intrinsics_from_dataset_file(toy: Path) -> None:
    source = TUMDatasetSource(toy)
    assert source.intrinsics_source == "intrinsics.json"
    assert (source.intrinsics.width, source.intrinsics.height) == (64, 48)
    assert source.intrinsics.cx == pytest.approx(31.5)


@pytest.mark.parametrize("sensor", ["freiburg1", "freiburg2", "freiburg3"])
def test_calibration_from_folder_name(tmp_path: Path, sensor: str) -> None:
    folder = tmp_path / f"rgbd_dataset_{sensor}_desk"
    folder.mkdir()
    calib, source = calibration_for(folder)
    assert source == sensor
    assert calib["fx"] == TUM_CALIBRATION[sensor]["fx"]
    assert (calib["width"], calib["height"], calib["depth_scale"]) == (640, 480, 5000.0)


def test_falls_back_to_settings_intrinsics(toy: Path, tmp_path: Path) -> None:
    other = tmp_path / "my_recording"
    toy.rename(other)
    (other / "intrinsics.json").unlink()
    source = TUMDatasetSource(other)
    assert source.intrinsics_source == "settings_default"
    assert source.intrinsics.fx == 525.0


def test_without_poses(toy: Path) -> None:
    (toy / "associations_with_gt.txt").unlink()
    with pytest.raises(ValueError, match="No usable frames"):
        TUMDatasetSource(toy)
    frames = list(TUMDatasetSource(toy, require_pose=False))
    assert len(frames) == 6 and frames[0].pose is None


def test_missing_folder_and_unreadable_image(tmp_path: Path, toy: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TUMDatasetSource(tmp_path / "nope")
    next((toy / "rgb").glob("*.png")).unlink()
    with pytest.raises(OSError, match="Could not read frame"):
        list(TUMDatasetSource(toy))


def test_close_is_safe(toy: Path) -> None:
    source = TUMDatasetSource(toy, max_frames=1)
    assert len(list(source)) == 1
    source.close()


def test_parse_tum_file(tmp_path: Path) -> None:
    f = tmp_path / "rgb.txt"
    f.write_text("# header\n1.5 rgb/a.png\nbad line\n2.0 rgb/b.png\n")
    assert parse_tum_file(f) == {1.5: "rgb/a.png", 2.0: "rgb/b.png"}
    assert parse_tum_file(tmp_path / "missing.txt") == {}
