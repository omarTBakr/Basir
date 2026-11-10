"""Tests for basir/spatial/pointcloud.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from basir.ingest.datasets.toy import create_toy_dataset
from basir.ingest.sources.tum import TUMDatasetSource
from basir.spatial.pointcloud import average_consistency, build_cloud, top_down_image


@pytest.fixture
def frames(tmp_path: Path) -> list:
    return list(TUMDatasetSource(create_toy_dataset(tmp_path / "toy", num_frames=4)))


def test_build_cloud_counts_and_colours(frames: list) -> None:
    cloud, raw = build_cloud(frames, pixel_stride=1, voxel=0)
    assert raw == len(cloud.points) == 4 * 64 * 48
    colors = np.asarray(cloud.colors)
    assert colors.min() >= 0 and colors.max() <= 1
    # world z = camera depth (1.5 + 0.05 i) + camera tz (0.5 + 0.02 i)
    assert np.allclose(np.unique(np.round(np.asarray(cloud.points)[:, 2], 4)), [2.0 + 0.07 * i for i in range(4)])


def test_voxel_downsampling_reduces_points(frames: list) -> None:
    dense, raw = build_cloud(frames, pixel_stride=1, voxel=0)
    sparse, raw2 = build_cloud(frames, pixel_stride=1, voxel=0.2)
    assert raw == raw2 and len(sparse.points) < len(dense.points)


def test_average_consistency_of_identical_frames_is_zero(frames: list) -> None:
    score, pairs = average_consistency([frames[0], frames[0], frames[0]], gap=1)
    assert pairs == 2
    assert score.median_cm == pytest.approx(0, abs=1e-6) and score.within_5cm == 1.0


def test_average_consistency_without_overlap(frames: list) -> None:
    assert average_consistency(frames[:1], gap=1) == (None, 0)


def test_top_down_image(frames: list) -> None:
    cloud, _ = build_cloud(frames, pixel_stride=2, voxel=0)
    image = top_down_image(cloud, frames, size=200)
    assert image.shape == (200, 200, 3) and image.dtype == np.uint8
    assert (image[..., 2] == 255).any()  # the red camera path is drawn
