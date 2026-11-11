"""Tests for basir/spatial/fusion.py (Phase 0 targets, fusion and saved outputs)."""

from __future__ import annotations

from pathlib import Path

import pytest

from basir.ingest.datasets.toy import create_toy_dataset
from basir.ingest.sources.tum import TUMDatasetSource
from basir.spatial.fusion import CloudResult, fuse, meets_targets, save_outputs


@pytest.mark.parametrize(
    ("median", "within", "expected"),
    [(1.9, 0.84, True), (3.0, 0.80, True), (3.01, 0.9, False), (1.0, 0.79, False), (None, 0.9, False)],
)
def test_meets_targets(median: float | None, within: float | None, expected: bool) -> None:
    assert meets_targets(median, within) is expected


def test_cloud_result_metrics_skip_missing_scores() -> None:
    result = CloudResult(3, 100, 50, 0.1, None, None, None, 0)
    assert not result.passes
    assert result.metrics() == {
        "frames_used": 3,
        "points_raw": 100,
        "points_downsampled": 50,
        "seconds": 0.1,
        "consistency_pairs": 0,
        "passes": 0.0,
    }


def test_fuse_and_save(tmp_path: Path) -> None:
    frames = list(TUMDatasetSource(create_toy_dataset(tmp_path / "toy", num_frames=4)))
    cloud, result = fuse(frames, pixel_stride=1, voxel=0)
    assert result.frames_used == 4 and result.points_raw == result.points_downsampled == len(cloud.points)
    assert result.consistency_pairs == 3

    outputs = save_outputs(cloud, frames, tmp_path / "out" / "toy.ply")
    assert [p.name for p in outputs] == ["toy.ply", "toy.topdown.png", "toy.trajectory.csv"]
    assert all(p.is_file() for p in outputs)
