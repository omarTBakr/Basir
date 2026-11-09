"""Smoke tests for the Phase 0 scripts on the toy dataset (headless, temporary MLflow store)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import replay, visualize_pointcloud
from scripts.download_dataset import create_toy_dataset, dataset_dir


@pytest.fixture
def toy(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> Path:
    clean_env.setenv("BASIR_DATA_DIR", str(tmp_path))  # MLflow store + outputs go to tmp
    return create_toy_dataset(tmp_path / "toy", num_frames=6)


def _runs(tmp_path: Path, experiment: str) -> list:
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    exp = client.get_experiment_by_name(experiment)
    return client.search_runs([exp.experiment_id]) if exp else []


def test_dataset_dir_resolution(toy: Path, tmp_path: Path) -> None:
    assert dataset_dir(str(toy)) == toy.resolve()
    with pytest.raises(FileNotFoundError, match="not downloaded yet"):
        dataset_dir("fr1/desk")
    with pytest.raises(FileNotFoundError, match="neither a folder"):
        dataset_dir("no_such_thing")


def test_replay_headless(toy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert replay.main([str(toy), "--speed", "0", "--no-window"]) == 0
    assert "Replayed 6 frames" in capsys.readouterr().out
    (run,) = _runs(tmp_path, "phase0-replay")
    assert run.data.metrics["frames"] == 6 and run.data.metrics["depth_valid_pct"] == 100.0


def test_replay_respects_speed(toy: Path) -> None:
    from basir.ingest.sources.tum import TUMDatasetSource

    stats = replay.replay(TUMDatasetSource(toy), speed=1.0, show=False)
    assert stats.wall_seconds >= 5 * 0.0333 * 0.9  # 6 frames span ~0.167 s of sensor time


def test_replay_missing_dataset_fails_cleanly(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("BASIR_DATA_DIR", str(tmp_path))
    assert replay.main(["fr1/desk", "--no-window"]) == 1


def test_render_shapes(toy: Path) -> None:
    from basir.ingest.sources.tum import TUMDatasetSource

    canvas = replay.render(TUMDatasetSource(toy)[0])
    assert canvas.shape == (48, 128, 3)


def test_visualize_pointcloud_headless(toy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import open3d as o3d

    assert visualize_pointcloud.main([str(toy), "--every", "1", "--pixel-stride", "1", "--voxel", "0", "--no-window"]) == 0
    out = capsys.readouterr().out
    assert "6 frames" in out and "Depth consistency" in out

    ply = tmp_path / "outputs" / "toy.ply"
    cloud = o3d.io.read_point_cloud(str(ply))
    points = np.asarray(cloud.points)
    assert len(points) == 6 * 64 * 48
    # Toy frames are fronto-parallel planes; the pose only translates, so each point's
    # world z is camera depth + camera tz (0.5 + 0.02 i).
    assert np.allclose(np.unique(np.round(points[:, 2], 4)), [2.0 + 0.07 * i for i in range(6)])
    assert (tmp_path / "outputs" / "toy.topdown.png").is_file()
    assert (tmp_path / "outputs" / "toy.trajectory.csv").read_text().startswith("timestamp,x,y,z")

    (run,) = _runs(tmp_path, "phase0-pointcloud")
    assert run.data.params["intrinsics_source"] == "intrinsics.json"
    assert run.data.metrics["frames_used"] == 6
    assert {"toy.ply", "toy.topdown.png", "toy.trajectory.csv"} <= {
        a.path for a in __import__("mlflow").tracking.MlflowClient(
            tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}").list_artifacts(run.info.run_id)
    }
