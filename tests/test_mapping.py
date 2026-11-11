"""Tests for basir/api/mapping.py: posed RGB-D frames from a phone grow a live map that browsers receive."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

from basir.api.framebus import QueuedFrame
from basir.api.mapping import frame_points
from basir.api.phone_simulator import RecordedFrame, record
from basir.api.protocol import encode_frame
from basir.api.rgbd import unpack_points
from basir.ingest.datasets.toy import TOY_BASE_DEPTH_M, create_toy_dataset
from basir.ingest.sources.tum import TUMDatasetSource
from utilities.config import get_settings


@pytest.fixture
def toy_frames(tmp_path: Path) -> list[RecordedFrame]:
    return record(TUMDatasetSource(create_toy_dataset(tmp_path / "toy", num_frames=3)), depth_width=32)


def queued(frame: RecordedFrame, device: str = "phone-1") -> QueuedFrame:
    return QueuedFrame(device, frame.header, frame.jpeg, 0.0, frame.depth)


def test_frame_points_are_in_the_world_at_the_toy_depth(toy_frames: list[RecordedFrame]) -> None:
    first = toy_frames[0]
    xyz, rgb = frame_points(queued(first))
    assert len(xyz) == 32 * 24 and rgb.dtype == np.uint8  # depth (32x24) is smaller than colour (64x48)
    camera = np.asarray(first.header.pose).reshape(4, 4)
    forward = (xyz - camera[:3, 3]) @ camera[:3, 2]  # distance along the optical axis
    assert np.allclose(forward, TOY_BASE_DEPTH_M, atol=0.002)


def test_frames_without_depth_are_not_mapped(toy_frames: list[RecordedFrame]) -> None:
    with pytest.raises(ValueError):
        frame_points(QueuedFrame("p", toy_frames[0].header, toy_frames[0].jpeg, 0.0, None))


def receive_points(ws: object, until: int) -> tuple[int, dict]:
    """Read map messages until `until` points arrived; return the count and the last stats message."""
    received, stats = 0, {}
    while received < until:
        message = ws.receive()  # type: ignore[attr-defined]
        if message.get("bytes") is not None:
            received += len(unpack_points(message["bytes"])[0])
        elif message.get("text"):
            stats = json.loads(message["text"])
    return received, stats


def test_live_map_grows_and_streams_to_browsers(client: TestClient, toy_frames: list[RecordedFrame]) -> None:
    first = toy_frames[0]
    with client.websocket_connect("/ws/map/map-phone") as viewer:
        meta = json.loads(viewer.receive_text())
        assert meta["type"] == "meta" and meta["voxels"] == 0 and meta["trail"] == []
        with client.websocket_connect("/ws/stream/map-phone") as phone:
            phone.send_bytes(encode_frame(first.header, first.jpeg, first.depth))
            phone.receive_text()  # the inference result
            points, _ = receive_points(viewer, 1)
            stats = json.loads(viewer.receive_text())
    assert points > 0 and stats["type"] == "stats" and stats["keyframes"] == 1 and stats["voxels"] == points
    sessions = client.get("/api/map/sessions").json()
    assert sessions == [{"device": "map-phone", "voxels": points, "keyframes": 1, "fuse_ms": stats["fuse_ms"]}]

    with client.websocket_connect("/ws/map/map-phone") as late:  # a new browser gets the whole map first
        meta = json.loads(late.receive_text())
        assert meta["voxels"] == points and len(meta["trail"]) == 1
        assert receive_points(late, points)[0] == points
    assert "basir_keyframes_fused_total" in client.get("/metrics").text


def test_save_writes_a_ply_and_an_mlflow_run_then_reset_clears(
    client: TestClient, toy_frames: list[RecordedFrame], tmp_path: Path, clean_env: pytest.MonkeyPatch
) -> None:
    clean_env.setenv("BASIR_DATA_DIR", str(tmp_path))  # utilities.experiment logs with the global settings
    get_settings.cache_clear()
    maps = client.app.state.ingest.maps  # type: ignore[attr-defined]
    session = maps.session("saver")
    xyz, rgb = frame_points(queued(toy_frames[0], "saver"))
    session.vmap.insert(xyz, rgb)
    session.trail.append([0.0, 0.0, 0.0])

    saved = client.post("/api/map/saver/save").json()
    assert Path(saved["ply"]).is_file() and saved["ply"].startswith(str(tmp_path))
    assert saved["voxels"] == len(session.vmap) and saved["run_id"]
    runs = client.get("/api/map/runs").json()
    assert [r["run_id"] for r in runs] == [saved["run_id"]]
    cloud = client.get(f"/api/map/runs/{saved['run_id']}/cloud")
    assert cloud.status_code == 200 and len(unpack_points(cloud.content)[0]) == saved["voxels"]

    assert client.post("/api/map/saver/reset").json() == {"reset": True}
    assert len(maps.sessions["saver"].vmap) == 0
    assert client.post("/api/map/saver/save").status_code == 404
    assert client.post("/api/map/nobody/reset").json() == {"reset": False}
