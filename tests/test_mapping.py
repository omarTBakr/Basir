"""Tests for basir/api/mapping.py: posed RGB-D frames from a phone grow a live map that browsers receive."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import threading
import time

from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect
import numpy as np
import pytest

from basir.api.framebus import QueuedFrame
from basir.api.mapping import MapManager, frame_points
from basir.api.phone_simulator import RecordedFrame, record
from basir.api.protocol import encode_frame
from basir.api.rgbd import unpack_points
from basir.ingest.datasets.toy import TOY_BASE_DEPTH_M, create_toy_dataset
from basir.ingest.sources.tum import TUMDatasetSource
from basir.spatial.live_map import VoxelMap
from tests.conftest import server_settings
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


@pytest.mark.slow
def test_save_writes_a_ply_and_an_mlflow_run_then_reset_clears(
    client: TestClient, toy_frames: list[RecordedFrame], tmp_path: Path
) -> None:
    maps = client.app.state.ingest.maps  # type: ignore[attr-defined]
    session = maps.session("saver")
    xyz, rgb = frame_points(queued(toy_frames[0], "saver"))
    session.vmap.insert(xyz, rgb)
    session.trail.append([0.0, 0.0, 0.0])

    saved = client.post("/api/map/saver/save").json()
    assert Path(saved["ply"]).is_file() and saved["ply"].startswith(str(tmp_path))
    assert saved["voxels"] == len(session.vmap) and saved["run_id"]
    runs = client.get("/api/map/runs").json()  # the app's own store (tmp_path), not the global settings' one
    assert [r["run_id"] for r in runs] == [saved["run_id"]]
    cloud = client.get(f"/api/map/runs/{saved['run_id']}/cloud")
    assert cloud.status_code == 200 and len(unpack_points(cloud.content)[0]) == saved["voxels"]

    assert client.post("/api/map/saver/reset").json() == {"reset": True}
    assert len(maps.sessions["saver"].vmap) == 0
    assert client.post("/api/map/saver/save").status_code == 404
    assert client.post("/api/map/nobody/reset").json() == {"reset": False}


@pytest.mark.slow
def test_save_logs_to_the_apps_store_not_the_global_one(
    client: TestClient, toy_frames: list[RecordedFrame], tmp_path: Path, clean_env: pytest.MonkeyPatch
) -> None:
    from mlflow.tracking import MlflowClient

    clean_env.setenv("BASIR_DATA_DIR", str(tmp_path / "global"))  # a different global store
    get_settings.cache_clear()
    global_uri = get_settings().mlflow_uri
    assert str(tmp_path / "global") in global_uri
    maps = client.app.state.ingest.maps  # type: ignore[attr-defined]
    session = maps.session("own-store")
    session.vmap.insert(*frame_points(queued(toy_frames[0], "own-store")))
    saved = client.post("/api/map/own-store/save").json()
    assert saved["run_id"]
    assert MlflowClient(global_uri).get_experiment_by_name("console-live-map") is None


@pytest.mark.slow
def test_opening_a_saved_cloud_leaves_no_temp_dir(
    client: TestClient, toy_frames: list[RecordedFrame], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile

    maps = client.app.state.ingest.maps  # type: ignore[attr-defined]
    maps.session("cloud").vmap.insert(*frame_points(queued(toy_frames[0], "cloud")))
    run_id = client.post("/api/map/cloud/save").json()["run_id"]
    scratch = tmp_path / "tmp"  # every temporary directory from here on is made in here
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    for _ in range(2):
        assert client.get(f"/api/map/runs/{run_id}/cloud").status_code == 200
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("device", ["bad id", "<img>", "x" * 65, "a%0Ab"])
def test_invalid_device_ids_create_no_map(client: TestClient, device: str) -> None:
    with pytest.raises(WebSocketDisconnect) as closed, client.websocket_connect(f"/ws/map/{device}") as ws:
        ws.receive_text()
    assert closed.value.code == 1008
    assert client.post(f"/api/map/{device}/reset").status_code == 404
    assert client.post(f"/api/map/{device}/save").status_code == 404
    assert client.get("/api/map/sessions").json() == []


def manager(tmp_path: Path, **overrides: object) -> MapManager:
    return MapManager(server_settings(tmp_path, **overrides))


def test_idle_maps_are_evicted_after_the_ttl(tmp_path: Path) -> None:
    maps = manager(tmp_path, map_idle_ttl_s=60)
    session = maps.session("gone")
    assert session is not None
    session.last_active = 1000.0
    maps.disconnected("gone")  # the phone left: the clock restarts now
    now = session.last_active
    assert maps.evict_idle(now + 59) == [] and "gone" in maps.sessions
    session.subscribers.add(asyncio.Queue())  # a browser still watching keeps it
    assert maps.evict_idle(now + 3600) == []
    session.subscribers.clear()
    assert maps.evict_idle(now + 61) == ["gone"] and maps.sessions == {}


def test_session_count_is_capped(tmp_path: Path) -> None:
    maps = manager(tmp_path, map_max_sessions=2)
    a, b = maps.session("a"), maps.session("b")
    assert a is not None and b is not None
    a.last_active, b.last_active = 1.0, 2.0
    assert maps.session("c") is not None  # the longest-idle map ("a") made room
    assert sorted(maps.sessions) == ["b", "c"]
    for s in maps.sessions.values():
        s.subscribers.add(asyncio.Queue())
    assert maps.session("d") is None  # every map is watched: no room
    assert maps.session("not valid") is None


def test_eviction_cancels_the_fuse_task(tmp_path: Path, toy_frames: list[RecordedFrame]) -> None:
    async def scenario() -> None:
        maps = manager(tmp_path, map_idle_ttl_s=1)
        maps.submit(queued(toy_frames[0], "phone"))
        task = maps.sessions["phone"].task
        assert task is not None
        await asyncio.sleep(0)
        assert maps.evict_idle(time.monotonic() + 10) == ["phone"]
        await asyncio.gather(task, return_exceptions=True)
        assert task.cancelled() or task.done()
        await maps.stop()

    asyncio.run(scenario())


def test_reset_waits_for_a_running_fuse_step(
    tmp_path: Path, toy_frames: list[RecordedFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reset during fusion must not swap the map under the fuse thread (that broadcast NaN points)."""
    entered, release = threading.Event(), threading.Event()
    real_fuse = MapManager._fuse

    def slow_fuse(vmap: VoxelMap, frame: QueuedFrame) -> tuple[np.ndarray, np.ndarray]:
        entered.set()
        release.wait(10)
        return real_fuse(vmap, frame)

    monkeypatch.setattr(MapManager, "_fuse", staticmethod(slow_fuse))

    async def scenario() -> list[bytes | str]:
        maps = manager(tmp_path)
        queue: asyncio.Queue[bytes | str] = asyncio.Queue()
        maps.submit(queued(toy_frames[0], "phone"))
        maps.sessions["phone"].subscribers.add(queue)
        await asyncio.to_thread(entered.wait, 10)
        reset = asyncio.create_task(maps.reset("phone"))
        await asyncio.sleep(0.05)
        assert not reset.done()  # blocked on the session lock while the fuse step runs
        release.set()
        assert await reset
        await maps.stop()
        return [queue.get_nowait() for _ in range(queue.qsize())]

    messages = asyncio.run(scenario())
    kinds = [json.loads(m)["type"] if isinstance(m, str) else "points" for m in messages]
    assert kinds == ["points", "stats", "reset"]  # the whole fuse step, then the reset; never interleaved
    xyz, _ = unpack_points(messages[0])  # type: ignore[arg-type]
    assert np.isfinite(xyz).all()
