"""Live map: fuse each phone's posed RGB-D keyframes into a voxel map and stream the new voxels to browsers.

Fusion runs in a thread per phone, off the ingest path; only the newest waiting frame is
fused (latest wins), and only when the camera moved >= MAP_KEYFRAME_M or turned >= MAP_KEYFRAME_DEG.

Browser socket `/ws/map/{device_id}` (Console listener only). Server -> browser:

    text   {"type": "meta", "voxels", "keyframes", "voxel_m", "trail": [[x, y, z], ...]}   on connect
    binary b"P" + u32 n + float32 xyz * n + uint8 rgb * n                                     points
    text   {"type": "stats", "voxels", "keyframes", "camera": [x, y, z], "fuse_ms"}           per keyframe
    text   {"type": "reset"}                                                                  map cleared

The first `P` messages after `meta` are the whole map; later ones are only the new voxels.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any

import cv2
from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
import numpy as np

from basir.api.framebus import QueuedFrame
from basir.api.metrics import KEYFRAMES_FUSED, MAP_UPDATE_SECONDS, MAP_VOXELS, VIEWERS
from basir.api.rgbd import decode_colour, decode_depth, pack_points
from basir.core.geometry import depth_to_points, transform_points
from basir.spatial.live_map import KeyframeGate, VoxelMap
from utilities.config import Settings

logger = logging.getLogger("basir.mapping")

MIN_DEPTH_M, MAX_DEPTH_M = 0.2, 4.0
TARGET_COLUMNS = 160  # pixel stride is chosen so about this many columns are fused per keyframe
SNAPSHOT_CHUNK = 200_000  # points per binary message when sending the whole map
SUBSCRIBER_QUEUE = 64
SAVED_CLOUD_MAX_POINTS = 600_000
MAP_EXPERIMENT = "console-live-map"
CLOUD_EXPERIMENTS = ("phase0-pointcloud", MAP_EXPERIMENT)

router = APIRouter()


def frame_points(frame: QueuedFrame) -> tuple[np.ndarray, np.ndarray]:
    """World points and colours of one posed RGB-D frame, sampled at the depth image's resolution."""
    header = frame.header
    if frame.depth is None or header.pose is None or header.intrinsics is None:
        raise ValueError("frame needs depth, pose and intrinsics")
    depth = decode_depth(frame.depth, header)
    rgb = decode_colour(frame.jpeg)
    h, w = depth.shape
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    k = header.intrinsics
    sx, sy = w / k["width"], h / k["height"]
    K = np.array(
        [[k["fx"] * sx, 0, (k["cx"] + 0.5) * sx - 0.5], [0, k["fy"] * sy, (k["cy"] + 0.5) * sy - 0.5], [0, 0, 1]]
    )
    stride = max(1, round(w / TARGET_COLUMNS))
    points, rows, cols = depth_to_points(depth, K, stride, MIN_DEPTH_M, MAX_DEPTH_M)
    pose = np.asarray(header.pose, dtype=np.float64).reshape(4, 4)
    return transform_points(pose, points), rgb[rows, cols]


@dataclass
class MapSession:
    """One phone's live map and the browsers watching it."""

    device: str
    vmap: VoxelMap
    gate: KeyframeGate
    trail: list[list[float]] = field(default_factory=list)
    keyframes: int = 0
    fuse_ms: float = 0.0
    pending: QueuedFrame | None = None
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    subscribers: set[asyncio.Queue[bytes | str]] = field(default_factory=set)
    task: asyncio.Task[None] | None = None

    def meta(self) -> dict[str, Any]:
        return {
            "type": "meta",
            "device": self.device,
            "voxels": len(self.vmap),
            "keyframes": self.keyframes,
            "voxel_m": self.vmap.voxel,
            "trail": self.trail,
        }

    def broadcast(self, message: bytes | str) -> None:
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:  # a stalled browser: tell it to reconnect for a fresh snapshot
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(json.dumps({"type": "resync"}))


class MapManager:
    """Live maps for every phone that sends posed RGB-D frames."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.sessions: dict[str, MapSession] = {}

    def session(self, device: str) -> MapSession:
        if device not in self.sessions:
            s = self.settings
            self.sessions[device] = MapSession(
                device, VoxelMap(s.map_voxel_m, s.map_max_points), KeyframeGate(s.map_keyframe_m, s.map_keyframe_deg)
            )
        return self.sessions[device]

    def submit(self, frame: QueuedFrame) -> None:
        """Offer a frame for fusion; frames without depth, pose or intrinsics are ignored."""
        header = frame.header
        if frame.depth is None or header.pose is None or header.intrinsics is None:
            return
        session = self.session(frame.device)
        session.pending = frame
        session.wake.set()
        if session.task is None or session.task.done():
            session.task = asyncio.create_task(self._fuse_loop(session))

    async def _fuse_loop(self, session: MapSession) -> None:
        while True:
            await session.wake.wait()
            session.wake.clear()
            frame, session.pending = session.pending, None
            if frame is None or frame.header.pose is None:
                continue
            pose = np.asarray(frame.header.pose).reshape(4, 4)
            if not session.gate.accept(pose):
                continue
            started = time.perf_counter()
            try:
                xyz, rgb = await asyncio.to_thread(self._fuse, session, frame)
            except Exception:
                logger.exception("Fusing a keyframe from %s failed", session.device)
                continue
            seconds = time.perf_counter() - started
            session.keyframes += 1
            session.fuse_ms = round(seconds * 1000, 1)
            camera = [round(float(v), 4) for v in pose[:3, 3]]
            session.trail.append(camera)
            MAP_UPDATE_SECONDS.observe(seconds)
            KEYFRAMES_FUSED.labels(session.device).inc()
            MAP_VOXELS.labels(session.device).set(len(session.vmap))
            if len(xyz):
                session.broadcast(pack_points(xyz, rgb))
            session.broadcast(
                json.dumps(
                    {
                        "type": "stats",
                        "voxels": len(session.vmap),
                        "keyframes": session.keyframes,
                        "camera": camera,
                        "fuse_ms": session.fuse_ms,
                    }
                )
            )

    @staticmethod
    def _fuse(session: MapSession, frame: QueuedFrame) -> tuple[np.ndarray, np.ndarray]:
        points, colours = frame_points(frame)
        return session.vmap.points(session.vmap.insert(points, colours))

    def reset(self, device: str) -> bool:
        session = self.sessions.get(device)
        if session is None:
            return False
        s = self.settings
        session.vmap = VoxelMap(s.map_voxel_m, s.map_max_points)
        session.gate.reset()
        session.trail.clear()
        session.keyframes = 0
        MAP_VOXELS.labels(device).set(0)
        session.broadcast(json.dumps({"type": "reset"}))
        return True

    def save(self, device: str) -> dict[str, Any]:
        """Write the map (.ply) and camera trail (.csv) and log them as an MLflow run. Blocking."""
        import open3d as o3d

        from utilities.experiment import log_artifact, log_metrics, log_params, start_run

        session = self.sessions.get(device)
        if session is None or not len(session.vmap):
            raise ValueError(f"no live map for {device}")
        xyz, rgb = session.vmap.points()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_dir: Path = self.settings.live_map_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        ply, csv = out_dir / f"{device}-{stamp}.ply", out_dir / f"{device}-{stamp}.trajectory.csv"
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
        cloud.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255)
        o3d.io.write_point_cloud(ply, cloud)
        np.savetxt(csv, np.array(session.trail).reshape(-1, 3), delimiter=",", header="x,y,z", comments="", fmt="%.4f")
        with start_run(MAP_EXPERIMENT, run_name=f"{device}-{stamp}", tags={"device": device}) as run_id:
            s = self.settings
            log_params({"voxel_m": s.map_voxel_m, "keyframe_m": s.map_keyframe_m, "keyframe_deg": s.map_keyframe_deg})
            log_metrics({"voxels": len(session.vmap), "keyframes": session.keyframes})
            log_artifact(ply)
            log_artifact(csv)
        return {"ply": str(ply), "run_id": run_id, "voxels": len(session.vmap)}

    async def stop(self) -> None:
        tasks = [s.task for s in self.sessions.values() if s.task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _maps(request: Request | WebSocket) -> MapManager:
    return request.app.state.ingest.maps


@router.websocket("/ws/map/{device_id}")
async def map_socket(websocket: WebSocket, device_id: str) -> None:
    """Send the whole map, then new voxels and stats as keyframes are fused."""
    session = _maps(websocket).session(device_id)
    queue: asyncio.Queue[bytes | str] = asyncio.Queue(SUBSCRIBER_QUEUE)
    await websocket.accept()
    VIEWERS.labels("map").inc()
    session.subscribers.add(queue)
    closed: asyncio.Task[None] | None = None
    try:
        await websocket.send_text(json.dumps(session.meta()))
        xyz, rgb = session.vmap.points()
        for start in range(0, len(xyz), SNAPSHOT_CHUNK):
            await websocket.send_bytes(
                pack_points(xyz[start : start + SNAPSHOT_CHUNK], rgb[start : start + SNAPSHOT_CHUNK])
            )
        closed = asyncio.create_task(_wait_closed(websocket))
        while True:
            getter = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait({getter, closed}, return_when=asyncio.FIRST_COMPLETED)
            if closed in done:
                getter.cancel()
                return
            message = getter.result()
            if isinstance(message, bytes):
                await websocket.send_bytes(message)
            else:
                await websocket.send_text(message)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if closed is not None:
            closed.cancel()
        session.subscribers.discard(queue)
        VIEWERS.labels("map").dec()


async def _wait_closed(websocket: WebSocket) -> None:
    """Return when the browser disconnects (it sends nothing else on this socket)."""
    while (await websocket.receive())["type"] != "websocket.disconnect":
        pass


@router.get("/api/map/sessions")
def map_sessions(request: Request) -> list[dict[str, Any]]:
    return [
        {"device": s.device, "voxels": len(s.vmap), "keyframes": s.keyframes, "fuse_ms": s.fuse_ms}
        for s in _maps(request).sessions.values()
    ]


@router.post("/api/map/{device_id}/reset")
def map_reset(request: Request, device_id: str) -> dict[str, bool]:
    return {"reset": _maps(request).reset(device_id)}


@router.post("/api/map/{device_id}/save")
async def map_save(request: Request, device_id: str) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(_maps(request).save, device_id)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e


@router.get("/api/map/runs")
async def map_runs(request: Request) -> list[dict[str, Any]]:
    """Recent MLflow runs of the point-cloud experiments (their .ply can be opened in the 3D map)."""
    return await asyncio.to_thread(_list_cloud_runs, request.app.state.settings.mlflow_uri)


@router.get("/api/map/runs/{run_id}/cloud")
async def map_run_cloud(request: Request, run_id: str) -> Response:
    """A saved run's first .ply artifact as one binary points message (voxel-limited)."""
    try:
        body = await asyncio.to_thread(_load_run_cloud, request.app.state.settings.mlflow_uri, run_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, str(e)) from e
    return Response(body, media_type="application/octet-stream")


def _list_cloud_runs(tracking_uri: str) -> list[dict[str, Any]]:
    from mlflow.tracking import MlflowClient

    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        ids = [e.experiment_id for name in CLOUD_EXPERIMENTS if (e := client.get_experiment_by_name(name))]
        runs = client.search_runs(ids, max_results=30, order_by=["attributes.start_time DESC"]) if ids else []
    except Exception as e:
        logger.warning("Listing point-cloud runs failed: %s", e)
        return []
    return [
        {
            "run_id": r.info.run_id,
            "name": r.info.run_name,
            "started": r.info.start_time,
            "points": r.data.metrics.get("points_downsampled") or r.data.metrics.get("voxels"),
        }
        for r in runs
    ]


def _load_run_cloud(tracking_uri: str, run_id: str) -> bytes:
    import mlflow
    import open3d as o3d

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    ply = next((a.path for a in client.list_artifacts(run_id) if a.path.endswith(".ply")), None)
    if ply is None:
        raise FileNotFoundError(f"run {run_id} has no .ply artifact")
    local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=ply, tracking_uri=tracking_uri)
    cloud = o3d.io.read_point_cloud(Path(local))
    voxel = 0.01
    while len(cloud.points) > SAVED_CLOUD_MAX_POINTS:
        cloud, voxel = cloud.voxel_down_sample(voxel), voxel * 1.5
    xyz = np.asarray(cloud.points, np.float32)
    rgb = (np.asarray(cloud.colors) * 255).astype(np.uint8) if cloud.has_colors() else np.full_like(xyz, 200, np.uint8)
    return pack_points(xyz, rgb)
