"""Live map: fuse each phone's posed RGB-D keyframes into a voxel map and stream the new voxels to browsers.

Fusion runs in a thread per phone, off the ingest path; only the newest waiting frame is
fused (latest wins), and only when the camera moved >= MAP_KEYFRAME_M or turned >= MAP_KEYFRAME_DEG.
A map lives until it has had no frames and no browsers for MAP_IDLE_TTL_S; at most
MAP_MAX_SESSIONS maps are kept.

Browser socket `/ws/map/{device_id}` (Console listener only; invalid ids are closed with 1008).
Server -> browser:

    text   {"type": "meta", "voxels", "keyframes", "voxel_m", "trail": [[x, y, z], ...]}   on connect
    binary b"P" + u32 n + float32 xyz * n + uint8 rgb * n                                     points
    text   {"type": "stats", "voxels", "keyframes", "camera": [x, y, z], "fuse_ms"}           per keyframe
    text   {"type": "reset"}                                                                  map cleared

The first `P` messages after `meta` are the whole map; later ones are only the new voxels.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import tempfile
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
import numpy as np

from basir.api.framebus import QueuedFrame
from basir.api.metrics import KEYFRAMES_FUSED, MAP_UPDATE_SECONDS, MAP_VOXELS, VIEWERS
from basir.api.protocol import valid_device_id
from basir.api.rgbd import pack_points
from basir.core.geometry import depth_to_points, transform_points
from basir.ingest.phone_frame import frame_from_message
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
POLICY_VIOLATION, TRY_AGAIN_LATER = 1008, 1013  # WebSocket close codes

router = APIRouter()


def frame_points(frame: QueuedFrame) -> tuple[np.ndarray, np.ndarray]:
    """World points and colours of one posed RGB-D frame, sampled at the depth image's resolution."""
    if frame.depth is None or frame.header.pose is None or frame.header.intrinsics is None:
        raise ValueError("frame needs depth, pose and intrinsics")
    posed = frame_from_message(frame.header, frame.jpeg, frame.depth)
    assert posed.depth is not None and posed.pose is not None
    stride = max(1, round(posed.intrinsics.width / TARGET_COLUMNS))
    points, rows, cols = depth_to_points(posed.depth, posed.intrinsics.K, stride, MIN_DEPTH_M, MAX_DEPTH_M)
    return transform_points(posed.pose.matrix, points), posed.rgb[rows, cols]


@dataclass
class MapSession:
    """One phone's live map and the browsers watching it.

    `lock` serialises everything that reads or replaces `vmap`: a fuse step and its broadcast,
    reset, save and a browser's snapshot, so a reset never lands in the middle of a fuse step.
    """

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
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_active: float = field(default_factory=time.monotonic)  # last frame, or last browser leaving

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

    def idle_for(self, now: float) -> float:
        """Seconds without frames and without browsers (0 while a browser watches)."""
        return 0.0 if self.subscribers else now - self.last_active


class MapManager:
    """Live maps for every phone that sends posed RGB-D frames.

    A session is evicted (fuse task cancelled, map freed) once it has had no frames and no
    browsers for `map_idle_ttl_s`, e.g. that long after its phone disconnected. At most
    `map_max_sessions` are kept: the longest-idle unwatched one makes room for a new phone.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.sessions: dict[str, MapSession] = {}
        self._janitor: asyncio.Task[None] | None = None

    def session(self, device: str) -> MapSession | None:
        """The device's session, created if needed; None for an invalid id or when every slot is watched."""
        if not valid_device_id(device):
            return None
        if device not in self.sessions:
            self.evict_idle()
            if len(self.sessions) >= self.settings.map_max_sessions and not self._evict_one():
                logger.warning("No room for a live map for %s (%d sessions)", device, len(self.sessions))
                return None
            s = self.settings
            self.sessions[device] = MapSession(
                device, VoxelMap(s.map_voxel_m, s.map_max_points), KeyframeGate(s.map_keyframe_m, s.map_keyframe_deg)
            )
            self._start_janitor()
        return self.sessions[device]

    def submit(self, frame: QueuedFrame) -> None:
        """Offer a frame for fusion; frames without depth, pose or intrinsics are ignored."""
        header = frame.header
        if frame.depth is None or header.pose is None or header.intrinsics is None:
            return
        session = self.session(frame.device)
        if session is None:
            return
        session.pending = frame
        session.last_active = time.monotonic()
        session.wake.set()
        if session.task is None or session.task.done():
            session.task = asyncio.create_task(self._fuse_loop(session))

    def disconnected(self, device: str) -> None:
        """The phone left: its map stays for `map_idle_ttl_s` (so it can still be saved), then is evicted."""
        if (session := self.sessions.get(device)) is not None:
            session.last_active = time.monotonic()

    def evict_idle(self, now: float | None = None) -> list[str]:
        """Evict every session idle for longer than `map_idle_ttl_s`; return their device ids."""
        now = time.monotonic() if now is None else now
        ttl = self.settings.map_idle_ttl_s
        stale = [d for d, s in self.sessions.items() if s.idle_for(now) > ttl]
        for device in stale:
            self._evict(device)
        return stale

    def _evict_one(self) -> bool:
        """Make room: evict the longest-idle session nobody watches."""
        unwatched = [s for s in self.sessions.values() if not s.subscribers]
        if not unwatched:
            return False
        self._evict(min(unwatched, key=lambda s: s.last_active).device)
        return True

    def _evict(self, device: str) -> None:
        session = self.sessions.pop(device)
        if session.task is not None:
            session.task.cancel()
        with suppress(KeyError):
            MAP_VOXELS.remove(device)
        logger.info("Evicted the live map of %s (%d voxels)", device, len(session.vmap))

    def _start_janitor(self) -> None:
        if self._janitor is not None and not self._janitor.done():
            return
        try:
            self._janitor = asyncio.get_running_loop().create_task(self._sweep())
        except RuntimeError:  # no running loop (a direct call): idle maps are swept on the next create
            self._janitor = None

    async def _sweep(self) -> None:
        period = max(1.0, min(60.0, self.settings.map_idle_ttl_s / 4))
        while self.sessions:
            await asyncio.sleep(period)
            self.evict_idle()

    async def _fuse_loop(self, session: MapSession) -> None:
        while True:
            await session.wake.wait()
            session.wake.clear()
            frame, session.pending = session.pending, None
            if frame is None:
                continue
            async with session.lock:
                await self._fuse_step(session, frame)

    async def _fuse_step(self, session: MapSession, frame: QueuedFrame) -> None:
        """Gate, fuse (in a thread) and broadcast one frame; the caller holds `session.lock`."""
        if frame.header.pose is None:
            return
        pose = np.asarray(frame.header.pose).reshape(4, 4)
        if not session.gate.accept(pose):
            return
        started = time.perf_counter()
        try:
            xyz, rgb = await asyncio.to_thread(self._fuse, session.vmap, frame)
        except Exception:
            logger.exception("Fusing a keyframe from %s failed", session.device)
            return
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
    def _fuse(vmap: VoxelMap, frame: QueuedFrame) -> tuple[np.ndarray, np.ndarray]:
        """Runs in a thread on one fixed `vmap` (the caller holds the session lock, so reset cannot swap it)."""
        points, colours = frame_points(frame)
        return vmap.points(vmap.insert(points, colours))

    async def reset(self, device: str) -> bool:
        """Clear a phone's map (waits for a running fuse step to finish); False if there is none."""
        session = self.sessions.get(device)
        if session is None:
            return False
        async with session.lock:
            s = self.settings
            session.vmap = VoxelMap(s.map_voxel_m, s.map_max_points)
            session.gate.reset()
            session.trail.clear()
            session.keyframes = 0
            MAP_VOXELS.labels(device).set(0)
            session.broadcast(json.dumps({"type": "reset"}))
        return True

    async def save(self, device: str) -> dict[str, Any]:
        """Write the map (.ply) and camera trail (.csv) and log them as an MLflow run (blocking work in a thread)."""
        session = self.sessions.get(device)
        if session is None:
            raise ValueError(f"no live map for {device}")
        async with session.lock:
            return await asyncio.to_thread(self._save, session)

    def _save(self, session: MapSession) -> dict[str, Any]:
        import open3d as o3d

        from utilities.experiment import log_artifact, log_metrics, log_params, start_run

        device = session.device
        if not len(session.vmap):
            raise ValueError(f"no live map for {device}")
        xyz, rgb = session.vmap.points()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        s = self.settings
        out_dir: Path = s.live_map_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        ply, csv = out_dir / f"{device}-{stamp}.ply", out_dir / f"{device}-{stamp}.trajectory.csv"
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
        cloud.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255)
        o3d.io.write_point_cloud(ply, cloud)
        np.savetxt(csv, np.array(session.trail).reshape(-1, 3), delimiter=",", header="x,y,z", comments="", fmt="%.4f")
        # The app's own settings, not the global ones: a Console built with explicit Settings logs to its store.
        with start_run(MAP_EXPERIMENT, run_name=f"{device}-{stamp}", tags={"device": device}, settings=s) as run_id:
            log_params({"voxel_m": s.map_voxel_m, "keyframe_m": s.map_keyframe_m, "keyframe_deg": s.map_keyframe_deg})
            log_metrics({"voxels": len(session.vmap), "keyframes": session.keyframes})
            log_artifact(ply)
            log_artifact(csv)
        return {"ply": str(ply), "run_id": run_id, "voxels": len(session.vmap)}

    async def stop(self) -> None:
        tasks = [s.task for s in self.sessions.values() if s.task is not None]
        if self._janitor is not None:
            tasks.append(self._janitor)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _maps(request: Request | WebSocket) -> MapManager:
    return request.app.state.ingest.maps


def _check_device(device_id: str) -> None:
    if not valid_device_id(device_id):
        raise HTTPException(404, "invalid device id")


@router.websocket("/ws/map/{device_id}")
async def map_socket(websocket: WebSocket, device_id: str) -> None:
    """Send the whole map, then new voxels and stats as keyframes are fused."""
    if not valid_device_id(device_id):
        await websocket.close(POLICY_VIOLATION, "invalid device id")
        return
    session = _maps(websocket).session(device_id)
    if session is None:
        await websocket.close(TRY_AGAIN_LATER, "too many live maps")
        return
    queue: asyncio.Queue[bytes | str] = asyncio.Queue(SUBSCRIBER_QUEUE)
    await websocket.accept()
    VIEWERS.labels("map").inc()
    async with session.lock:  # snapshot and subscribe at once: no fuse step is missed or sent twice
        meta = session.meta()
        xyz, rgb = session.vmap.points()
        session.subscribers.add(queue)
    closed: asyncio.Task[None] | None = None
    try:
        await websocket.send_text(json.dumps(meta))
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
        session.last_active = time.monotonic()
        VIEWERS.labels("map").dec()


async def _wait_closed(websocket: WebSocket) -> None:
    """Return when the browser disconnects (it sends nothing else on this socket)."""
    while (await websocket.receive())["type"] != "websocket.disconnect":
        pass


@router.get("/api/map/sessions")
async def map_sessions(request: Request) -> list[dict[str, Any]]:
    return [
        {"device": s.device, "voxels": len(s.vmap), "keyframes": s.keyframes, "fuse_ms": s.fuse_ms}
        for s in _maps(request).sessions.values()
    ]


@router.post("/api/map/{device_id}/reset")
async def map_reset(request: Request, device_id: str) -> dict[str, bool]:
    _check_device(device_id)
    return {"reset": await _maps(request).reset(device_id)}


@router.post("/api/map/{device_id}/save")
async def map_save(request: Request, device_id: str) -> dict[str, Any]:
    _check_device(device_id)
    try:
        return await _maps(request).save(device_id)
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
    with tempfile.TemporaryDirectory(prefix="basir-cloud-") as tmp:  # deleted once read, so nothing leaks
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=ply, tracking_uri=tracking_uri, dst_path=tmp
        )
        cloud = o3d.io.read_point_cloud(Path(local))
    voxel = 0.01
    while len(cloud.points) > SAVED_CLOUD_MAX_POINTS:
        cloud, voxel = cloud.voxel_down_sample(voxel), voxel * 1.5
    xyz = np.asarray(cloud.points, np.float32)
    rgb = (np.asarray(cloud.colors) * 255).astype(np.uint8) if cloud.has_colors() else np.full_like(xyz, 200, np.uint8)
    return pack_points(xyz, rgb)
