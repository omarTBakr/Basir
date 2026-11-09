"""Round 3: three 3D treatments of the real fr1/room scan for the Basir logo.

A. Voxel blocks   the scan rebuilt as isometric cubes (visible faces only, painter's order)
B. Perspective    a corner view with near walls cut away, nearer points darker (z-buffer)
C. Exploded       height slices (floor / furniture / upper walls) stacked in isometric

Each view is rendered to PNG alpha masks (ink + target), themed by CSS, with a vector
overlay (route, target ring, Basir's bubble). Writes index.html.
"""

from __future__ import annotations

import base64
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d

HERE = Path(__file__).parent
PROJECT = Path(sys.argv[1])
sys.argv = [sys.argv[0], str(PROJECT)]
from generate import bubble_svg  # noqa: E402  (same speech bubble as round 2)

PX = 1024          # raster size of every view
K, DY = 0.84, 0.16  # scene shrinks to the lower-left so the bubble has clear space
COS30, SIN30 = math.cos(math.radians(30)), 0.5


# ---------------------------------------------------------------------------
# Load and square up the scan
# ---------------------------------------------------------------------------

def load_scene() -> dict:
    out = PROJECT / "outputs"
    P = np.asarray(o3d.io.read_point_cloud(str(out / "rgbd_dataset_freiburg1_room.ply")).points)
    traj = np.loadtxt(out / "rgbd_dataset_freiburg1_room.trajectory.csv", delimiter=",", skiprows=1)[:, 1:]
    lo, hi = np.percentile(P[:, :2], 2, axis=0), np.percentile(P[:, :2], 98, axis=0)
    P = P[(P[:, 0] > lo[0]) & (P[:, 0] < hi[0]) & (P[:, 1] > lo[1]) & (P[:, 1] < hi[1])
          & (P[:, 2] > -0.05) & (P[:, 2] < 2.3)]

    # rotate so the walls line up with x / y (minimum-area rectangle of the wall points)
    walls = P[(P[:, 2] > 1.0)][:, :2].astype(np.float32)
    (cx, cy), _, angle = cv2.minAreaRect(walls)
    a = math.radians(-angle)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])

    def square(Q: np.ndarray) -> np.ndarray:
        Q = Q.copy()
        Q[:, :2] = (Q[:, :2] - [cx, cy]) @ R.T
        return Q

    P, traj = square(P), square(traj)
    shift = np.array([*P[:, :2].min(0), 0.0])
    P, traj = P - shift, traj - shift

    table = P[np.abs(P[:, 2] - 0.77) < 0.04]
    target = np.array([*np.median(table[:, :2], axis=0), 0.79])
    far = traj[np.argmax(np.linalg.norm(traj[:, :2] - target[:2], axis=1))]
    route = np.array([[far[0], far[1], 0.02], [far[0], target[1], 0.02], [target[0], target[1], 0.02]])
    return dict(P=P, target=target, route=route, size=P.max(0))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def iso(Q: np.ndarray) -> np.ndarray:
    Q = np.atleast_2d(Q)
    return np.column_stack([(Q[:, 0] - Q[:, 1]) * COS30, (Q[:, 0] + Q[:, 1]) * SIN30 - Q[:, 2]])


class Fit:
    """Maps projected 2D coordinates into the PX x PX canvas (with a margin)."""

    def __init__(self, uv: np.ndarray, margin: float = 0.06):
        lo, hi = uv.min(0), uv.max(0)
        self.s = PX * (1 - 2 * margin) / max(hi - lo)
        self.o = PX * margin + (PX * (1 - 2 * margin) - (hi - lo) * self.s) / 2 - lo * self.s

    def __call__(self, uv: np.ndarray) -> np.ndarray:
        return np.atleast_2d(uv) * self.s + self.o


def png_mask(alpha: np.ndarray) -> str:
    rgba = np.zeros((*alpha.shape, 4), np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", rgba, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def overlay(route_px: np.ndarray, target_px: np.ndarray, extra: str = "") -> str:
    path = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in route_px)
    tx, ty = target_px
    scene = (
        extra
        + f'<path d="{path}" style="fill:none;stroke:var(--route);stroke-width:13;stroke-linecap:round;'
          f'stroke-linejoin:round;stroke-dasharray:0.1 28"/>'
        f'<circle cx="{route_px[0][0]:.1f}" cy="{route_px[0][1]:.1f}" r="17" style="fill:var(--route)"/>'
        f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="56" style="fill:none;stroke:var(--target);stroke-width:11"/>'
    )
    return (f'<svg viewBox="0 0 {PX} {PX}" xmlns="http://www.w3.org/2000/svg">'
            f'<g transform="translate(0 {PX * DY:.1f}) scale({K})">{scene}</g>'
            + bubble_svg(PX, (tx * K, ty * K + PX * DY)) + "</svg>")


# ---------------------------------------------------------------------------
# A. Voxel blocks
# ---------------------------------------------------------------------------

def voxel_view(sc: dict, vs: float = 0.12) -> tuple[np.ndarray, np.ndarray, str]:
    P, size = sc["P"], sc["size"]
    keys, counts = np.unique(np.floor(P / vs).astype(int), axis=0, return_counts=True)
    keys = keys[counts >= 3]
    keys = keys[keys[:, 2] >= 1]  # the floor is drawn as one plate instead
    # cut the two near walls down to 0.9 m so the room is visible from the viewer's corner
    near = (keys[:, 0] * vs > size[0] - 0.6) | (keys[:, 1] * vs > size[1] - 0.6)
    keys = keys[~near | (keys[:, 2] * vs < 0.9)]
    occupied = {tuple(k) for k in keys}
    keys = keys[np.argsort(keys.sum(1), kind="stable")]  # back to front

    corners = iso(np.array([[0, 0, 0], [size[0], 0, 0], [0, size[1], 0], [size[0], size[1], 0],
                            [0, 0, size[2]], [size[0], size[1], size[2]]]))
    fit = Fit(corners)
    label = np.zeros((PX, PX), np.uint8)  # 0 empty, 1 top, 2 +x face, 3 +y face, 4 floor, 5 target
    floor = fit(iso(np.array([[0, 0, 0], [size[0], 0, 0], [size[0], size[1], 0], [0, size[1], 0]])))
    cv2.fillConvexPoly(label, np.round(floor).astype(np.int32), 4)

    t_key = np.floor(sc["target"] / vs).astype(int)
    for k in keys:
        x0, y0, z0 = k * vs
        x1, y1, z1 = x0 + vs, y0 + vs, z0 + vs
        is_target = np.abs(k - t_key).max() <= 1
        faces = []
        if (k[0], k[1], k[2] + 1) not in occupied:
            faces.append((1, [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]))
        if (k[0] + 1, k[1], k[2]) not in occupied:
            faces.append((2, [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]]))
        if (k[0], k[1] + 1, k[2]) not in occupied:
            faces.append((3, [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]]))
        for tone, quad in faces:
            pts = np.round(fit(iso(np.array(quad)))).astype(np.int32)
            cv2.fillConvexPoly(label, pts, 5 if is_target else tone)

    ink = np.choose(label, [0, .42, .70, .93, .07, 0]).astype(np.float32)
    target = (label == 5).astype(np.float32)
    route = fit(iso(sc["route"]))
    return ink, target, route, fit(iso(sc["target"]))[0], ""


# ---------------------------------------------------------------------------
# B. Corner perspective
# ---------------------------------------------------------------------------

def perspective_view(sc: dict) -> tuple[np.ndarray, np.ndarray, str]:
    P, size = sc["P"], sc["size"]
    near = ((P[:, 0] > size[0] - 0.7) | (P[:, 1] > size[1] - 0.7)) & (P[:, 2] > 0.35)
    P = P[~near]

    cam = np.array([size[0] + 1.6, size[1] + 1.6, 3.4])
    look = np.array([size[0] * 0.45, size[1] * 0.45, 0.5])
    f = (look - cam) / np.linalg.norm(look - cam)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)

    def project(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        D = np.atleast_2d(Q) - cam
        z = D @ f
        return np.column_stack([(D @ r) / z, -(D @ u) / z]), z

    uv, depth = project(P)
    fit = Fit(uv, margin=0.05)
    ij = np.round(fit(uv)).astype(int)
    d = (depth - depth.min()) / (np.ptp(depth) + 1e-9)
    tone = 1.0 - 0.62 * d                       # nearer = darker
    tone[P[:, 2] < 0.12] *= 0.28                # floor stays faint
    img = np.zeros((PX, PX), np.float32)
    order = np.argsort(-depth)                  # far first, near points overwrite
    for dx in (0, 1, 2):
        for dy in (0, 1, 2):
            jj = np.clip(ij[order, 0] + dx, 0, PX - 1)
            ii = np.clip(ij[order, 1] + dy, 0, PX - 1)
            img[ii, jj] = tone[order]
    img = cv2.GaussianBlur(img, (0, 0), 0.7)

    target = np.zeros_like(img)
    t_uv = fit(project(sc["target"])[0])[0]
    near_t = np.linalg.norm(P - sc["target"], axis=1) < 0.14
    tj = np.clip(np.round(fit(uv[near_t])).astype(int), 0, PX - 1)
    for dx in (0, 1, 2):
        for dy in (0, 1, 2):
            target[np.clip(tj[:, 1] + dy, 0, PX - 1), np.clip(tj[:, 0] + dx, 0, PX - 1)] = 1
    return img, target, fit(project(sc["route"])[0]), t_uv, ""


# ---------------------------------------------------------------------------
# C. Exploded height layers
# ---------------------------------------------------------------------------

LAYERS = [  # (z from, z to, stacked height in the drawing, tone)
    (-0.05, 0.25, 0.0, 0.16),
    (0.25, 1.10, 1.5, 0.80),
    (1.10, 2.30, 3.0, 0.48),
]


def exploded_view(sc: dict, cell: float = 0.03) -> tuple[np.ndarray, np.ndarray, str]:
    P, size = sc["P"], sc["size"]
    outline_pts = []
    for _, _, h, _ in LAYERS:
        outline_pts += [[0, 0, h], [size[0], 0, h], [size[0], size[1], h], [0, size[1], h]]
    fit = Fit(iso(np.array(outline_pts)), margin=0.05)

    img = np.zeros((PX, PX), np.float32)
    outlines = ""
    for z0, z1, h, tone in LAYERS:
        layer = P[(P[:, 2] >= z0) & (P[:, 2] < z1)]
        cells = np.unique(np.floor(layer[:, :2] / cell).astype(int), axis=0)
        flat = np.column_stack([(cells + 0.5) * cell, np.full(len(cells), h)])
        ij = np.round(fit(iso(flat))).astype(int)
        for dx in (0, 1, 2):
            for dy in (0, 1):
                img[np.clip(ij[:, 1] + dy, 0, PX - 1), np.clip(ij[:, 0] + dx, 0, PX - 1)] = tone
        quad = fit(iso(np.array([[0, 0, h], [size[0], 0, h], [size[0], size[1], h], [0, size[1], h]])))
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in quad)
        outlines += f'<polygon points="{pts}" style="fill:none;stroke:var(--ink);stroke-width:3;opacity:.35"/>'
    img = cv2.GaussianBlur(img, (0, 0), 0.6)

    mid_h = LAYERS[1][2]
    target_mid = sc["target"].copy()
    target_mid[2] = mid_h
    t_px = fit(iso(target_mid))[0]
    floor_t = fit(iso(np.array([*sc["target"][:2], 0.0])))[0]
    route = fit(iso(sc["route"] * [1, 1, 0]))
    riser = (f'<path d="M{floor_t[0]:.1f} {floor_t[1]:.1f} L{t_px[0]:.1f} {t_px[1]:.1f}" '
             f'style="fill:none;stroke:var(--route);stroke-width:7;stroke-linecap:round;stroke-dasharray:0.1 18"/>')
    target = np.zeros_like(img)
    cv2.circle(target, tuple(np.round(t_px).astype(int)), 20, 1.0, -1)
    return img, target, route, t_px, outlines + riser


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

OPTIONS = [
    ("a", "A · VOXEL BLOCKS", "The scan rebuilt as 12 cm cubes",
     "Every cube is a 12 cm cell of the real reconstruction that held at least three points, drawn in isometric with only its visible faces. The two near walls are cut to 0.9 m so you look into the room. The amber cubes are the table where the target sits.",
     "The most logo-like of the three: solid shapes, three clean tones, and it still reads at 32 px. It looks deliberately digital, like Basir's memory of the room."),
    ("b", "B · CORNER PERSPECTIVE", "Looking in from above one corner",
     "A real perspective camera above the room's corner, aimed at its centre. Every point is drawn with a depth test, so nearer surfaces are darker and cover the ones behind. The near walls are cut away and the floor is kept faint.",
     "The most cinematic and the most like what a person sees standing in the doorway. Too fine below 64 px, so it fits a README hero or a splash screen."),
    ("c", "C · EXPLODED LAYERS", "The room sliced by height and stacked",
     "The scan split into three height bands: floor (0–0.25 m), furniture (0.25–1.1 m) and upper walls and shelves (1.1–2.3 m), each flattened and stacked in isometric. The route runs on the floor layer, and a dotted riser climbs to the target on the furniture layer.",
     "It explains how Basir thinks: a top view and a 3D view in one picture. Most diagram-like; strong for the README and slides, busy as a favicon."),
]


def main() -> None:
    sc = load_scene()
    renders = {"a": voxel_view(sc), "b": perspective_view(sc), "c": exploded_view(sc)}

    css, cards = [], []
    for key, tag, title, body, verdict in OPTIONS:
        ink, target, route_px, target_px, extra = renders[key]
        svg = overlay(route_px, target_px, extra)
        css.append(f".ink-{key}{{-webkit-mask-image:url({png_mask(ink)});mask-image:url({png_mask(ink)})}}")
        css.append(f".tgt-{key}{{-webkit-mask-image:url({png_mask(target)});mask-image:url({png_mask(target)})}}")
        mark = (f'<div class="scan"><div class="layer ink ink-{key}"></div>'
                f'<div class="layer tgt tgt-{key}"></div>{svg}</div>')
        cards.append(f"""
  <section class="concept" aria-labelledby="t-{key}">
    <div class="hero" role="img" aria-label="{title}">{mark}</div>
    <div class="info">
      <div class="tag">{tag}</div>
      <h2 id="t-{key}">{title}</h2>
      <p>{body}</p>
      <p class="says">{verdict}</p>
      <div class="row">
        <div class="size"><div class="box" style="width:64px">{mark}</div>64</div>
        <div class="size"><div class="box" style="width:32px">{mark}</div>32</div>
        <div class="light-tile"><div class="box" style="width:72px">{mark}</div></div>
        <div class="dark-tile"><div class="box" style="width:72px">{mark}</div></div>
      </div>
      <div class="lockup"><div class="box" style="width:56px">{mark}</div><div class="word">Basir<small>بصير</small></div></div>
    </div>
  </section>""")

    page = (HERE / "template3.html").read_text()
    page = page.replace("/*{{masks}}*/", "\n".join(css)).replace("<!--{{cards}}-->", "\n".join(cards))
    (HERE / "index.html").write_text(page)
    print(f"wrote index.html {(HERE / 'index.html').stat().st_size / 1024:.0f} KB; room {sc['size'].round(2)}")


if __name__ == "__main__":
    main()
