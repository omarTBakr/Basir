"""Generate Basir logo concepts from real 3D geometry.

1. Modelled scenes (a cluttered room, an apartment) as 3D boxes -> isometric view
   (painter's algorithm) and orthographic top view, as SVG.
2. The real fr1/room point cloud Basir reconstructed -> oblique 3D view and top view,
   as PNG alpha masks (themed by CSS) with vector overlays.
Writes logo-round2.html next to this file.
"""

from __future__ import annotations

import base64
import math
from pathlib import Path
import sys

import cv2
import numpy as np

HERE = Path(__file__).parent
PROJECT = Path(sys.argv[1])

COS30, SIN30 = math.cos(math.radians(30)), 0.5

# ---------------------------------------------------------------------------
# Scenes: boxes (x0, y0, z0, x1, y1, z1, kind). Metres, z up. Viewer at +x,+y.
# kinds: wall, lowwall, furn, flat, target
# ---------------------------------------------------------------------------

ROOM = dict(
    size=(5.0, 4.0, 2.6),
    boxes=[
        (0, 0, 0, 0.12, 4.0, 2.6, "wall"), (0, 0, 0, 5.0, 0.12, 2.6, "wall"),
        (1.2, 2.2, 0, 3.2, 3.6, 0.02, "flat"),                     # rug
        (0.15, 0.15, 0, 2.0, 1.55, 0.5, "furn"), (0.15, 0.15, 0.5, 0.3, 1.55, 1.05, "furn"),   # bed
        (0.15, 1.65, 0, 0.55, 2.05, 0.55, "furn"),                 # nightstand
        (2.3, 0.15, 0, 3.3, 0.5, 1.9, "furn"),                     # shelf
        (3.6, 0.15, 0.7, 4.9, 0.85, 0.78, "furn"),                 # desk top
        (3.65, 0.2, 0, 3.75, 0.3, 0.7, "furn"), (4.75, 0.2, 0, 4.85, 0.3, 0.7, "furn"),
        (3.65, 0.7, 0, 3.75, 0.8, 0.7, "furn"), (4.75, 0.7, 0, 4.85, 0.8, 0.7, "furn"),
        (4.0, 0.2, 0.78, 4.6, 0.27, 1.15, "furn"),                 # monitor
        (3.9, 1.0, 0, 4.4, 1.5, 0.45, "furn"), (3.9, 1.42, 0.45, 4.4, 1.5, 0.95, "furn"),     # chair
        (1.1, 1.75, 0, 1.45, 2.0, 0.45, "furn"),                   # backpack
        (0.3, 2.25, 0, 0.36, 2.31, 1.5, "furn"), (0.2, 2.15, 1.5, 0.46, 2.41, 1.75, "furn"),  # lamp
        (1.8, 2.6, 0, 2.6, 3.2, 0.4, "furn"),                      # coffee table
        (0.2, 3.35, 0, 0.6, 3.75, 0.4, "furn"), (0.25, 3.4, 0.4, 0.55, 3.7, 1.1, "furn"),    # plant
        (3.2, 2.9, 0, 3.6, 3.3, 0.3, "furn"), (2.9, 3.45, 0, 3.2, 3.75, 0.2, "furn"),        # scattered boxes
        (4.3, 2.4, 0, 4.8, 2.9, 0.45, "furn"), (4.4, 2.95, 0, 4.7, 3.25, 0.25, "furn"),
        (2.15, 2.85, 0.4, 2.29, 2.99, 0.54, "target"),             # the mug
    ],
    start=(4.0, 3.85),
    route=[(4.0, 3.85), (4.0, 2.15), (2.85, 2.15), (2.5, 2.6)],
)

APARTMENT = dict(
    size=(9.0, 7.0, 2.4),
    boxes=[
        (0, 0, 0, 0.15, 7.0, 2.4, "wall"), (0, 0, 0, 9.0, 0.15, 2.4, "wall"),
        # interior walls (cut low so the rooms stay visible), with door gaps
        (5.45, 0.15, 0, 5.6, 3.2, 0.9, "lowwall"), (5.45, 4.0, 0, 5.6, 5.3, 0.9, "lowwall"),
        (5.45, 6.0, 0, 5.6, 7.0, 0.9, "lowwall"),
        (0.15, 2.45, 0, 2.0, 2.6, 0.9, "lowwall"), (3.5, 2.45, 0, 5.45, 2.6, 0.9, "lowwall"),
        (5.6, 4.45, 0, 9.0, 4.6, 0.9, "lowwall"),
        (8.85, 0.15, 0, 9.0, 7.0, 0.3, "lowwall"),
        (0.15, 6.85, 0, 1.0, 7.0, 0.3, "lowwall"), (2.2, 6.85, 0, 8.85, 7.0, 0.3, "lowwall"),
        # kitchen
        (0.2, 0.2, 0, 4.5, 0.8, 0.9, "furn"), (1.8, 1.3, 0, 3.4, 1.9, 0.9, "furn"), (4.6, 0.2, 0, 5.3, 0.9, 2.0, "furn"),
        # living
        (1.0, 3.2, 0, 3.0, 4.6, 0.02, "flat"),
        (0.25, 4.0, 0, 1.1, 6.4, 0.45, "furn"), (0.25, 4.0, 0.45, 0.5, 6.4, 0.9, "furn"),
        (1.6, 4.6, 0, 2.5, 5.6, 0.4, "furn"),
        (4.6, 4.7, 0, 5.2, 6.0, 0.5, "furn"), (4.95, 4.9, 0.5, 5.05, 5.8, 1.2, "furn"),
        (2.8, 2.9, 0.72, 4.4, 3.9, 0.78, "furn"), (3.4, 3.2, 0, 3.8, 3.6, 0.72, "furn"),
        (2.45, 3.1, 0, 2.75, 3.4, 0.45, "furn"), (4.45, 3.1, 0, 4.75, 3.4, 0.45, "furn"),
        (3.0, 5.8, 0, 3.3, 6.1, 0.25, "furn"), (3.6, 6.2, 0, 4.0, 6.5, 0.35, "furn"), (2.7, 5.0, 0, 2.95, 5.25, 0.2, "furn"),
        (0.3, 2.8, 0, 0.7, 3.2, 1.2, "furn"),
        # bedroom
        (6.8, 0.25, 0, 8.8, 2.4, 0.5, "furn"), (8.65, 0.25, 0.5, 8.8, 2.4, 1.0, "furn"),
        (5.7, 0.2, 0, 6.4, 1.8, 2.0, "furn"), (5.8, 3.4, 0, 7.0, 3.9, 0.75, "furn"),
        (7.4, 3.0, 0, 7.8, 3.4, 0.35, "furn"), (8.2, 3.5, 0, 8.7, 4.0, 0.4, "furn"),
        # bath
        (7.2, 5.4, 0, 8.8, 6.7, 0.55, "furn"), (5.8, 6.2, 0, 6.5, 6.7, 0.85, "furn"),
        (6.3, 3.55, 0.75, 6.46, 3.71, 0.82, "target"),             # the keys
    ],
    start=(1.6, 6.6),
    route=[(1.6, 6.6), (2.6, 6.6), (2.6, 4.1), (5.25, 4.1), (5.9, 3.6), (6.38, 3.3)],
)

FACE_STYLE = {  # (top, +x face, +y face) as (token, opacity)
    "wall": (("ink", .10), ("ink", .16), ("ink", .22)),
    "lowwall": (("ink", .55), ("ink", .75), ("ink", .9)),
    "furn": (("ink", .38), ("ink", .66), ("ink", .88)),
    "flat": (("ink", .10), ("ink", .10), ("ink", .10)),
    "target": (("target", .8), ("target", 1), ("target", 1)),
}


def iso(x: float, y: float, z: float) -> tuple[float, float]:
    return (x - y) * COS30, (x + y) * SIN30 - z


def fill(token: str, opacity: float) -> str:
    return f'style="fill:var(--{token});fill-opacity:{opacity}"'


def fit(points: np.ndarray, size: float, pad: float) -> tuple[float, float, float]:
    lo, hi = points.min(0), points.max(0)
    scale = (size - 2 * pad) / max(hi - lo)
    off = pad + ((size - 2 * pad) - (hi - lo) * scale) / 2 - lo * scale
    return scale, off[0], off[1]


def poly(pts: list[tuple[float, float]], tf: tuple[float, float, float], style: str) -> str:
    s, ox, oy = tf
    coords = " ".join(f"{x * s + ox:.2f},{y * s + oy:.2f}" for x, y in pts)
    return f'<polygon points="{coords}" {style}/>'


def iso_svg(scene: dict, size: int = 120, bubble: bool = True) -> str:
    W, D, H = scene["size"]
    corners = [iso(x, y, z) for x in (0, W) for y in (0, D) for z in (0, H)]
    tf = fit(np.array(corners), size, 10)
    s, ox, oy = tf
    out = [poly([iso(0, 0, 0), iso(W, 0, 0), iso(W, D, 0), iso(0, D, 0)], tf, fill("ink", .05))]
    # route on the floor, drawn before furniture (furniture may occlude it, as in reality)
    r = [iso(x, y, 0.01) for x, y in scene["route"]]
    path = "M" + " L".join(f"{x * s + ox:.2f} {y * s + oy:.2f}" for x, y in r)
    route = (f'<path d="{path}" style="fill:none;stroke:var(--route);stroke-width:{size / 55:.2f};'
             f'stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:0.1 {size / 30:.2f}"/>')
    sx, sy = iso(*scene["start"], 0)
    start = f'<circle cx="{sx * s + ox:.2f}" cy="{sy * s + oy:.2f}" r="{size / 45:.2f}" style="fill:var(--route)"/>'

    boxes = sorted(scene["boxes"], key=lambda b: (b[6] != "wall", b[0] + b[1] + b[2] * 0.01, b[2]))
    flats = [b for b in boxes if b[6] in ("wall", "flat")]
    solids = [b for b in boxes if b[6] not in ("wall", "flat")]
    target_center = None
    for group in (flats, [None], solids):
        for b in group:
            if b is None:
                out += [route, start]
                continue
            x0, y0, z0, x1, y1, z1, kind = b
            top, fx, fy = FACE_STYLE[kind]
            out.append(poly([iso(x0, y0, z1), iso(x1, y0, z1), iso(x1, y1, z1), iso(x0, y1, z1)], tf, fill(*top)))
            out.append(poly([iso(x1, y0, z0), iso(x1, y1, z0), iso(x1, y1, z1), iso(x1, y0, z1)], tf, fill(*fx)))
            out.append(poly([iso(x0, y1, z0), iso(x1, y1, z0), iso(x1, y1, z1), iso(x0, y1, z1)], tf, fill(*fy)))
            if kind == "target":
                cx, cy = iso((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
                target_center = (cx * s + ox, cy * s + oy)
    tx, ty = target_center
    out.append(f'<circle cx="{tx:.2f}" cy="{ty:.2f}" r="{size / 16:.2f}" '
               f'style="fill:none;stroke:var(--target);stroke-width:{size / 70:.2f}"/>')
    body = "".join(out)
    if bubble:
        body, t = make_room_for_bubble(out, size, (tx, ty))
        body += bubble_svg(size, t)
    return f'<svg viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">{body}</svg>'



def make_room_for_bubble(svg_body: list[str], size: float, target: tuple[float, float]) -> tuple[str, tuple[float, float]]:
    """Scale the scene to 84% toward the bottom-left so the bubble sits in clear space."""
    k, dy = 0.84, size * 0.16
    t = (target[0] * k, target[1] * k + dy)
    return f'<g transform="translate(0 {dy:.2f}) scale({k})">{"".join(svg_body)}</g>', t


def bubble_svg(size: float, target: tuple[float, float], corner: str = "tr") -> str:
    """Basir's speech bubble in a top corner, its tail pointing at the target, with a direction arrow."""
    w, h, r = size * 0.30, size * 0.17, size * 0.05
    x = size - w - size * 0.03 if corner == "tr" else size * 0.03
    y = size * 0.03
    tx, ty = target
    bx = min(max(tx, x + r * 1.5), x + w - r * 1.5)
    tail_tip = (bx + (tx - bx) * 0.35, y + h + (ty - y - h) * 0.35)
    body = (f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{r:.2f}" style="fill:var(--ink)"/>'
            f'<polygon points="{bx - size * .03:.2f},{y + h - 1:.2f} {bx + size * .03:.2f},{y + h - 1:.2f} '
            f'{tail_tip[0]:.2f},{tail_tip[1]:.2f}" style="fill:var(--ink)"/>')
    ax, ay, a = x + w * 0.22, y + h * 0.72, h * 0.45
    arrow = (f'<path d="M{ax:.2f} {ay:.2f} L{ax + a:.2f} {ay - a:.2f} M{ax + a * .35:.2f} {ay - a:.2f} '
             f'H{ax + a:.2f} V{ay - a * .65:.2f}" style="fill:none;stroke:var(--paper);stroke-width:{size / 60:.2f};'
             f'stroke-linecap:round;stroke-linejoin:round"/>')
    lines = (f'<path d="M{x + w * .55:.2f} {y + h * .38:.2f} H{x + w * .84:.2f} M{x + w * .55:.2f} {y + h * .64:.2f} '
             f'H{x + w * .74:.2f}" style="fill:none;stroke:var(--paper);stroke-width:{size / 70:.2f};'
             f'stroke-linecap:round;opacity:.65"/>')
    return body + arrow + lines


def top_svg(scene: dict, size: int = 120, tile: bool = False, bubble: bool = True) -> str:
    W, D, _ = scene["size"]
    pad = 16 if tile else 8
    s = (size - 2 * pad) / max(W, D)
    ox, oy = pad + ((size - 2 * pad) - W * s) / 2, pad + ((size - 2 * pad) - D * s) / 2
    P = lambda x, y: (x * s + ox, y * s + oy)  # noqa: E731
    out = []
    if tile:
        out.append(f'<rect width="{size}" height="{size}" rx="{size * .23:.2f}" style="fill:var(--ink)"/>')
    wall_tok, furn_tok, floor_tok = ("paper", "paper", "paper") if tile else ("ink", "ink", "ink")
    x0, y0 = P(0, 0)
    x1, y1 = P(W, D)
    out.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{x1 - x0:.2f}" height="{y1 - y0:.2f}" '
               f'style="fill:var(--{floor_tok});fill-opacity:.06"/>')
    target = None
    for bx0, by0, bz0, bx1, by1, bz1, kind in sorted(scene["boxes"], key=lambda b: b[5]):
        a, b = P(bx0, by0), P(bx1, by1)
        if kind in ("wall", "lowwall"):
            style = f'style="fill:var(--{wall_tok})"'
        elif kind == "flat":
            style = f'style="fill:var(--{furn_tok});fill-opacity:.12"'
        elif kind == "target":
            target = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            continue
        else:
            style = f'style="fill:var(--{furn_tok});fill-opacity:{0.25 + 0.3 * min(bz1, 2) / 2:.2f}"'
        out.append(f'<rect x="{a[0]:.2f}" y="{a[1]:.2f}" width="{b[0] - a[0]:.2f}" height="{b[1] - a[1]:.2f}" rx="0.6" {style}/>')
    # the front walls of the room scene are open in the 3D view; close the plan here
    if scene is ROOM:
        t = 0.12 * s
        out.append(f'<rect x="{x1 - t:.2f}" y="{y0:.2f}" width="{t:.2f}" height="{y1 - y0:.2f}" style="fill:var(--{wall_tok})"/>')
        dx0, dx1 = P(3.6, 0)[0], P(4.5, 0)[0]
        out.append(f'<rect x="{x0:.2f}" y="{y1 - t:.2f}" width="{dx0 - x0:.2f}" height="{t:.2f}" style="fill:var(--{wall_tok})"/>')
        out.append(f'<rect x="{dx1:.2f}" y="{y1 - t:.2f}" width="{x1 - dx1:.2f}" height="{t:.2f}" style="fill:var(--{wall_tok})"/>')
    path = "M" + " L".join(f"{P(x, y)[0]:.2f} {P(x, y)[1]:.2f}" for x, y in scene["route"])
    out.append(f'<path d="{path}" style="fill:none;stroke:var(--route);stroke-width:{size / 50:.2f};stroke-linecap:round;'
               f'stroke-linejoin:round;stroke-dasharray:0.1 {size / 28:.2f}"/>')
    sx, sy = P(*scene["start"])
    out.append(f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="{size / 40:.2f}" style="fill:var(--route)"/>')
    tx, ty = target
    out.append(f'<circle cx="{tx:.2f}" cy="{ty:.2f}" r="{size / 16:.2f}" style="fill:none;stroke:var(--target);stroke-width:{size / 65:.2f}"/>')
    out.append(f'<circle cx="{tx:.2f}" cy="{ty:.2f}" r="{size / 34:.2f}" style="fill:var(--target)"/>')
    body = "".join(out)
    if bubble:
        body, t = make_room_for_bubble(out, size, (tx, ty))
        body += bubble_svg(size, t)
    return f'<svg viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">{body}</svg>'


# ---------------------------------------------------------------------------
# The real scan
# ---------------------------------------------------------------------------

def real_views(px: int = 512) -> dict[str, str]:
    import open3d as o3d

    root = PROJECT / "outputs"
    cloud = o3d.io.read_point_cloud(str(root / "rgbd_dataset_freiburg1_room.ply"))
    P = np.asarray(cloud.points)
    traj = np.loadtxt(root / "rgbd_dataset_freiburg1_room.trajectory.csv", delimiter=",", skiprows=1)[:, 1:]
    lo, hi = np.percentile(P[:, :2], 2, axis=0), np.percentile(P[:, :2], 98, axis=0)
    keep = (P[:, 0] > lo[0]) & (P[:, 0] < hi[0]) & (P[:, 1] > lo[1]) & (P[:, 1] < hi[1]) & (P[:, 2] < 2.3)
    P = P[keep]
    furniture = P[P[:, 2] > 0.15]

    # target: centre of the largest table-height surface (the plane found at z ~ 0.77)
    table = P[np.abs(P[:, 2] - 0.77) < 0.04]
    target = np.array([*np.median(table[:, :2], axis=0), 0.79])
    start = traj[np.argmax(np.linalg.norm(traj[:, :2] - target[:2], axis=1))].copy()
    start[2] = 0.05
    route_world = np.array([start, [start[0], target[1], 0.05], [target[0], target[1], 0.05]])

    views = {}
    for name, project in (("top", lambda Q: Q[:, :2] * [1, -1]),
                          ("oblique", lambda Q: _oblique(Q))):
        uv_f = project(furniture)
        uv_all = project(P)
        lo2, hi2 = uv_all.min(0), uv_all.max(0)
        scale = (px * 0.86) / max(hi2 - lo2)
        off = px * 0.07 + ((px * 0.86) - (hi2 - lo2) * scale) / 2 - lo2 * scale
        to_px = lambda Q: project(Q) * scale + off  # noqa: E731

        density = np.zeros((px, px), np.float32)
        for uv, weight in ((uv_all, 0.35), (uv_f, 1.0)):
            ij = np.clip((uv * scale + off).astype(int), 0, px - 1)
            np.add.at(density, (ij[:, 1], ij[:, 0]), weight)
        density = cv2.GaussianBlur(density, (0, 0), 0.6)
        alpha = np.clip(density / np.percentile(density[density > 0], 85), 0, 1) ** 0.8
        rgba = np.zeros((px, px, 4), np.uint8)
        rgba[..., :3] = 255
        rgba[..., 3] = (alpha * 255).astype(np.uint8)
        ok, buf = cv2.imencode(".png", rgba)
        mask = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()

        r = to_px(route_world)
        t = to_px(target[None])[0]
        sp = r[0]
        path = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in r)
        overlay = (
            f'<path d="{path}" style="fill:none;stroke:var(--route);stroke-width:7;stroke-linecap:round;'
            f'stroke-linejoin:round;stroke-dasharray:0.1 15"/>'
            f'<circle cx="{sp[0]:.1f}" cy="{sp[1]:.1f}" r="9" style="fill:var(--route)"/>'
            f'<circle cx="{t[0]:.1f}" cy="{t[1]:.1f}" r="30" style="fill:none;stroke:var(--target);stroke-width:6"/>'
            f'<circle cx="{t[0]:.1f}" cy="{t[1]:.1f}" r="12" style="fill:var(--target)"/>'
        )
        k, dy = 0.84, px * 0.16
        overlay = (f'<g transform="translate(0 {dy:.1f}) scale({k})">{overlay}</g>'
                   + bubble_svg(px, (t[0] * k, t[1] * k + dy)))
        views[name] = (mask, f'<svg viewBox="0 0 {px} {px}" xmlns="http://www.w3.org/2000/svg">{overlay}</svg>')
    return views


def _oblique(Q: np.ndarray, yaw: float = 35.0, elevation: float = 38.0) -> np.ndarray:
    a, e = math.radians(yaw), math.radians(elevation)
    x = Q[:, 0] * math.cos(a) - Q[:, 1] * math.sin(a)
    y = Q[:, 0] * math.sin(a) + Q[:, 1] * math.cos(a)
    return np.column_stack([x, -(Q[:, 2] * math.cos(e) + y * math.sin(e))])


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def real_mark(view: tuple[str, str], label: str) -> str:
    mask, overlay = view
    return (f'<div class="scan" role="img" aria-label="{label}"><div class="scan-dots" '
            f'style="-webkit-mask-image:url({mask});mask-image:url({mask})"></div>{overlay}</div>')


def main() -> None:
    marks = {
        "room_iso": iso_svg(ROOM), "room_top": top_svg(ROOM), "room_tile": top_svg(ROOM, tile=True, bubble=False),
        "apt_iso": iso_svg(APARTMENT), "apt_top": top_svg(APARTMENT), "apt_tile": top_svg(APARTMENT, tile=True, bubble=False),
        "room_iso_plain": iso_svg(ROOM, bubble=False), "apt_iso_plain": iso_svg(APARTMENT, bubble=False),
    }
    real = real_views()
    template = (HERE / "template.html").read_text()
    for key, svg in marks.items():
        template = template.replace(f"{{{{{key}}}}}", svg)
    template = template.replace("{{real_oblique}}", real_mark(real["oblique"], "Real scan of the TUM fr1 room, 3D view"))
    template = template.replace("{{real_top}}", real_mark(real["top"], "Real scan of the TUM fr1 room, top view"))
    (HERE / "index.html").write_text(template)
    print("wrote", HERE / "index.html", f"{(HERE / 'index.html').stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
