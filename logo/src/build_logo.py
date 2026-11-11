"""Build Basir's logo from the room it reconstructed in Phase 0.

The scanned room as isometric voxels + a two-bubble chat + a pin on the found object,
written as light and dark SVG (colours baked in, the scan embedded as a tinted PNG)
plus PNG sizes, rasterised with headless Google Chrome.

    uv run python logo/src/build_logo.py            # writes logo/*.svg and logo/*.png
    uv run python logo/src/build_logo.py --out /tmp/logo-check
"""

from __future__ import annotations

import argparse
import base64
import math
from pathlib import Path
import re
import subprocess
import tempfile

import cv2
import numpy as np
import open3d as o3d

PROJECT = Path(__file__).resolve().parents[2]
SCAN = PROJECT / "outputs" / "rgbd_dataset_freiburg1_room.ply"
TRAJECTORY = PROJECT / "outputs" / "rgbd_dataset_freiburg1_room.trajectory.csv"

PX = 1024  # canvas size
VOXEL = 0.12  # metres
COS30, SIN30 = math.cos(math.radians(30)), 0.5

PALETTES = {
    "light": {"ink": "#15212b", "paper": "#f9fbfa", "route": "#12a39a", "target": "#f08a24"},
    "dark": {"ink": "#e6ecea", "paper": "#15212b", "route": "#2cc7bc", "target": "#ff9d42"},
}

# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------


def load_scene() -> dict:
    """The reconstructed room, squared to the axes, with the target (table surface) and a route to it."""
    if not SCAN.is_file():
        raise SystemExit(f"{SCAN} not found. Run: uv run python scripts/visualize_pointcloud.py fr1/room --no-window")
    P = np.asarray(o3d.io.read_point_cloud(str(SCAN)).points)
    traj = np.loadtxt(TRAJECTORY, delimiter=",", skiprows=1)[:, 1:]
    lo, hi = np.percentile(P[:, :2], 2, axis=0), np.percentile(P[:, :2], 98, axis=0)
    P = P[
        (P[:, 0] > lo[0])
        & (P[:, 0] < hi[0])
        & (P[:, 1] > lo[1])
        & (P[:, 1] < hi[1])
        & (P[:, 2] > -0.05)
        & (P[:, 2] < 2.3)
    ]

    # rotate so the walls line up with x / y (minimum-area rectangle of the wall points)
    (cx, cy), _, angle = cv2.minAreaRect(P[P[:, 2] > 1.0][:, :2].astype(np.float32))
    a = math.radians(-angle)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])

    def square(Q: np.ndarray) -> np.ndarray:
        Q = Q.copy()
        Q[:, :2] = (Q[:, :2] - [cx, cy]) @ R.T
        return Q

    P, traj = square(P), square(traj)
    shift = np.array([*P[:, :2].min(0), 0.0])
    P, traj = P - shift, traj - shift

    table = P[np.abs(P[:, 2] - 0.77) < 0.04]  # the table plane found in Phase 0
    target = np.array([*np.median(table[:, :2], axis=0), 0.79])
    far = traj[np.argmax(np.linalg.norm(traj[:, :2] - target[:2], axis=1))]
    route = np.array([[far[0], far[1], 0.02], [far[0], target[1], 0.02], [target[0], target[1], 0.02]])
    return {"P": P, "target": target, "route": route, "size": P.max(0)}


def iso(Q: np.ndarray) -> np.ndarray:
    Q = np.atleast_2d(Q)
    return np.column_stack([(Q[:, 0] - Q[:, 1]) * COS30, (Q[:, 0] + Q[:, 1]) * SIN30 - Q[:, 2]])


class Fit:
    """Maps projected 2D coordinates into the PX x PX canvas, with a margin."""

    def __init__(self, uv: np.ndarray, margin: float = 0.06):
        lo, hi = uv.min(0), uv.max(0)
        self.s = PX * (1 - 2 * margin) / max(hi - lo)
        self.o = PX * margin + (PX * (1 - 2 * margin) - (hi - lo) * self.s) / 2 - lo * self.s

    def __call__(self, uv: np.ndarray) -> np.ndarray:
        return np.atleast_2d(uv) * self.s + self.o


def voxel_render(sc: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Isometric voxels of the scan (visible faces only, back to front).

    Returns (ink alpha, target alpha, route in pixels, target in pixels).
    """
    P, size = sc["P"], sc["size"]
    keys, counts = np.unique(np.floor(P / VOXEL).astype(int), axis=0, return_counts=True)
    keys = keys[counts >= 3]
    keys = keys[keys[:, 2] >= 1]  # the floor is drawn as one plate instead
    near = (keys[:, 0] * VOXEL > size[0] - 0.6) | (keys[:, 1] * VOXEL > size[1] - 0.6)
    keys = keys[~near | (keys[:, 2] * VOXEL < 0.9)]  # cut the two near walls to 0.9 m
    occupied = {tuple(k) for k in keys}
    keys = keys[np.argsort(keys.sum(1), kind="stable")]

    corners = iso(
        np.array(
            [
                [0, 0, 0],
                [size[0], 0, 0],
                [0, size[1], 0],
                [size[0], size[1], 0],
                [0, 0, size[2]],
                [size[0], size[1], size[2]],
            ]
        )
    )
    fit = Fit(corners)
    label = np.zeros((PX, PX), np.uint8)  # 0 empty, 1 top, 2 +x face, 3 +y face, 4 floor, 5 target
    floor = fit(iso(np.array([[0, 0, 0], [size[0], 0, 0], [size[0], size[1], 0], [0, size[1], 0]])))
    cv2.fillConvexPoly(label, np.round(floor).astype(np.int32), 4)

    t_key = np.floor(sc["target"] / VOXEL).astype(int)
    for k in keys:
        x0, y0, z0 = k * VOXEL
        x1, y1, z1 = x0 + VOXEL, y0 + VOXEL, z0 + VOXEL
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

    ink = np.choose(label, [0, 0.42, 0.70, 0.93, 0.07, 0]).astype(np.float32)
    target = (label == 5).astype(np.float32)
    return ink, target, fit(iso(sc["route"])), fit(iso(sc["target"]))[0]


# ---------------------------------------------------------------------------
# Drawing elements (SVG on the PX x PX canvas, colours as var(--token))
# ---------------------------------------------------------------------------


def pin(x: float, y: float, s: float) -> str:
    """Location pin with its tip at (x, y); s is the head radius."""
    hy = y - 1.5 * s
    d = (
        f"M{x:.1f} {y:.1f} C{x - 0.5 * s:.1f} {y - 0.55 * s:.1f} {x - s:.1f} {hy + 0.55 * s:.1f} {x - s:.1f} {hy:.1f} "
        f"A{s:.1f} {s:.1f} 0 1 1 {x + s:.1f} {hy:.1f} C{x + s:.1f} {hy + 0.55 * s:.1f} {x + 0.5 * s:.1f} "
        f"{y - 0.55 * s:.1f} {x:.1f} {y:.1f}Z"
    )
    return (
        f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{s * 0.7:.1f}" ry="{s * 0.25:.1f}" '
        'style="fill:var(--ink);opacity:.25"/>'
        f'<path d="{d}" style="fill:var(--target)"/>'
        f'<circle cx="{x:.1f}" cy="{hy:.1f}" r="{s * 0.42:.1f}" style="fill:var(--paper)"/>'
    )


def bubble(x: float, y: float, w: float, h: float, filled: bool, tail: str, content: str) -> str:
    """A chat bubble. tail: 'bl' (bottom-left, Basir) or 'br' (bottom-right, you)."""
    r = h * 0.32
    tx = x + w * (0.22 if tail == "bl" else 0.78)
    tip = (tx + (-h * 0.28 if tail == "bl" else h * 0.28), y + h + h * 0.34)
    style = "fill:var(--ink)" if filled else f"fill:var(--paper);stroke:var(--ink);stroke-width:{h * 0.09:.1f}"
    tail_pts = f"{tx - h * 0.18:.1f},{y + h - 2:.1f} {tx + h * 0.18:.1f},{y + h - 2:.1f} {tip[0]:.1f},{tip[1]:.1f}"
    shape = (
        f'<polygon points="{tail_pts}" style="{style};stroke-linejoin:round"/>'
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r:.1f}" style="{style}"/>'
    )
    if not filled:  # hide the tail's inner edge so the outline reads as one shape
        shape += (
            f'<rect x="{tx - h * 0.16:.1f}" y="{y + h - h * 0.12:.1f}" width="{h * 0.32:.1f}" height="{h * 0.14:.1f}" '
            f'style="fill:var(--paper)"/>'
        )
    return shape + content


def text_lines(x: float, y: float, w: float, h: float, color: str) -> str:
    """Two short strokes standing in for a line of chat text."""
    out = ""
    for i in range(2):
        ly = y + h * (0.38 + 0.26 * i)
        lw = w * (0.62 if i == 0 else 0.4)
        out += (
            f'<path d="M{x + w * 0.18:.1f} {ly:.1f} H{x + w * 0.18 + lw:.1f}" style="fill:none;stroke:var(--{color});'
            f'stroke-width:{h * 0.11:.1f};stroke-linecap:round"/>'
        )
    return out


def arrow(x: float, y: float, a: float, color: str, width: float) -> str:
    return (
        f'<path d="M{x:.1f} {y + a:.1f} L{x + a:.1f} {y:.1f} M{x + a * 0.35:.1f} {y:.1f} H{x + a:.1f} '
        f'V{y + a * 0.65:.1f}" style="fill:none;stroke:var(--{color});stroke-width:{width:.1f};'
        f'stroke-linecap:round;stroke-linejoin:round"/>'
    )


def chat_pair(x: float, y: float, scale: float) -> str:
    """Your question (outlined, right) above Basir's answer (filled, left, with the direction arrow)."""
    w, h = 300 * scale, 120 * scale
    you = bubble(x + 90 * scale, y, w, h, False, "br", text_lines(x + 90 * scale, y, w, h, "ink"))
    by, bw = y + h + 70 * scale, 330 * scale
    basir = bubble(
        x,
        by,
        bw,
        h * 1.05,
        True,
        "bl",
        arrow(x + bw * 0.14, by + h * 0.25, h * 0.55, "paper", h * 0.12)
        + text_lines(x + bw * 0.30, by, bw * 0.68, h * 1.05, "paper"),
    )
    return you + basir


def scene(route: np.ndarray, target: np.ndarray, left: float, top: float, w: float) -> str:
    """The route and pin, placed with the scan (fractions of the canvas)."""
    path = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in route)
    return (
        f'<g transform="translate({left * PX:.1f} {top * PX:.1f}) scale({w})">'
        f'<path d="{path}" style="fill:none;stroke:var(--route);stroke-width:{14 / w:.1f};stroke-linecap:round;'
        f'stroke-linejoin:round;stroke-dasharray:0.1 {30 / w:.1f}"/>'
        f'<circle cx="{route[0][0]:.1f}" cy="{route[0][1]:.1f}" r="{18 / w:.1f}" style="fill:var(--route)"/>'
        + pin(target[0], target[1], 46 / w)
        + "</g>"
    )


def compositions(route: np.ndarray, target: np.ndarray) -> dict[str, dict]:
    """Where the scan sits (left, top, width as canvas fractions) and what is drawn over it."""
    logo_layer = (0.0, 0.16, 0.84)
    return {
        "basir-logo": {
            "title": "Basir logo",
            "layer": logo_layer,
            "svg": scene(route, target, *logo_layer) + chat_pair(PX * 0.56, PX * 0.03, 0.85),
            "sizes": [512, 256, 128, 64, 32],
        },
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def tinted_png(alpha: np.ndarray, hex_color: str) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    bgra = np.zeros((*alpha.shape, 4), np.uint8)
    bgra[..., 0], bgra[..., 1], bgra[..., 2] = b, g, r
    bgra[..., 3] = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    _, buf = cv2.imencode(".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def standalone_svg(comp: dict, ink: np.ndarray, target: np.ndarray, palette: dict[str, str]) -> str:
    left, top, w = comp["layer"]
    x, y, size = left * PX, top * PX, w * PX
    overlay = re.sub(r"var\(--(ink|paper|route|target)\)", lambda m: palette[m.group(1)], comp["svg"])
    images = "".join(
        f'<image href="{tinted_png(mask, palette[key])}" x="{x:.1f}" y="{y:.1f}" '
        f'width="{size:.1f}" height="{size:.1f}"/>'
        for mask, key in ((ink, "ink"), (target, "target"))
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {PX} {PX}" width="{PX}" height="{PX}" role="img">'
        f"<title>{comp['title']}</title>{images}{overlay}</svg>"
    )


def render_png(svg_path: Path, png_path: Path, size: int) -> None:
    """Rasterise with headless Chrome on a transparent background."""
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "page.html"
        page.write_text(
            f"<!doctype html><html><head><style>html,body{{margin:0;background:transparent}}"
            f"img{{display:block;width:{size}px;height:{size}px}}</style></head>"
            f'<body><img src="file://{svg_path}"></body></html>'
        )
        subprocess.run(
            [
                "google-chrome",
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--default-background-color=00000000",
                f"--window-size={size},{size}",
                "--allow-file-access-from-files",
                f"--screenshot={png_path}",
                f"file://{page}",
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=PROJECT / "logo", help="Output folder (default: logo/).")
    parser.add_argument("--no-png", action="store_true", help="Write the SVGs only.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ink, target, route_px, target_px = voxel_render(load_scene())
    for name, comp in compositions(route_px, target_px).items():
        for theme, palette in PALETTES.items():
            svg_path = (args.out / f"{name}-{theme}.svg").resolve()
            svg_path.write_text(standalone_svg(comp, ink, target, palette))
            if not args.no_png:
                for size in comp["sizes"] if theme == "light" else comp["sizes"][:1]:
                    render_png(svg_path, args.out / f"{name}-{theme}-{size}.png", size)
    print(f"Wrote {len(list(args.out.glob('basir-*')))} files to {args.out}")


if __name__ == "__main__":
    main()
