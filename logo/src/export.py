"""Export the chosen Basir marks as standalone files.

  logo   = composition 1 (Conversation): the real voxel scan + two-bubble chat + pin
  readme = composition 3 (Phone chat):   the scan on a phone + the conversation beside it

Writes light and dark SVGs (colours baked in, scan embedded as tinted PNG) and PNG sizes
into <project>/assets/logo/.
"""

from __future__ import annotations

import base64
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import cv2
import numpy as np

PROJECT = Path(sys.argv[1])
from generate3 import PX, load_scene, voxel_view  # noqa: E402
from generate4 import compositions  # noqa: E402

PALETTES = {
    "light": {"ink": "#15212b", "paper": "#f9fbfa", "route": "#12a39a", "target": "#f08a24"},
    "dark": {"ink": "#e6ecea", "paper": "#15212b", "route": "#2cc7bc", "target": "#ff9d42"},
}
OUT = PROJECT / "logo"


def tinted_png(alpha: np.ndarray, hex_color: str) -> str:
    rgb = [int(hex_color[i : i + 2], 16) for i in (1, 3, 5)]
    rgba = np.zeros((*alpha.shape, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = rgb[2], rgb[1], rgb[0]  # OpenCV writes BGRA
    rgba[..., 3] = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", rgba, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def standalone_svg(comp: dict, ink: np.ndarray, target: np.ndarray, palette: dict[str, str], title: str) -> str:
    left, top, w = comp["layer"]
    x, y, size = left * PX, top * PX, w * PX
    overlay = re.sub(r"var\(--(ink|paper|route|target)\)", lambda m: palette[m.group(1)], comp["svg"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {PX} {PX}" width="{PX}" height="{PX}" role="img">'
        f"<title>{title}</title>"
        f'<image href="{tinted_png(ink, palette["ink"])}" x="{x:.1f}" y="{y:.1f}" width="{size:.1f}" height="{size:.1f}"/>'
        f'<image href="{tinted_png(target, palette["target"])}" x="{x:.1f}" y="{y:.1f}" width="{size:.1f}" height="{size:.1f}"/>'
        f"{overlay}</svg>"
    )


def render_png(svg_path: Path, png_path: Path, size: int) -> None:
    """Rasterise with headless Chrome on a transparent background."""
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "page.html"
        page.write_text(
            f'<!doctype html><html><head><style>html,body{{margin:0;background:transparent}}'
            f'img{{display:block;width:{size}px;height:{size}px}}</style></head>'
            f'<body><img src="file://{svg_path}"></body></html>'
        )
        subprocess.run(
            ["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars",
             "--default-background-color=00000000", f"--window-size={size},{size}",
             "--allow-file-access-from-files", f"--screenshot={png_path}", f"file://{page}"],
            check=True, capture_output=True, timeout=120,
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sc = load_scene()
    ink, target, route_px, target_px, _ = voxel_view(sc)
    metres = float(np.sum(np.linalg.norm(np.diff(sc["route"][:, :2], axis=0), axis=1)))
    comps = {c["key"]: c for c in compositions(route_px, target_px, metres)}

    chosen = {
        "basir-logo": (comps["conversation"], "Basir logo", [512, 256, 128, 64, 32]),
        "basir-readme": (comps["phone"], "Basir: the scanned room on your phone, with the conversation", [1024]),
    }
    for name, (comp, title, sizes) in chosen.items():
        for theme, palette in PALETTES.items():
            svg_path = OUT / f"{name}-{theme}.svg"
            svg_path.write_text(standalone_svg(comp, ink, target, palette, title))
            for size in sizes if theme == "light" else sizes[:1]:
                render_png(svg_path, OUT / f"{name}-{theme}-{size}.png", size)
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:<32} {f.stat().st_size / 1024:7.1f} KB")


if __name__ == "__main__":
    main()
