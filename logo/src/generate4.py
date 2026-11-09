"""Round 4: the real voxel scan combined with conversation and guidance elements.

Compositions (all on the same voxel render of the fr1/room scan):
  1 Conversation   scan + a two-bubble chat icon (your question, Basir's answer) + pin
  2 Eye badge      scan inside an eye-ring ("Basir" = the one who sees) + chat bubble + pin
  3 Phone chat     scan on a phone screen with the conversation beside it
  4 Answer chip    scan + pin + Basir's actual reply chip ("↗ <real route length> m")
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).parent
PROJECT = sys.argv[1]
from generate3 import PX, load_scene, png_mask, voxel_view  # noqa: E402

# ---------------------------------------------------------------------------
# Elements (1024 x 1024 canvas)
# ---------------------------------------------------------------------------


def pin(x: float, y: float, s: float) -> str:
    """Location pin with its tip at (x, y); s is the head radius."""
    hy = y - 1.5 * s
    d = (f"M{x:.1f} {y:.1f} C{x - .5 * s:.1f} {y - .55 * s:.1f} {x - s:.1f} {hy + .55 * s:.1f} {x - s:.1f} {hy:.1f} "
         f"A{s:.1f} {s:.1f} 0 1 1 {x + s:.1f} {hy:.1f} C{x + s:.1f} {hy + .55 * s:.1f} {x + .5 * s:.1f} "
         f"{y - .55 * s:.1f} {x:.1f} {y:.1f}Z")
    return (f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{s * .7:.1f}" ry="{s * .25:.1f}" style="fill:var(--ink);opacity:.25"/>'
            f'<path d="{d}" style="fill:var(--target)"/>'
            f'<circle cx="{x:.1f}" cy="{hy:.1f}" r="{s * .42:.1f}" style="fill:var(--paper)"/>')


def bubble(x: float, y: float, w: float, h: float, filled: bool, tail: str, content: str) -> str:
    """A chat bubble. tail: 'bl' (bottom-left, Basir) or 'br' (bottom-right, you)."""
    r = h * 0.32
    tx = x + w * (0.22 if tail == "bl" else 0.78)
    tip = (tx + (-h * .28 if tail == "bl" else h * .28), y + h + h * .34)
    style = "fill:var(--ink)" if filled else f"fill:var(--paper);stroke:var(--ink);stroke-width:{h * .09:.1f}"
    tail_pts = f"{tx - h * .18:.1f},{y + h - 2:.1f} {tx + h * .18:.1f},{y + h - 2:.1f} {tip[0]:.1f},{tip[1]:.1f}"
    shape = (f'<polygon points="{tail_pts}" style="{style};stroke-linejoin:round"/>'
             f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r:.1f}" style="{style}"/>')
    if not filled:  # hide the tail's inner edge so the outline reads as one shape
        shape += (f'<rect x="{tx - h * .16:.1f}" y="{y + h - h * .12:.1f}" width="{h * .32:.1f}" height="{h * .14:.1f}" '
                  f'style="fill:var(--paper)"/>')
    return shape + content


def text_lines(x: float, y: float, w: float, h: float, color: str, n: int = 2) -> str:
    out = ""
    for i in range(n):
        ly = y + h * (0.38 + 0.26 * i)
        lw = w * (0.62 if i == 0 else 0.4)
        out += (f'<path d="M{x + w * .18:.1f} {ly:.1f} H{x + w * .18 + lw:.1f}" style="fill:none;stroke:var(--{color});'
                f'stroke-width:{h * .11:.1f};stroke-linecap:round"/>')
    return out


def arrow(x: float, y: float, a: float, color: str, width: float) -> str:
    return (f'<path d="M{x:.1f} {y + a:.1f} L{x + a:.1f} {y:.1f} M{x + a * .35:.1f} {y:.1f} H{x + a:.1f} V{y + a * .65:.1f}" '
            f'style="fill:none;stroke:var(--{color});stroke-width:{width:.1f};stroke-linecap:round;stroke-linejoin:round"/>')


def chat_pair(x: float, y: float, scale: float) -> str:
    """Your question (outlined, right) above Basir's answer (filled, left, with the direction arrow)."""
    w, h = 300 * scale, 120 * scale
    you = bubble(x + 90 * scale, y, w, h, False, "br", text_lines(x + 90 * scale, y, w, h, "ink"))
    by, bw = y + h + 70 * scale, 330 * scale
    basir = bubble(x, by, bw, h * 1.05, True, "bl",
                   arrow(x + bw * .14, by + h * .25, h * .55, "paper", h * .12)
                   + text_lines(x + bw * .30, by, bw * .68, h * 1.05, "paper"))
    return you + basir


def route_and_pin(route: np.ndarray, target: np.ndarray, k: float = 1.0) -> str:
    path = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in route)
    return (f'<path d="{path}" style="fill:none;stroke:var(--route);stroke-width:{14 / k:.1f};stroke-linecap:round;'
            f'stroke-linejoin:round;stroke-dasharray:0.1 {30 / k:.1f}"/>'
            f'<circle cx="{route[0][0]:.1f}" cy="{route[0][1]:.1f}" r="{18 / k:.1f}" style="fill:var(--route)"/>'
            + pin(target[0], target[1], 46 / k))


# ---------------------------------------------------------------------------
# Compositions: where the scan sits (left, top, width as fractions) + extra elements
# ---------------------------------------------------------------------------


def compositions(route: np.ndarray, target: np.ndarray, metres: float) -> list[dict]:
    def scene(left: float, top: float, w: float) -> str:
        return (f'<g transform="translate({left * PX:.1f} {top * PX:.1f}) scale({w})">'
                + route_and_pin(route, target, w) + "</g>")

    chip_x, chip_y = target[0] * .9 + PX * .05, target[1] * .9 + PX * .10
    chip = (f'<g><path d="M{chip_x:.1f} {chip_y - 92:.1f} V{chip_y - 150:.1f}" style="stroke:var(--ink);stroke-width:6;opacity:.5"/>'
            f'<rect x="{chip_x - 60:.1f}" y="{chip_y - 270:.1f}" width="510" height="124" rx="62" style="fill:var(--ink)"/>'
            + arrow(chip_x - 12, chip_y - 240, 62, "paper", 14)
            + f'<text x="{chip_x + 80:.1f}" y="{chip_y - 186:.1f}" style="fill:var(--paper);font-family:Readex Pro,Segoe UI,'
              f'system-ui,sans-serif;font-weight:700;font-size:68px">{metres:.1f} m</text>'
            + bubble(chip_x + 330, chip_y - 252, 92, 64, False, "br", "") + "</g>")

    return [
        dict(key="conversation", tag="1 · CONVERSATION", title="The scan, with a two-person chat in the corner",
             body="Your question in an outlined bubble, Basir's answer in a filled one carrying the direction arrow. "
                  "The amber pin marks the found object on the real table, with the route leading to it.",
             verdict="Says \"you ask, Basir answers, here it is\". The most direct version of your idea, and the chat icon "
                     "still reads at 32 px.",
             layer=(0, .16, .84), svg=scene(0, .16, .84) + chat_pair(PX * .56, PX * .03, .85)),
        dict(key="badge", tag="2 · EYE BADGE", title="The room inside an eye, with Basir speaking",
             body="The scan sits inside an almond eye ring, since Basir (بصير) means \"the one who sees\". "
                  "A small filled chat bubble breaks out of the ring at the top-right.",
             verdict="The most emblem-like: a self-contained badge for an app icon, a sticker or a profile picture. "
                     "The eye carries the name; the bubble carries the chat.",
             layer=(.23, .28, .54),
             svg=(f'<path d="M40 540 C210 200 814 200 984 540 C814 880 210 880 40 540Z" '
                  f'style="fill:none;stroke:var(--ink);stroke-width:34;stroke-linejoin:round"/>'
                  + scene(.23, .28, .54)
                  + bubble(PX * .66, PX * .06, 300, 136, True, "bl", arrow(PX * .66 + 50, PX * .06 + 34, 70, "paper", 16)
                           + text_lines(PX * .66 + 108, PX * .06, 200, 136, "paper")))),
        dict(key="phone", tag="3 · PHONE CHAT", title="The scan on your phone, with the conversation beside it",
             body="A phone outline frames the scan the way the realme app will show it. The two chat bubbles sit "
                  "next to the phone, as if the conversation is happening on it.",
             verdict="Tells the whole product in one picture: phone camera, 3D memory, chat. Best for slides, the README "
                     "hero and app-store art; too detailed for a favicon.",
             layer=(.10, .27, .54),
             svg=(f'<rect x="80" y="70" width="600" height="890" rx="92" style="fill:none;stroke:var(--ink);stroke-width:30"/>'
                  f'<rect x="300" y="108" width="160" height="22" rx="11" style="fill:var(--ink)"/>'
                  + scene(.10, .27, .54) + chat_pair(PX * .52, PX * .40, .78))),
        dict(key="chip", tag="4 · ANSWER CHIP", title="The scan, a pin and Basir's actual reply",
             body=f"A pill above the pin shows Basir's answer: a direction arrow and the distance, {metres:.1f} m, which is "
                  "the real length of the route through the scanned room. A tiny outlined bubble on the pill marks "
                  "it as a message.",
             verdict="The most informative and the most product-like, with a real number in it. Works as a hero image "
                     "or a marketing tile; the chip text is too small below 64 px.",
             layer=(.05, .10, .90), svg=scene(.05, .10, .90) + chip),
    ]


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def main() -> None:
    sc = load_scene()
    ink, target_mask, route_px, target_px, _ = voxel_view(sc)
    metres = float(np.sum(np.linalg.norm(np.diff(sc["route"][:, :2], axis=0), axis=1)))

    css = [f".ink-v{{-webkit-mask-image:url({png_mask(ink)});mask-image:url({png_mask(ink)})}}",
           f".tgt-v{{-webkit-mask-image:url({png_mask(target_mask)});mask-image:url({png_mask(target_mask)})}}"]
    cards = []
    for c in compositions(route_px, target_px, metres):
        left, top, w = c["layer"]
        pos = f"left:{left * 100:.1f}%;top:{top * 100:.1f}%;width:{w * 100:.1f}%;height:{w * 100:.1f}%"
        mark = (f'<div class="scan"><div class="layer ink ink-v" style="{pos}"></div>'
                f'<div class="layer tgt tgt-v" style="{pos}"></div>'
                f'<svg viewBox="0 0 {PX} {PX}" xmlns="http://www.w3.org/2000/svg">{c["svg"]}</svg></div>')
        cards.append(f"""
  <section class="concept" aria-labelledby="t-{c['key']}">
    <div class="hero" role="img" aria-label="{c['title']}">{mark}</div>
    <div class="info">
      <div class="tag">{c['tag']}</div>
      <h2 id="t-{c['key']}">{c['title']}</h2>
      <p>{c['body']}</p>
      <p class="says">{c['verdict']}</p>
      <div class="row">
        <div class="size"><div class="box" style="width:64px">{mark}</div>64</div>
        <div class="size"><div class="box" style="width:32px">{mark}</div>32</div>
        <div class="light-tile"><div class="box" style="width:72px">{mark}</div></div>
        <div class="dark-tile"><div class="box" style="width:72px">{mark}</div></div>
      </div>
      <div class="lockup"><div class="box" style="width:60px">{mark}</div><div class="word">Basir<small>بصير</small></div></div>
    </div>
  </section>""")

    page = (HERE / "template3.html").read_text()
    page = page.replace("Basir · identity · round 3 · the real scan", "Basir · identity · round 4 · scan + chat")
    page = page.replace("Three 3D views <span>of the room Basir rebuilt</span>",
                        "The real scan, <span>now talking to you</span>")
    page = page.replace(
        "All three come from the same Phase 0 reconstruction of the TUM fr1/room sequence, rotated so the room is square to the axes.",
        "Every composition uses the voxel view of the room Basir rebuilt in Phase 0, plus the elements that say "
        "what Basir does: a chat with someone, a pin on the found object, and Basir's reply.")
    page = page.replace("/*{{masks}}*/", "\n".join(css)).replace("<!--{{cards}}-->", "\n".join(cards))
    page = page.replace(
        "<p><strong>Picking one:</strong> A is the strongest logo (solid, readable small). B is the strongest hero image. C best explains what Basir does. A common pairing is A as the mark with B or C as the README banner.</p>",
        "<p><strong>Picking one:</strong> 1 is your idea at its simplest. 2 is the best app icon. 3 explains the whole product. "
        "4 shows a real answer. A strong set is 1 or 2 as the logo, with 3 or 4 as the README banner.</p>")
    page = page.replace("(voxel size, camera angle, cut height)", "(bubble placement, pin size, voxel size)")
    (HERE / "index.html").write_text(page)
    print(f"wrote index.html {(HERE / 'index.html').stat().st_size / 1024:.0f} KB; route {metres:.2f} m")


if __name__ == "__main__":
    main()
