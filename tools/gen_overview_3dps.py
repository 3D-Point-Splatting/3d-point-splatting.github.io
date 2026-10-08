#!/usr/bin/env python3
"""3DPS method-overview figure in the 4DPS paper's visual grammar.

Four columns, the same reading order the 4DPS overview uses:

  1 the scene at a glance -- the fitted point set with the cascaded radar
    at the origin on its boresight, and the radar's own inputs below it
  2 the scene's inputs -- geometry, normals, materials -- rows aligned
    with the computations they feed
  3 the four closed-form computations, each on a white tile keyed by a
    strip in its own colour, naming the part of the signal it makes
  4 rendered and real products: the complex range profile, the raw ADC
    one invertible transform away, and the |RA| map the loss is taken on

Differences from 4DPS, which the physics forces: 3DPS is static, so the
velocity input and the Doppler splat are gone, and the products are
CRP / ADC / |RA| rather than RD / RAED.

Data (read-only, nothing re-fitted):
  static/viewer/<scene>.bin   the shipped checkpoint's 20,000 oriented
                              points, as packed by tools/export_viewer.py
  teaser_panels/<scene>/v4    the CRP / ADC / |RA| panels already used by
                              the teaser and the current pipeline figure

Usage: gen_overview_3dps.py light|dark <out_dir>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import root  # noqa: E402
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from PIL import Image                                        # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig4dps_common as fc                                  # noqa: E402

THEME = sys.argv[1] if len(sys.argv) > 1 else "light"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "."
SCENE = "seq_1_frame_185"
WEB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(WEB, "static", "viewer")
PANELS = os.path.join(root("TDPS_ROOT"), "output/teaser_panels", SCENE, "v4")
PATTERN = os.path.join(root("TDPS_ROOT"), "assets/antenna_pattern/MMWCAS")

fc.apply_paper_font()

# ── palette ───────────────────────────────────────────────────────────────
# one colour per computation (4DPS's Okabe-Ito keys), plus the page chrome
C_GAIN, C_BSDF, C_PHASE, C_SPLAT = "#E69F00", "#009E73", "#CC79A7", "#E8820C"
T_AMP = "#3b6e55"
T_GAIN, T_BSDF, T_PHASE, T_SPLAT = "#A86F00", "#007656", "#A8508A", "#9A5405"
if THEME == "light":
    BG, TILE, INK, MUTED = "#f5f5f7", "#ffffff", "#1f2024", "#5b5c63"
    BLOCK, RULE = "#eaebef", "#d4d4da"
else:
    BG, TILE, INK, MUTED = "#1a1a1e", "#26272c", "#e8e8ec", "#9696a0"
    BLOCK, RULE = "#32333b", "#3c3d45"
GRN, RED, VIO = "#1E7A46", "#B31B1B", "#6A3FB5"
RULE_INK = "#8a8c94"
TRAJ_C = "#2f7fd0"
HILITE = "#F6D9C0"
T_LEARN = "#8a4b12"
if THEME == "dark":
    GRN, RED, VIO = "#5ecb7c", "#e0584c", "#a98cf0"
    RULE_INK = "#7e8089"
    HILITE = "#4a3426"
    T_LEARN = "#f0b887"

MESH_GREY = "#b9bac2" if THEME == "light" else "#565761"
POSE_BOX = 1.10             # pose cell is slightly wider than tall

W, H = 7.1, 2.52            # 24% shorter than v8; see draw() for the budget
R_TILE = 0.05


# ── primitives (the 4DPS grammar, re-implemented on plain matplotlib) ─────
def _rrect_path(x, y, w, h, radius, corners):
    """Rounded-rect Path in figure fractions with PHYSICALLY circular corners
    and per-corner control (TL, TR, BR, BL) -- 4DPS's _rounded_rect_path. A
    key strip needs its outer corners round and its inner ones square so it
    sits flush inside the tile's edge instead of on top of it."""
    from matplotlib.path import Path
    x0, y0 = x / W, y / H
    x1, y1 = (x + w) / W, (y + h) / H
    rx, ry = min(radius / W, w / W / 2), min(radius / H, h / H / 2)
    k = 0.5523
    tl, tr, br, bl = corners
    v, c = [], []
    if bl:
        v += [(x0, y0 + ry), (x0, y0 + ry * (1 - k)),
              (x0 + rx * (1 - k), y0), (x0 + rx, y0)]
        c += [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    else:
        v.append((x0, y0)); c.append(Path.MOVETO)
    if br:
        v += [(x1 - rx, y0), (x1 - rx * (1 - k), y0),
              (x1, y0 + ry * (1 - k)), (x1, y0 + ry)]
        c += [Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    else:
        v.append((x1, y0)); c.append(Path.LINETO)
    if tr:
        v += [(x1, y1 - ry), (x1, y1 - ry * (1 - k)),
              (x1 - rx * (1 - k), y1), (x1 - rx, y1)]
        c += [Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    else:
        v.append((x1, y1)); c.append(Path.LINETO)
    if tl:
        v += [(x0 + rx, y1), (x0 + rx * (1 - k), y1),
              (x0, y1 - ry * (1 - k)), (x0, y1 - ry)]
        c += [Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    else:
        v.append((x0, y1)); c.append(Path.LINETO)
    v.append(v[0]); c.append(Path.CLOSEPOLY)
    return Path(v, c)


def rrect(fig, x, y, w, h, color, radius=R_TILE, z=1, ec="none", lw=0.0,
          corners=(True, True, True, True)):
    from matplotlib.patches import PathPatch
    p = PathPatch(_rrect_path(x, y, w, h, radius, corners),
                  transform=fig.transFigure, facecolor=color,
                  edgecolor=ec, linewidth=lw, zorder=z)
    fig.patches.append(p)
    return p


def ax_in(fig, x, y, w, h, z=3, polar=False):
    kw = {"projection": "polar"} if polar else {}
    a = fig.add_axes([x / W, y / H, w / W, h / H], zorder=z, **kw)
    a.set_facecolor("none")
    if not polar:                 # a polar panel keeps its grid and spine
        a.axis("off")
    return a


def txt(fig, x, y, s, size=5.6, color=None, ha="center", va="center",
        weight="normal", z=10):
    return fig.text(x / W, y / H, s, ha=ha, va=va, zorder=z,
                    fontsize=size, color=color or INK, fontweight=weight)


HEAD = 0.032


def _poly(fig, pts, color, lw, head_end, head_start, z, ls="-"):
    from matplotlib.patches import FancyArrowPatch
    pts = [tuple(map(float, q)) for q in pts]
    shaft = list(pts)

    def pull(a, b):                      # stop the shaft at the head's base
        d = np.subtract(a, b); L = np.hypot(*d)
        return tuple(np.subtract(a, d / max(L, 1e-9) * HEAD * 0.85))
    if head_end:
        shaft[-1] = pull(pts[-1], pts[-2])
    if head_start:
        shaft[0] = pull(pts[0], pts[1])
    xs, ys = zip(*shaft)
    ln = fig.add_artist(plt.Line2D([x / W for x in xs], [y / H for y in ys],
                                   color=color, lw=lw, zorder=z, ls=ls,
                                   solid_capstyle="butt",
                                   solid_joinstyle="miter"))
    ln.set_snap(False)
    for a, b in ([(pts[-2], pts[-1])] if head_end else []) + \
                ([(pts[1], pts[0])] if head_start else []):
        d = np.subtract(b, a); d = d / max(np.hypot(*d), 1e-9)
        base = np.subtract(b, d * HEAD)
        hp = FancyArrowPatch((base[0] / W, base[1] / H), (b[0] / W, b[1] / H),
                             transform=fig.transFigure, arrowstyle="-|>",
                             mutation_scale=4.4, lw=0, color=color,
                             shrinkA=0, shrinkB=0, zorder=z)
        hp.set_snap(False)
        fig.patches.append(hp)


def arrow(fig, pts, color, lw=0.7, z=20, ls="-"):
    _poly(fig, pts, color, lw, True, False, z, ls)


def arrow2(fig, p0, p1, color, lw=0.7, z=20):
    """Double-headed: a transform, not a gradient path (4DPS's real side
    and its loss both use this)."""
    _poly(fig, [p0, p1], color, lw, True, True, z)


def fwd_back(fig, p0, p1, gap=0.05, lw=0.75, z=21):
    """Straight differentiable run: GREEN forward, RED gradient back,
    side by side -- the 4DPS pair."""
    d = np.subtract(p1, p0); L = np.hypot(*d)
    n = np.array([-d[1], d[0]]) / max(L, 1e-9) * (gap / 2)
    arrow(fig, [np.add(p0, n), np.add(p1, n)], GRN, lw=lw, z=z)
    arrow(fig, [np.subtract(p1, n), np.subtract(p0, n)], RED, lw=lw, z=z)


def _offset_ortho(pts, g):
    """Shift an orthogonal polyline sideways by ``g``, mitring the corners."""
    seg = []
    for a, b in zip(pts, pts[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = max(np.hypot(dx, dy), 1e-9)
        seg.append((-dy / L * g, dx / L * g))
    out = [(pts[0][0] + seg[0][0], pts[0][1] + seg[0][1])]
    for i in range(1, len(pts) - 1):
        a, b = seg[i - 1], seg[i]
        out.append((pts[i][0] + a[0] + b[0], pts[i][1] + a[1] + b[1]))
    out.append((pts[-1][0] + seg[-1][0], pts[-1][1] + seg[-1][1]))
    return out


def fwd_back_path(fig, pts, gap=0.034, lw=0.55, z=21):
    """Green forward and red gradient along one orthogonal route."""
    g = gap / 2
    arrow(fig, _offset_ortho(pts, g), GRN, lw=lw, z=z)
    arrow(fig, _offset_ortho(pts[::-1], g), RED, lw=lw, z=z)


def fwd_back_route(fig, p0, p1, x_mid, gap=0.05, lw=0.7, z=21):
    """The same pair, routed orthogonally (out, across, in) so runs between
    rows stack instead of crossing as diagonals."""
    g = gap / 2
    (x0, y0), (x1, y1) = p0, p1
    arrow(fig, [(x0, y0 + g), (x_mid + g, y0 + g), (x_mid + g, y1 + g),
                (x1, y1 + g)], GRN, lw=lw, z=z)
    arrow(fig, [(x1, y1 - g), (x_mid - g, y1 - g), (x_mid - g, y0 - g),
                (x0, y0 - g)], RED, lw=lw, z=z)


GUARDS = []          # (x0, y0, x1, y1) in inches: arrow runs labels must clear


def guard(x0, y0, x1, y1):
    GUARDS.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))


def bus(fig, pts, color=None, lw=0.5, z=18, dots=True):
    """A fixed (non-learned) input routed orthogonally into a tile, with
    junction dots at the bends -- the grey bus of the 4DPS reference."""
    color = color or RULE_INK
    _poly(fig, pts, color, lw, True, False, z)
    if dots:
        for q in pts[1:-1]:
            fig.add_artist(plt.Line2D([q[0] / W], [q[1] / H], marker="o",
                                      ms=1.4, color=color, zorder=z + 1))


KEY_W = 0.026          # strip width; 4DPS uses 0.0225 on a wider figure


def key_tile(fig, x, y, w, h, col, z=2):
    """Tile keyed by a strip down its LEFT EDGE in the computation's colour.
    The strip is CLIPPED TO THE TILE'S OWN PATH, so it fills the left edge
    exactly -- including the corner curves -- instead of a rounded pill
    sitting on top of the tile. (4DPS gets away with a separately rounded
    strip because its tile radius is close to the strip's; ours is 0.05 vs
    0.014, so the strip would overshoot both rounded ends.) ``col=None``
    keys it with the CRP's own colour bar -- the value axis it makes --
    exactly as 4DPS keys its Amplitude tile with |RD|."""
    tile = rrect(fig, x, y, w, h, TILE, z=z)
    clip = (tile.get_path(), tile.get_transform())
    if col is None:
        a = ax_in(fig, x, y, KEY_W, h, z=z + 1)
        im = a.imshow(np.linspace(0, 1, 128)[:, None], cmap="viridis",
                      aspect="auto", origin="lower", extent=(0, 1, 0, 1))
        a.set_xlim(0, 1); a.set_ylim(0, 1); a.axis("off")
        a.patch.set_visible(False)
        im.set_clip_path(*clip)
    else:
        strip = rrect(fig, x, y, KEY_W, h, col, radius=0.0, z=z + 1)
        strip.set_clip_path(*clip)


# ── data ──────────────────────────────────────────────────────────────────
def load_points():
    meta = json.load(open(os.path.join(VIEWER, f"{SCENE}.json")))
    raw = open(os.path.join(VIEWER, f"{SCENE}.bin"), "rb").read()
    o, N = meta["offsets"], meta["points"]
    pos = np.frombuffer(raw, "<f4", o["pos"][1] // 4, o["pos"][0]).reshape(N, 3)
    nrm = np.frombuffer(raw, np.uint8, o["nrm"][1], o["nrm"][0]).reshape(N, 3)
    mat = np.frombuffer(raw, np.uint8, o["mat"][1], o["mat"][0]).reshape(N, 3)
    mv = np.frombuffer(raw, "<f4", o["mv"][1] // 4, o["mv"][0]).reshape(-1, 3)
    return meta, pos.astype(float), nrm / 255.0, mat / 255.0, mv.astype(float)


# The "Ray Tracing vs Splatting" clips render the scene frame through
# scene_build's hero camera; the point tiles here use the SAME camera so
# the figure and the clips show one view of one scene.
HERO_POS = np.array([-9.0, -5.5, 7.0])
HERO_TGT = np.array([1.8, 4.4, 0.4])
HERO_LENS, SENSOR_MM, ASPECT = 30.0, 36.0, 16.0 / 9.0


def project(p):
    """Perspective projection through the clips' hero camera -> NDC."""
    p = np.atleast_2d(np.asarray(p, float))
    fwd = HERO_TGT - HERO_POS
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    v = p - HERO_POS
    zc = v @ fwd
    half_h = SENSOR_MM / 2.0 / HERO_LENS          # tan(hfov/2), sensor fit = horizontal
    half_v = half_h / ASPECT
    safe = np.maximum(zc, 1e-6)
    u = (v @ right) / safe / half_h
    w = (v @ up) / safe / half_v
    return u, w, zc


def cloud(ax, pos, col, s=0.08, sub=7000, box=1.75, seed=0, pad=1.04,
          keep_lim=False):
    """Scatter the points through the hero camera, auto-fitted to a tile of
    aspect ``box`` so the scene fills it without distorting the view."""
    idx = np.random.default_rng(seed).choice(len(pos), min(sub, len(pos)),
                                             replace=False)
    u, v, d = project(pos[idx])
    keep = d > 0.1
    u, v, d = u[keep] * ASPECT, v[keep], d[keep]      # square up the NDC
    c = np.asarray(col)[idx][keep]
    order = np.argsort(-d)
    ax.scatter(u[order], v[order], s=s, c=c[order], linewidths=0,
               marker="o", rasterized=True)
    if keep_lim:
        ax.set_aspect("equal"); ax.axis("off")
        return
    cx, cy = (u.min() + u.max()) / 2, (v.min() + v.max()) / 2
    hw, hh = np.ptp(u) / 2 * pad, np.ptp(v) / 2 * pad
    if hw / hh < box:
        hw = hh * box
    else:
        hh = hw / box
    ax.set_xlim(cx - hw, cx + hw)
    ax.set_ylim(cy - hh, cy + hh)
    ax.set_aspect("equal")
    ax.axis("off")


def panel(name, invert_for_dark=False):
    a = np.asarray(Image.open(os.path.join(PANELS, name)).convert("RGB"))
    if invert_for_dark and THEME == "dark":
        # saturation-preserving luma inversion: the white ground goes to the
        # deck dark band, the coloured traces keep their own brightness
        rgb = a.astype(np.float32) / 255.0
        mx, mn = rgb.max(axis=2), rgb.min(axis=2)
        L = (mx + mn) / 2.0
        sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
        w = np.clip(sat * 2.5, 0, 1)
        Lout = w * L + (1 - w) * (1.0 - L)
        sc = np.where(L > 1e-5, Lout / np.maximum(L, 1e-5), 0.0)
        a = np.clip(rgb * sc[..., None], 0, 1) * 255.0
        a = a.astype(np.uint8)
    return a


def img_in(fig, x, y, w, h, arr, z=4, frame=None):
    a = ax_in(fig, x, y, w, h, z=z)
    a.imshow(arr, aspect="auto", interpolation="bilinear")
    if frame:
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_color(frame); sp.set_linewidth(0.5)
        a.set_xticks([]); a.set_yticks([]); a.axis("on")
    return a


def stack(fig, x, y, w, h, arr, depth=2, sd=0.028, z=4):  # noqa: E501
    for k in range(depth, -1, -1):
        a = ax_in(fig, x + k * sd, y + k * sd, w, h, z=z - 0.1 * k)
        a.imshow(arr, aspect="auto", interpolation="bilinear",
                 alpha=(1.0, 0.8, 0.6)[k])


# ── the figure ────────────────────────────────────────────────────────────
POSE_RED, BORE_GRN = "#d4211d", "#1d8a3a"      # mmIR's [1,0,0] / [0,0.8,0]
# TX / RX across the radar row. The poster's blue-amber pair, NOT the red-
# green the array cell used to use: in this figure red and green already
# mean "backward" and "forward" gradient, so TX/RX in those colours read as
# a dataflow claim they do not make.
TX_C = "#1565C0" if THEME == "light" else "#40C8FF"
RX_C = "#B45F17" if THEME == "light" else "#FF9E28"


def radar_art(axes, meta):
    """Antenna patterns, array and pose, drawn into the radar row's cells.
    ``radar_art.mesh`` / ``.grey`` / ``.box`` are set by draw() first."""
    a = axes[0]
    # mmIR poster, make_beams_overlay.py: ONE polar axes with TX and RX
    # overlaid, each contributing both cuts -- column 0 (elevation) solid,
    # column 1 (azimuth) at half alpha -- plotted as RAW dBi radius with no
    # normalisation and no dB floor, boresight up, every tick label off.
    ang = np.radians(np.linspace(0.0, 360.0, 361) - 180.0)
    for fname, col in (("tx1_76.npy", TX_C), ("rx1_76.npy", RX_C)):
        d = np.load(os.path.join(PATTERN, fname))
        a.plot(ang, d[:, 0], color=col, lw=0.85)
        a.plot(ang, d[:, 1], color=col, lw=0.6, alpha=0.5)
    a.set_theta_zero_location("N")
    a.set_xticklabels([]); a.set_yticklabels([])
    a.grid(alpha=0.35, linewidth=0.3, color=RULE)
    a.spines["polar"].set_color(RULE)
    a.spines["polar"].set_linewidth(0.4)
    a.patch.set_visible(False)

    a = axes[1]
    pose = [q for q in meta["poses"] if q["test"]][0]
    tx, rx = np.array(pose["tx"]), np.array(pose["rx"])
    el = np.vstack([tx, rx]); pad = 0.013
    a.add_patch(plt.Rectangle((el[:, 0].min() - pad, el[:, 1].min() - pad),
                              np.ptp(el[:, 0]) + 2 * pad, np.ptp(el[:, 1]) + 2 * pad,
                              facecolor=BLOCK, edgecolor=RULE, lw=0.35, zorder=1))
    a.scatter(rx[:, 0], rx[:, 1], s=1.6, c=RX_C, linewidths=0, zorder=3)
    a.scatter(tx[:, 0], tx[:, 1], s=1.6, c=TX_C, linewidths=0, zorder=3)
    a.set_aspect("equal", adjustable="datalim"); a.margins(0.12); a.axis("off")

    a = axes[2]
    # mmIR's pose glyph (visualization_open3d.create_radar_visualization_
    # geometries): red marker at the sensor centre, green boresight arrow,
    # mesh behind it for context. Open3D offscreen has no EGL here, so the
    # same grammar is drawn in matplotlib through the hero camera: the mesh
    # vertices stand in for the semi-transparent grey surface.
    mv_art = radar_art.mesh
    mu, mw, md = project(mv_art)
    keep = md > 0.1
    a.scatter(mu[keep] * ASPECT, mw[keep], s=0.05,
              c=MESH_GREY, linewidths=0, marker="o", rasterized=True)
    # ONE pose, as in mmIR: a single sensor in the scene, not a track
    k = [i for i, q in enumerate(meta["poses"]) if q["test"]][0]
    P = np.array(meta["poses"][k]["pos"], float)
    B = np.array(meta["poses"][k]["bore"], float)
    pu, pw, _ = project(P)
    allP = np.array([q["pos"] for q in meta["poses"]], float)
    sc = float(np.ptp(np.linalg.norm(allP - allP.mean(0), axis=1))) * 1.6 + 0.9
    tu, tw_, _ = project(P + B * sc)
    a.annotate("", (tu[0] * ASPECT, tw_[0]), (pu[0] * ASPECT, pw[0]),
               arrowprops=dict(arrowstyle="-|>", color=BORE_GRN, lw=0.6,
                               mutation_scale=3.0, shrinkA=0, shrinkB=0),
               zorder=5)
    a.plot([pu[0] * ASPECT], [pw[0]], marker="o", ms=2.4, color=POSE_RED,
           mec=TILE, mew=0.25, zorder=6)
    a.set_aspect("equal"); a.axis("off")
    _cu = np.array([pu[0] * ASPECT, tu[0] * ASPECT])
    _cw = np.array([pw[0], tw_[0]])
    _cx, _cy = (_cu.min() + _cu.max()) / 2, (_cw.min() + _cw.max()) / 2
    _r = max(np.ptp(_cu), np.ptp(_cw)) * 1.9 + 0.05   # keep mesh context
    a.set_xlim(_cx - _r * POSE_BOX, _cx + _r * POSE_BOX)
    a.set_ylim(_cy - _r, _cy + _r)


def text_box(size, weight="bold", sample="Input: Scene Representation"):
    """Rendered (width, height) of a title string, in INCHES. Title bands
    are built from this instead of a guessed constant, so the margin above
    a title and the margin below it are equal by construction."""
    f = plt.figure(figsize=(4, 1), dpi=200)
    t = f.text(0.1, 0.5, sample, fontsize=size, weight=weight)
    f.canvas.draw()
    bb = t.get_window_extent(f.canvas.get_renderer())
    plt.close(f)
    return bb.width / 200.0, bb.height / 200.0


def inner_tile(fig, x, y, w, h, label, learned=False, polar=False):
    """One cell of the input grid: centred label at the top, artefact under
    it. A learned parameter gets its label on a highlight chip, the way the
    mmIR poster marks what the optimiser touches."""
    rrect(fig, x, y, w, h, TILE, z=1)
    ly = y + h - 0.068
    if learned:
        rrect(fig, x + w / 2 - 0.175, ly - 0.044, 0.35, 0.089, HILITE,
              radius=0.022, z=10)
    txt(fig, x + w / 2, ly, label, size=5.0, weight="bold",
        color=T_LEARN if learned else INK, z=11)
    return ax_in(fig, x + 0.035, y + 0.03, w - 0.07, h - 0.145, polar=polar)


def draw():
    meta, pos, nrm, mat, mv = load_points()
    grey = np.tile(np.array([0.42, 0.43, 0.48] if THEME == "light"
                            else [0.70, 0.71, 0.75]), (len(pos), 1))
    mgrey = np.tile(np.array([0.63, 0.64, 0.69] if THEME == "light"
                             else [0.46, 0.47, 0.51]), (len(mv), 1))

    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("none")
    rrect(fig, 0.03, 0.03, W - 0.06, H - 0.06, BG, radius=0.08, z=-2)

    # ===== shared layout constants =======================================
    # One margin everywhere: PAD above and below every tile title, MG from
    # an inner tile to its parent's edge, VGAP between the column's three
    # tiles and to the column floor. Title bands are measured, not guessed,
    # so "margin above the title" == "margin below it" exactly.
    PAD = MG = VGAP = 0.07
    OUT = 0.10                   # figure edge -> any tile, all four sides
    TH6 = text_box(6.0, sample="Differentiable Forward Model")[1]
    TW58, TH58 = text_box(5.8, sample="Input: Radar Representation")
    TH55 = text_box(5.5, sample="Amplitude")[1]
    OB6, OB58 = PAD + TH6 + PAD, PAD + TH58 + PAD      # outer title bands
    IB = MG + TH55 + 0.035                             # inner tile band
    CY0, CY1 = OUT, H - OUT            # forward-model column, top and bottom

    # ===== the inputs: two row tiles, three cells each ====================
    gx, gw = OUT, 2.72
    ROW_AR = 2.38 / 0.84               # the v9 row aspect, locked
    RH = gw / ROW_AR
    R2_Y, R1_Y = CY0, CY1 - RH
    G0, G1 = R2_Y + RH, R1_Y           # the row gap
    GC = (G0 + G1) / 2                 # ... and its centre
    cw_i = (gw - 4 * 0.07) / 3.0
    xs_i = [gx + 0.07 + k * (cw_i + 0.07) for k in range(3)]
    cxs = [xi + cw_i / 2 for xi in xs_i]
    CELL_H = RH - OB58 - MG            # both rows: title on top, MG below
    S_CELL_Y, R_CELL_Y = R1_Y + MG, R2_Y + MG
    S_TOP, R_TOP = S_CELL_Y + CELL_H, R_CELL_Y + CELL_H

    for ry, lab in ((R1_Y, "Input: Scene Representation"),
                    (R2_Y, "Input: Radar Representation")):
        rrect(fig, gx, ry, gw, RH, BLOCK, z=0)
        txt(fig, gx + gw / 2, ry + RH - PAD - TH58 / 2, lab, size=5.8,
            weight="bold", z=11)
    scene_cells = [("Geometry", grey, False), ("Normals", nrm, True),
                   ("Materials", mat, True)]
    box_i = (cw_i - 0.07) / (CELL_H - 0.145)
    for (lab, col, learned), xi in zip(scene_cells, xs_i):
        a = inner_tile(fig, xi, S_CELL_Y, cw_i, CELL_H, lab, learned)
        if lab == "Geometry":
            cloud(a, mv, mgrey, s=0.03, sub=len(mv), box=box_i, keep_lim=True)
        cloud(a, pos, col, s=0.022, sub=8000, box=box_i)
    radar_axes = [inner_tile(fig, xi, R_CELL_Y, cw_i, CELL_H, lab,
                             polar=(lab == "Antenna patterns"))
                  for lab, xi in zip(("Pose", "Array", "Antenna patterns"),
                                     xs_i)]
    radar_art.mesh = mv
    radar_art(radar_axes[::-1], meta)   # radar_art draws patterns, array, pose

    # ===== the forward model =============================================
    cx, cw = 2.98, 1.50
    rrect(fig, cx, CY0, cw, CY1 - CY0, BLOCK, z=0)
    txt(fig, cx + cw / 2, CY1 - PAD - TH6 / 2, "Differentiable Forward Model",
        size=6.0, weight="bold", z=11)
    RS_TOP = CY1 - OB6                 # ... and the scene cells' top: equal
    IN_H, IGAP = 0.27, 0.05
    BS_Y = GC - IN_H / 2               # BSDF centred on the row gap
    GA_Y = BS_Y - IGAP - IN_H
    AM_Y = GA_Y - MG
    AM_H = (BS_Y + IN_H) + IB - AM_Y
    RS_Y = AM_Y + AM_H + VGAP
    RS_H = RS_TOP - RS_Y
    PH_Y = CY0 + VGAP
    PH_H = AM_Y - VGAP - PH_Y
    tw = cw - 0.10
    ix, iw2 = cx + 0.05 + MG, tw - 2 * MG

    key_tile(fig, cx + 0.05, RS_Y, tw, RS_H, C_SPLAT)
    txt(fig, cx + cw / 2, RS_Y + RS_H - MG - TH55 / 2, "Range splat",
        size=5.5, weight="bold", color=INK, z=11)
    _ih = RS_H - 0.295                  # equation 0.055, inset, title band
    a = ax_in(fig, cx + 0.34, RS_Y + 0.130, tw - 0.52, _ih, z=12)
    centre = 0.35
    xk = np.linspace(-4.2, 4.2, 400)
    yk = np.abs(np.sinc(xk - centre)) * np.exp(-((xk - centre) / 3.1) ** 2)
    a.fill_between(xk, 0, yk, color=C_SPLAT, alpha=0.18, lw=0)
    a.plot(xk, yk, color=C_SPLAT, lw=0.7)
    kk = np.arange(-4, 5)
    ak = np.abs(np.sinc(kk - centre)) * np.exp(-((kk - centre) / 3.1) ** 2)
    big = ak > 0.04
    a.vlines(kk[big], 0, ak[big], color=T_SPLAT, lw=0.6)
    a.plot(kk[big], ak[big], "o", ms=1.5, color=T_SPLAT, mec="white", mew=0.25)
    a.vlines(kk, -0.07, 0, color=MUTED, lw=0.35)
    a.plot([centre], [-0.03], marker="^", ms=2.2, color=INK, clip_on=False)
    a.annotate("", (4.7, 0), (-4.4, 0), arrowprops=dict(arrowstyle="-|>",
               color=MUTED, lw=0.4, mutation_scale=3, shrinkA=0, shrinkB=0))
    a.text(4.85, -0.02, "$k$", ha="left", va="center", fontsize=4.4, color=T_SPLAT)
    a.set_xlim(-4.6, 5.4); a.set_ylim(-0.14, 1.1); a.axis("off")
    txt(fig, cx + cw / 2, RS_Y + 0.055, r"$k_i=(R^{TX}_i+R^{RX}_i)/2\Delta r$",
        size=5.0, color=T_SPLAT, z=11)

    key_tile(fig, cx + 0.05, AM_Y, tw, AM_H, None)
    txt(fig, cx + cw / 2, AM_Y + AM_H - MG - TH55 / 2, "Amplitude", size=5.5,
        weight="bold", color=INK, z=11)
    rrect(fig, ix, BS_Y, iw2, IN_H, BLOCK, radius=0.03, z=3)
    txt(fig, cx + cw / 2, BS_Y + IN_H - 0.068, "BSDF  (ITU-R P.2040)",
        size=4.7, weight="bold", color=T_BSDF, z=11)
    txt(fig, cx + cw / 2, BS_Y + 0.072,
        r"$f_r(\theta_i,\theta_o,\mathbf{n}_i,\varepsilon'_r,\sigma,t)$",
        size=4.8, color=T_BSDF, z=11)
    rrect(fig, ix, GA_Y, iw2, IN_H, BLOCK, radius=0.03, z=3)
    txt(fig, cx + cw / 2, GA_Y + IN_H - 0.068, "antenna gain", size=4.7,
        weight="bold", color=T_GAIN, z=11)
    txt(fig, cx + cw / 2, GA_Y + 0.072, r"$G_{TX}(\theta_i)\,G_{RX}(\theta_o)$",
        size=4.8, color=T_GAIN, z=11)

    key_tile(fig, cx + 0.05, PH_Y, tw, PH_H, C_PHASE)
    txt(fig, cx + cw / 2, PH_Y + PH_H - MG - TH55 / 2, "Phase", size=5.5,
        weight="bold", color=INK, z=11)
    txt(fig, cx + cw / 2, PH_Y + 0.13,
        r"$\varphi_i=-2\pi(R^{TX}_i+R^{RX}_i)/\lambda$", size=5.0,
        color=T_PHASE, z=11)

    # margins are constructed, not eyeballed -- prove it
    def _eq(name, *v):
        assert max(v) - min(v) < 1e-9, f"{name}: {v}"
    _eq("column title band", CY1 - (CY1 - PAD - TH6 / 2) - TH6 / 2,
        (CY1 - PAD - TH6 / 2) - TH6 / 2 - RS_TOP, PAD)
    for ry in (R1_Y, R2_Y):
        _eq("row title band", ry + RH - (ry + RH - PAD - TH58 / 2) - TH58 / 2,
            (ry + RH - PAD - TH58 / 2) - TH58 / 2 - (ry + MG + CELL_H), PAD)
    _eq("column tile gaps", PH_Y - CY0, AM_Y - (PH_Y + PH_H),
        RS_Y - (AM_Y + AM_H), VGAP)
    _eq("amplitude margins", ix - (cx + 0.05), (cx + 0.05 + tw) - (ix + iw2),
        GA_Y - AM_Y, MG)
    assert RS_TOP == CY1 - OB6 and abs(RS_TOP - (R1_Y + MG + CELL_H)) < 1e-9, \
        "column and scene-row content tops must line up"
    _eq("outer margin", gx, CY0, H - CY1, OUT)
    print(f"  [layout] PAD=MG=VGAP={PAD}; Phase {PH_H:.3f} / Ampl {AM_H:.3f}"
          f" / Rsplat {RS_H:.3f}; content top {RS_TOP:.3f}")

    # ===== wiring =========================================================
    # Pose rises straight into the node: at cxs[0] it is clear of the radar
    # title. Array cannot -- the title spans its whole cell -- so it rises
    # into the band under the text, runs right beneath it, and turns up
    # only past the title's measured right edge. The node moves there to
    # meet it, which is why Materials now enters the BSDF ABOVE the bus.
    T_L, T_R = gx + gw / 2 - TW58 / 2, gx + gw / 2 + TW58 / 2
    assert cxs[0] + 0.02 < T_L, "pose riser would cross the radar title"
    JX, JY = T_R + 0.085, GC
    SPX, PAT_X = 2.92, 2.86
    CORR = R_TOP + PAD / 2                 # the run under the title text
    BS_N, BS_G, BS_M = GC - 0.08, GC, GC + 0.08

    def line(pts):
        _poly(fig, pts, RULE_INK, 0.42, False, False, 18)

    line([(cxs[0], S_CELL_Y), (cxs[0], JY), (JX, JY)])       # geometry, down
    line([(cxs[0], R_TOP), (cxs[0], JY)])                    # pose, up
    line([(cxs[1], R_TOP), (cxs[1], CORR), (JX, CORR), (JX, JY)])   # array
    fig.add_artist(plt.Line2D([JX / W], [JY / H], marker="o", ms=2.6,
                              color=RULE_INK, zorder=19))

    for ex, ey in ((cx + 0.02, RS_Y + RS_H / 2),            # range splat
                   (cx + 0.02, PH_Y + PH_H / 2),            # phase
                   (ix, GA_Y + IN_H - 0.07),                # antenna gain
                   (ix, BS_G)):                             # BSDF
        bus(fig, [(JX, JY), (SPX, JY), (SPX, ey), (ex, ey)], lw=0.42,
            dots=False)

    bus(fig, [(cxs[2], R_TOP), (cxs[2], G0 + 0.055), (PAT_X, G0 + 0.055),
              (PAT_X, GA_Y + 0.07), (ix, GA_Y + 0.07)], lw=0.42, dots=False)

    for xc, ey in ((cxs[1] + 0.09, BS_N), (cxs[2] + 0.09, BS_M)):
        fwd_back_path(fig, [(xc, S_CELL_Y), (xc, ey), (ix, ey)])

    # ===== the products ==================================================
    crp_o, crp_g = panel("panel_crp_ours.png"), panel("panel_crp_gt.png")
    ra_o, ra_g = panel("panel_ra_ours_train.png"), panel("panel_ra_gt_train.png")
    adc_o = panel("panel_adc_ours_train.png", True)
    adc_g = panel("panel_adc_gt_train.png", True)
    # Rendered / Real keep their exact structure -- three shadowed stacks,
    # same order, same labels, same arrows. Only the scale follows the
    # figure: this column is the tallest thing in it, so a 26% height cut
    # cannot leave it at absolute size (0.56 -> 0.50 in, 11% smaller).
    ps, xo, xg, SD = 0.50, 4.94, 5.90, 0.028
    shadow = 2 * SD
    # Rendered / Real sit at the EXACT midpoint of the band between the
    # figure's top edge and the top of the ADC stacks, so the text no
    # longer hugs the boundary. Stack spacing follows from that.
    Y_RA = CY0
    Y_ADC = (CY1 - 0.185) - ps - shadow
    _slack = Y_ADC - (Y_RA + ps + shadow) - (ps + shadow)
    Y_CRP = Y_RA + ps + shadow + _slack * 0.557   # more room under the CRP
    TITLE_Y = (H + Y_ADC + ps + shadow) / 2
    txt(fig, xo + ps / 2, TITLE_Y, "Rendered", size=6.0, weight="bold", color=GRN)
    txt(fig, xg + ps / 2, TITLE_Y, "Real", size=6.0, weight="bold", color=MUTED)
    prod = [("ADC", Y_ADC, adc_o, adc_g), ("CRP", Y_CRP, crp_o, crp_g),
            ("RA", Y_RA, ra_o, ra_g)]
    for label, y, o_img, g_img in prod:
        stack(fig, xo, y, ps, ps, o_img, sd=SD)
        stack(fig, xg, y, ps, ps, g_img, sd=SD)
        txt(fig, xg + ps + shadow + 0.12, y + ps / 2, label, size=5.6,
            weight="bold", color=MUTED, ha="left")
    cxo, cxg = xo + ps / 2, xg + ps / 2
    # ADC <-> CRP is an invertible transform on both sides: grey, two-headed
    arrow2(fig, (cxo, Y_CRP + ps + shadow + 0.025), (cxo, Y_ADC - 0.025),
           RULE_INK, lw=0.6)
    arrow2(fig, (cxg, Y_CRP + ps + shadow + 0.025), (cxg, Y_ADC - 0.025),
           RULE_INK, lw=0.6)
    txt(fig, cxo + 0.09, (Y_CRP + ps + shadow + Y_ADC) / 2,
        r"$\mathcal{F}_r^{-1}$", size=5.0, ha="left", color=MUTED)
    # the loss sits on |RA|, so only that leg carries gradients
    # the F_theta pair stops 0.10 in BELOW the CRP so the range-k axis
    # arrow, which spans the stack's full width, passes clear above it
    # right of the "range k" label, which now sits under the range axis
    FTX = cxo + 0.05
    fwd_back(fig, (FTX, Y_CRP - 0.075), (FTX, Y_RA + ps + shadow + 0.025))
    guard(FTX - 0.017, Y_RA + ps + shadow + 0.025, FTX + 0.017, Y_CRP - 0.075)
    arrow2(fig, (cxg, Y_CRP - 0.025), (cxg, Y_RA + ps + shadow + 0.025),
           RULE_INK, lw=0.6)
    txt(fig, FTX + 0.09, (Y_RA + ps + shadow + Y_CRP - 0.050) / 2,
        r"$\mathcal{F}_\theta$", size=5.0, ha="left", color=MUTED)
    fwd_back(fig, (cx + cw + 0.04, Y_CRP + ps / 2), (xo - 0.26, Y_CRP + ps / 2))
    ly = Y_RA + ps / 2
    arrow2(fig, (xo + ps + shadow + 0.05, ly), (xg - 0.05, ly), VIO, lw=0.8)
    txt(fig, (xo + ps + shadow + xg) / 2, ly + 0.14, r"$\mathcal{L}$",
        size=6.2, color=VIO)

    # the CRP's axes, keyed to the computations that make them
    cy0, cy1 = Y_CRP, Y_CRP + ps
    arrow2(fig, (xo, cy0 - 0.045), (xo + ps, cy0 - 0.045), T_SPLAT, lw=0.7)
    # under the arrow and flush with its left end, with the F_theta pair
    # to its right (user's mark-up, 2026-10-07)
    fig.text(xo / W, (cy0 - 0.125) / H, "range  $k$", ha="left", va="center",
             fontsize=4.6, color=T_SPLAT, zorder=11)
    arrow2(fig, (xo - 0.045, cy0), (xo - 0.045, cy1), T_PHASE, lw=0.7)
    fig.text((xo - 0.115) / W, (cy0 + cy1) / 2 / H, "(TX, RX) pairs",
             rotation=90, ha="center", va="center", fontsize=4.6,
             color=T_PHASE, zorder=11)
    # the slices lean up-right, so the frames key runs parallel to them,
    # starting at the top tip of the (TX, RX) axis
    # the whole key slides 0.030 in up-left along the arrow's own normal:
    # swept against the clearance check, that is the smallest offset that
    # leaves the label a comfortable 0.021 in off the CRP stack.
    _nx, _ny = -0.7071 * 0.030, 0.7071 * 0.030
    arrow(fig, [(xo - 0.045 + _nx, cy1 + 0.035 + _ny),
                (xo + 0.055 + _nx, cy1 + 0.135 + _ny)], TRAJ_C, lw=0.6)
    # centred on the arrow and offset along its up-left normal, so the
    # label rides on top of the shaft instead of starting at the tip
    _fx = xo + 0.005 + _nx - 0.7071 * 0.042
    _fy = cy1 + 0.085 + _ny + 0.7071 * 0.042
    fig.text(_fx / W, _fy / H, "frames", rotation=45, rotation_mode="anchor",
             ha="center", va="center", fontsize=4.6, color=TRAJ_C, zorder=11)

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    texts = [t for t in fig.texts if t.get_text().strip()]
    boxes = [t.get_window_extent(rend) for t in texts]
    # clearance, not just overlap: a label that merely fails to intersect
    # by a pixel is still a defect, so the tightest pair is reported and a
    # hard floor of MIN_GAP inches is enforced.
    MIN_GAP = 0.008 * fig.dpi
    def gap(a, b):
        return max(max(a.x0 - b.x1, b.x0 - a.x1),
                   max(a.y0 - b.y1, b.y0 - a.y1))
    from matplotlib.transforms import Bbox
    gboxes = [Bbox.from_extents(x0 * fig.dpi, y0 * fig.dpi,
                                x1 * fig.dpi, y1 * fig.dpi)
              for x0, y0, x1, y1 in GUARDS]
    pairs = []
    for t, tb in zip(texts, boxes):
        for ax in fig.axes:
            pairs.append((gap(tb, ax.get_window_extent()),
                          f"{t.get_text()[:26]} / panel"))
        for gb in gboxes:
            pairs.append((gap(tb, gb), f"{t.get_text()[:26]} / arrow"))
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            pairs.append((gap(boxes[i], boxes[j]),
                          f"{texts[i].get_text()[:16]} / {texts[j].get_text()[:16]}"))
    pairs.sort()
    bad = [f"{n} ({g / fig.dpi:+.4f} in)" for g, n in pairs if g < MIN_GAP]
    assert not bad, "too close: " + "; ".join(bad[:6])
    print(f"  [check] {len(texts)} texts + {len(GUARDS)} guards, tightest {pairs[0][0] / fig.dpi:.4f} in "
          f"({pairs[0][1]}); all >= {MIN_GAP / fig.dpi:.3f} in")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"overview_3dps_{THEME}")
    fig.savefig(out + ".png", dpi=600, facecolor="none", edgecolor="none")
    fig.savefig(out + ".pdf", facecolor="none", edgecolor="none")
    plt.close(fig)
    print("wrote", out + ".{png,pdf}")


draw()
