#!/usr/bin/env python3
"""The two application panels the static teaser does not have separately.

The paper teaser folds compression, material reconstruction and normals
estimation into one column. The animated teaser lists four applications,
so material reconstruction and normal estimation get a panel each. They
are drawn to the SAME box as the cropped columns (gen_teaser_tiles.py),
in the same typography and theme tokens, from the same point data and the
same hero camera the overview figure uses -- so a panel built here and a
panel cropped from the teaser sit side by side without seams.

Open3D offscreen has no EGL here, so the clouds are matplotlib scatters
through the hero camera, exactly as gen_overview_3dps.py draws them.

Usage: gen_teaser_apps.py light|dark [out_dir]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import root  # noqa: E402
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THEME = sys.argv[1]
assert THEME in ("light", "dark")
OUT = sys.argv[2] if len(sys.argv) > 2 else f"static/teaser/{THEME}"
VIEWER, SCENE = "static/viewer", "seq_1_frame_185"

# the teaser's own tokens, through gen_hires_figs.py's theme map
if THEME == "light":
    TILE, HEAD, SUB = "#ffffff", "#1f2024", "#5b5c63"
    MESH = np.array([0.72, 0.73, 0.78])
else:
    TILE, HEAD, SUB = "#26272c", "#e8e8ec", "#9696a0"
    MESH = np.array([0.34, 0.35, 0.39])

# the cropped application columns are 2.2067 x 1.5111 in at 600 dpi
W, H, DPI = (7.1 - 4 * 0.06 - 2 * 0.12) / 3.0, 1.51111, 600
R = 0.08 * 0.55                                   # the column tile's radius

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "mathtext.fontset": "dejavuserif"})

meta = json.load(open(os.path.join(VIEWER, f"{SCENE}.json")))
raw = open(os.path.join(VIEWER, f"{SCENE}.bin"), "rb").read()
o, N = meta["offsets"], meta["points"]
pos = np.frombuffer(raw, "<f4", o["pos"][1] // 4, o["pos"][0]).reshape(N, 3).astype(float)
nrm = np.frombuffer(raw, np.uint8, o["nrm"][1], o["nrm"][0]).reshape(N, 3) / 255.0
mat = np.frombuffer(raw, np.uint8, o["mat"][1], o["mat"][0]).reshape(N, 3) / 255.0
mv = np.frombuffer(raw, "<f4", o["mv"][1] // 4, o["mv"][0]).reshape(-1, 3).astype(float)

HERO_POS = np.array([-9.0, -5.5, 7.0])
HERO_TGT = np.array([1.8, 4.4, 0.4])
HERO_LENS, SENSOR_MM, ASPECT = 30.0, 36.0, 16.0 / 9.0


def project(p):
    p = np.atleast_2d(np.asarray(p, float))
    fwd = HERO_TGT - HERO_POS
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    v = p - HERO_POS
    zc = np.maximum(v @ fwd, 1e-6)
    half_h = SENSOR_MM / 2.0 / HERO_LENS
    return (v @ right) / zc / half_h, (v @ up) / zc / (half_h / ASPECT), v @ fwd


def panel(name, colours):
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.patch.set_facecolor("none")
    ax = fig.add_axes([0.03, 0.035, 0.94, 0.93], zorder=2)
    ax.set_facecolor("none"); ax.axis("off")
    for p, c, s in ((mv, np.tile(MESH, (len(mv), 1)), 0.035),
                    (pos, colours, 0.05)):
        u, v, d = project(p)
        k = d > 0.1
        ax.scatter(u[k] * ASPECT, v[k], s=s, c=np.asarray(c)[k],
                   linewidths=0, marker="o", rasterized=True)
    u, v, d = project(pos)
    k = d > 0.1
    cx, cy = (u[k].min() + u[k].max()) / 2 * ASPECT, (v[k].min() + v[k].max()) / 2
    hw = np.ptp(u[k]) * ASPECT / 2 * 0.90      # fill the tile, not 4% inside it
    hh = np.ptp(v[k]) / 2 * 0.90
    box = (W * 0.94) / (H * 0.93)
    if hw / hh < box:
        hw = hh * box
    else:
        hh = hw / box
    ax.set_xlim(cx - hw, cx + hw); ax.set_ylim(cy - hh, cy + hh)
    ax.set_aspect("equal")
    os.makedirs(OUT, exist_ok=True)
    p_out = os.path.join(OUT, name)
    fig.savefig(p_out, dpi=DPI, facecolor="none", edgecolor="none")
    plt.close(fig)
    from PIL import Image
    print(f"  {name:<20s} {Image.open(p_out).size[0]} x "
          f"{Image.open(p_out).size[1]} px")


panel("app_material.png", mat)
# normals are rgb=(n+1)/2 and so inherently pastel; on the light tile they
# wash out, the same effect measured in the viewer (median 2.27:1 against
# the background). Same correction, same factor, hue preserved.
NRM = nrm * 0.75 if THEME == "light" else nrm
panel("app_normals.png", NRM)


# ── product-agnostic rendering ───────────────────────────────────────────
# Rebuilt rather than cropped: the teaser's column carries a GT row that is
# not the point here, and three rows left the ADC / CRP / RA cells small.
# Two rows -- ours and the prior methods -- make each cell ~46% larger.
PANELS = os.path.join(root("TDPS_ROOT"), "output/teaser_panels", SCENE, "v4")
RULE = "#c9cad2" if THEME == "light" else "#4a4b54"
OURS_C = "#1d8a3a" if THEME == "light" else "#5ecb7c"
PRIOR_C = "#b3271e" if THEME == "light" else "#e0584c"


def rendering():
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.patch.set_facecolor("none")

    # Two things were wasting the panel. The row labels ran HORIZONTALLY,
    # so "Prior Methods" claimed 0.45in of a 2.21in panel; rotated upright
    # they cost their text height instead. And after the title moved to
    # HTML the block still hung from the top, leaving ~0.42in dead at the
    # bottom. The grid is now sized by whichever axis binds and centred in
    # both, with the margins cut to the minimum that keeps the arrows clear.
    MG, AW, GAP, ROW_PT, HDR = 0.025, 0.115, 0.035, 5.6, 0.095
    LW = 0.10                                  # upright label: its height
    c = min((W - 2 * MG - LW - 2 * AW) / 3.0,
            (H - 2 * MG - HDR - GAP) / 2.0)    # square cells
    gw, gh = LW + 3 * c + 2 * AW, HDR + 2 * c + GAP
    x0 = (W - gw) / 2 + LW                     # first cell's left edge
    top = (H + gh) / 2
    hdr_y = top - HDR / 2
    y1 = top - HDR - c                         # row 1 (3DPS)
    y0 = y1 - GAP - c                          # row 2 (prior methods)
    xs = [x0 + k * (c + AW) for k in range(3)]
    cw = ch = c

    def ax_at(x, y, img=None):
        a = fig.add_axes([x / W, y / H, cw / W, ch / H], zorder=2)
        a.set_facecolor("none")
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_color(RULE); sp.set_linewidth(0.6)
        a.set_xticks([]); a.set_yticks([])
        if img is not None:
            im = plt.imread(os.path.join(PANELS, img))
            a.imshow(im)
            a.set_xlim(0, im.shape[1]); a.set_ylim(im.shape[0], 0)
        return a

    for k, t in enumerate(("ADC", "CRP", "RA")):
        fig.text((xs[k] + cw / 2) / W, hdr_y / H, t, ha="center", va="center",
                 fontsize=5.8, color=SUB, zorder=3)
    ax_at(xs[0], y1, "panel_adc_ours_train.png")
    ax_at(xs[1], y1, "panel_crp_ours.png")
    ax_at(xs[2], y1, "panel_ra_ours_train.png")
    ax_at(xs[2], y0, "panel_ra_radarsplat_train.png")
    for k in (0, 1):                           # prior methods reach neither
        a = ax_at(xs[k], y0)
        a.set_xlim(0, 1); a.set_ylim(0, 1)
        # DejaVu Serif has no U+2715; U+00D7 is in the font
        a.text(0.5, 0.5, "\u00d7", ha="center", va="center", fontsize=13,
               color=PRIOR_C, transform=a.transAxes)
    lx = x0 - 0.045
    for y, txt_, col in ((y1, "3DPS", OURS_C), (y0, "Prior Methods", PRIOR_C)):
        t = fig.text(lx / W, (y + ch / 2) / H, txt_, ha="center", va="center",
                     rotation=90, rotation_mode="anchor", fontsize=ROW_PT,
                     fontweight="bold", color=col, zorder=3)
        fig.canvas.draw()
        bb = t.get_window_extent(fig.canvas.get_renderer())
        assert bb.x0 / DPI > MG * 0.4, (
            f"{txt_} clips: starts at {bb.x0 / DPI:.3f} in, margin {MG}")
        assert bb.height / DPI < ch + 0.02, (
            f"{txt_} is taller than its row: {bb.height / DPI:.3f} vs {ch:.3f}")

    def arrow(ax0, ax1, y, lab, rev):
        fig.patches.append(plt.matplotlib.patches.FancyArrowPatch(
            ((ax1 if rev else ax0) / W, y / H), ((ax0 if rev else ax1) / W, y / H),
            transform=fig.transFigure, arrowstyle="-|>", mutation_scale=4.2,
            lw=0.7, color=SUB, shrinkA=0, shrinkB=0, zorder=3))
        fig.text((ax0 + ax1) / 2 / W, (y + 0.062) / H, lab, ha="center",
                 va="center", fontsize=5.4, color=SUB, zorder=3)
    for y in (y1, y0):
        arrow(xs[0] + cw + 0.016, xs[1] - 0.016, y + ch / 2,
              r"$\mathcal{F}_r^{-1}$", True)
        arrow(xs[1] + cw + 0.016, xs[2] - 0.016, y + ch / 2,
              r"$\mathcal{F}_\theta$", False)
    out = os.path.join(OUT, "app_rendering.png")
    fig.savefig(out, dpi=DPI, facecolor="none", edgecolor="none")
    plt.close(fig)
    from PIL import Image
    print(f"  app_rendering.png    {Image.open(out).size[0]} x "
          f"{Image.open(out).size[1]} px   cells {cw:.3f} in")


rendering()
print(f"wrote 3 panels to {OUT}")
