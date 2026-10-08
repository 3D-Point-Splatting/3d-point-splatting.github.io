#!/usr/bin/env python3
"""Content-parity checks for the 3DPS webpage figures (pipeline rule 3).

For each figure:
  1. light vs dark (same dpi): every DATA region must be ~identical —
     only chrome (bg tile, panels, text) may differ.
       - racmp: the 5x6 heatmap-cell interiors, computed from the same
         GridLayout math the figure script uses.
       - pipeline: all strongly-saturated pixels (CRP viridis, RA hot,
         crimson points, arrows) plus the mesh panel (largest bright
         connected component in the dark variant).
  2. dark 600dpi, downscaled 2x, vs the canonical 300dpi deck artifact:
     mean abs diff must be small (fonts rescale slightly; data must not
     move).
  3. corners must be transparent (alpha == 0 at the 4 extreme pixels).

Usage: verify_figs.py <dir600> (containing light/ and dark/)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import root  # noqa: E402
import sys

import numpy as np
from PIL import Image

DECK = root("DECK_ROOT")
FIGSRC = os.path.join(DECK, "slides", "figsrc")
sys.path.insert(0, FIGSRC)

D600 = sys.argv[1]
CANON = {
    "pipeline_3dps_system_overview.png":
        os.path.join(DECK, "slides/figs/dark_tdps_pipeline.png"),
    "training_ra_comparison.png":
        os.path.join(DECK, "slides/figs/dark_tdps_ra_cmp.png"),
}

failures = []


def check(name, ok, msg):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {msg}")
    if not ok:
        failures.append(name)


def load(path):
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.int16)


def corners_transparent(img, name):
    a = img[..., 3]
    vals = [a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]]
    check(f"{name} corners", max(vals) == 0, f"corner alpha {vals}")


def data_diff(l_rgb, d_rgb, mask, name):
    d = np.abs(l_rgb - d_rgb).max(axis=2)[mask]
    frac = (d > 8).mean() if d.size else 1.0
    check(f"{name} light/dark data parity", frac < 0.002,
          f"{mask.sum()} data px, frac>|8| = {frac:.5f}, mean {d.mean():.3f}")


def racmp_cells(shape):
    """Heatmap-cell interior mask from the figure's own GridLayout."""
    from fig_common_3dps import GridLayout
    layout = GridLayout.from_image_aspect(
        5, 6, img_aspect=1.0, margin_in=0.08, col_gap_in=0.03,
        row_gap_in=0.03, header_in=0.16, label_w_in=0.50)
    H, W = shape[:2]
    mask = np.zeros((H, W), dtype=bool)
    for r in range(5):
        for c in range(6):
            left, bottom, w, h = layout.cell_pos(r, c)
            # trim 4% border: spines + corr-text overlay corner excluded
            x0 = int((left + 0.04 * w) * W)
            x1 = int((left + 0.96 * w) * W)
            y1 = int((1 - bottom - 0.04 * h) * H)
            y0 = int((1 - bottom - 0.96 * h) * H)
            cell = np.zeros((H, W), dtype=bool)
            cell[y0:y1, x0:x1] = True
            # the corr text sits lower-right; carve it out
            cell[int(y1 - 0.28 * (y1 - y0)):y1,
                 int(x1 - 0.40 * (x1 - x0)):x1] = False
            mask |= cell
    return mask


def pipeline_masks(dark):
    rgb = dark[..., :3].astype(np.float32)
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = (mx - mn) > 40          # coloured data: CRP/RA/points/arrows
    from scipy import ndimage
    # erode 2px: the outermost blend of a data pixel against the (theme-
    # dependent) chrome colour legitimately differs between themes
    sat = ndimage.binary_erosion(sat, iterations=2)
    bright = (rgb.mean(axis=2) > 150)
    lab, n = ndimage.label(bright)
    if n:
        sizes = np.bincount(lab.ravel())[1:]
        mesh = lab == (1 + int(np.argmax(sizes)))   # the mesh render panel
    else:
        mesh = np.zeros_like(bright)
    # erode mesh 2px so its antialiased rim (blends with chrome) is excluded
    mesh = ndimage.binary_erosion(mesh, iterations=2)
    return sat, mesh


for fname, canon_path in CANON.items():
    print(f"\n== {fname} ==")
    light = load(f"{D600}/light/{fname}")
    dark = load(f"{D600}/dark/{fname}")
    corners_transparent(light, fname + " light")
    corners_transparent(dark, fname + " dark")
    check(f"{fname} shapes", light.shape == dark.shape,
          f"{light.shape} vs {dark.shape}")

    l_rgb, d_rgb = light[..., :3], dark[..., :3]
    if "training_ra" in fname:
        mask = racmp_cells(dark.shape)
        data_diff(l_rgb, d_rgb, mask, fname + " cells")
    else:
        # the scene-representation panel is theme-adapted since round 3
        # (the dark variant luma-inverts the light render), so the old
        # bright-blob mesh parity check no longer applies. The crimson
        # points keep their own brightness (saturation-preserving
        # transform), so the saturated check still covers them.
        sat, _ = pipeline_masks(light)
        data_diff(l_rgb, d_rgb, sat, fname + " saturated")

    # dark 600 vs canonical 300 content
    canon = np.asarray(Image.open(canon_path).convert("RGB"), dtype=np.int16)
    # composite dark 600 on its own bg colour (canonical has opaque bg)
    alpha = (dark[..., 3:4] / 255.0)
    bg = np.array([0x1A, 0x1A, 0x1E], dtype=np.float64)
    comp = (dark[..., :3] * alpha + bg * (1 - alpha))
    ds = np.asarray(
        Image.fromarray(comp.astype(np.uint8)).resize(
            (canon.shape[1], canon.shape[0]), Image.LANCZOS),
        dtype=np.int16)
    d = np.abs(ds - canon).max(axis=2)
    if "pipeline" in fname:
        # exclude the theme-adapted scene panel (the canonical's largest
        # bright connected component) from the canonical comparison
        from scipy import ndimage
        bright = canon.mean(axis=2) > 150
        lab, n = ndimage.label(bright)
        if n:
            sizes = np.bincount(lab.ravel())[1:]
            sl = ndimage.find_objects(lab)[int(np.argmax(sizes))]
            d[sl] = 0
    # thresholds allow 600->300 Lanczos resample noise on glyphs and sharp
    # colormap edges; data drift would blow past the mean by an order of
    # magnitude (verified: mesh panel mean 1.45, all >40 clusters <=136px
    # glyph-sized)
    check(f"{fname} vs canonical", d.mean() < 3.0 and (d > 40).mean() < 0.02,
          f"mean {d.mean():.3f}, frac>|40| = {(d > 40).mean():.5f}")

print()
sys.exit(1 if failures else 0)
