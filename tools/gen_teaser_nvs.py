#!/usr/bin/env python3
"""Render the Novel View Synthesis panel instead of cropping it.

It used to be cut out of the composed paper teaser, which meant its row
labels (GT / 3DPS / RS) were stuck in the paper's serif while every other
piece of teaser text on the page is the page's own sans face. Cropping
also forced the panel to carry an opaque background.

So the column is drawn here directly, calling the paper generator's OWN
_draw_nvs_column so the artwork stays identical, with two changes: the
font is overridden to sans AFTER the module's apply_paper_font() runs,
and the figure is transparent. The column's header is placed above the
figure's top edge so matplotlib clips it -- the animated teaser sets that
title in HTML.

Usage: gen_teaser_nvs.py light|dark [out_dir]      (needs TDPS_ROOT)
"""
import ast
import json
import os
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import root, REPO  # noqa: E402

THEME = sys.argv[1]
assert THEME in ("light", "dark")
OUT = sys.argv[2] if len(sys.argv) > 2 else f"static/teaser/{THEME}"
ROOT = root("TDPS_ROOT")
SCENE, TEST_F = "seq_1_frame_185", 185
HIRES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "gen_hires_figs.py")

# the page's light/dark token maps live in gen_hires_figs.py; read them
# rather than keep a second copy that can drift
name = "TEASER_LIGHT" if THEME == "light" else "TEASER_DARK"
tokens = None
for node in ast.walk(ast.parse(open(HIRES).read())):
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == name:
        tokens = ast.literal_eval(node.value)
assert tokens, f"{name} not found in gen_hires_figs.py"

path = os.path.join(ROOT, "figures", "generate_fig_teaser_v9_highres.py")
src = open(path).read()
for k, v in tokens.items():
    src = src.replace(k, v)
G = types.ModuleType("teaser_nvs")
G.__file__ = path
sys.modules[G.__name__] = G
exec(compile(src, os.path.basename(path), "exec"), G.__dict__)

# the module calls apply_paper_font() at import, which pins a Times-like
# serif; the page is sans, so override once the module has loaded
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"],
                     "mathtext.fontset": "dejavusans"})

run = os.path.join(ROOT, "mm25DGS_v5_v4", "output_frame_nvs",
                   f"{SCENE}_train8frames_1loops_test{TEST_F}_loop0_pass2_N20000")
test_cc = float(json.load(open(os.path.join(run, "results.json")))
                .get("final_test_cc", 0.0))
panel_dir = os.path.join(ROOT, "output", "teaser_panels", SCENE, "v4")

# same box as the crop it replaces: the column width from the teaser's own
# layout, and the body height with the 0.22in header band removed
W = (7.1 - 4 * 0.06 - 2 * 0.12) / 3.0
H = 1.51111 - 0.22
HEAD = 0.18                      # header band inside _draw_nvs_column

# The row labels are drawn ha="right" and extend LEFT of the grid, into
# the 0.28in label column. In the composed teaser the column sat inside a
# padded tile, so there was slack; here x0 = 0 would put "3DPS" off the
# edge. Measure the widest label and shift the column by what it needs.
probe = plt.figure(figsize=(2, 1), dpi=600)
pw = max(probe.text(0.1, 0.5, t, fontsize=7.2, fontweight="bold")
         .get_window_extent(probe.canvas.get_renderer()).width / 600
         for t in ("GT", "3DPS", "RS"))
plt.close(probe)
X0 = max(0.0, pw - (0.28 - 0.04) + 0.015)

fig = plt.figure(figsize=(W, H), dpi=600)
fig.patch.set_facecolor("none")
# y0 = 0 with h = H + HEAD puts body_top exactly at the figure's top edge,
# so the column's own title is drawn above it and clipped away
G._draw_nvs_column(fig, X0, 0.0, W - X0, H + HEAD, W, H, panel_dir, test_cc)

fig.canvas.draw()
rend = fig.canvas.get_renderer()
for t in fig.texts:
    if not t.get_text().strip():
        continue
    bb = t.get_window_extent(rend)
    if bb.y0 / 600 > H - 0.05:          # the clipped-away column header
        continue
    assert bb.x0 >= -1 and bb.x1 <= W * 600 + 1, (
        f"{t.get_text()!r} runs off the panel: "
        f"x {bb.x0 / 600:.3f}..{bb.x1 / 600:.3f} in of {W:.3f}")
os.makedirs(OUT, exist_ok=True)
out = os.path.join(OUT, "app_nvs.png")
fig.savefig(out, dpi=600, facecolor="none", edgecolor="none")
plt.close(fig)

from PIL import Image  # noqa: E402
im = Image.open(out)
print(f"  app_nvs.png          {im.size[0]} x {im.size[1]} px   "
      f"sans, transparent; column shifted {X0:.3f} in so labels fit")
