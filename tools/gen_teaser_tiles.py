#!/usr/bin/env python3
"""Cut the composed teaser into the pieces the animated teaser reveals.

The animated teaser shows the same artwork as the static one, a tile at a
time, so the tiles are CROPPED FROM THE CANONICAL COMPOSED PNG rather than
re-rendered: whatever the paper teaser shows, this shows, byte for byte.

Boxes are computed from generate_fig_teaser_v9_highres.generate()'s own
layout arithmetic (FIG_WIDTH_INCHES 7.1, margin .06, row pad .06, row gap
.10, tile gap .12, tile_h = tile_w / 1.5, bottom header band .24) and then
CHECKED against the render: each method tile must actually have its red or
green border on the pixels the box lands on, or the tool fails.

Only the three method tiles are cropped. Every applications panel is
generated instead -- gen_teaser_apps.py and gen_teaser_nvs.py -- so they
can share the page's font and draw on transparency.

Usage: gen_teaser_tiles.py light|dark [out_dir]
"""
import os
import sys

import numpy as np
from PIL import Image

THEME = sys.argv[1]
assert THEME in ("light", "dark")
OUT = sys.argv[2] if len(sys.argv) > 2 else f"static/teaser/{THEME}"
SRC = f"static/images/teaser_{THEME}.png"

FW, M, RP, RG, TG = 7.1, 0.06, 0.06, 0.10, 0.12
INNER = FW - 2 * M - 2 * RP
TW = (INNER - 2 * TG) / 3.0
TH = TW / 1.5
ROW_H = 0.22 + TH + 2 * RP
FH = M + ROW_H + RG + ROW_H + M
TOP_ROW_Y = M + ROW_H + RG
TILE_Y = TOP_ROW_Y + RP
TITLE_TOP = TILE_Y + TH + 0.20          # title text sits above the tile
BODY_BOT = M + 0.06
COL_TOP = (M + ROW_H) - 0.24
XS = [M + RP + i * (TW + TG) for i in range(3)]

BORDER = {"light": {"mc": (0xb3, 0x27, 0x1e), "inr": (0xb3, 0x27, 0x1e),
                    "pts": (0x1d, 0x8a, 0x3a)},
          "dark":  {"mc": (0xe0, 0x58, 0x4c), "inr": (0xe0, 0x58, 0x4c),
                    "pts": (0x5e, 0xcb, 0x7c)}}[THEME]

im = Image.open(SRC).convert("RGBA")
W, H = im.size
PPI = W / FW
assert abs(H - FH * PPI) < 2, f"{H} vs {FH * PPI:.1f}: teaser layout drifted"
arr = np.asarray(im)


def crop(x_in, y_lo_in, w_in, h_in, name):
    """y_lo_in is measured from the FIGURE BOTTOM, as matplotlib reports it."""
    x0 = int(round(x_in * PPI))
    x1 = int(round((x_in + w_in) * PPI))
    y0 = int(round((FH - (y_lo_in + h_in)) * PPI))     # top edge, px from top
    y1 = int(round((FH - y_lo_in) * PPI))
    os.makedirs(OUT, exist_ok=True)
    im.crop((x0, y0, x1, y1)).save(os.path.join(OUT, name))
    return x0, y0, x1, y1, name


rows = []
for i, key in enumerate(("mc", "inr", "pts")):
    # the border check: a few px inside the tile's own top edge
    xm = int(round((XS[i] + TW / 2) * PPI))
    ym = int(round((FH - (TILE_Y + TH)) * PPI)) + 3
    got, want = arr[ym, xm, :3].astype(int), np.array(BORDER[key])
    d = int(np.abs(got - want).max())
    assert d <= 12, (f"{key}: box lands on {tuple(got)}, expected the "
                     f"{tuple(want)} border (diff {d})")
    rows.append(crop(XS[i], TILE_Y, TW, TITLE_TOP - TILE_Y, f"tile_{key}.png"))

# Only NVS is cropped now. Product-agnostic rendering is REBUILT by
# gen_teaser_apps.py (two rows instead of three, bigger cells), so cropping
# it here would overwrite that.
# Nothing from the applications row is cropped any more: all four panels
# are generated (gen_teaser_apps.py, gen_teaser_nvs.py) so they share the
# page's sans face and draw on transparency. Cropping could do neither.

for x0, y0, x1, y1, name in rows:
    print(f"  {name:<20s} {x1 - x0:4d} x {y1 - y0:4d} px   "
          f"@ ({x0}, {y0})")
print(f"  [check] all three method tiles land on their own border colour")
print(f"wrote {len(rows)} tiles to {OUT}")
