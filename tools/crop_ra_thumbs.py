#!/usr/bin/env python3
"""Re-cut the viewer's range-azimuth thumbnails to the DATA ONLY.

export_viewer.py's ra_thumb() just downsized the source figure, so every
thumbnail carried matplotlib's white margin, axes, tick labels and colour
bar. In a 180px overlay that chrome is unreadable and crowds out the map
it is supposed to frame.

The RA panel is the largest dark connected region in the figure (the
colour bar is dark too, but narrow and separate), so the crop needs no
hard-coded pixel boxes and survives a change of figure size. Only
static/viewer/ra/*.png is touched -- the point/mesh blobs and the JSON
are left alone.

Usage: crop_ra_thumbs.py            (needs TDPS_ROOT)
"""
import os
import re
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import root, REPO  # noqa: E402

ROOT = root("TDPS_ROOT")
RA_DIR = os.path.join(REPO, "static", "viewer", "ra")
NVS = os.path.join(ROOT, "mm25DGS_v5_v4", "output_frame_nvs")
OUT_PX = 320                      # ~2x the 180px slot, so it stays sharp
DARK = 110


def run_dir(scene):
    for cand in sorted(os.listdir(NVS)):
        if cand.startswith(scene):
            return os.path.join(NVS, cand)
    raise SystemExit(f"no run directory for {scene}")


def data_box(a):
    """Bounding box of the largest dark blob = the RA panel."""
    lab, n = ndimage.label(a.mean(axis=2) < DARK)
    if not n:
        return None
    k = int(np.argmax(np.bincount(lab.ravel())[1:]))
    y, x = ndimage.find_objects(lab)[k]
    # the blob's box still catches the anti-aliased axes spine, so pull in
    # by 1% a side -- a couple of pixels at source scale
    ix = max(2, round(0.01 * (x.stop - x.start)))
    iy = max(2, round(0.01 * (y.stop - y.start)))
    return x.start + ix, y.start + iy, x.stop - ix, y.stop - iy


done = skipped = 0
for fn in sorted(os.listdir(RA_DIR)):
    m = re.fullmatch(r"(seq_\d+_frame_(\d+))_f(\d+)_(gt|rd)\.png", fn)
    if not m:
        continue
    scene, test_f, fr, tag = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    d = run_dir(scene)
    if fr != test_f:
        d = os.path.join(d, "train_frames", f"frame_{fr}")
    src = os.path.join(d, f"{'gt' if tag == 'gt' else 'rendered'}_ra_linear.png")
    if not os.path.exists(src):
        skipped += 1
        continue
    im = Image.open(src).convert("RGB")
    box = data_box(np.asarray(im).astype(int))
    if box is None:
        skipped += 1
        continue
    cut = im.crop(box)
    cut.thumbnail((OUT_PX, OUT_PX), Image.LANCZOS)
    cut.save(os.path.join(RA_DIR, fn), optimize=True)
    done += 1

print(f"  [check] re-cut {done} thumbnails to the RA panel alone"
      f"{f', skipped {skipped} with no source' if skipped else ''}")
