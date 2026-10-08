#!/usr/bin/env python3
"""Derive the webpage variant of the p5b_pareto Blender clip.

Web edits (user requests, 2026-08-26):
  - no on-screen title (the page section heading carries it)
  - no staged build-up: the 3-D surface and ALL six spheres are present
    from frame 1
  - NO RadarSplat pink flash anywhere (round 2)
  - on the front-face projection RadarSplat's name sits UNDER its
    circle, off the frontier trace (round 2)
  - light|dark theme renders with film_transparent RGBA PNG frames
    (round 2): dark keeps the deck palette; light swaps the chrome
    (text/axes/grid/3DPS green) for the webpage light palette. Data
    colours (plasma surface + trace, sphere greys, mmIR amber) are
    untouched. Output goes to $PARETO_WEB_OUT (a PNG frame prefix).
  - acts 4-8 (traces, both projections, holds) are preserved EXACTLY,
    shifted -365 frames; new length 535 f @ 30 fps (~17.8 s)

The patched script is written to the given path and run with Blender
separately — the original p5b_pareto.py, the deck mp4, and
~/aexam_scratch previews are never touched (runs set AEXAM_SCRATCH to
the session scratchpad and PARETO_WEB_OUT for output).

Usage: gen_pareto_web.py <out_script.py> light|dark
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import root  # noqa: E402
import sys

BLENDER_DIR = os.path.join(root("DECK_ROOT"), "blender")
SRC = os.path.join(BLENDER_DIR, "p5b_pareto.py")
THEME = sys.argv[2] if len(sys.argv) > 2 else "dark"

# (old, new, expected occurrences)
EDITS = [
    # imports / data resolve from the real blender dir, not the copy's
    ("os.path.dirname(os.path.abspath(__file__))",
     f'"{BLENDER_DIR}"', 3),
    # timeline: acts 1-3 collapse to frame 1 (everything visible),
    # acts 4-8 shift by -365 with all durations preserved
    ("F_END = 900", "F_END = 535", 1),
    ("F_SURF = 55", "F_SURF = 1", 1),
    ("F_MARK = 150", "F_MARK = 30", 1),
    ("F_MMIR = 250", "F_MMIR = 70", 1),   # only times the flash end now
    ("F_DPS = 390", "F_DPS = 1", 1),
    ("F_TR1A, F_TR1B = 455, 500", "F_TR1A, F_TR1B = 90, 135", 1),
    ("F_SW1A, F_SW1B = 505, 540", "F_SW1A, F_SW1B = 140, 175", 1),
    ("F_FADE1 = 520", "F_FADE1 = 155", 1),
    ("F_SQ1A, F_SQ1B = 522, 540", "F_SQ1A, F_SQ1B = 157, 175", 1),
    ("F_HOLD1 = 660", "F_HOLD1 = 295", 1),
    ("F_LF1B = 676", "F_LF1B = 311", 1),
    ("F_SURF_IN = 678", "F_SURF_IN = 313", 1),
    ("F_RET = 700", "F_RET = 335", 1),
    ("F_TR2A, F_TR2B = 705, 738", "F_TR2A, F_TR2B = 340, 373", 1),
    ("F_SW2A, F_SW2B = 740, 780", "F_SW2A, F_SW2B = 375, 415", 1),
    ("F_FADE2 = 755", "F_FADE2 = 390", 1),
    ("F_SQ2A, F_SQ2B = 757, 780", "F_SQ2A, F_SQ2B = 392, 415", 1),
    ("F_WIG0, F_WIG1 = 55, F_SW1A", "F_WIG0, F_WIG1 = 1, F_SW1A", 1),
    # the wiggle window shrank 450 -> 139 frames; one loop keeps its pace
    ("R_WIG, N_LOOPS = 0.55, 2", "R_WIG, N_LOOPS = 0.55, 1", 1),
    # all spheres + labels visible from frame 1
    ("plot2d.hide_until(o, f0)",
     "pass  # web edit: visible from frame 1", 1),
    # no baked title
    ('_ttl.body = "3DPS: Pareto Optimal"', '_ttl.body = ""', 1),
    # no off-frontier pink flash, in 3-D or on the front face
    ("    if off:", "    if False:  # web edit: no pink flash", 1),
    # front-face projection: RadarSplat's name goes UNDER its circle
    # (dz != 0.0 selects the label entry; its sphere keeps dz = 0)
    ("""        p3d = (x_true + dx, y_true, pz(acc) + dz)
        for fr, loc in ((F_SQ1A, p3d),
                        (F_SQ1B, (x_true + dx, 0.0, pz(acc) + dz)),
                        (F_HOLD1, (x_true + dx, 0.0, pz(acc) + dz)),""",
     """        p3d = (x_true + dx, y_true, pz(acc) + dz)
        dz1 = -0.70 if (nm == "RadarSplat" and dz != 0.0) else dz
        for fr, loc in ((F_SQ1A, p3d),
                        (F_SQ1B, (x_true + dx, 0.0, pz(acc) + dz1)),
                        (F_HOLD1, (x_true + dx, 0.0, pz(acc) + dz1)),""", 1),
    # RGBA frames for the theme-adaptive web encode
    ("""OUT = (os.path.join(SCRATCH, "preview", "p5b_pareto.mp4") if PREVIEW
       else os.path.join(FINAL_DIR, "p5b_pareto.mp4"))""",
     'OUT = os.environ["PARETO_WEB_OUT"]', 1),
    ("""palette.setup_render(scene, OUT, preview=PREVIEW,
                     frame_start=1, frame_end=F_END)""",
     """palette.setup_render(scene, OUT, preview=PREVIEW,
                     frame_start=1, frame_end=F_END, transparent=True)""",
     1),
]

if THEME == "light":
    EDITS.append((
        "\nimport palette\n",
        """
import palette
# --- webpage light chrome (gen_pareto_web.py); data colours untouched ---
palette.ANNOT_COLOR = palette._srgb(0x1f, 0x20, 0x24)
palette.AXIS_COLOR = palette._srgb(0x5b, 0x5c, 0x63)
palette.GRID_COLOR = palette._srgb(0xd4, 0xd4, 0xda)
palette.NORMAL_COLOR = palette._srgb(0x1d, 0x8a, 0x3a)
palette.DIM_TEXT = palette._srgb(0x77, 0x78, 0x82)
""", 1))

src = open(SRC).read()
for old, new, n in EDITS:
    found = src.count(old)
    assert found == n, f"expected {n} of {old!r}, found {found}"
    src = src.replace(old, new)

out = sys.argv[1]
with open(out, "w") as f:
    f.write(src)
print(f"wrote {out}")
