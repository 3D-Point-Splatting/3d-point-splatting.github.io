#!/usr/bin/env python3
"""Derive theme-adaptive webpage variants of the deck Blender clips.

Covers p1a_pipeline, p1b_points, p1c_crp (Ray Tracing vs Splatting) and
p5a_payoff (3DPS Recovers per-point Materials and Normals). Same recipe
as gen_pareto_web.py round 2: film-transparent RGBA frames, then the
encode step composites onto the figure-tile colour per theme.

Light chrome swaps (BEFORE plot2d/scene_build/workflow/flowchart import,
so module-level derived colours pick them up):
  ANNOT/DIM/AXIS/GRID text+lines -> webpage light palette
  NORMAL_COLOR green, TX amber, RX blue -> the webpage's darker light-
  mode accents
  BACKGROUND -> light tile #f5f5f7 (backing plates / flowchart cards
  derive from it) — but setup_world is wrapped to keep the ORIGINAL
  dark world so the lit 3-D geometry (staircase mesh, point clouds)
  renders identically in both themes (the world lights the scene even
  when film-transparent).

Output frames go to $CLIP_WEB_OUT (a PNG frame prefix). The originals
in blender/ and the deck mp4s are never touched.

Usage: gen_clips_web.py <clip> light|dark <out_script.py>
       clip in {p1a_pipeline, p1b_points, p1c_crp, p5a_payoff}
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import root  # noqa: E402
import re
import sys

BLENDER_DIR = os.path.join(root("DECK_ROOT"), "blender")
CLIP, THEME, OUT_PATH = sys.argv[1], sys.argv[2], sys.argv[3]
assert CLIP in ("p1a_pipeline", "p1b_points", "p1c_crp", "p5a_payoff",
                "p4_nvs")
assert THEME in ("light", "dark")

src = open(f"{BLENDER_DIR}/{CLIP}.py").read()

# Tile titles (user, 2026-10-07), BOTH themes. The deck clips are beats
# inside one long pipeline walk, so they carry the talk's running titles:
# p1a shows "mmIR - the per-view pipeline" throughout (F_T1=650 is past
# its 648-frame window), and p1b's first frame -- the one its 0.8s
# lead-in holds -- still shows that title before switching at F_T1. On
# the webpage the tiles are labelled by what they ARE, so each clip gets
# a single constant title.
TITLES = {
    "p1a_pipeline": ("Ray Tracing", None),
    "p1b_points":   ("3D Point Splatting", "mat_ok"),
    "p1c_crp":      ("3D Point Splatting", "mat_ok"),
}
if CLIP in TITLES:
    _title, _mat = TITLES[CLIP]
    for _old in ('"mmIR \u2014 the per-view pipeline"', '"3DPS \u2014 persistent points"'):
        assert src.count(_old) == 1, f"title drifted: {_old}"
        src = src.replace(_old, f'"{_title}"')
    if _mat:   # keep one colour so the F_T1 swap is invisible
        _o = 'ov_coll, mat_annot, (12.65, 8.45), size=0.4'
        assert src.count(_o) == 1
        src = src.replace(_o, f'ov_coll, {_mat}, (12.65, 8.45), size=0.4')

if CLIP == "p4_nvs":
    # content edits (user, 2026-08-26), BOTH themes: the curve plot keeps
    # only its title + axes; legend and held-out readout decluttered.
    P4_EDITS = [
        ('pan.title("mean train |RA| corr — measured, this run", '
         'size=0.34)',
         'pan.title("mean train |RA| corr", size=0.34)'),
        # honesty footnote under the plot
        ('''t = plot2d.text_object("every frame is a real render from this run — "
                       "500 iterations, all 8 views", "t_hon", ov_coll,
                       mat_dim, (8.0, 0.52), size=0.28,
                       align='CENTER', font=freg)
plot2d.show_between(t, F_PLAY0, F_CLEAN)
''', ""),
        # densify caption (the dashed vlines themselves stay)
        ('''t = plot2d.text_object("densify @ 100/200/300/400", "t_dens", ov_coll,
                       mat_dim, (5.20, 0.52), size=0.28,
                       align='CENTER', font=freg)
plot2d.show_between(t, it2f(100) - 10, F_CLEAN)
''', ""),
        # RHS fit-cost line
        ('''t = plot2d.text_object(f"500 iters × 8 views · {MIN_WALL:.2f} min · "
                       f"RTX 4090", "t_fit",
                       ov_coll, mat_dim, (13.35, 1.72), size=0.24,
                       align='CENTER', font=freg)
plot2d.show_between(t, F_CURVE, F_CLEAN)
''', ""),
        ('"● train (8)"', '"● train"'),
        ('"● held out (F = 185)"', '"● held out"'),
        ('f"held out: {TEST0:.3f} → {TEST1:.3f}"',
         'f"Held Out Correlation: {TEST1:.3f}"'),
    ]
    for old, new in P4_EDITS:
        assert src.count(old) == 1, f"p4 edit drifted: {old[:60]!r}"
        src = src.replace(old, new)

# resolve sibling imports / data from the real blender dir
old = "os.path.dirname(os.path.abspath(__file__))"
assert src.count(old) >= 1
src = src.replace(old, f'"{BLENDER_DIR}"')

# output: RGBA PNG frame prefix from the environment
out_block = (f'OUT = (os.path.join(SCRATCH, "preview", "{CLIP}.mp4") '
             f'if PREVIEW\n       '
             f'else os.path.join(FINAL_DIR, "{CLIP}.mp4"))')
assert src.count(out_block) == 1, "OUT block drifted"
src = src.replace(out_block, 'OUT = os.environ["CLIP_WEB_OUT"]')

# transparent main render (the overlay scene is already transparent and
# alpha-overed in the compositor, so the composite keeps alpha)
pat = re.compile(r"palette\.setup_render\(scene, OUT, ([^)]+)\)")
assert len(pat.findall(src)) == 1, "setup_render(scene, OUT, ...) drifted"
src = pat.sub(r"palette.setup_render(scene, OUT, \1, transparent=True)",
              src, count=1)

if THEME == "light":
    anchor = "\nimport palette\n"
    assert src.count(anchor) == 1
    src = src.replace(anchor, """
import palette
# --- webpage light chrome (gen_clips_web.py); 3-D lighting kept dark ---
_BG_DARK = palette.BACKGROUND
_setup_world = palette.setup_world
def _setup_world_dark(scene):
    _b = palette.BACKGROUND
    palette.BACKGROUND = _BG_DARK
    _setup_world(scene)
    palette.BACKGROUND = _b
palette.setup_world = _setup_world_dark
palette.BACKGROUND = palette._srgb(0xf5, 0xf5, 0xf7)
palette.ANNOT_COLOR = palette._srgb(0x1f, 0x20, 0x24)
palette.DIM_TEXT = palette._srgb(0x6b, 0x6c, 0x76)
palette.AXIS_COLOR = palette._srgb(0x5b, 0x5c, 0x63)
palette.GRID_COLOR = palette._srgb(0xc4, 0xc5, 0xcc)
palette.NORMAL_COLOR = palette._srgb(0x1d, 0x8a, 0x3a)
palette.TX_COLOR = palette._srgb(0xd9, 0x74, 0x06)
palette.RX_COLOR = palette._srgb(0x0d, 0x7f, 0xc0)
palette.WAVE_COLOR = palette._srgb(0xc2, 0x6a, 0x10)
# light-grey mesh under the kept-dark world + sun (calibrated on stills)
palette.STAIR_COLOR = palette._srgb(0xd8, 0xda, 0xe2)
palette.WALL_COLOR = palette._srgb(0xca, 0xcc, 0xd6)
""")
    if CLIP == "p5a_payoff":
        # the deck dims the mesh 0.22x so the cloud reads against the
        # DARK bg; on the light tile the equivalent is a light, slightly
        # muted mesh — keep the light albedo, mild dim only
        old_dim = """bsdf_m.inputs['Base Color'].default_value = (_c0[0] * 0.22,
                                             _c0[1] * 0.22,
                                             _c0[2] * 0.22, 1.0)"""
        assert src.count(old_dim) == 1
        src = src.replace(old_dim, """bsdf_m.inputs['Base Color'].default_value = (_c0[0] * 0.85,
                                             _c0[1] * 0.85,
                                             _c0[2] * 0.85, 1.0)""")
    # flowchart cards: CARD_FILL derives 3x BACKGROUND at import, which
    # clips to white on the light tile — pin a visible light card grey.
    # Patch only the first (module-level) import; later ones re-import
    # the already-patched module.
    src = re.sub(
        r"^import flowchart$",
        "import flowchart\n"
        "flowchart.CARD_FILL = palette._srgb(0xe6, 0xe7, 0xed)",
        src, count=1, flags=re.M)

with open(OUT_PATH, "w") as f:
    f.write(src)
print(f"wrote {OUT_PATH}")
