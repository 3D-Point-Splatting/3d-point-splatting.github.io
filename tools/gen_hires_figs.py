#!/usr/bin/env python3
"""High-res (dpi 600) 3DPS webpage figures, light and dark palettes.

Both variants of each figure come from the same figsrc script (same
matplotlib fonts/layout); only the chrome colours differ — the RA
heatmaps keep their dark 'hot' colormap in both themes (dark is data).
PNGs saved with transparent rounded corners so the page colour shows
through.

Canonical artifacts these must content-match (verified 2026-08-26,
antialias-level diffs only, same corr overlays / column order):
  slides/figs/dark_tdps_pipeline.png  (= figsrc/out/pipeline_3dps_system_overview.png)
  slides/figs/dark_tdps_ra_cmp.png    (= figsrc/out/training_ra_comparison.png)
generated per figsrc/render_all.sh with TDPS_ROOT=~/Desktop/mm3DGS:
  --baselines_dir $TDPS_ROOT/baselines
  --ours_dir      $TDPS_ROOT/mm25DGS_v5_v4/output_frame_nvs
  --center_variant system_overview

The paper teaser (figures/generate_fig_teaser_v9_highres.py in the mm3DGS
repo) reproduces figs_highres/teaser_3dps_v9b.png BIT-EXACTLY from current
data (verified 2026-08-26); the web variants recolour only its chrome
tokens and downscale the native ~2000-dpi render to 4260 px wide.

Usage: gen_hires_figs.py pipeline|racmp|teaser light|dark <out_dir>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import root  # noqa: E402
import os
import sys
import types

FIGSRC = os.path.join(root("DECK_ROOT"), "slides", "figsrc")
TDPS_ROOT = os.path.expanduser("~/Desktop/mm3DGS")

WHICH, THEME = sys.argv[1], sys.argv[2]
OUT_DIR = sys.argv[3]

os.environ["FIGSRC_PROJECT_ROOT"] = TDPS_ROOT
sys.path.insert(0, TDPS_ROOT)
sys.path.insert(0, FIGSRC)

# fig_common_3dps deck-dark palette -> webpage light palette. The light
# values match the mmIR webpage: figure bg = light card #f5f5f7, panels
# white, text #1f2024, dim #5b5c63.
LIGHT_MAP = {
    "#1A1A1E": "#f5f5f7",   # BACKGROUND_COLOR (the baked figure tile)
    "#26272C": "#ffffff",   # PANEL_COLOR
    "#E8E8EC": "#1f2024",   # FG_COLOR
    "#9696A0": "#5b5c63",   # DIM_COLOR
}


def load_patched(name, extra=None):
    """Exec a figsrc module with dpi 600, transparent PNG corners, and
    (light theme) the palette substituted at source level, so constants
    captured in default args / module aliases all pick it up."""
    src = open(os.path.join(FIGSRC, name + ".py")).read()
    src = src.replace("dpi=300", "dpi=600")
    # transparent PNG corners: page colour shows through the rounding
    src = src.replace("facecolor=BACKGROUND_COLOR", 'facecolor="none"')
    src = src.replace('facecolor="none" if ext == "pdf" else BG_COLOR',
                      'facecolor="none"')
    if THEME == "light":
        for k, v in LIGHT_MAP.items():
            src = src.replace(f'"{k}"', f'"{v}"')
        for k, v in (extra or {}).items():
            src = src.replace(k, v)
    mod = types.ModuleType(name)
    mod.__file__ = os.path.join(FIGSRC, name + ".py")
    sys.modules[name] = mod
    exec(compile(src, name + ".py", "exec"), mod.__dict__)
    return mod


# fig_common_3dps must be patched BEFORE the figure script imports it.
load_patched("fig_common_3dps")


def darken_panel(src_path, dst_path):
    """Saturation-preserving luma inversion for a light scene render:
    neutral chrome (sky, mesh shading) inverts into the deck-dark band,
    coloured data (points, Gaussians) keeps its own brightness."""
    import numpy as np
    from PIL import Image
    rgb = np.asarray(Image.open(src_path).convert("RGB"),
                     dtype=np.float32) / 255.0
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    L = (mx + mn) / 2.0
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    w = np.clip(sat * 2.5, 0, 1)
    Lout = w * L + (1 - w) * (1.0 - L)
    scale = np.where(L > 1e-5, Lout / np.maximum(L, 1e-5), 0.0)
    out = np.clip(rgb * scale[..., None], 0, 1)
    Image.fromarray((out * 255).astype("uint8")).save(dst_path)


if WHICH == "pipeline":
    # BLOCK_COLOR: forward-model sub-blocks, "darker than tile" — invert
    # to slightly-darker-than-white on light. Data colours untouched.
    G = load_patched("generate_fig_pipeline_3dps_v2",
                     extra={'"#32333B"': '"#eaebef"'})
    if THEME == "dark":
        # the scene-representation render bakes a light sky: dark-adapt
        # a COPY (original panel_a_scene.png untouched)
        dark_panel = os.path.join(OUT_DIR, "panel_a_scene_dark.png")
        darken_panel(G.PANEL_A_PATH, dark_panel)
        G.PANEL_A_PATH = dark_panel
    sys.argv = ["x", "--output_dir", OUT_DIR,
                "--center_variant", "system_overview"]
    G.main()
elif WHICH == "racmp":
    # the corr overlay sits ON the heatmap cells, which keep their dark
    # 'hot' colormap in both themes (dark is data) — keep it light
    G = load_patched("generate_fig_tdps_ra_cmp",
                     extra={"fontsize=7, color=FG_COLOR":
                            'fontsize=7, color="#E8E8EC"'})
    sys.argv = ["x",
                "--baselines_dir", os.path.join(TDPS_ROOT, "baselines"),
                "--ours_dir",
                os.path.join(TDPS_ROOT, "mm25DGS_v5_v4/output_frame_nvs"),
                "--output_dir", OUT_DIR]
    G.main()
elif WHICH == "teaser":
    # Paper teaser, chrome tokens only — data panels (RA/CRP/ADC embeds,
    # point renders) are untouched; the light variant differs from the
    # canonical paper build only in its grey chrome.
    TEASER_LIGHT = {
        '"#ECEBEB"': '"#f5f5f7"',   # row tiles
        '"#f5f5f5"': '"#f5f5f7"',   # bottom row outer
        '"#ebebeb"': '"#ffffff"',   # inner column tiles
        '"#dadada"': '"#eaebef"',   # inset panel
        '"#1a1a1a"': '"#1f2024"',   # main text
        '"#222222"': '"#1f2024"',   # applications header
        '"#5a5a5a"': '"#5b5c63"',   # subtle text
        '"#6a6a6a"': '"#5b5c63"',   # captions
        'facecolor="white"': 'facecolor="none"',
    }
    TEASER_DARK = {
        '"#ECEBEB"': '"#1a1a1e"',
        '"#f5f5f5"': '"#1a1a1e"',
        '"#ebebeb"': '"#26272c"',
        '"#dadada"': '"#32333B"',
        '"#1a1a1a"': '"#e8e8ec"',
        '"#222222"': '"#e8e8ec"',
        '"#5a5a5a"': '"#9696a0"',
        '"#6a6a6a"': '"#9696a0"',
        '"black"': '"#e8e8ec"',
        '"#444"': '"#c8c8d0"',      # GT labels / arrows
        '"#666"': '"#9696a0"',      # arrows
        '"#bbb"': '"#4a4b52"',      # separator rule
        '"#3b6f8e"': '"#7fb8dc"',   # blue caption text
        '"#c5dbeb"': '"#22303c"',   # NVS train sub-tiles
        '"#f3c4c4"': '"#3a2528"',   # NVS test sub-tile
        '"#FBE7E6"': '"#3a2528"',   # variant-a fills (unused in b)
        '"#E5F4E5"': '"#24321f"',
        '"#1d8a3a"': '"#5ecb7c"',   # GREEN_OK / green border
        '"#b3271e"': '"#e0584c"',   # RED_BAD / red border
        '"#0e6b2c"': '"#3dbb63"',   # ours accent
        '"#4d8eb8"': '"#5da8d8"',   # trajectory blue
        'facecolor="white"': 'facecolor="none"',
    }

    def load_teaser():
        path = os.path.join(TDPS_ROOT, "figures",
                            "generate_fig_teaser_v9_highres.py")
        src = open(path).read()
        for k, v in (TEASER_LIGHT if THEME == "light"
                     else TEASER_DARK).items():
            src = src.replace(k, v)
        mod = types.ModuleType("generate_fig_teaser_v9_highres")
        mod.__file__ = path
        sys.modules[mod.__name__] = mod
        exec(compile(src, os.path.basename(path), "exec"), mod.__dict__)
        return mod

    panel_args = []
    if THEME == "dark":
        # The three top-row primitive renders bake a light sky/backdrop.
        # For dark, saturation-preserving luma inversion: neutral chrome
        # (sky, mesh shading) inverts into the deck-dark band, coloured
        # data (pink points, pastel Gaussians) keeps its own brightness.
        # Adapted COPIES go to a scratch panel tree; originals untouched.
        import shutil
        SCENE = "seq_1_frame_185"
        src_dir = os.path.join(TDPS_ROOT, "output/teaser_panels",
                               SCENE, "v4")
        tmp_root = os.path.join(OUT_DIR, "panels_dark")
        tmp_dir = os.path.join(tmp_root, SCENE, "v4")
        os.makedirs(tmp_dir, exist_ok=True)
        for f in os.listdir(src_dir):
            shutil.copy2(os.path.join(src_dir, f), os.path.join(tmp_dir, f))
        for name in ("panel_mesh_only.png", "panel_implicit.png",
                     "panel_3dps_points.png"):
            darken_panel(os.path.join(src_dir, name),
                         os.path.join(tmp_dir, name))
        panel_args = ["--panel_dir", tmp_root]

    G = load_teaser()
    sys.argv = ["x", "--variant", "b", "--output_dir", OUT_DIR] + panel_args
    G.main()

    # downscale the native render (~14200 px wide) for the web
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    src_png = os.path.join(OUT_DIR, "teaser_3dps_v9b.png")
    im = Image.open(src_png).convert("RGBA")
    w = 4260
    im.resize((w, round(im.size[1] * w / im.size[0])),
              Image.LANCZOS).save(
        os.path.join(OUT_DIR, f"teaser_web_{THEME}.png"))
    print(f"web PNG: {OUT_DIR}/teaser_web_{THEME}.png")
else:
    sys.exit(f"unknown figure {WHICH!r}")
