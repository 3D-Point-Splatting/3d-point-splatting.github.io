# 3D Point Splatting for mmWave Radar Novel View Synthesis

Project page for **3D Point Splatting for mmWave Radar Novel View Synthesis**.

- **Paper:** [arXiv:2609.11894](https://arxiv.org/abs/2609.11894)
- **Authors:** Adnan Armouti, Yixuan Gao, Rajalakshmi Nandakumar — Cornell Tech, New York, NY, USA
- **Site:** <https://3d-point-splatting.github.io/>

3DPS is a differentiable point renderer for radar, derived from the standard
solid-angle form of the radar equation. Each oriented 3D point carries an
ITU-R P.2040 material model, evaluated in closed form, and its complex phasor
is splatted into range bins through a precomputed point spread function.
Because the output is complex-valued, one fitted scene yields raw ADC, complex
range profile and range–azimuth outputs through standard FFT pipelines, with no
retraining per format.

## Running locally

A static site — no build step and no dependencies:

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

## Layout

```
index.html          the page: markup, styles and the clip sequencer
static/js/          three.js, the animated teaser and the interactive 3-D viewer
static/images/      figures (light/dark pairs)
static/videos/      rendered clips (light/dark pairs)
static/teaser/      the animated teaser's tiles and application panels
static/viewer/      packed point clouds and range-azimuth maps for the viewer
static/data/        the Pareto frontier numbers and fitted surface
tools/              the generators that produce the assets above
```

The generators read from research and slide-deck trees that are not part
of this repository. Their locations come from the environment rather than
being hard-coded — see `tools/_paths.py`:

```bash
export TDPS_ROOT=/path/to/mm3DGS     # assets/, output/teaser_panels/
export DECK_ROOT=/path/to/a_exam     # blender/, slides/figsrc/, slides/figs/
```

Figures and clips are generated as matched light/dark pairs from a single
source per asset, so typography and data are identical across themes. Any
asset changed under an unchanged filename must bump its `?v=N` query in
`index.html`, or browsers will serve the stale copy.

## License

Copyright © 2025–2026 Cornell University. All Rights Reserved.
See [LICENSE](LICENSE) (Cornell CTL docket 12012).

Page template adapted from [Nerfies](https://github.com/nerfies/nerfies.github.io).
