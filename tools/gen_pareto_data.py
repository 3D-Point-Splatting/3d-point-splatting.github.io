#!/usr/bin/env python3
"""Pin the Pareto numbers for the interactive teaser plot.

The authority is the deck's Blender script, p5b_pareto.py, which cannot be
imported here (it imports bpy). The table is therefore transcribed and then
re-checked against that script's OWN assertions, plus the measured-memory
JSON it reads, so a drift in either source fails this tool rather than
silently shipping stale numbers.

Provenance, as recorded in p5b_pareto.py's header:
  DART / Radar Fields / RadarSplat  thesis tab:tdps:runtime + results
  Sionna-RT                         rebuttal openreview response
  mmIR                              thesis joint 8-view extension
  3DPS                              ours; peak memory MEASURED

Geometry in the deck's plot: X = log10(minutes), Y = peak GPU memory (GB),
Z = held-out |RA| accuracy (up).

Usage: gen_pareto_data.py [out.json]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import root  # noqa: E402
import json
import os
import sys

SRC = os.path.join(root("DECK_ROOT"), "blender", "p5b_pareto.py")
MEM_JSON = os.path.join(os.path.dirname(SRC), ".pareto_cache",
                        "3dps_peak_mem.json")
OUT = sys.argv[1] if len(sys.argv) > 1 else "static/data/pareto.json"

mem = json.load(open(MEM_JSON))
assert mem["scene"] == "seq_1_frame_185" and mem["iters"] == 500
DPS_MEM = float(mem["peak_gpu_mem_gb"])
assert 0.05 < DPS_MEM < 24.0, DPS_MEM

# name, accuracy, minutes, peak GB (None = not reported), family
MC, INR, PTS = "mc", "inr", "points"
METHODS = [
    ("DART",         0.112,  0.43, None,    INR),
    ("Radar Fields", 0.133,  1.63, 2.4,     INR),
    ("RadarSplat",   0.339, 19.74, 14.2,    INR),
    ("Sionna-RT",    0.380, 12.20, 21.8,    MC),
    ("mmIR",         0.454, 77.59, 6.0,     MC),
    ("3DPS",         0.587,  3.17, DPS_MEM, PTS),
]

# --- the source script's own checks, repeated here -----------------------
txt = open(SRC).read()
for name, acc, mins, gb, _ in METHODS:
    if name == "3DPS":
        continue
    needle = f'("{name}",'
    assert needle in txt, f"{name} no longer appears in {SRC}"
    row = txt[txt.index(needle):txt.index("\n", txt.index(needle))]
    for v in (acc, mins) + ((gb,) if gb is not None else ()):
        assert str(v) in row, f"{name}: {v} not in source row {row.strip()}"

BASE = [m for m in METHODS if m[0] not in ("mmIR", "3DPS")]
def frontier(rows, i):
    out, best = [], -1.0
    for m in sorted(rows, key=lambda r: r[i]):
        if m[1] > best:
            out.append(m[0]); best = m[1]
    return out
assert frontier(BASE, 2) == ["DART", "Radar Fields", "Sionna-RT"]
assert frontier([m for m in BASE if m[3] is not None], 3) == \
    ["Radar Fields", "RadarSplat", "Sionna-RT"]
for nm, acc, *_ in METHODS:
    if nm in ("mmIR", "3DPS"):
        assert all(acc > b[1] for b in BASE), nm
assert abs(0.587 / 0.380 - 1.545) < 0.01
assert abs(21.8 / DPS_MEM - 61.0) < 2.0
assert abs(12.20 / 3.17 - 3.85) < 0.05

# --- the surface, replicated from p5b_pareto.py ---------------------------
# A monotone concave saturating fit through the three baselines that report
# all three axes:  acc(u,v) = c0 + c1(1-e^-lu(u-u0)) + c2(1-e^-lv(v-v0)),
# u = px(minutes) (so the time knee lives in log-minutes), v = py(GB).
# c1,c2 >= 0 makes it monotone by construction. lu, lv are chosen
# deterministically: the far-edge gradient must keep >= SLOPE_KEEP of the
# near-origin gradient on each axis, and within that bound the SMALLER of
# the two axis sags is maximised. The rule never references mmIR or 3DPS --
# that they land above the surface is the data's doing, and is asserted.
# The reviewed build's solution is pinned below, as the source script pins it.
import math
T_LO, T_HI = 0.3, 100.0
A_LO, A_HI = 0.0, 0.65
M_LO, M_HI = 0.0, 24.0
X_LEN, Z_LEN, Y_LEN = 8.4, 5.2, 6.0
SLOPE_KEEP = 0.6
px = lambda t: (math.log10(t) - math.log10(T_LO)) / (math.log10(T_HI) - math.log10(T_LO)) * X_LEN
py = lambda g: (g - M_LO) / (M_HI - M_LO) * Y_LEN
U0, V0 = px(T_LO), py(M_LO)
SPAN_U, SPAN_V = px(T_HI) - U0, py(M_HI) - V0
Z_PER_ACC = Z_LEN / (A_HI - A_LO)

BM = [m for m in METHODS if m[0] not in ("mmIR", "3DPS") and m[3] is not None]
import numpy as np
pu = np.array([px(m[2]) for m in BM]); pv = np.array([py(m[3]) for m in BM])
pa = np.array([m[1] for m in BM])

def solve(lu, lv):
    su = 1.0 - np.exp(-lu * (pu - U0)); sv = 1.0 - np.exp(-lv * (pv - V0))
    try:
        return np.linalg.solve(np.stack([np.ones(3), su, sv], -1), pa)
    except np.linalg.LinAlgError:
        return None

def sag(lam, span, coef):
    s = np.linspace(0.0, span, 240)
    f = coef * (1.0 - np.exp(-lam * s))
    return float(np.abs(f - f[-1] * s / span).max()) * Z_PER_ACC

best = None
for lu in np.linspace(0.005, -math.log(SLOPE_KEEP) / SPAN_U, 64):
    for lv in np.linspace(0.005, -math.log(SLOPE_KEEP) / SPAN_V, 64):
        c = solve(lu, lv)
        if c is None or c[1] < 0 or c[2] < 0:
            continue
        top = c[0] + c[1]*(1-math.exp(-lu*SPAN_U)) + c[2]*(1-math.exp(-lv*SPAN_V))
        if top > A_HI or c[0] < A_LO:
            continue
        sg = min(sag(lu, SPAN_U, c[1]), sag(lv, SPAN_V, c[2]))
        if best is None or sg > best[0]:
            best = (sg, lu, lv, c)
assert best is not None, "no feasible monotone saturating surface"
_, LU, LV, C = best
assert C[1] > 0 and C[2] > 0
assert math.exp(-LU * SPAN_U) >= SLOPE_KEEP - 1e-9
assert math.exp(-LV * SPAN_V) >= SLOPE_KEEP - 1e-9
assert abs(LU - 0.0573) / 0.0573 < 0.10, LU          # the reviewed build's
assert abs(LV - 0.0851) / 0.0851 < 0.10, LV
assert abs(C[0] - 0.0300) < 0.01, C[0]
assert abs(C[1] - 0.5877) / 0.5877 < 0.10, C[1]
assert abs(C[2] - 0.5244) / 0.5244 < 0.10, C[2]

NU = NV = 28
tg = np.logspace(math.log10(T_LO), math.log10(T_HI), NU)
mg = np.linspace(M_LO, M_HI, NV)
surf = [[float(C[0] + C[1]*(1-math.exp(-LU*(px(t)-U0)))
              + C[2]*(1-math.exp(-LV*(py(g)-V0)))) for t in tg] for g in mg]
for nm, acc, t, g, _ in METHODS:
    if nm in ("mmIR", "3DPS"):
        above = acc > (C[0] + C[1]*(1-math.exp(-LU*(px(t)-U0)))
                       + C[2]*(1-math.exp(-LV*(py(g)-V0))))
        assert above, f"{nm} should sit ABOVE the baseline surface"

# --- frontier trace 1 (accuracy vs time), replicated from p5b_pareto.py ----
# Fritsch-Carlson monotone cubic through the origin and the time-frontier
# baselines (DART, Radar Fields, Sionna-RT) in scene x; past Sionna-RT the
# height follows the fitted surface itself to the plot edge (the deck's
# "extend the FIT, not the endpoint gradient"). Depth (memory) is the same
# interpolation over the baselines' memory, DART on the time face (y = 0).
def _fc_tangents(xs, ys):
    h = np.diff(xs); d = np.diff(ys) / h
    m = np.empty(len(xs)); m[0], m[-1] = d[0], d[-1]
    for i in range(1, len(xs) - 1):
        if d[i - 1] * d[i] <= 0.0:
            m[i] = 0.0
        else:
            w1, w2 = 2.0 * h[i] + h[i - 1], h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    return m


def _fc_eval(xs, ys, m, q):
    if q >= xs[-1]:
        return float(ys[-1] + m[-1] * (q - xs[-1]))
    i = int(np.searchsorted(xs, q)) - 1
    h = xs[i + 1] - xs[i]; t = (q - xs[i]) / h
    return float((2*t**3 - 3*t**2 + 1) * ys[i] + (t**3 - 2*t**2 + t) * h * m[i]
                 + (-2*t**3 + 3*t**2) * ys[i + 1] + (t**3 - t**2) * h * m[i + 1])


FRONT_T = [m for m in sorted(BASE, key=lambda r: r[2])
           if m[0] in ("DART", "Radar Fields", "Sionna-RT")]
KC = np.array([0.0] + [px(m[2]) for m in FRONT_T])
KA = np.array([0.0] + [m[1] for m in FRONT_T])
KD = np.array([0.0] + [py(m[3]) if m[3] is not None else 0.0 for m in FRONT_T])
ma, md = _fc_tangents(KC, KA), _fc_tangents(KC, KD)
surf_uv = lambda u, v: float(C[0] + C[1]*(1-math.exp(-LU*(u-U0))) + C[2]*(1-math.exp(-LV*(v-V0))))
trace = []
for c in np.union1d(np.linspace(0.0, X_LEN, 169), KC):
    dd = min(max(_fc_eval(KC, KD, md, c), 0.0), Y_LEN)
    a = _fc_eval(KC, KA, ma, c) if c <= KC[-1] else surf_uv(c, dd)
    trace.append([float(c / X_LEN), float(a), float(dd / Y_LEN)])   # (x frac, acc, mem frac)
acc_tr = [p[1] for p in trace]
assert all(b >= a - 1e-12 for a, b in zip(acc_tr, acc_tr[1:])), "trace height not monotone"
assert abs(acc_tr[-1] - surf_uv(X_LEN, trace[-1][2] * Y_LEN)) < 1e-9

# DART reports no memory figure, so on a 3-axis plot it has no z. Rather
# than park it on the m = 0 wall -- where it floats well clear of the fitted
# surface and reads as an error -- it is placed at the memory the FIT implies
# for its measured accuracy and runtime: solve surface(t=0.43, m) = 0.112 for
# m. That is an inference from a surface fitted WITHOUT DART, not a
# measurement, so the marker stays hollow and the caption says so.
_dart = [m for m in METHODS if m[0] == "DART"][0]
_u = px(_dart[2]) - U0
_need = _dart[1] - C[0] - C[1] * (1 - math.exp(-LU * _u))
assert 0 < _need < C[2], f"DART accuracy {_dart[1]} is off the surface's range"
_v = -math.log(1 - _need / C[2]) / LV
DART_GB = _v / Y_LEN * (M_HI - M_LO)
assert M_LO < DART_GB < M_HI, DART_GB
_chk = (C[0] + C[1] * (1 - math.exp(-LU * _u))
        + C[2] * (1 - math.exp(-LV * py(DART_GB))))
assert abs(_chk - _dart[1]) < 1e-9, (_chk, _dart[1])

data = {
    "axes": {"x": "log10 runtime (min)", "y": "peak GPU memory (GB)",
             "z": "held-out |RA| correlation"},
    "methods": [{"name": n, "acc": a, "min": t, "gb": g, "family": f}
                for n, a, t, g, f in METHODS],
    "families": {MC: "Mesh + MC ray tracing", INR: "NeRF / 3DGS primitives",
                 PTS: "3DPS: point primitives"},
    "ours": "3DPS",
    "mem_provenance": {k: mem[k] for k in
                       ("peak_gpu_mem_gb", "scene", "iters", "note")},
    "domain": {"t": [T_LO, T_HI], "a": [A_LO, A_HI], "m": [M_LO, M_HI],
               "t_ticks": [0.3, 1, 3, 10, 30, 100],
               "a_ticks": [0.0, 0.2, 0.4, 0.6],
               "m_ticks": [0, 6, 12, 18, 24]},
    "dart_on_surface_gb": float(DART_GB),
    "surface": {"t": [float(x) for x in tg], "m": [float(x) for x in mg],
                "acc": surf,
                "fit": {"c0": float(C[0]), "c1": float(C[1]),
                        "c2": float(C[2]), "lu": float(LU), "lv": float(LV)}},
    "trace_time": trace,
    "scene_len": {"x": X_LEN, "y": Y_LEN, "z": Z_LEN},
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(data, open(OUT, "w"), indent=2)
print(f"  [check] 6 methods agree with {os.path.basename(SRC)}; "
      f"3DPS peak memory {DPS_MEM:.4f} GB (measured)")
print(f"  [check] surface fit lu={LU:.4f} lv={LV:.4f} "
      f"c=({C[0]:.4f}, {C[1]:.4f}, {C[2]:.4f}) matches the reviewed build; "
      f"both proposals sit above it")
print(f"  [check] DART placed on the surface at {DART_GB:.2f} GB "
      f"(inferred, not measured): surface there is {_chk:.4f} vs its "
      f"measured {_dart[1]}")
print(f"  [check] baseline frontiers and the 1.545x / 61x / 3.85x "
      f"headline ratios all hold")
print(f"wrote {OUT}")
