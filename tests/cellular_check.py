"""Cellular (Worley/Voronoi) tiling checks.

Guards the bug where feature offsets were wrapped by one cell instead of by the
tiling period: that aliased neighbouring cells' features into the current cell
and made the field discontinuous at every cell boundary as soon as an octave's
period reached 4 (e.g. L=256 with base freq 0.01 and lacunarity > 1).

Runs on the CPU path only (stdlib), so it needs no pyopencl or display.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import generate_volumetric as gv

SEED = 1926328
MODES = ["F1", "F2", "F1 - F2", "Jitter", "Edge"]
# Only F1 is a continuous distance field; F2/F1-F2/Jitter/Edge legitimately
# step where the nearest or containing cell changes, so they are checked for
# seamless wrapping but not for smoothness.
SMOOTH_MODES = ["F1"]
PERIODS = [2, 3, 4, 5, 6, 9, 16]
N = 512
MAX_STEP = 0.05
PROBES = [(0.37, 0.61), (0.11, 0.83), (0.5, 0.5)]


def scan_line(sample, period):
    """(worst step between consecutive samples, mismatch across the wrap edge).

    The wrap mismatch must be exactly zero: sampling at 0 and at one full period
    is the same point of the tiling.
    """
    vals = [sample(i / N * period) for i in range(N)]
    step = max(abs(vals[i + 1] - vals[i]) for i in range(N - 1))
    return step, abs(sample(0.0) - sample(float(period)))


fails = []

for period in PERIODS:
    table = gv._precompute_worley_table(period, SEED)

    for py, pz in PROBES:
        step, wrap = scan_line(lambda c, y=py, z=pz: gv._sample_worley_table(c, y, z, table, period),
                               period)
        if step > MAX_STEP:
            fails.append(("worley continuity", period, py, pz, round(step, 4)))
        if wrap:
            fails.append(("worley seamless wrap", period, py, pz, wrap))

        for mode in MODES:
            sampler = lambda c, m=mode, y=py, z=pz: gv._voronoi_sample_table(c, y, z, table, period, m)
            step, wrap = scan_line(sampler, period)
            if mode in SMOOTH_MODES and step > MAX_STEP:
                fails.append(("voronoi continuity", period, mode, py, pz, round(step, 4)))
            if wrap:
                fails.append(("voronoi seamless wrap", period, mode, py, pz, wrap))

for f in fails:
    print("FAIL:", *f)
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILURES")
sys.exit(0 if not fails else 1)
