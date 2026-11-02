import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import generate_volumetric as gv
import generate_volumetric_gpu as gvg

PARAMS = [
    dict(size=16, octaves=4, base_freq=0.01, lacunarity=2.0, seed=42),
    dict(size=16, octaves=6, base_freq=0.05, lacunarity=2.0, seed=7),
    dict(size=32, octaves=3, base_freq=0.5, lacunarity=1.5, seed=123),
    dict(size=8, octaves=5, base_freq=0.02, lacunarity=3.0, seed=999),
    dict(size=16, octaves=5, base_freq=1.0, lacunarity=2.0, seed=5),
    # Cellular tiling regressions: periods 5/9 only occur when the octave
    # period is not a power of two (the case that used to alias features).
    dict(size=64, octaves=4, base_freq=0.078125, lacunarity=2.0, seed=1926328),
    dict(size=32, octaves=4, base_freq=0.140625, lacunarity=1.5, seed=1926328),
    dict(size=16, octaves=2, base_freq=0.078125, lacunarity=0.5, seed=1926328),
]
NOISE = [("Value Noise", ["F1"]), ("Worley Noise", ["F1"]),
         ("FBM Perlin Noise", ["F1"]),
         ("Voronoi Noise", ["F1", "F2", "F1 - F2", "Jitter", "Edge"])]

def same(a, b):
    return all(ay == by for ay, by in zip(a, b))

def cases():
    for nt, modes in NOISE:
        for p in PARAMS:
            for m in modes:
                yield nt, m, p

fails = []

orig = gvg.OPENCL_AVAILABLE

gvg.OPENCL_AVAILABLE = False
n_fb = 0
for nt, m, p in cases():
    ref = gv._generate_volume(p["size"], p["octaves"], p["base_freq"],
                              p["lacunarity"], p["seed"], nt, None, m)
    got = gvg.generate_volume_gpu(p["size"], p["octaves"], p["base_freq"],
                                  p["lacunarity"], p["seed"], nt, None, m)
    n_fb += 1
    if not same(ref, got):
        fails.append(("fallback", nt, m, p))
print(f"fallback-vs-CPU bit-exact: {n_fb} cases, {len(fails)} fail")
gvg.OPENCL_AVAILABLE = orig

if not orig or not gvg._get_mgr().is_ready():
    print("GPU unavailable — skipping GPU bit-exact phase")
    sys.exit(1 if fails else 0)

def boom(*a, **k):
    raise AssertionError("silent fallback!")
saved = gvg._generate_volume_cpu
gvg._generate_volume_cpu = boom
n_gpu, gpu_fail = 0, 0
for nt, m, p in cases():
    if gvg._any_octave_capped(p["size"], p["octaves"], p["base_freq"], p["lacunarity"]):
        continue
    ref = gv._generate_volume(p["size"], p["octaves"], p["base_freq"],
                              p["lacunarity"], p["seed"], nt, None, m)
    try:
        got = gvg.generate_volume_gpu(p["size"], p["octaves"], p["base_freq"],
                                      p["lacunarity"], p["seed"], nt, None, m)
    except AssertionError:
        fails.append(("gpu-fallback", nt, m, p))
        gpu_fail += 1
        continue
    n_gpu += 1
    if not same(ref, got):
        fails.append(("gpu", nt, m, p))
gvg._generate_volume_cpu = saved
print(f"GPU (fallback sabotaged) bit-exact: {n_gpu} cases, {gpu_fail} fell back")

for f in fails:
    print("FAIL:", f)
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILURES")
sys.exit(0 if not fails else 1)
