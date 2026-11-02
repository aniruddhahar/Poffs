#!/usr/bin/env bash
# Verification suite (no test framework; stdlib only, optional pyopencl/DISPLAY).
# Usage: ./tests/run_all.sh
#   GUI_PY=python3.12 GPU_PY=/tmp/opencode/gputest/bin/python ./tests/run_all.sh
set -u
cd "$(dirname "$0")/.."
fails=0

PY=${PY:-python3}

echo "== CLI smoke =="
for spec in "--size 16" "--size 16 --noise-type worley" "--size 16 --noise-type perlin" \
            "--size 16 --noise-type voronoi --voronoi-mode Jitter" \
            "--size 16 --noise-type voronoi --voronoi-mode Edge"; do
  if $PY generate_volumetric.py $spec -o /tmp/vp_cli_test.png >/dev/null 2>&1; then
    echo "PASS: cli $spec"
  else
    echo "FAIL: cli $spec"; fails=$((fails+1))
  fi
done

echo "== CLI validation =="
if $PY generate_volumetric.py --size 0 >/dev/null 2>&1; then
  echo "FAIL: size 0 accepted"; fails=$((fails+1))
else
  echo "PASS: size 0 rejected"
fi
if $PY generate_volumetric.py --octaves 0 --size 8 >/dev/null 2>&1; then
  echo "FAIL: octaves 0 accepted"; fails=$((fails+1))
else
  echo "PASS: octaves 0 rejected"
fi

echo "== Cellular tiling (continuity + seamless wrap) =="
$PY tests/cellular_check.py || fails=$((fails+1))

echo "== GPU/CPU bit-exact equivalence (optional: needs pyopencl + OpenCL device) =="
GPU_PY=${GPU_PY:-$PY}
if $GPU_PY -c "import pyopencl" >/dev/null 2>&1; then
  $GPU_PY tests/gpu_equiv.py || fails=$((fails+1))
else
  echo "SKIP: pyopencl not importable by $GPU_PY"
fi

echo "== GUI end-to-end (optional: needs DISPLAY + tkinter) =="
GUI_PY=${GUI_PY:-$PY}
if [ -n "${DISPLAY:-}" ] && $GUI_PY -c "import tkinter" >/dev/null 2>&1; then
  $GUI_PY tests/gui_smoke.py || fails=$((fails+1))
else
  echo "SKIP: DISPLAY unset or tkinter missing for $GUI_PY"
fi

if [ "$fails" -eq 0 ]; then
  echo "RESULT: ALL PASS"
else
  echo "RESULT: $fails FAILURES"
fi
exit "$fails"
