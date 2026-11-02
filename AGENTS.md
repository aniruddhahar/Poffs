# Agent Instructions

## Project Overview
Volumetric 3D texture generator. Produces greyscale PNGs containing a grid of Z-axis slices from a 3D noise volume. Zero external dependencies (Python stdlib only); optional `pyopencl` for GPU acceleration.

## Entry Points
- **Launcher (Linux/macOS):** `./run.sh` (checks/installs deps: picks a Python ≥3.10 with tkinter for GUI, `--gpu` adds pyopencl+numpy; portable across bash 3.2/macOS and bash 4/5)
- **Launcher (Windows):** `run.cmd` → `run.ps1` (same checks/semantics; uses the `py` launcher then `python`)
- **CLI:** `python generate_volumetric.py --size 64 --output texture.png`
- **GUI:** `python generate_volumetric_gui.py` (Tkinter)
- **GPU Backend:** `generate_volumetric_gpu.py` (OpenCL, optional, auto-fallback to CPU)

## Key Commands
```bash
# Launch with dependency check/install (GUI default; `cli` subcommand for CLI,
# `--gpu` installs optional pyopencl+numpy)
./run.sh

# Generate a 64³ texture (default)
python generate_volumetric.py

# Generate with custom params
python generate_volumetric.py --size 32 --seed 123 --octaves 6 --base-freq 0.5

# Generate Worley noise
python generate_volumetric.py --noise-type worley

# Generate Voronoi noise with specific mode
python generate_volumetric.py --noise-type voronoi --voronoi-mode "F1 - F2"

# Run GUI
python generate_volumetric_gui.py

# Verification suite (CLI smokes + validation, cellular tiling checks, GPU/CPU bit-exact, GUI end-to-end)
./tests/run_all.sh   # optional: GUI_PY=python3.12 GPU_PY=<python with pyopencl>
```

## CLI Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--size`, `-s` | int | 64 | Cube dimension L |
| `--output`, `-o` | str | volumetric_texture.png | Output PNG path |
| `--octaves` | int | 4 | Number of noise octaves for detail |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--base-freq` | float | 0.01 | Base noise frequency |
| `--lacunarity` | float | 2.0 | Frequency multiplier between octaves |
| `--noise-type` | str | value | Noise algorithm: `value`, `worley`, `perlin`, or `voronoi` |
| `--voronoi-mode` | str | F1 | Voronoi output mode (only with `--noise-type voronoi`): `F1`, `F2`, `F1 - F2`, `Jitter`, `Edge` |

The CLI rejects `--size` < 1 and `--octaves` < 1 (argparse errors). For L outside {4, 16, 64, 256} it prints a note when output dimensions are not powers of two.

## GPU Acceleration
Install `pyopencl` (`pip install pyopencl`) to enable optional GPU-accelerated generation. The GUI shows a "Use OpenCL (GPU)" checkbox when available. Output is bit-exact identical to CPU (kernels use IEEE double + `volatile` guards to block FMA contraction; 8-bit quantization matches). Falls back to CPU automatically if OpenCL is unavailable or fails, any octave's table period exceeds 128, or the seed is outside 0–2³¹−1. GPU launches run one Z-slice at a time so cancel/progress work.

## GUI Features
- **Noise Types:** Value Noise, Worley Noise (cellular), FBM Perlin Noise, Voronoi Noise (5 output modes)
- **Controls:** Reference Size (L ∈ {4,16,64,256}), Noise Type dropdown, Voronoi Mode selector (shown when Voronoi Noise is selected), Base Freq, Seed, Octaves, Lacunarity (0-10 integer steps), OpenCL toggle, Invert Output checkbox, Save/Load Preset buttons (JSON parameter files)
- **Output:** Browseable PNG path
- **Preview Workflow:** Generates full L³ volume; preview panel has a View toggle — **2D Slices** (single Z-slice at full resolution scaled to fit window, slider cycles all slices) or **3D Volume** (ray-marched cube)
- **Preview Performance:** Pre-computes quantized slices on volume generation, caches upscaled grids per (slice, scale) pair, debounced slider (30ms) and resize (50ms) updates
- **3D Preview (ray-march):** Orthographic front-to-back march of a downsampled flat copy (`volume_to_flat`, ≤ `MAX_RENDER_RES`=128) via `render_volume_3d` (CPU) or the `raymarch` OpenCL kernel (`render_volume_3d_gpu`, float, CPU fallback); preview-only, not bit-exact. Controls (bottom of preview, mirror Unity SimpleVolumetric names): Yaw, Pitch (0-360°), Density (0-100 → /DENSITY_SCALE), Steps (8-32). Progressive coarse→fine passes (48→96→128) posted via `_post_ui`; single-slot worker superseded on each change; frames FIFO-capped (`MAX_FRAME_CACHE`=12); invert applied at blit; cancel between rows (CPU) / passes (GPU). Never modifies the noise functions/kernels.
- **Invert Output:** Checkbox inverts pixel values (255 - value); applies to render and 2D preview at generation, and is applied at blit time in the 3D view (instant re-composite, no re-march)
- **Layout:** Controls panel on left, live preview on right with a 2D/3D View toggle at the top
- **Cancel:** Threaded generation; cancel works on CPU (between rows) and GPU (between Z-slices); progress reported per slice via `progress_cb`
- **Fonts:** On X11/XWayland, Tk named fonts are forced to the `fixed` generic family (Tk 9 renders the default Helvetica→Nimbus match aliased and ignores Xft resources); Windows/macOS keep native fonts
- **Completion:** File browser for save path, completion dialog with stats

## Architecture Notes
- Both CLI and GUI share identical noise algorithms via imports (`generate_volumetric_gui.py` imports from `generate_volumetric.py`)
- Generation logic is centralized in `generate_volumetric.py`; GUI adds only Tkinter UI layer
- All textures use seamless 3D tiling by default (modulo-based wrapping) — no option to disable
- Pre-computes hash/value/gradient/feature tables per octave for performance; caps table size at `MAX_TABLE_PERIOD = 128` to prevent freeze at high frequencies
- GUI uses threading for generation; worker threads post UI updates to a `queue.Queue` pumped by the main thread (`_post_ui`/`_poll_ui`) — `root.after()` from worker threads is unreliable on thread-aware Tcl builds
- Seed is passed as function parameter to all generation functions
- Output is always greyscale PNG: each cell in the grid is one Z-slice
- FBM (Fractal Brownian Motion) combines multiple octaves with decreasing amplitude (amplitude halves each octave)
- Reference size L is chosen so that `L × √L` is always a power of two, ensuring output PNG dimensions are powers of two
- GPU backend (`generate_volumetric_gpu.py`) provides OpenCL-accelerated generation with four specialized kernels (one per noise type), auto-fallback to CPU. Kernels receive the seed as `np.int32`; seeds outside 0–2³¹−1 route to CPU. GPU generation launches one Z-slice at a time so `cancel_event`/`progress_cb` work on the GPU path. A fifth `raymarch` kernel (`render_volume_3d_gpu`) powers the 3D preview only — it runs in `float` (preview need not be bit-exact) and falls back to the CPU march; it never touches the four generation kernels or PNG output
- **Preview caching:** Quantized slices pre-computed during volume generation, upscaled grids cached per (slice, scale) pair, debounced UI updates for smooth slider/resize interaction
- **Invert Output:** Applied after noise generation and quantization, affects both preview and render

## Noise Algorithms

### Value Noise
Deterministic scalar values are placed at integer grid points in 3D space using a hash function, then trilinearly interpolated using smoothstep (`t²(3−2t)`) for smooth transitions. Supports both direct per-voxel sampling and pre-computed lookup tables (capped at 128³ entries) for performance. Fastest of all algorithms. Produces smooth, amorphous, cloud-like patterns suitable for generic texture fills.

### Worley Noise (Cellular)
For each query point, computes the distance to the nearest "feature point" — pseudo-random locations in 3D space generated deterministically from a hash. Searches a 3×3×3 cell neighborhood (5×5×5 when the tiling is 2 cells wide) and returns the minimum Euclidean distance, taken to the nearest periodic image of the tiling (offsets wrapped by the period, not by one cell) and scaled by `CELLULAR_SCALE` = 2/√3 to [0, 1]. Produces organic, cell-like, cratered, or stone-like patterns. Commonly used for terrain, stone, and cellular textures.

### FBM Perlin Noise
Gradient-based noise: deterministic gradient vectors are generated per grid point, then the dot product between each gradient and the position offset to the unit cube corner is computed. Results are trilinearly interpolated using smoothstep. Combined via FBM (Fractal Brownian Motion) across multiple octaves with halving amplitude for rich, multi-scale natural detail. Produces the most natural-looking patterns — cloudy, wooden, or marbled depending on parameters. Slowest algorithm due to gradient computation.

### Voronoi Noise (Cellular)
Same feature-point search as Worley noise, but with 5 distinct output modes computed from the nearest (F1) and second-nearest (F2) feature distances:
- **F1:** Distance to nearest feature point (identical to Worley noise output)
- **F2:** Distance to second nearest feature point
- **F1 - F2:** Difference between first and second nearest — highlights cell boundaries as thin lines
- **Jitter:** Distance from query point to the feature point of the cell containing the query point (nearest periodic image, normalized)
- **Edge:** Normalized edge detection: F1 / (F1 + F2), clamped to [0, 1]

## Constraints
- Smoke/verification suite in `tests/run_all.sh` (no test framework, no linting, no type checking)
- Python 3.10+ (uses `list[list[int]]` type hints)
- GUI requires display server (won't work in headless environments)
- Large sizes (256³) may be slow and consume significant memory
- `volumetric_texture.png` is the tracked README sample: the GUI/CLI defaults write to it when run from the repo root — generate with `-o <other path>` to avoid churn; restore with `git checkout -- volumetric_texture.png`
- GPU backend requires `pyopencl` and an OpenCL-compatible GPU (NVIDIA, AMD, or Intel)
