# Agent Instructions

## Project Overview
Volumetric 3D texture generator. Produces greyscale PNGs containing a grid of Z-axis slices from a 3D noise volume. Zero external dependencies (Python stdlib only); optional `pyopencl` for GPU acceleration.

## Entry Points
- **CLI:** `python generate_volumetric.py --size 64 --output texture.png`
- **GUI:** `python generate_volumetric_gui.py` (Tkinter)
- **GPU Backend:** `generate_volumetric_gpu.py` (OpenCL, optional, auto-fallback to CPU)

## Key Commands
```bash
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

## GPU Acceleration
Install `pyopencl` (`pip install pyopencl`) to enable optional GPU-accelerated generation. The GUI shows a "Use OpenCL (GPU)" checkbox when available. Output is 100% identical to CPU (8-bit quantization matches). Falls back to CPU automatically if OpenCL is unavailable or fails.

## GUI Features
- **Noise Types:** Value Noise, Worley Noise (cellular), FBM Perlin Noise, Voronoi Noise (5 output modes)
- **Controls:** Reference Size (L ∈ {4,16,64,256}), Noise Type dropdown, Voronoi Mode selector (shown when Voronoi Noise is selected), Base Freq, Seed, Octaves, Lacunarity, OpenCL toggle
- **Output:** Browseable PNG path
- **Workflow:** Preview (fixed 16³ volume, upscaled to 256px) → Render (full L³ size to file)
- **Layout:** Controls panel on left, live preview on right
- **Cancel:** Threaded generation with cancel support via progress bar
- **Output:** File browser for save path, completion dialog with stats

## Architecture Notes
- Both CLI and GUI share identical noise algorithms via imports (`generate_volumetric_gui.py` imports from `generate_volumetric.py`)
- Generation logic is centralized in `generate_volumetric.py`; GUI adds only Tkinter UI layer
- All textures use seamless 3D tiling by default (modulo-based wrapping) — no option to disable
- Pre-computes hash/value/gradient/feature tables per octave for performance; caps table size at `MAX_TABLE_PERIOD = 128` to prevent freeze at high frequencies
- GUI uses threading for generation; updates UI via `root.after()`
- Seed is passed as function parameter to all generation functions
- Output is always greyscale PNG: each cell in the grid is one Z-slice
- FBM (Fractal Brownian Motion) combines multiple octaves with decreasing amplitude (amplitude halves each octave)
- Reference size L is chosen so that `L × √L` is always a power of two, ensuring output PNG dimensions are powers of two
- GPU backend (`generate_volumetric_gpu.py`) provides OpenCL-accelerated generation with four specialized kernels (one per noise type), auto-fallback to CPU

## Noise Algorithms

### Value Noise
Deterministic scalar values are placed at integer grid points in 3D space using a hash function, then trilinearly interpolated using smoothstep (`t²(3−2t)`) for smooth transitions. Supports both direct per-voxel sampling and pre-computed lookup tables (capped at 128³ entries) for performance. Fastest of all algorithms. Produces smooth, amorphous, cloud-like patterns suitable for generic texture fills.

### Worley Noise (Cellular)
For each query point, computes the distance to the nearest "feature point" — pseudo-random locations in 3D space generated deterministically from a hash. Searches a 3×3×3 neighborhood (including wrapping for seamless tiling) and returns the normalized minimum Euclidean distance scaled to [0, 1]. Produces organic, cell-like, cratered, or stone-like patterns. Commonly used for terrain, stone, and cellular textures.

### FBM Perlin Noise
Gradient-based noise: deterministic gradient vectors are generated per grid point, then the dot product between each gradient and the position offset to the unit cube corner is computed. Results are trilinearly interpolated using smoothstep. Combined via FBM (Fractal Brownian Motion) across multiple octaves with halving amplitude for rich, multi-scale natural detail. Produces the most natural-looking patterns — cloudy, wooden, or marbled depending on parameters. Slowest algorithm due to gradient computation.

### Voronoi Noise (Cellular)
Same feature-point search as Worley noise, but with 5 distinct output modes computed from the nearest (F1) and second-nearest (F2) feature distances:
- **F1:** Distance to nearest feature point (identical to Worley noise output)
- **F2:** Distance to second nearest feature point
- **F1 - F2:** Difference between first and second nearest — highlights cell boundaries as thin lines
- **Jitter:** Distance from query point to its assigned feature point (within the nearest cell)
- **Edge:** Normalized edge detection: F1 / (F1 + F2), clamped to [0, 1]

## Constraints
- No tests, no linting, no type checking configured
- Python 3.10+ (uses `list[list[int]]` type hints)
- GUI requires display server (won't work in headless environments)
- Large sizes (256³) may be slow and consume significant memory
- GPU backend requires `pyopencl` and an OpenCL-compatible GPU (NVIDIA, AMD, or Intel)
