# Agent Instructions

## Project Overview
Volumetric 3D texture generator. Produces greyscale PNGs containing a grid of Z-axis slices from a 3D noise volume. Zero external dependencies (Python stdlib only).

## Entry Points
- **CLI:** `python generate_volumetric.py --size 64 --output texture.png --seamless true`
- **GUI:** `python generate_volumetric_gui.py` (Tkinter)

## Key Commands
```bash
# Generate a 64³ texture (default)
python generate_volumetric.py

# Generate with custom params
python generate_volumetric.py --size 32 --seed 123 --octaves 6 --base-freq 0.5

# Run GUI
python generate_volumetric_gui.py
```

## CLI Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--size`, `-s` | int | 64 | Cube dimension L |
| `--output`, `-o` | str | volumetric_texture.png | Output PNG path |
| `--seamless` | str | true | Seamless 3D tiling (true/false) |
| `--octaves` | int | 4 | Number of noise octaves for detail |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--base-freq` | float | 1.0 | Base noise frequency |
| `--lacunarity` | float | 2.0 | Frequency multiplier between octaves |
| `--noise-type` | str | value | Noise algorithm: `value`, `worley`, `perlin` |

## GUI Features
- **Noise Types:** Value Noise, Worley Noise (cellular), FBM Perlin Noise
- **Controls:** Reference Size (L ∈ {4,16,64,256}), Noise Type dropdown, Base Freq, Seed, Octaves, Lacunarity
- **Output:** Browseable PNG path
- **Workflow:** Preview (fixed 16³ volume, upscaled to 256px) → Render (full L³ size to file)
- **Layout:** Controls panel on left, live preview on right
- **Cancel:** Threaded generation with cancel support via progress bar
- **Output:** File browser for save path, completion dialog with stats

## Architecture Notes
- Both CLI and GUI share identical noise algorithms via imports (`generate_volumetric_gui.py` imports from `generate_volumetric.py`)
- Generation logic is centralized in `generate_volumetric.py`; GUI adds only Tkinter UI layer
- All textures use seamless 3D tiling by default (modulo-based wrapping)
- Pre-computes hash/value/gradient tables per octave for performance; caps table size at `MAX_TABLE_PERIOD = 128` to prevent freeze at high frequencies
- GUI uses threading for generation; updates UI via `root.after()`
- Seed is passed as function parameter to all generation functions
- Output is always greyscale PNG: each cell in the grid is one Z-slice
- FBM (Fractal Brownian Motion) combines multiple octaves with decreasing amplitude
- Reference size L is chosen so that `L * sqrt(L)` is always a power of two

## Noise Algorithms
- **Value Noise:** Interpolated random values at grid points
- **Worley (Cellular):** Distance to nearest feature point
- **Perlin (FBM):** Gradient-based noise with dot products

## Constraints
- No tests, no linting, no type checking configured
- Python 3.10+ (uses `list[list[int]]` type hints)
- GUI requires display server (won't work in headless environments)
- Large sizes (256³) may be slow and consume significant memory
