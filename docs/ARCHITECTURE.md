# Architecture

## High-Level Flow

```
User Input (CLI args / GUI controls)
    │
    ▼
_generate_volume()              ← Core generation: L×L×L noise volume
    │
    ▼
compute_grid_dims()             ← Calculate grid layout (cols, rows)
    │
    ▼
volume_to_grid()                ← Flatten volume slices into 2D grid
    │
    ▼ (GUI only)
upscale_grid()                  ← Scale up for preview display
    │
    ▼
write_png()                     ← Minimal PNG writer (zlib + struct)
```

## Module Structure

### `generate_volumetric.py` (659 lines)

The main module containing all generation logic and the CLI entry point.

**Sections:**
1. **PNG Writer** (lines 22–58) — Minimal PNG encoder using `struct` and `zlib`
2. **Noise Primitives** (lines 61–103) — Hash, lerp, smoothstep, Worley hash, Perlin gradient
3. **Value Noise** (lines 106–171) — Direct and table-based sampling
4. **Worley Noise** (lines 174–257) — Cellular noise with feature points
5. **Voronoi Noise** (lines 260–388) — Cellular noise with 5 output modes
6. **FBM Perlin Noise** (lines 391–462) — Gradient-based noise with dot products
7. **Volume Generation** (lines 465–551) — FBM loop with table pre-computation
8. **Grid Layout** (lines 554–587) — Dimension calculation and slice arrangement
9. **Upscaling** (lines 590–603) — Neighbor-free pixel scaling for preview
10. **CLI Entry Point** (lines 606–659) — argparse-based argument handling

### `generate_volumetric_gui.py` (341 lines)

A **thin wrapper** around `generate_volumetric.py`. Contains zero noise algorithms or generation logic — all of that is imported from the CLI module. The GUI adds only:
- Tkinter controls (dropdowns, sliders, entries, buttons)
- Threading with cancel support via `threading.Event`
- Live preview rendering and file completion dialogs
- Progress bar and status updates
- Optional OpenCL GPU acceleration toggle

**Sections:**
1. **Constants** (lines 44–47) — Valid sizes, noise types, preview size
2. **App class** (lines 54–331) — Main GUI application
3. **UI Builder** (lines 75–168) — Control panel and preview area layout
4. **Generation Workers** (lines 216–303) — Threaded preview and render
5. **Callbacks** (lines 305–331) — Status updates, completion, error handling

### `generate_volumetric_gpu.py` (972 lines, OpenCL Backend)

Optional GPU acceleration via OpenCL. Provides identical results to the CPU backend with four specialized kernels (one per noise type). Falls back to CPU if OpenCL is unavailable or fails.

**Kernels:**
- `generate_value` — Value noise with precomputed hash tables
- `generate_worley` — Worley/cellular noise with 3×3×3 neighbor search
- `generate_perlin` — FBM Perlin noise with gradient dot products
- `generate_voronoi` — Voronoi cellular noise with 5 output modes

**Data layout:**
- Octave tables are merged into single contiguous GPU arrays
- Metadata (periods, p2s, offsets) passed as `constant int*`
- One thread per voxel, launched as 3D grid

**CPU fallback:**
- Contains duplicate noise sampling functions that mirror `generate_volumetric.py`
- These are used when OpenCL is unavailable or fails
- Imports hash/gradient primitives from `generate_volumetric.py` via `from generate_volumetric import ...`

**Integration:**
- Imported by GUI as optional backend
- Selected via "Use OpenCL (GPU)" checkbox
- Automatic CPU fallback on any error

## Key Constants

| Constant | Value | Location | Description |
|----------|-------|----------|-------------|
| `MAX_TABLE_PERIOD` | 128 | `generate_volumetric.py:470` | Max pre-computed table dimension |
| `VALID_SIZES` | `[4, 16, 64, 256]` | `generate_volumetric_gui.py:44` | GUI dropdown options |
| `DEFAULT_PREVIEW_SIZE` | 16 | `generate_volumetric_gui.py:47` | Fixed size for preview generation |
| `NOISE_TYPES` | 4 types | `generate_volumetric_gui.py:46` | GUI noise type options |
| `VORONOI_MODES` | 5 modes | `generate_volumetric.py:264` | Voronoi output mode options |
| `OPENCL_AVAILABLE` | bool | `generate_volumetric_gpu.py:17` | Whether pyopencl + GPU detected |

## Data Flow

### Volume Generation

Each voxel `(x, y, z)` in the L×L×L volume is computed by:

```python
volume[z][y][x] = sum(amplitude_i * noise(coord_x_i, coord_y_i, coord_z_i) for each octave) / max_amplitude
```

Where:
- `coord_x = (x / size) * hash_period` (normalized to table period)
- `amplitude` halves each octave: `1.0, 0.5, 0.25, ...`
- `max_val` accumulates: `1.0 + 0.5 + 0.25 + ...`
- Result is normalized to `[0, 1]`

### Grid Layout

The L×L×L volume is flattened into a 2D grid for PNG export:

```
Grid cells = L (one per Z-slice)
Grid layout = ceil(sqrt(L)) × ceil(L / ceil(sqrt(L)))
Total grid pixels = (cols × L) × (rows × L)
```

Each cell `C` (where `C = slice_index`) maps to:
- Column: `C % cols`
- Row: `C // cols`
- Pixel value: `int(round(volume[C][y][x] * 255))`

## Noise Table Pre-computation

For performance, each octave's hash/gradient/feature data is pre-computed into a 3D table:

```python
for octave in range(octaves):
    period = max(2, round(size * base_freq * lacunarity^octave))
    if period <= 128:
        table = precompute(period, seed)  # (period)³ entries
    else:
        table = None  # Use direct sampling
```

## Thread Model (GUI)

The GUI uses Python's `threading` module for non-blocking generation:

```
Main Thread          Worker Thread
───────             ────────────
User clicks Preview/Render
    │
    ├─► _generate_volume() ──► volume
    │   (checks cancel_event)
    │
    ├─► volume_to_grid()
    │
    ├─► root.after(0, _preview_ready) ──► _grid_to_photo()
    │
Main Thread updates UI via root.after() callbacks
Worker thread is daemon=True (dies with main thread)
```

## PNG Writer

A minimal PNG encoder with no external dependencies:

1. Build raw pixel data: `b"\x00"` (filter: none) + pixel row bytes
2. Compress with `zlib`
3. Assemble chunks: `IHDR` (header) → `IDAT` (compressed data) → `IEND` (end)
4. Write signature: `89 50 4E 47 0D 0A 1A 0A`

Each chunk includes:
- 4-byte length
- 4-byte type
- data
- 4-byte CRC32

## Type Hints

The codebase uses Python 3.10+ type hints:

- `list[list[int]]` — 2D grid (rows × cols)
- `list[list[list[float]]]` — 3D volume (z × y × x)
- `tuple[float, float, float]` — 3D vectors (gradients, feature points)
- `tuple[int, int]` — 2D coordinates (dimensions, positions)
