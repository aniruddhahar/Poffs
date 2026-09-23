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

### `generate_volumetric.py` (512 lines)

The main module containing all generation logic and the CLI entry point.

**Sections:**
1. **PNG Writer** (lines 22–58) — Minimal PNG encoder using `struct` and `zlib`
2. **Noise Primitives** (lines 61–103) — Hash, lerp, smoothstep, Worley hash, Perlin gradient
3. **Value Noise** (lines 106–171) — Direct and table-based sampling
4. **Worley Noise** (lines 174–257) — Cellular noise with feature points
5. **Perlin Noise** (lines 260–331) — Gradient-based noise with dot products
6. **Volume Generation** (lines 334–409) — FBM loop with table pre-computation
7. **Grid Layout** (lines 312–445) — Dimension calculation and slice arrangement
8. **Upscaling** (lines 448–461) — Neighbor-free pixel scaling for preview
9. **CLI Entry Point** (lines 464–510) — argparse-based argument handling

### `generate_volumetric_gui.py` (283 lines)

Tkinter-based GUI that imports generation logic from `generate_volumetric.py`.

**Sections:**
1. **Constants** (lines 24–29) — Valid sizes, noise types, preview size
2. **App class** (lines 36–273) — Main GUI application
3. **UI Builder** (lines 56–129) — Control panel and preview area layout
4. **Generation Workers** (lines 162–245) — Threaded preview and render
5. **Callbacks** (lines 247–273) — Status updates, completion, error handling

## Key Constants

| Constant | Value | Location | Description |
|----------|-------|----------|-------------|
| `MAX_TABLE_PERIOD` | 128 | `generate_volumetric.py:339` | Max pre-computed table dimension |
| `VALID_SIZES` | `[4, 16, 64, 256]` | `generate_volumetric_gui.py:26` | GUI dropdown options |
| `DEFAULT_PREVIEW_SIZE` | 16 | `generate_volumetric_gui.py:29` | Fixed size for preview generation |
| `NOISE_TYPES` | 3 types | `generate_volumetric_gui.py:28` | GUI noise type options |
| `DEFAULT_PREVIEW_SIZE` | 16 | `generate_volumetric_gui.py:29` | Fixed preview volume size |

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
