# Architecture

## High-Level Flow

```
User Input (CLI args / GUI controls)
    │
    ▼
_generate_volume()              ← Core generation: L×L×L noise volume
    │                             (or generate_volume_gpu() — OpenCL, bit-exact,
    │                             auto-falls back to _generate_volume)
    ▼
compute_grid_dims()             ← Calculate grid layout (cols, rows)
    │
    ▼
volume_to_grid()                ← Flatten volume slices into 2D grid
    │
    ▼ (GUI preview only)
upscale_grid()                  ← Scale slice up for display (cached per slice+scale)
    │
    ▼
write_png()                     ← Minimal PNG writer (zlib + struct)
```

Both `_generate_volume()` and `generate_volume_gpu()` accept `cancel_event` (checked between rows / Z-slices) and `progress_cb(fraction)` (invoked once per completed Z-slice).

## Module Structure

### `generate_volumetric.py` (~700 lines)

The main module containing all generation logic and the CLI entry point.

**Sections:**
1. **PNG Writer** — Minimal encoder: rows built with `b"".join`, chunk CRCs via `zlib.crc32`, output via `struct`
2. **Noise Primitives** — Hash, lerp, smoothstep, Worley hash, Perlin gradient
3. **Value Noise** — Direct and table-based sampling
4. **Worley Noise** — Cellular noise with feature points
5. **Voronoi Noise** — Cellular noise with 5 output modes (table + direct samplers)
6. **FBM Perlin Noise** — Gradient-based noise with dot products
7. **Volume Generation** — FBM loop with table pre-computation, `cancel_event`, `progress_cb`
8. **Grid Layout** — Dimension calculation and slice arrangement
9. **Upscaling** — Neighbor-free pixel scaling for preview
10. **CLI Entry Point** — argparse with validation (`--size` ≥ 1, `--octaves` ≥ 1, non-power-of-two output note)

### `generate_volumetric_gui.py` (~630 lines)

A **thin wrapper** around `generate_volumetric.py`. Contains zero noise algorithms or generation logic — all of that is imported from the CLI module (the GPU module is imported optionally). The GUI adds only:
- Tkinter controls (dropdowns, sliders, entries, buttons, preset save/load)
- Threading with cancel support via `threading.Event`
- Live preview rendering (full-volume slice slider), preview caching, completion dialogs
- Progress bar driven by the generator's `progress_cb`
- Optional OpenCL GPU acceleration toggle
- X11-only font setup (`_configure_fonts`: forces the `fixed` generic family on X11/XWayland where Tk 9's default match renders aliased; Windows/macOS untouched)

**Key mechanisms:**
- **UI event queue:** worker threads never touch Tkinter; they call `_post_ui(callback, *args)`, which enqueues onto a `queue.Queue` drained by the main thread's `_poll_ui` (every 50 ms). `root.after()` from worker threads is silently dropped on thread-aware Tcl/Tk builds.
- **Parameter capture:** all Tk variables are read on the main thread before spawning a worker.
- **Preview caching:** quantized 8-bit slices pre-computed when a volume is cached; upscaled grids cached per (slice, scale) with a FIFO cap (`MAX_UPSCALE_CACHE = 12`); slider (30 ms) and resize (50 ms) callbacks debounced via `after_cancel`.
- **Presets:** `_current_preset()` / `_apply_preset()` round-trip all parameters through a plain JSON file (`json` stdlib); loaded values are validated or clamped.

### `generate_volumetric_gpu.py` (~725 lines, OpenCL Backend)

Optional GPU acceleration via OpenCL. Produces **bit-exact** results identical to the CPU backend.

**Kernels:**
- `generate_value`, `generate_worley`, `generate_perlin`, `generate_voronoi` (5 modes via int parameter)
- Written in IEEE double precision; `volatile` stores block FMA contraction (GPU FMA would otherwise diverge from Python's separate mul+add floats)
- `sqrt`/double math only (driver-safe); no `sqrtf`/`fabsf`

**Data layout:**
- Per-octave tables pre-computed with `_precompute_*_table()` in `(tx, ty, tz)` component-array form (needed for GPU buffers), then merged by `_merge_octave_tables()` into single contiguous `cl.Buffer`s
- Metadata (periods, p2s, offsets) passed as `constant int*`
- Launched **one Z-slice at a time** (`global_size=[L, L, 1]` + `global_work_offset=[0, 0, z]`) so `cancel_event` and `progress_cb` work; buffer cleanup in `finally`

**Fallback guards (all route to CPU before any kernel runs):**
- OpenCL unavailable or manager not ready
- Unknown noise type
- Any octave's hash period > `MAX_TABLE_PERIOD` (`_any_octave_capped`) — kernels only support table sampling
- Seed outside 0–2³¹−1 (kernels take `np.int32`)
- Any exception during the OpenCL path

**CPU fallback:** `_generate_volume_cpu()` is a thin delegate to `generate_volumetric._generate_volume()` — single source of truth for noise sampling, no duplicated samplers.

**Integration:** imported by the GUI as an optional backend; selected via the "Use OpenCL (GPU)" checkbox.

## Launchers & Tooling

- `run.sh` — Linux/macOS launcher: picks a Python ≥ 3.10 (with tkinter for GUI), apt/brew tkinter install, optional `--gpu` pyopencl+numpy install with `.venv` fallback; portable across bash 3.2 (macOS) and bash 4/5
- `run.cmd` → `run.ps1` — Windows equivalent (uses the `py` launcher then `python`)
- `tests/run_all.sh` — verification suite: CLI smokes + validation, `tests/gpu_equiv.py` (40 fallback + 32 GPU bit-exact cases with the fallback sabotaged to prove kernel execution), `tests/gui_smoke.py` (14-check GUI state machine)
- `SimpleVolumetric.unitypackage` — URP Shader Graph (`SimpleVolumetric.shadergraph` + `raymarchloop.shadersubgraph`) for raymarching the generated volumes in Unity

## Key Constants

| Constant | Value | Module | Description |
|----------|-------|--------|-------------|
| `MAX_TABLE_PERIOD` | 128 | `generate_volumetric.py` | Max pre-computed table dimension (period > 128 → direct sampling on CPU; whole generation routes to CPU on the GPU path) |
| `VALID_SIZES` | `[4, 16, 64, 256]` | `generate_volumetric_gui.py` | GUI dropdown options (power-of-two output guarantee) |
| `NOISE_TYPES` | 4 types | `generate_volumetric_gui.py` | GUI noise type options |
| `VORONOI_MODES` | 5 modes | `generate_volumetric.py` | Voronoi output mode options |
| `MIN_BASE_FREQ` / `MAX_BASE_FREQ` | 0.001 / 2.0 | `generate_volumetric_gui.py` | Base Freq slider bounds; typed values clamped on focus-out |
| `MAX_UPSCALE_CACHE` | 12 | `generate_volumetric_gui.py` | FIFO cap on cached upscaled preview grids |
| `OPENCL_AVAILABLE` | bool | `generate_volumetric_gpu.py` | Whether pyopencl + an OpenCL platform were detected |

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
        table = None  # Use direct sampling (CPU only; GPU path routes to CPU)
```

## Thread Model (GUI)

```
Main Thread                     Worker Thread (daemon)
───────────                     ─────────────────────
User clicks Preview/Render
    │  (params captured on main thread)
    ├─► spawn worker ──────────►  _generate_volume() / generate_volume_gpu()
    │                             • checks cancel_event (rows / Z-slices)
    │                             • progress_cb(f) ──► _post_ui(...)
    │                             • volume done ─────► _post_ui(_cache_volume)
    │
    ◄─ queue.Queue ◄────────────  (worker never touches Tkinter)
    │
_poll_ui() drains queue every 50 ms → status/progress/preview updates on main thread
```

Cancel sets `_cancel_event`; the generator aborts between rows (CPU) or between Z-slice launches (GPU), and the worker posts `_generation_cancelled`.

## PNG Writer

A minimal PNG encoder with no external dependencies:

1. Build raw pixel data: `b"".join(b"\x00" + bytes(row) for row in pixels)` (filter byte 0 per row)
2. Compress with `zlib.compress`
3. Assemble chunks: `IHDR` (header) → `IDAT` (compressed data) → `IEND` (end)
4. Write signature: `89 50 4E 47 0D 0A 1A 0A`

Each chunk includes:
- 4-byte length
- 4-byte type
- data
- 4-byte CRC32 (`zlib.crc32`)

## Type Hints

The codebase uses Python 3.10+ type hints:

- `list[list[int]]` — 2D grid (rows × cols)
- `list[list[list[float]]]` — 3D volume (z × y × x)
- `tuple[float, float, float]` — 3D vectors (gradients, feature points)
- `tuple[int, int]` — 2D coordinates (dimensions, positions)
