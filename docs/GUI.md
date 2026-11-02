# GUI Guide

## Launch

```bash
python3 generate_volumetric_gui.py
```

## Layout

- **Left panel:** Controls (Reference Size, Noise Type, parameters, output path, buttons)
- **Right panel:** Live preview of the generated texture

## Controls

### Reference Size (L)

Dropdown with fixed options: `4`, `16`, `64`, `256`.

These sizes are chosen so that `L × √L` is always a power of two, ensuring the output PNG dimensions are always powers of two.

### Noise Type

Dropdown with four options:
- **Value Noise** — Fast, simple, smooth patterns
- **Worley Noise** — Cellular, organic, cell-like patterns
- **FBM Perlin Noise** — Rich, natural multi-octave detail
- **Voronoi Noise** — Cellular noise with 5 selectable output modes

### Voronoi Mode

When Voronoi Noise is selected, a mode selector appears below the noise type dropdown with these options:

| Mode | Description |
|------|-------------|
| **F1** | Distance to nearest feature point (identical to Worley Noise) |
| **F2** | Distance to second nearest feature point |
| **F1 - F2** | Difference between first and second nearest — highlights cell boundaries |
| **Jitter** | Distance from query point to the feature point of its containing cell |
| **Edge** | Normalized edge detection: F1 / (F1 + F2) |

### Base Frequency (Base Freq)

Controls the base frequency of the noise. Range: 0.001–2.0 (via slider), adjustable via text input (typed values are clamped into range when the entry loses focus).

Lower values produce larger-scale features; higher values produce finer detail.

### Seed

Deterministic seed for reproducible results.

- Type a specific integer for consistent output
- Click **Randomize** to generate a random seed (0 to 2³¹−1)

### Octaves

Number of FBM octaves (detail levels). More octaves add finer detail at higher frequencies.

### Lacunarity

Frequency multiplier between octaves. Range: 0–10 (via slider, snapped to integer steps). Default: 2.0.

Higher values increase the frequency jump between octaves, producing coarser detail at higher levels.

### Use OpenCL (GPU)

When `pyopencl` is installed and an OpenCL-compatible GPU is available, a checkbox appears below Lacunarity.

- **Enabled:** Uses `generate_volumetric_gpu.py` for GPU-accelerated generation
- **Disabled:** Uses the CPU backend (`generate_volumetric.py`)
- **Output is 100% identical** regardless of which backend is used
- Requires starting a new Preview/Render after toggling (no live switch)

**Installation:** `pip install pyopencl` (requires OpenCL runtime on the system)

### Invert Output

Checkbox that inverts pixel values (`255 − value`) after noise generation and quantization. Applies to both the live preview and the rendered PNG.

### Output

File path for the saved PNG.

- Type a path directly, or click **Browse...** to open the file picker
- Default: `volumetric_texture.png` in the current directory

### Buttons

| Button | Action |
|--------|--------|
| **Preview** | Generates the full L³ volume and displays it in the live preview without writing a file (see Workflow below) |
| **Render** | Generates the full volume and writes it to the output PNG file |
| **Cancel** | Enabled during generation; aborts it and returns to `Ready` state |
| **Save Preset** | Saves all generation parameters to a JSON file via the file picker |
| **Load Preset** | Loads generation parameters from a preset JSON file |

### Presets

Presets are plain JSON files (stdlib `json`, no schema library) capturing: `size`, `noise_type`, `voronoi_mode`, `base_freq`, `seed`, `octaves`, `lacunarity`, `invert`, `use_opencl`, `output`. On load, values are validated: `size`/`noise_type`/`voronoi_mode`/`octaves` must be valid (error dialog otherwise); `base_freq` and `lacunarity` are clamped to their slider ranges; unknown keys are ignored.

### Progress Bar & Status

- Displays real generation progress, updated once per completed Z-slice on both CPU and GPU paths (volume phase maps to 0–90% for Preview, 0–60% for Render)
- Shows status text: `Ready`, `Computing volume...`, `Building grid...`, `Writing PNG...`, `Preview ready`, `Done!`, `Cancelled`, `Preset saved/loaded: <name>`, or `Error`
- All buttons (including preset buttons) are disabled during generation

## Workflow

### Preview

1. Click **Preview**
2. The full L³ volume is generated at the selected Reference Size (same generation as Render)
3. The right panel shows one Z-slice at full resolution, scaled to fit the window; the slider cycles through all L slices
4. No file is written — useful for rapid parameter exploration
5. Quantized slices are pre-computed and upscaled grids cached, so slider/resize interaction is smooth

### Render

1. Set parameters (Size, Noise Type, Base Freq, Seed, Octaves, Lacunarity)
2. Set output path (or use default)
3. Click **Render**
4. The full L³ volume is generated at the selected Reference Size
5. The PNG is written to the output path
6. A completion dialog appears with generation stats (output path, volume size, grid dimensions, resolution)

## Threaded Generation

Generation runs in a background worker thread; the UI remains responsive. Worker threads never touch Tkinter directly — they post UI updates to a thread-safe `queue.Queue` that the main thread drains every 50 ms (`_post_ui`/`_poll_ui`), because `root.after()` from a non-main thread is silently dropped on thread-aware Tcl/Tk builds. Parameters are captured on the main thread before the worker starts.

Clicking **Cancel** sets the `_cancel_event` threading flag. The CPU generator checks it between rows; the GPU path launches one Z-slice per kernel launch and checks between slices. Either way generation aborts promptly and the UI returns to idle.

## Limitations

- Requires a display server (won't work in headless environments)
- Large sizes (256³) may be slow and consume significant memory
- On X11/XWayland the GUI forces the `fixed` generic font family for Tk named fonts: Tk 9 renders the default Helvetica→Nimbus Sans L match aliased and ignores Xft resources there. Windows/macOS keep native system fonts (antialiased by default)

## OpenCL GPU Backend

The GPU backend (`generate_volumetric_gpu.py`) provides optional OpenCL-accelerated generation.

**How it works:**
- Four specialized OpenCL kernels (one per noise type) replace the CPU loops
- Kernels compute in IEEE double precision with `volatile` guards that block FMA contraction, keeping output bit-exact vs Python floats
- Octave tables are merged into contiguous GPU arrays; one thread per voxel, launched one Z-slice at a time (so `cancel_event` and progress work)
- Falls back to CPU (the canonical `generate_volumetric.py` generator) if OpenCL is unavailable, a kernel fails, any octave's table period exceeds 128, or the seed is outside 0–2³¹−1
- Output is bitwise identical to CPU (8-bit quantization matches 100%)

**Performance:**
- For small volumes (4³–64³), CPU and GPU have similar runtime (overhead dominates)
- Larger volumes (>128³) benefit more from GPU acceleration
- Requires `pip install pyopencl` and an OpenCL-compatible GPU
