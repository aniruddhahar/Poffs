# Volumetric Texture Generator

A tool for generating volumetric 3D textures as greyscale PNG image grids. Each cell in the output image represents a Z-axis slice of a 3D noise volume, allowing you to visualize and use 3D textures in games, VFX, and procedural content generation.

![Example Output](volumetric_texture.png)

*64³ volume generated with Worley Noise (seed 1337, 4 octaves) — 512×512 px, 8×8 grid of Z-slices.*

## Features

- **Zero dependencies** — Python standard library only (3.10+); optional `pyopencl` for GPU acceleration
- **Four noise algorithms** — Value Noise, Worley (Cellular), FBM Perlin Noise, and Voronoi (5 output modes)
- **GPU acceleration** — Optional OpenCL backend with 100% identical output (install `pyopencl` to enable)
- **Command-line interface** — Batch generation and scripting support
- **Interactive GUI** — Live 2D slice preview and a ray-marched 3D volume view, multiple noise types, parameter controls, and OpenCL toggle
- **Seamless tiling** — Generate wrap-around 3D textures with no edge seams
- **Power-of-two output** — Reference sizes chosen so the output PNG is always a power-of-two dimension
- **Configurable FBM** — Control octaves, frequency, lacunarity, and seed for fine-grained detail
- **Cancel support** — Threaded generation with real per-slice progress reporting and cancel on both CPU and GPU paths
- **Presets** — Save/restore all generation parameters as simple JSON files (GUI buttons)
- **Cross-platform launchers** — `run.sh` (Linux/macOS), `run.cmd`/`run.ps1` (Windows) with dependency checks
- **Verification suite** — `tests/run_all.sh`: CLI smokes, cellular tiling checks, GPU↔CPU bit-exactness, GUI end-to-end

## Noise Types

| Type | Description |
|------|-------------|
| **Value Noise** | Smooth interpolation of random values at grid points. Fast and simple. |
| **Worley Noise** | Cellular noise based on distance to nearest feature point. Produces organic, cell-like patterns. |
| **FBM Perlin Noise** | Fractal Brownian Motion with gradient-based Perlin noise. Rich, natural-looking detail across multiple octaves. |
| **Voronoi Noise** | Cellular noise with 5 output modes: `F1` (nearest distance), `F2` (second nearest), `F1 - F2` (edge highlight), `Jitter` (feature distance), `Edge` (normalized edge detection) |

## Installation

No installation required. The tool uses only the Python standard library.

**Requirements:**
- Python 3.10 or later
- A display server for the GUI (X11/Wayland on Linux, Windows, macOS)

**Optional — GPU Acceleration:**
- Install `pyopencl` for OpenCL GPU support: `pip install pyopencl`
- Requires an OpenCL-compatible GPU (NVIDIA, AMD, or Intel)
- When available, the GUI shows a "Use OpenCL (GPU)" checkbox

When OpenCL is not installed, all generation runs on CPU automatically.

## Running

The easiest start is the launcher — it checks/installs dependencies (tkinter, optional `pyopencl`+`numpy` with `--gpu`) and launches the GUI:

| Platform | Command |
|----------|---------|
| Linux / macOS | `./run.sh` (macOS: `bash run.sh`; works with the stock bash 3.2) |
| Windows | `run.cmd` (double-click, or from cmd/PowerShell — wraps `run.ps1`) |

Use `./run.sh cli ...` or `run.cmd cli ...` to pass through to the CLI. On macOS, tkinter comes from the python.org installer or `brew install python-tk@3.x`; on Windows it is the optional "tcl/tk and IDLE" feature of the Python installer.

### CLI

Generate a volumetric texture from the command line:

```bash
# Generate default 64³ texture
python generate_volumetric.py

# Custom parameters
python generate_volumetric.py --size 128 --seed 123 --octaves 6 --base-freq 0.5 --output my_texture.png
```

#### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--size`, `-s` | int | 64 | Cube dimension L |
| `--output`, `-o` | str | volumetric_texture.png | Output PNG path |
| `--octaves` | int | 4 | Number of noise octaves for detail |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--base-freq` | float | 0.01 | Base noise frequency |
| `--lacunarity` | float | 2.0 | Frequency multiplier between octaves |
| `--noise-type` | str | `value` | Noise algorithm: `value`, `worley`, `perlin`, or `voronoi` |
| `--voronoi-mode` | str | `F1` | Voronoi output mode (only with `--noise-type voronoi`): `F1`, `F2`, `F1 - F2`, `Jitter`, `Edge` |

### GUI

Launch the interactive GUI:

```bash
python generate_volumetric_gui.py
```

#### GUI Controls

| Control | Description |
|---------|-------------|
| **Reference Size (L)** | Cube dimension. Options: 4, 16, 64, 256. Output dimensions are always a power of two (`L × √L`). |
| **Noise Type** | Select between Value Noise, Worley Noise, FBM Perlin Noise, and Voronoi Noise. |
| **Voronoi Mode** | When Voronoi Noise is selected, choose output mode: F1, F2, F1-F2, Jitter, or Edge. |
| **Base Freq** | Base frequency for the noise (0.001–2.0). |
| **Seed** | Random seed for reproducible results. Use the **Randomize** button for a new random seed. |
| **Octaves** | Number of FBM octaves (detail levels). |
| **Lacunarity** | Frequency multiplier between octaves (0–10, integer steps). Default: 2.0. |
| **Use OpenCL (GPU)** | When `pyopencl` is installed and an OpenCL GPU is available, enable GPU-accelerated generation. Requires a restart of the generation to take effect. |
| **Invert Output** | Inverts pixel values (255 − value). Applies to the rendered PNG and the 2D preview (next generation); in the 3D view it re-composites the current frame instantly. |
| **Output** | File path for the saved PNG. Use **Browse...** to pick a location. |
| **Save Preset / Load Preset** | Save or restore all generation parameters as a simple JSON preset file. Loaded values are validated (invalid values rejected, out-of-range frequencies clamped). |

#### GUI Workflow

1. **Preview** — Generates the full L³ volume at the selected Reference Size and displays it in the live preview. The preview panel has a **View** toggle: **2D Slices** shows one Z-slice at full resolution with a slider through all L slices; **3D Volume** ray-marches the volume into a shaded cube you can orient with Yaw/Pitch and tune with Density/Steps. No file is written.
2. **Render** — Generates the full-resolution volume at the selected Reference Size and writes it to the output PNG file.

A live preview is displayed on the right side of the window and updates after each generation. The 3D view is progressive (coarse→fine), cancellable, and uses the OpenCL ray-march kernel when GPU acceleration is on (CPU software march otherwise).

## Output Format

The output is a greyscale PNG where:

- Each cell is one Z-axis slice of the L×L×L volume
- Cells are arranged in a `√L × √L` grid
- Total dimensions: `(L × √L) × (L × √L)` pixels
- Pixel values range from 0 (black) to 255 (white)

| Reference Size (L) | Grid | Output Resolution |
|--------------------|------|-------------------|
| 4 | 2×2 | 8×8 px |
| 16 | 4×4 | 64×64 px |
| 64 | 8×8 | 512×512 px |
| 256 | 16×16 | 4096×4096 px |

## Using the Textures in Unity

This repo ships **`SimpleVolumetric.unitypackage`** — a ready-made URP Shader Graph for raymarching the generated volumes:

- `Assets/Shaders/SimpleVolumetric.shadergraph` — URP shader with a `RaymarchLoop` custom function
- `Assets/Shaders/raymarchloop.shadersubgraph` — the reusable raymarch loop sub-graph

Exposed material properties: `Texture3D` (the volume), `Density`, `Step Count`, `Step Size`, `Main Scale`/`Main Wind`, `Detail Scale`/`Detail Wind`, `Time`, `Minimum`/`Maximum`.

**Workflow (Unity 6+, URP):**

1. Generate a texture with this tool (GUI or CLI).
2. In Unity: **Assets ▸ Import Package ▸ Custom Package…** → select `SimpleVolumetric.unitypackage`.
3. Create a Unity `Texture3D` from the sliced PNG — each grid cell is one Z-slice — e.g. with a small editor script that reads the PNG cells into a `Texture3D` (greyscale, sRGB off recommended).
4. Create a material with the `SimpleVolumetric` shader, assign your 3D texture to its `Texture3D` slot, and tune density/step controls.

## Project Structure

```
generate_volumetric.py        # CLI entry point + all noise algorithms and generation logic
generate_volumetric_gui.py    # GUI entry point (thin wrapper, imports from generate_volumetric.py)
generate_volumetric_gpu.py    # Optional OpenCL GPU backend (auto-falls back to CPU)
run.sh                        # Dependency-checking launcher (Linux/macOS: gui | cli, --gpu)
run.cmd / run.ps1             # Windows launcher (same checks/semantics)
SimpleVolumetric.unitypackage # URP Shader Graph for raymarching textures in Unity
docs/                         # CLI / GUI / architecture / noise-algorithm reference
tests/                        # Verification scripts (run_all.sh, gui_smoke.py, gpu_equiv.py, cellular_check.py)
```

- `generate_volumetric.py` is self-contained with inline PNG writing and noise algorithms
- `generate_volumetric_gui.py` imports all generation logic and adds only the Tkinter UI layer
- `generate_volumetric_gpu.py` provides GPU-accelerated generation via OpenCL (optional dependency)

## Verification

No test framework required — run the smoke/verification suite with:

```bash
./tests/run_all.sh
# Optional overrides:
GUI_PY=python3.12 GPU_PY=python3 ./tests/run_all.sh
```

It runs CLI smokes + validation, cellular tiling checks (`tests/cellular_check.py` — continuity across cell boundaries and seamless wrapping), GPU↔CPU bit-exact equivalence (`tests/gpu_equiv.py`, skipped without `pyopencl`), and a GUI end-to-end check (`tests/gui_smoke.py`, skipped without a display; set `GUI_PY` to an interpreter that has `tkinter`).

**Note:** `volumetric_texture.png` is the tracked README sample — the GUI/CLI defaults write to it when run from the repo root. Generate with `-o <other path>` to avoid churn; restore with `git checkout -- volumetric_texture.png`.
