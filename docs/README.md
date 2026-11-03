# Volumetric Texture Generator Documentation

## Overview

A Python tool that generates L×L×L volumetric 3D textures and exports them as greyscale PNG grids. Each cell in the output image represents one Z-axis slice of a 3D noise volume.

**Key characteristics:**
- No third-party packages for core use: the CLI runs on the Python standard library alone; the GUI needs `tkinter` (a stdlib module that ships as a separate OS package on many Linux distributions); optional GPU acceleration uses `pyopencl` + `numpy`
- Four noise algorithms: Value Noise, Worley (Cellular), FBM Perlin Noise, and Voronoi (5 output modes)
- GPU acceleration via OpenCL (bit-exact identical output to CPU; auto-fallback if unavailable)
- Seamless 3D tiling enabled by default (modulo-based wrapping)
- Command-line and GUI entry points; cross-platform launchers (`run.sh` Linux/macOS, `run.cmd`/`run.ps1` Windows)
- GUI: live 2D slice preview + ray-marched 3D volume view, presets (JSON), real progress + cancel (CPU and GPU), OpenCL toggle
- `SimpleVolumetric.unitypackage`: URP Shader Graph for raymarching the textures in Unity
- Verification suite in `tests/run_all.sh` (CLI smokes, cellular tiling checks, GPU↔CPU bit-exactness, GUI end-to-end)

## Requirements

- Python 3.10+ (uses `list[list[int]] | None` type hints — the `X | None` unions require 3.10)
- `tkinter` for the GUI — bundled with Windows/macOS python.org installers; on Linux it is usually a separate package (`python3-tk` on Debian/Ubuntu, `python3-tkinter` on Fedora, `tk` on Arch, `python-tk@3.x` via Homebrew). The launchers detect and try to install it.
- Display server for GUI (X11/Wayland on Linux, native on Windows/macOS)
- **Optional — GPU Acceleration:** `pip install pyopencl numpy` and an OpenCL-compatible GPU (NVIDIA, AMD, or Intel)

## Quick Start

```bash
# Easiest: launcher with dependency checks (Linux/macOS)
./run.sh                     # GUI
./run.sh cli --size 16 ...   # CLI
# Windows: run.cmd (same semantics)

# Or directly:
python3 generate_volumetric.py --size 16 --seed 123 --noise-type perlin
python3 generate_volumetric_gui.py
```

## Project Structure

```
generate_volumetric.py        # CLI entry point + all noise algorithms and generation logic
generate_volumetric_gui.py    # Thin Tkinter wrapper (imports all logic from generate_volumetric.py)
generate_volumetric_gpu.py    # Optional OpenCL GPU backend (delegates fallback to CPU module)
run.sh                        # Launcher with dependency checks (Linux/macOS)
run.cmd / run.ps1             # Launcher with dependency checks (Windows)
SimpleVolumetric.unitypackage # URP Shader Graph (raymarch) for use in Unity
tests/                        # run_all.sh + GUI smoke + GPU/CPU equivalence + cellular tiling checks
docs/                         # Documentation
```

## Documentation Index

- [CLI Reference](CLI.md) — Command-line arguments, usage examples
- [GUI Guide](GUI.md) — Controls, workflow, features
- [Noise Algorithms](NOISE_ALGORITHMS.md) — Technical explanation of value, Worley, Perlin, and Voronoi noise
- [Architecture](ARCHITECTURE.md) — Code structure, data flow, OpenCL GPU backend
