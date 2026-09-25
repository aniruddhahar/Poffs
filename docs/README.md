# Volumetric Texture Generator Documentation

## Overview

A Python tool that generates L×L×L volumetric 3D textures and exports them as greyscale PNG grids. Each cell in the output image represents one Z-axis slice of a 3D noise volume.

**Key characteristics:**
- Zero external dependencies (Python standard library only); optional `pyopencl` for GPU acceleration
- Four noise algorithms: Value Noise, Worley (Cellular), FBM Perlin Noise, and Voronoi (5 output modes)
- GPU acceleration via OpenCL (100% identical output to CPU; auto-fallback if unavailable)
- Seamless 3D tiling enabled by default (modulo-based wrapping)
- Command-line and GUI entry points
- GUI includes "Use OpenCL (GPU)" toggle when OpenCL is available

## Requirements

- Python 3.10+ (uses `list[list[int]]` type hints)
- Display server for GUI (X11/Wayland on Linux, native on Windows/macOS)
- **Optional — GPU Acceleration:** `pip install pyopencl` and an OpenCL-compatible GPU (NVIDIA, AMD, or Intel)

## Quick Start

```bash
# Generate default 64³ texture (value noise)
python3 generate_volumetric.py

# Generate with custom parameters
python3 generate_volumetric.py --size 16 --seed 123 --noise-type perlin

# Launch GUI
python3 generate_volumetric_gui.py
```

## Project Structure

```
generate_volumetric.py        # CLI entry point + all noise algorithms and generation logic
generate_volumetric_gui.py    # Thin Tkinter wrapper (imports all logic from generate_volumetric.py)
generate_volumetric_gpu.py    # Optional OpenCL GPU backend (falls back to CPU)
docs/                         # Documentation
```

## Documentation Index

- [CLI Reference](CLI.md) — Command-line arguments, usage examples
- [GUI Guide](GUI.md) — Controls, workflow, features
- [Noise Algorithms](NOISE_ALGORITHMS.md) — Technical explanation of value, Worley, Perlin, and Voronoi noise
- [Architecture](ARCHITECTURE.md) — Code structure, data flow, OpenCL GPU backend
