# Volumetric Texture Generator Documentation

## Overview

A Python tool that generates L×L×L volumetric 3D textures and exports them as greyscale PNG grids. Each cell in the output image represents one Z-axis slice of a 3D noise volume.

**Key characteristics:**
- Zero external dependencies (Python standard library only)
- Three noise algorithms: Value Noise, Worley (Cellular), and FBM Perlin Noise
- Seamless 3D tiling enabled by default (modulo-based wrapping)
- Command-line and GUI entry points

## Requirements

- Python 3.10+ (uses `list[list[int]]` type hints)
- Display server for GUI (X11/Wayland on Linux, native on Windows/macOS)

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
generate_volumetric.py        # CLI entry point + shared noise algorithms
generate_volumetric_gui.py    # GUI entry point (Tkinter)
docs/                         # Documentation
```

## Documentation Index

- [CLI Reference](CLI.md) — Command-line arguments, usage examples
- [GUI Guide](GUI.md) — Controls, workflow, features
- [Noise Algorithms](NOISE_ALGORITHMS.md) — Technical explanation of value, Worley, and Perlin noise
- [Architecture](ARCHITECTURE.md) — Code structure, data flow, internals
