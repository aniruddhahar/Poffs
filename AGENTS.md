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

## Architecture Notes
- Both files contain duplicated PNG writer and noise functions (not shared via import)
- GUI uses threading for generation; updates UI via `root.after()`
- Seed is set via module-level global `_hash_seed` (GUI) or function param (CLI)
- Output is always greyscale PNG: each cell in the grid is one Z-slice

## Constraints
- No tests, no linting, no type checking configured
- Python 3.10+ (uses `list[list[int]]` type hints)
- GUI requires display server (won't work in headless environments)
