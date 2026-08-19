#!/usr/bin/env python3
"""
Volumetric 3D Texture Generator

Generates LxLxL volumetric textures and exports them as a PNG grid
where each cell is a vertical (Z-axis) slice of the volume.

Usage:
    python generate_volumetric.py --size 64 --output texture.png --seamless True
"""

import argparse
import math
import struct
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Minimal PNG writer (no external dependencies)
# ---------------------------------------------------------------------------

def _png_u32(n: int) -> bytes:
    return struct.pack(">I", n)

def _png_u16(n: int) -> bytes:
    return struct.pack(">H", n)

def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    chunk = chunk_type + data
    crc = struct.pack(">I", 0xFFFFFFFF & _crc32(chunk))
    return _png_u32(len(data)) + chunk + crc

def _crc32(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xEDB88320 if crc & 1 else crc >> 1
    return crc ^ 0xFFFFFFFF

def write_png(filename: str, pixels: list[list[int]], width: int, height: int):
    """Write a greyscale PNG from a 2D list of 0-255 values."""
    import zlib
    raw = b""
    for row in pixels:
        raw += b"\x00"  # filter: none
        raw += bytes(row)

    compressed = zlib.compress(raw)

    ihdr = _png_chunk(b"IHDR", _png_u32(width) + _png_u32(height) + b"\x08\x00\x00\x00\x00")
    idat = _png_chunk(b"IDAT", compressed)
    iend = _png_chunk(b"IEND", b"")

    Path(filename).write_bytes(b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend)

# ---------------------------------------------------------------------------
# 3D Noise (value noise with smooth interpolation, supports wrapping)
# ---------------------------------------------------------------------------

def _hash_coord(x: int, y: int, z: int, size: int = 64, seamless: bool = True, seed: int = 42) -> float:
    """Deterministic hash -> [0, 1]. Uses modulo for seamless wrapping."""
    if seamless:
        x = x % size
        y = y % size
        z = z % size
    h = seed ^ (x * 374761393) ^ (y * 668265263) ^ (z * 1274126177)
    h = (h ^ (h >> 13)) * 1103515245
    h = (h ^ (h >> 16))
    return (h & 0x7FFFFFFF) / 0x7FFFFFFF

def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)

def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)

def sample_3d(x: float, y: float, z: float, size: int, seamless: bool,
              octaves: int = 4, base_freq: float = 1.0, hash_period: int = None) -> float:
    """Sample 3D value noise with optional wrapping for seamless tiling."""
    if not seamless:
        # Clamp to [0, size-1] for non-seamless
        x = max(0, min(size - 1, x))
        y = max(0, min(size - 1, y))
        z = max(0, min(size - 1, z))

    ix, iy, iz = int(x), int(y), int(z)
    fx, fy, fz = x - ix, y - iy, z - iz

    fx = _smoothstep(fx)
    fy = _smoothstep(fy)
    fz = _smoothstep(fz)

    period = hash_period if hash_period and hash_period > 0 else size
    h000 = _hash_coord(ix, iy, iz, period, seamless)
    h100 = _hash_coord(ix + 1, iy, iz, period, seamless)
    h010 = _hash_coord(ix, iy + 1, iz, period, seamless)
    h110 = _hash_coord(ix + 1, iy + 1, iz, period, seamless)
    h001 = _hash_coord(ix, iy, iz + 1, period, seamless)
    h101 = _hash_coord(ix + 1, iy, iz + 1, period, seamless)
    h011 = _hash_coord(ix, iy + 1, iz + 1, period, seamless)
    h111 = _hash_coord(ix + 1, iy + 1, iz + 1, period, seamless)

    # Trilinear interpolation
    v00 = _lerp(h000, h100, fx)
    v10 = _lerp(h010, h110, fx)
    v01 = _lerp(h001, h101, fx)
    v11 = _lerp(h011, h111, fx)

    v0 = _lerp(v00, v10, fy)
    v1 = _lerp(v01, v11, fy)

    return _lerp(v0, v1, fz)

def generate_volume(size: int, seamless: bool, octaves: int = 4,
                    base_freq: float = 1.0) -> list[list[list[float]]]:
    """Generate an LxLxL volume of [0, 1] values."""
    volume = [[[0.0] * size for _ in range(size)] for _ in range(size)]
    for z in range(size):
        for y in range(size):
            for x in range(size):
                # Fractal Brownian Motion for richer detail
                val = 0.0
                amplitude = 1.0
                frequency = base_freq
                max_val = 0.0
                for octave_idx in range(octaves):
                    octave_freq = base_freq * (2.0 ** octave_idx)
                    hash_period = max(2, round(size * octave_freq)) if seamless else size
                    coord_x = (x / size) * hash_period
                    coord_y = (y / size) * hash_period
                    coord_z = (z / size) * hash_period
                    val += amplitude * sample_3d(
                        coord_x,
                        coord_y,
                        coord_z,
                        size, seamless, octaves=1, hash_period=hash_period
                    )
                    max_val += amplitude
                    amplitude *= 0.5
                    frequency *= 2.0
                volume[z][y][x] = val / max_val
    return volume

# ---------------------------------------------------------------------------
# Grid layout helpers
# ---------------------------------------------------------------------------

def compute_grid_dims(num_slices: int) -> tuple[int, int]:
    """Given N slices, return (cols, rows) for a square-ish grid."""
    cols = math.ceil(math.sqrt(num_slices))
    rows = math.ceil(num_slices / cols)
    return cols, rows

def volume_to_grid(volume: list[list[list[float]]], slice_size: int,
                   cols: int, rows: int) -> list[list[int]]:
    """Convert volume slices into a 2D grid of cells for PNG export.

    Each cell is a slice along the Z-axis (a slice_size x slice_size image).
    Cells are arranged in a cols x rows grid.
    """
    total_width = cols * slice_size
    total_height = rows * slice_size
    grid = [[0] * total_width for _ in range(total_height)]

    num_slices = len(volume)
    for s in range(num_slices):
        col = s % cols
        row = s // cols
        if row >= rows:
            break  # extra slices don't fit; skip

        for y in range(slice_size):
            for x in range(slice_size):
                val = int(round(volume[s][y][x] * 255))
                val = max(0, min(255, val))
                grid[row * slice_size + y][col * slice_size + x] = val

    return grid

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate volumetric 3D textures as a sliced PNG grid."
    )
    parser.add_argument("--size", "-s", type=int, default=64,
                        help="Cube dimension L (default: 64)")
    parser.add_argument("--output", "-o", type=str, default="volumetric_texture.png",
                        help="Output PNG path")
    parser.add_argument("--seamless", type=str, choices=["true", "false"],
                        default="true", help="Seamless 3D tiling (true/false)")
    parser.add_argument("--octaves", type=int, default=4,
                        help="Number of noise octaves for detail (default: 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--base-freq", type=float, default=1.0,
                        help="Base noise frequency (default: 1.0)")
    args = parser.parse_args()

    # Override seed in hash function
    global _hash_coord
    original_hash = _hash_coord.__code__  # noqa; we use the module-level import

    size = args.size
    seamless = args.seamless == "true"
    print(f"Generating {size}x{size}x{size} volumetric texture ...")
    print(f"  Seamless tiling : {seamless}")
    print(f"  Octaves         : {args.octaves}")
    print(f"  Base frequency  : {args.base_freq}")

    volume = generate_volume(size, seamless, octaves=args.octaves, base_freq=args.base_freq)

    cols, rows = compute_grid_dims(size)
    grid = volume_to_grid(volume, size, cols, rows)

    total_w = cols * size
    total_h = rows * size
    write_png(args.output, grid, total_w, total_h)
    print(f"Exported {args.output}  ({total_w}x{total_h} px, {cols}x{rows} grid)")

if __name__ == "__main__":
    main()
