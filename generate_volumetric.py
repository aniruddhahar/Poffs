#!/usr/bin/env python3
"""
Volumetric 3D Texture Generator

Generates LxLxL volumetric textures and exports them as a PNG grid
where each cell is a vertical (Z-axis) slice of the volume.

Supports three noise types: Value Noise, Worley (Cellular), and FBM Perlin Noise.

Usage:
    python generate_volumetric.py --size 64 --output texture.png
    python generate_volumetric.py --size 64 --noise-type worley --lacunarity 2.5 --seed 123
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
# Shared noise primitives
# ---------------------------------------------------------------------------

def _hash_coord(x: int, y: int, z: int, seed: int = 42) -> float:
    """Deterministic hash -> [0, 1]."""
    h = seed ^ (x * 374761393) ^ (y * 668265263) ^ (z * 1274126177)
    h = (h ^ (h >> 13)) * 1103515245
    h = (h ^ (h >> 16))
    return (h & 0x7FFFFFFF) / 0x7FFFFFFF

def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)

def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)

def _worley_hash_coord(x: int, y: int, z: int, seed: int) -> tuple[float, float, float]:
    """Generate a deterministic pseudo-random point for Worley noise."""
    h = seed ^ (x * 374761393) ^ (y * 668265263) ^ (z * 1274126177)
    h = (h ^ (h >> 13)) * 1103515245
    h = h ^ (h >> 16)
    rx = (h & 0x7FFFFFFF) / 0x7FFFFFFF
    h = (h ^ (h >> 13)) * 1103515245
    h = h ^ (h >> 16)
    ry = (h & 0x7FFFFFFF) / 0x7FFFFFFF
    h = (h ^ (h >> 13)) * 1103515245
    h = h ^ (h >> 16)
    rz = (h & 0x7FFFFFFF) / 0x7FFFFFFF
    return (rx, ry, rz)

def _perlin_gradient(x: int, y: int, z: int, seed: int) -> tuple[float, float, float]:
    """Generate a deterministic gradient vector for Perlin noise."""
    h1 = seed ^ (x * 374761393) ^ (y * 668265263) ^ (z * 1274126177)
    h1 = (h1 ^ (h1 >> 13)) * 1103515245
    h1 = h1 ^ (h1 >> 16)
    gx = ((h1 & 0x7FFFFFFF) / 0x7FFFFFFF) * 2.0 - 1.0
    h2 = (h1 ^ (h1 >> 8)) * 1103515245
    h2 = h2 ^ (h2 >> 16)
    gy = ((h2 & 0x7FFFFFFF) / 0x7FFFFFFF) * 2.0 - 1.0
    h3 = ((h2 >> 4) ^ (h2 >> 12)) * 1103515245
    h3 = h3 ^ (h3 >> 16)
    gz = ((h3 & 0x7FFFFFFF) / 0x7FFFFFFF) * 2.0 - 1.0
    return (gx, gy, gz)

# ---------------------------------------------------------------------------
# Value Noise
# ---------------------------------------------------------------------------

def _sample_value_3d(sx: float, sy: float, sz: float, period: int, seed: int) -> float:
    """Sample 3D value noise at floating-point coordinates (seamless)."""
    ix, iy, iz = int(sx), int(sy), int(sz)
    fx = _smoothstep(sx - ix)
    fy = _smoothstep(sy - iy)
    fz = _smoothstep(sz - iz)

    i0x, i1x = ix % period, (ix + 1) % period
    i0y, i1y = iy % period, (iy + 1) % period
    i0z, i1z = iz % period, (iz + 1) % period

    h000 = _hash_coord(i0x, i0y, i0z, seed)
    h100 = _hash_coord(i1x, i0y, i0z, seed)
    h010 = _hash_coord(i0x, i1y, i0z, seed)
    h110 = _hash_coord(i1x, i1y, i0z, seed)
    h001 = _hash_coord(i0x, i0y, i1z, seed)
    h101 = _hash_coord(i1x, i0y, i1z, seed)
    h011 = _hash_coord(i0x, i1y, i1z, seed)
    h111 = _hash_coord(i1x, i1y, i1z, seed)

    v00 = _lerp(h000, h100, fx)
    v10 = _lerp(h010, h110, fx)
    v01 = _lerp(h001, h101, fx)
    v11 = _lerp(h011, h111, fx)
    v0 = _lerp(v00, v10, fy)
    v1 = _lerp(v01, v11, fy)
    return _lerp(v0, v1, fz)

def _precompute_value_table(period: int, seed: int) -> list[list[list[float]]]:
    table = [[[0.0] * period for _ in range(period)] for _ in range(period)]
    for z in range(period):
        for y in range(period):
            for x in range(period):
                table[z][y][x] = _hash_coord(x, y, z, seed)
    return table

def _sample_value_table(cx: float, cy: float, cz: float, table: list[list[list[float]]],
                        period: int) -> float:
    ix, iy, iz = int(cx), int(cy), int(cz)
    fx = _smoothstep(cx - ix)
    fy = _smoothstep(cy - iy)
    fz = _smoothstep(cz - iz)

    i0x, i1x = ix % period, (ix + 1) % period
    i0y, i1y = iy % period, (iy + 1) % period
    i0z, i1z = iz % period, (iz + 1) % period

    h000 = table[i0z][i0y][i0x]
    h100 = table[i0z][i0y][i1x]
    h010 = table[i0z][i1y][i0x]
    h110 = table[i0z][i1y][i1x]
    h001 = table[i1z][i0y][i0x]
    h101 = table[i1z][i0y][i1x]
    h011 = table[i1z][i1y][i0x]
    h111 = table[i1z][i1y][i1x]

    v00 = _lerp(h000, h100, fx)
    v10 = _lerp(h010, h110, fx)
    v01 = _lerp(h001, h101, fx)
    v11 = _lerp(h011, h111, fx)
    v0 = _lerp(v00, v10, fy)
    v1 = _lerp(v01, v11, fy)
    return _lerp(v0, v1, fz)

# ---------------------------------------------------------------------------
# Worley Noise (Cellular)
# ---------------------------------------------------------------------------

def _precompute_worley_table(period: int, seed: int) -> list[list[tuple[float, float, float]]]:
    table = [[[None] * period for _ in range(period)] for _ in range(period)]
    for z in range(period):
        for y in range(period):
            for x in range(period):
                table[z][y][x] = _worley_hash_coord(x, y, z, seed)
    return table

def _sample_worley_table(cx: float, cy: float, cz: float, table: list[list[tuple[float, float, float]]],
                         period: int) -> float:
    ix, iy, iz = int(cx), int(cy), int(cz)
    min_dist = float("inf")

    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                cx_i = (ix + dx) % period
                cy_i = (iy + dy) % period
                cz_i = (iz + dz) % period
                feat = table[cz_i][cy_i][cx_i]
                fx = cx - ix - feat[0]
                fy = cy - iy - feat[1]
                fz = cz - iz - feat[2]
                if fx > 0.5:
                    fx -= 1.0
                elif fx < -0.5:
                    fx += 1.0
                if fy > 0.5:
                    fy -= 1.0
                elif fy < -0.5:
                    fy += 1.0
                if fz > 0.5:
                    fz -= 1.0
                elif fz < -0.5:
                    fz += 1.0

                dist_sq = fx * fx + fy * fy + fz * fz
                if dist_sq < min_dist:
                    min_dist = dist_sq

    return max(0.0, min(1.0, math.sqrt(min_dist) * 2.0))

def _sample_worley_direct(sx: float, sy: float, sz: float, hash_period: int, seed: int) -> float:
    """Direct Worley noise sampling without precomputed table."""
    ix, iy, iz = int(sx), int(sy), int(sz)

    neighbors = {}
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                cx_i = (ix + dx) % hash_period
                cy_i = (iy + dy) % hash_period
                cz_i = (iz + dz) % hash_period
                neighbors[(dx, dy, dz)] = _worley_hash_coord(cx_i, cy_i, cz_i, seed)

    min_dist = float("inf")
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                feat = neighbors[(dx, dy, dz)]
                fx = sx - ix - feat[0]
                fy = sy - iy - feat[1]
                fz = sz - iz - feat[2]
                if fx > 0.5:
                    fx -= 1.0
                elif fx < -0.5:
                    fx += 1.0
                if fy > 0.5:
                    fy -= 1.0
                elif fy < -0.5:
                    fy += 1.0
                if fz > 0.5:
                    fz -= 1.0
                elif fz < -0.5:
                    fz += 1.0

                dist_sq = fx * fx + fy * fy + fz * fz
                if dist_sq < min_dist:
                    min_dist = dist_sq

    return max(0.0, min(1.0, math.sqrt(min_dist) * 2.0))

# ---------------------------------------------------------------------------
# FBM Perlin Noise
# ---------------------------------------------------------------------------

def _precompute_perlin_table(period: int, seed: int) -> list[list[tuple[float, float, float]]]:
    table = [[[None] * period for _ in range(period)] for _ in range(period)]
    for z in range(period):
        for y in range(period):
            for x in range(period):
                table[z][y][x] = _perlin_gradient(x, y, z, seed)
    return table

def _sample_perlin_table(cx: float, cy: float, cz: float, table: list[list[tuple[float, float, float]]],
                         period: int) -> float:
    ix, iy, iz = int(cx), int(cy), int(cz)
    dx = _smoothstep(cx - ix)
    dy = _smoothstep(cy - iy)
    dz = _smoothstep(cz - iz)

    i0x, i1x = ix % period, (ix + 1) % period
    i0y, i1y = iy % period, (iy + 1) % period
    i0z, i1z = iz % period, (iz + 1) % period

    def dot(g, dx_val, dy_val, dz_val):
        return g[0] * dx_val + g[1] * dy_val + g[2] * dz_val

    g000 = table[i0z][i0y][i0x]
    g100 = table[i0z][i0y][i1x]
    g010 = table[i0z][i1y][i0x]
    g110 = table[i0z][i1y][i1x]
    g001 = table[i1z][i0y][i0x]
    g101 = table[i1z][i0y][i1x]
    g011 = table[i1z][i1y][i0x]
    g111 = table[i1z][i1y][i1x]

    nx00 = _lerp(dot(g000, cx - ix, cy - iy, cz - iz), dot(g100, cx - ix - 1, cy - iy, cz - iz), dx)
    nx01 = _lerp(dot(g010, cx - ix, cy - iy - 1, cz - iz), dot(g110, cx - ix - 1, cy - iy - 1, cz - iz), dx)
    nx10 = _lerp(dot(g001, cx - ix, cy - iy, cz - iz - 1), dot(g101, cx - ix - 1, cy - iy, cz - iz - 1), dx)
    nx11 = _lerp(dot(g011, cx - ix, cy - iy - 1, cz - iz - 1), dot(g111, cx - ix - 1, cy - iy - 1, cz - iz - 1), dx)
    nx0 = _lerp(nx00, nx01, dy)
    nx1 = _lerp(nx10, nx11, dy)
    return _lerp(nx0, nx1, dz)

def _sample_perlin_direct(sx: float, sy: float, sz: float, hash_period: int, seed: int) -> float:
    """Direct Perlin noise sampling without precomputed table."""
    ix, iy, iz = int(sx), int(sy), int(sz)
    dx = _smoothstep(sx - ix)
    dy = _smoothstep(sy - iy)
    dz = _smoothstep(sz - iz)

    i0x, i1x = ix % hash_period, (ix + 1) % hash_period
    i0y, i1y = iy % hash_period, (iy + 1) % hash_period
    i0z, i1z = iz % hash_period, (iz + 1) % hash_period

    def dot(g, dx_val, dy_val, dz_val):
        return g[0] * dx_val + g[1] * dy_val + g[2] * dz_val

    g000 = _perlin_gradient(i0x, i0y, i0z, seed)
    g100 = _perlin_gradient(i1x, i0y, i0z, seed)
    g010 = _perlin_gradient(i0x, i1y, i0z, seed)
    g110 = _perlin_gradient(i1x, i1y, i0z, seed)
    g001 = _perlin_gradient(i0x, i0y, i1z, seed)
    g101 = _perlin_gradient(i1x, i0y, i1z, seed)
    g011 = _perlin_gradient(i0x, i1y, i1z, seed)
    g111 = _perlin_gradient(i1x, i1y, i1z, seed)

    nx00 = _lerp(dot(g000, sx - ix, sy - iy, sz - iz), dot(g100, sx - ix - 1, sy - iy, sz - iz), dx)
    nx01 = _lerp(dot(g010, sx - ix, sy - iy - 1, sz - iz), dot(g110, sx - ix - 1, sy - iy - 1, sz - iz), dx)
    nx10 = _lerp(dot(g001, sx - ix, sy - iy, sz - iz - 1), dot(g101, sx - ix - 1, sy - iy, sz - iz - 1), dx)
    nx11 = _lerp(dot(g011, sx - ix, sy - iy - 1, sz - iz - 1), dot(g111, sx - ix - 1, sy - iy - 1, sz - iz - 1), dx)
    nx0 = _lerp(nx00, nx01, dy)
    nx1 = _lerp(nx10, nx11, dy)
    return _lerp(nx0, nx1, dz)

# ---------------------------------------------------------------------------
# Volume generation
# ---------------------------------------------------------------------------

# Max precomputed table dimension (period^3 entries).
# 128^3 = 2M entries (~16 MB for floats). Larger periods cause freeze at high freq.
MAX_TABLE_PERIOD = 128

def _generate_volume(size: int, octaves: int, base_freq: float,
                     lacunarity: float, seed: int, noise_type: str,
                     cancel_event=None) -> list[list[list[float]]]:
    """Generate LxLxL volume using pre-computed tables for performance."""
    octave_tables = []
    for octave_idx in range(octaves):
        octave_freq = base_freq * (lacunarity ** octave_idx)
        hash_period = max(2, round(size * octave_freq))

        # Cap table size to prevent freeze at high frequencies
        if hash_period > MAX_TABLE_PERIOD:
            octave_tables.append((hash_period, None))
            continue

        if noise_type == "Worley Noise":
            table = _precompute_worley_table(hash_period, seed)
        elif noise_type == "FBM Perlin Noise":
            table = _precompute_perlin_table(hash_period, seed)
        else:
            table = _precompute_value_table(hash_period, seed)

        octave_tables.append((hash_period, table))

    volume = [[[0.0] * size for _ in range(size)] for _ in range(size)]
    for z in range(size):
        if cancel_event and cancel_event.is_set():
            break
        for y in range(size):
            if cancel_event and cancel_event.is_set():
                break
            for x in range(size):
                val = 0.0
                amplitude = 1.0
                max_val = 0.0
                for octave_idx in range(octaves):
                    hash_period, table = octave_tables[octave_idx]
                    coord_x = (x / size) * hash_period
                    coord_y = (y / size) * hash_period
                    coord_z = (z / size) * hash_period

                    if table is None:
                        # Use direct sampling for capped tables
                        if noise_type == "Worley Noise":
                            val += amplitude * _sample_worley_direct(
                                coord_x, coord_y, coord_z, hash_period, seed)
                        elif noise_type == "FBM Perlin Noise":
                            val += amplitude * _sample_perlin_direct(
                                coord_x, coord_y, coord_z, hash_period, seed)
                        else:
                            val += amplitude * _sample_value_3d(
                                coord_x, coord_y, coord_z, hash_period, seed)
                        max_val += amplitude
                        amplitude *= 0.5
                        continue

                    if noise_type == "Worley Noise":
                        val += amplitude * _sample_worley_table(
                            coord_x, coord_y, coord_z, table, hash_period)
                    elif noise_type == "FBM Perlin Noise":
                        val += amplitude * _sample_perlin_table(
                            coord_x, coord_y, coord_z, table, hash_period)
                    else:
                        val += amplitude * _sample_value_table(
                            coord_x, coord_y, coord_z, table, hash_period)

                    max_val += amplitude
                    amplitude *= 0.5
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
# Upscaling for preview
# ---------------------------------------------------------------------------

def upscale_grid(grid: list[list[int]], scale: int) -> list[list[int]]:
    h = len(grid)
    w = len(grid[0])
    upscaled = [[0] * (w * scale) for _ in range(h * scale)]
    for y in range(h):
        for x in range(w):
            val = grid[y][x]
            for dy in range(scale):
                for dx in range(scale):
                    upscaled[y * scale + dy][x * scale + dx] = val
    return upscaled

# ---------------------------------------------------------------------------
# Main (CLI)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate volumetric 3D textures as a sliced PNG grid."
    )
    parser.add_argument("--size", "-s", type=int, default=64,
                        help="Cube dimension L (default: 64)")
    parser.add_argument("--output", "-o", type=str, default="volumetric_texture.png",
                        help="Output PNG path")
    parser.add_argument("--octaves", type=int, default=4,
                        help="Number of noise octaves for detail (default: 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--base-freq", type=float, default=0.01,
                        help="Base noise frequency (default: 0.01)")
    parser.add_argument("--lacunarity", type=float, default=2.0,
                        help="Frequency multiplier between octaves (default: 2.0)")
    parser.add_argument("--noise-type", type=str, choices=["value", "worley", "perlin"],
                        default="value", help="Noise algorithm (default: value)")
    args = parser.parse_args()

    size = args.size
    noise_type_map = {"value": "Value Noise", "worley": "Worley Noise", "perlin": "FBM Perlin Noise"}
    noise_type = noise_type_map[args.noise_type]
    
    print(f"Generating {size}x{size}x{size} volumetric texture ...")
    print(f"  Noise type      : {noise_type}")
    print(f"  Octaves         : {args.octaves}")
    print(f"  Base frequency  : {args.base_freq}")
    print(f"  Lacunarity      : {args.lacunarity}")
    print(f"  Seed            : {args.seed}")

    volume = _generate_volume(
        size, octaves=args.octaves, base_freq=args.base_freq,
        lacunarity=args.lacunarity, seed=args.seed, noise_type=noise_type
    )

    cols, rows = compute_grid_dims(size)
    grid = volume_to_grid(volume, size, cols, rows)

    total_w = cols * size
    total_h = rows * size
    write_png(args.output, grid, total_w, total_h)
    print(f"Exported {args.output}  ({total_w}x{total_h} px, {cols}x{rows} grid)")

if __name__ == "__main__":
    main()
