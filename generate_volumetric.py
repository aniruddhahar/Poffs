#!/usr/bin/env python3
"""
Volumetric 3D Texture Generator

Generates LxLxL volumetric textures and exports them as a PNG grid
where each cell is a vertical (Z-axis) slice of the volume.

Supports four noise types: Value Noise, Worley (Cellular), FBM Perlin Noise, and Voronoi.

Usage:
    python generate_volumetric.py --size 64 --output texture.png
    python generate_volumetric.py --size 64 --noise-type worley --lacunarity 2.5 --seed 123
"""

import argparse
import math
import struct
import sys
import zlib
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
    crc = struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
    return _png_u32(len(data)) + chunk + crc

def write_png(filename: str, pixels: list[list[int]], width: int, height: int):
    """Write a greyscale PNG from a 2D list of 0-255 values."""
    raw = b"".join(b"\x00" + bytes(row) for row in pixels)
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

# Normaliser for cellular (Worley/Voronoi) distances: the nearest periodic
# feature is never farther than half a cell diagonal, sqrt(3)/2, so dividing by
# it maps the raw distance into [0, 1]. Written as exact hex floats (identical
# bit patterns to the OpenCL kernel literals) so GPU output stays bit-exact.
CELLULAR_SCALE = float.fromhex("0x1.279a74590331dp+0")   # 2 / sqrt(3)
CELLULAR_SCALE2 = float.fromhex("0x1.279a74590331dp-1")  # 1 / sqrt(3)

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

def _cellular_offset(q: float, cell: int, feat: float, period: int) -> float:
    """Signed offset from a query point to a feature point, taken to the nearest
    periodic image of the tiling (period is in the same cell-sized units).

    The wrap must use the tiling period, not one cell: wrapping by 1.0 aliases
    neighbour cells' features into the current cell, which makes the field
    discontinuous at cell boundaries for any period >= 4.
    """
    half_p = period / 2.0
    d = q - cell - feat
    if d > half_p:
        d -= period
    elif d < -half_p:
        d += period
    return d

def _cellular_radius(period: int) -> int:
    """Half-width of the cell search window.

    A radius of 1 covers every feature within half a cell diagonal (sqrt(3)/2)
    only when the tiling is at least 3 cells wide; a 2-cell tiling needs radius
    2 for the same coverage.
    """
    return 1 if period >= 3 else 2

def _sample_worley_table(cx: float, cy: float, cz: float, table: list[list[tuple[float, float, float]]],
                         period: int) -> float:
    ix, iy, iz = int(cx), int(cy), int(cz)
    min_dist = float("inf")
    r = _cellular_radius(period)

    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                cx_i = (ix + dx) % period
                cy_i = (iy + dy) % period
                cz_i = (iz + dz) % period
                feat = table[cz_i][cy_i][cx_i]
                fx = _cellular_offset(cx, ix + dx, feat[0], period)
                fy = _cellular_offset(cy, iy + dy, feat[1], period)
                fz = _cellular_offset(cz, iz + dz, feat[2], period)

                dist_sq = fx * fx + fy * fy + fz * fz
                if dist_sq < min_dist:
                    min_dist = dist_sq

    return max(0.0, min(1.0, math.sqrt(min_dist) * CELLULAR_SCALE))

def _sample_worley_direct(sx: float, sy: float, sz: float, hash_period: int, seed: int) -> float:
    """Direct Worley noise sampling without precomputed table."""
    ix, iy, iz = int(sx), int(sy), int(sz)
    r = _cellular_radius(hash_period)

    neighbors = {}
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                cx_i = (ix + dx) % hash_period
                cy_i = (iy + dy) % hash_period
                cz_i = (iz + dz) % hash_period
                neighbors[(dx, dy, dz)] = _worley_hash_coord(cx_i, cy_i, cz_i, seed)

    min_dist = float("inf")
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                feat = neighbors[(dx, dy, dz)]
                fx = _cellular_offset(sx, ix + dx, feat[0], hash_period)
                fy = _cellular_offset(sy, iy + dy, feat[1], hash_period)
                fz = _cellular_offset(sz, iz + dz, feat[2], hash_period)

                dist_sq = fx * fx + fy * fy + fz * fz
                if dist_sq < min_dist:
                    min_dist = dist_sq

    return max(0.0, min(1.0, math.sqrt(min_dist) * CELLULAR_SCALE))

# ---------------------------------------------------------------------------
# Voronoi Noise (Cellular)
# ---------------------------------------------------------------------------

# Voronoi output modes
VORONOI_MODES = ["F1", "F2", "F1 - F2", "Jitter", "Edge"]

def _voronoi_sample_table(cx: float, cy: float, cz: float, table: list[list[tuple[float, float, float]]],
                          period: int, mode: str) -> float:
    """Sample Voronoi noise from precomputed table.

    Modes:
        F1      - Distance to nearest feature point (same as Worley)
        F2      - Distance to second nearest feature point
        F1 - F2 - Difference between first and second nearest
        Jitter  - Distance to the feature point of the cell containing the query point
        Edge    - Normalized edge detection: F1 / (F1 + F2)
    """
    ix, iy, iz = int(cx), int(cy), int(cz)
    f1 = float("inf")
    f2 = float("inf")
    r = _cellular_radius(period)

    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                cx_i = (ix + dx) % period
                cy_i = (iy + dy) % period
                cz_i = (iz + dz) % period
                feat = table[cz_i][cy_i][cx_i]
                fx = _cellular_offset(cx, ix + dx, feat[0], period)
                fy = _cellular_offset(cy, iy + dy, feat[1], period)
                fz = _cellular_offset(cz, iz + dz, feat[2], period)

                dist_sq = fx * fx + fy * fy + fz * fz
                if dist_sq < f1:
                    f2 = f1
                    f1 = dist_sq
                elif dist_sq < f2:
                    f2 = dist_sq

    f1 = math.sqrt(f1)
    f2 = math.sqrt(f2)

    if mode == "F1":
        return max(0.0, min(1.0, f1 * CELLULAR_SCALE))
    elif mode == "F2":
        return max(0.0, min(1.0, f2 * CELLULAR_SCALE2))
    elif mode == "F1 - F2":
        return max(0.0, min(1.0, abs(f1 - f2) * CELLULAR_SCALE))
    elif mode == "Jitter":
        return _jitter_distance_table(table, ix, iy, iz, cx, cy, cz, period)
    elif mode == "Edge":
        denom = f1 + f2
        if denom < 1e-10:
            return 0.5
        return max(0.0, min(1.0, (f1 / denom) * 2.0))

    return max(0.0, min(1.0, f1 * CELLULAR_SCALE))

def _jitter_distance_table(cx_table: list[list[tuple[float, float, float]]],
                           ix: int, iy: int, iz: int,
                           cx: float, cy: float, cz: float, period: int) -> float:
    """Jitter mode: distance from query point to the feature point of the
    cell that contains it (nearest periodic image), normalized to [0, 1]."""
    feat = cx_table[iz % period][iy % period][ix % period]
    jx = _cellular_offset(cx, ix, feat[0], period)
    jy = _cellular_offset(cy, iy, feat[1], period)
    jz = _cellular_offset(cz, iz, feat[2], period)
    dist = math.sqrt(jx * jx + jy * jy + jz * jz)
    return max(0.0, min(1.0, dist * CELLULAR_SCALE))

def _voronoi_sample_direct(sx: float, sy: float, sz: float, hash_period: int, seed: int, mode: str) -> float:
    """Sample Voronoi noise without precomputed table (direct hash)."""
    ix, iy, iz = int(sx), int(sy), int(sz)
    f1 = float("inf")
    f2 = float("inf")
    r = _cellular_radius(hash_period)

    neighbors = {}
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                cx_i = (ix + dx) % hash_period
                cy_i = (iy + dy) % hash_period
                cz_i = (iz + dz) % hash_period
                neighbors[(dx, dy, dz)] = _worley_hash_coord(cx_i, cy_i, cz_i, seed)

    for (dx, dy, dz), feat in neighbors.items():
        fx = _cellular_offset(sx, ix + dx, feat[0], hash_period)
        fy = _cellular_offset(sy, iy + dy, feat[1], hash_period)
        fz = _cellular_offset(sz, iz + dz, feat[2], hash_period)

        dist_sq = fx * fx + fy * fy + fz * fz
        if dist_sq < f1:
            f2 = f1
            f1 = dist_sq
        elif dist_sq < f2:
            f2 = dist_sq

    f1 = math.sqrt(f1)
    f2 = math.sqrt(f2)

    if mode == "F1":
        return max(0.0, min(1.0, f1 * CELLULAR_SCALE))
    elif mode == "F2":
        return max(0.0, min(1.0, f2 * CELLULAR_SCALE2))
    elif mode == "F1 - F2":
        return max(0.0, min(1.0, abs(f1 - f2) * CELLULAR_SCALE))
    elif mode == "Jitter":
        feat = _worley_hash_coord(ix % hash_period, iy % hash_period, iz % hash_period, seed)
        jx = _cellular_offset(sx, ix, feat[0], hash_period)
        jy = _cellular_offset(sy, iy, feat[1], hash_period)
        jz = _cellular_offset(sz, iz, feat[2], hash_period)
        dist = math.sqrt(jx * jx + jy * jy + jz * jz)
        return max(0.0, min(1.0, dist * CELLULAR_SCALE))
    elif mode == "Edge":
        denom = f1 + f2
        if denom < 1e-10:
            return 0.5
        return max(0.0, min(1.0, (f1 / denom) * 2.0))

    return max(0.0, min(1.0, f1 * CELLULAR_SCALE))

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
                      cancel_event=None, voronoi_mode: str = "F1",
                      progress_cb=None) -> list[list[list[float]]]:
    """Generate LxLxL volume using pre-computed tables for performance.

    Args:
        voronoi_mode: Only used when noise_type is "Voronoi Noise".
            One of: "F1", "F2", "F1 - F2", "Jitter", "Edge"
        progress_cb: Optional callable(fraction 0.0-1.0) invoked once per
            completed Z-slice (worker thread — must not touch UI directly).
    """
    octave_tables = []
    for octave_idx in range(octaves):
        octave_freq = base_freq * (lacunarity ** octave_idx)
        hash_period = max(2, round(size * octave_freq))

        # Cap table size to prevent freeze at high frequencies
        if hash_period > MAX_TABLE_PERIOD:
            octave_tables.append((hash_period, None))
            continue

        if noise_type == "Worley Noise" or noise_type == "Voronoi Noise":
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
                        if noise_type == "Voronoi Noise":
                            val += amplitude * _voronoi_sample_direct(
                                coord_x, coord_y, coord_z, hash_period, seed, voronoi_mode)
                        elif noise_type == "Worley Noise":
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

                    if noise_type == "Voronoi Noise":
                        val += amplitude * _voronoi_sample_table(
                            coord_x, coord_y, coord_z, table, hash_period, voronoi_mode)
                    elif noise_type == "Worley Noise":
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
        if progress_cb:
            progress_cb((z + 1) / size)
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
# 3D preview: ray-march renderer (CPU software rasterizer, stdlib only)
#
# Rendering is separate from noise generation: it only *reads* an already
# generated volume. Preview output need not be bit-exact with the PNG export.
# ---------------------------------------------------------------------------

# Max cube dimension used for the ray-marched preview (256^3 is downsampled
# to this before marching to keep the pure-Python march interactive-ish).
MAX_RENDER_RES = 128


def volume_to_flat(volume: list[list[list[float]]],
                   max_res: int = MAX_RENDER_RES) -> tuple[list[float], int]:
    """Flatten a [z][y][x] volume into a 1D list (index = z*res*res + y*res + x).

    The volume is downsampled (nearest) so the cube never exceeds max_res per
    axis, keeping the CPU ray-march tractable for large L (e.g. 256 -> 128).
    Returns (flat, res).
    """
    L = len(volume)
    if L == 0:
        return [], 0
    res = L if L <= max_res else max_res
    step = max(1, L // res)
    flat = [0.0] * (res * res * res)
    i = 0
    for z in range(0, res * step, step):
        vz = volume[z]
        for y in range(0, res * step, step):
            row = vz[y]
            for x in range(0, res * step, step):
                flat[i] = row[x]
                i += 1
    return flat, res


def _raymarch_cpu(flat: list[float], res: int, yaw: float, pitch: float,
                  density: float, step_count: int, image_size: int,
                  cancel_event=None) -> list[list[int]] | None:
    """Orthographic front-to-back ray-march of a flat volume into a grayscale
    grid (image_size x image_size, 0-255, composited over black).

    yaw/pitch are in degrees. density is an opacity multiplier. Returns None
    if cancel_event is set mid-render.
    """
    if res <= 0 or image_size <= 0 or step_count <= 0:
        return [[0] * image_size for _ in range(image_size)]

    cy = math.cos(math.radians(yaw)); sy = math.sin(math.radians(yaw))
    cp = math.cos(math.radians(pitch)); sp = math.sin(math.radians(pitch))
    r2 = res * res
    half = 0.5
    frustum = 0.9            # half-extent of the ortho view box (cube fits when rotated)
    cam_z = 1.0             # camera distance along +Z (outside the unit cube)
    inv_img = 1.0 / image_size

    grid = [[0] * image_size for _ in range(image_size)]

    for py in range(image_size):
        if cancel_event is not None and (py & 7) == 0 and cancel_event.is_set():
            return None
        sy_ndc = 1.0 - (py + 0.5) * inv_img * 2.0
        row_out = grid[py]
        for px in range(image_size):
            sx_ndc = (px + 0.5) * inv_img * 2.0 - 1.0
            # Camera-space ray (orthographic): origin on the view plane, dir -Z.
            ox = sx_ndc * frustum
            oy = sy_ndc * frustum
            oz = cam_z
            dx = 0.0; dy = 0.0; dz = -1.0
            # Rotate origin and direction by yaw (about Y) then pitch (about X).
            x1 = ox * cy + oz * sy
            z1 = -ox * sy + oz * cy
            y1 = oy * cp - z1 * sp
            z2 = oy * sp + z1 * cp
            ox, oy, oz = x1, y1, z2
            x1 = dx * cy + dz * sy
            z1 = -dx * sy + dz * cy
            y1 = dy * cp - z1 * sp
            z2 = dy * sp + z1 * cp
            dx, dy, dz = x1, y1, z2

            # Slab intersection with the axis-aligned cube [-half, half]^3.
            tmin = -1e30; tmax = 1e30
            miss = False
            for o, d in ((ox, dx), (oy, dy), (oz, dz)):
                if d < 1e-9 and d > -1e-9:
                    if o < -half or o > half:
                        miss = True
                        break
                else:
                    inv = 1.0 / d
                    ta = (-half - o) * inv
                    tb = (half - o) * inv
                    if ta > tb:
                        ta, tb = tb, ta
                    if ta > tmin:
                        tmin = ta
                    if tb < tmax:
                        tmax = tb
                    if tmin > tmax:
                        miss = True
                        break
            if miss or tmax < 0.0:
                continue

            tstart = tmin if tmin > 0.0 else 0.0
            ds = (tmax - tstart) / step_count
            if ds <= 0.0:
                continue

            C = 0.0; T = 1.0
            for i in range(step_count):
                t = tstart + (i + 0.5) * ds
                ppx = ox + t * dx + half
                ppy = oy + t * dy + half
                ppz = oz + t * dz + half
                # Wrap into [0, 1) for seamless tiling, then to texel space.
                ppx -= int(ppx); ppy -= int(ppy); ppz -= int(ppz)
                fx = ppx * res; fy = ppy * res; fz = ppz * res
                x0 = int(fx); y0 = int(fy); z0 = int(fz)
                rx = fx - x0; ry = fy - y0; rz = fz - z0
                x1 = x0 + 1; y1 = y0 + 1; z1 = z0 + 1
                x0 %= res; x1 %= res
                y0 %= res; y1 %= res
                z0 %= res; z1 %= res
                zb0 = z0 * r2; zb1 = z1 * r2
                r00 = zb0 + y0 * res; r10 = zb0 + y1 * res
                r01 = zb1 + y0 * res; r11 = zb1 + y1 * res
                c00 = flat[r00 + x0] + (flat[r00 + x1] - flat[r00 + x0]) * rx
                c10 = flat[r10 + x0] + (flat[r10 + x1] - flat[r10 + x0]) * rx
                c01 = flat[r01 + x0] + (flat[r01 + x1] - flat[r01 + x0]) * rx
                c11 = flat[r11 + x0] + (flat[r11 + x1] - flat[r11 + x0]) * rx
                a0 = c00 + (c10 - c00) * ry
                a1 = c01 + (c11 - c01) * ry
                dens = a0 + (a1 - a0) * rz

                alpha = dens * density * ds
                if alpha > 1.0:
                    alpha = 1.0
                elif alpha < 0.0:
                    alpha = 0.0
                if alpha > 0.0:
                    C += T * alpha * dens
                    T *= 1.0 - alpha
                    if T < 0.02:
                        break

            if C > 1.0:
                C = 1.0
            elif C < 0.0:
                C = 0.0
            row_out[px] = int(round(C * 255.0))

    return grid


def render_volume_3d(flat: list[float], res: int, yaw: float = 30.0,
                     pitch: float = 20.0, density: float = 2.0,
                     step_count: int = 64, image_size: int = 128,
                     cancel_event=None) -> list[list[int]] | None:
    """Public CPU entry point for the 3D preview (mirrors the GPU signature).

    Returns an image_size x image_size 0-255 grayscale grid, or None on cancel.
    """
    return _raymarch_cpu(flat, res, yaw, pitch, density, step_count,
                         image_size, cancel_event)


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
    parser.add_argument("--noise-type", type=str, choices=["value", "worley", "perlin", "voronoi"],
                        default="value", help="Noise algorithm (default: value)")
    parser.add_argument("--voronoi-mode", type=str, choices=VORONOI_MODES, default="F1",
                        help="Voronoi output mode (only used with --noise-type voronoi)")
    args = parser.parse_args()

    size = args.size
    if size < 1:
        parser.error("--size must be a positive integer")
    if args.octaves < 1:
        parser.error("--octaves must be at least 1")

    noise_type_map = {"value": "Value Noise", "worley": "Worley Noise", "perlin": "FBM Perlin Noise", "voronoi": "Voronoi Noise"}
    noise_type = noise_type_map[args.noise_type]
    
    print(f"Generating {size}x{size}x{size} volumetric texture ...")
    print(f"  Noise type      : {noise_type}")
    if noise_type == "Voronoi Noise":
        print(f"  Voronoi mode    : {args.voronoi_mode}")
    print(f"  Octaves         : {args.octaves}")
    print(f"  Base frequency  : {args.base_freq}")
    print(f"  Lacunarity      : {args.lacunarity}")
    print(f"  Seed            : {args.seed}")

    volume = _generate_volume(
        size, octaves=args.octaves, base_freq=args.base_freq,
        lacunarity=args.lacunarity, seed=args.seed, noise_type=noise_type,
        voronoi_mode=args.voronoi_mode
    )

    cols, rows = compute_grid_dims(size)
    grid = volume_to_grid(volume, size, cols, rows)

    total_w = cols * size
    total_h = rows * size
    if total_w & (total_w - 1) or total_h & (total_h - 1):
        print("  Note: output is not a power of two "
              "(guaranteed only for L in {4, 16, 64, 256})")
    write_png(args.output, grid, total_w, total_h)
    print(f"Exported {args.output}  ({total_w}x{total_h} px, {cols}x{rows} grid)")

if __name__ == "__main__":
    main()
