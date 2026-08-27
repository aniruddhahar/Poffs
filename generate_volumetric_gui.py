#!/usr/bin/env python3
"""
Volumetric 3D Texture Generator - GUI
Generates LxLxL volumetric textures as greyscale PNG grids.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import math
import struct
import threading
import zlib
from pathlib import Path

# Valid cube dimensions where output texture is always Power of Two.
# For cube size L, grid is sqrt(L) x sqrt(L), output = L * sqrt(L).
VALID_SIZES = [4, 16, 64, 256]

# Max precomputed table dimension (period^3 entries).
# 128^3 = 2M entries (~16 MB for floats). Larger periods cause freeze at high freq.
MAX_TABLE_PERIOD = 128
NOISE_TYPES = ["Value Noise", "Worley Noise", "FBM Perlin Noise"]
DEFAULT_PREVIEW_SIZE = 16


# ---------------------------------------------------------------------------
# PNG writer (stdlib only)
# ---------------------------------------------------------------------------

def _png_u32(n: int) -> bytes:
    return struct.pack(">I", n)

def _png_u16(n: int) -> bytes:
    return struct.pack(">H", n)

def _crc32(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xEDB88320 if crc & 1 else crc >> 1
    return crc ^ 0xFFFFFFFF

def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    chunk = chunk_type + data
    crc = struct.pack(">I", 0xFFFFFFFF & _crc32(chunk))
    return _png_u32(len(data)) + chunk + crc

def write_png(filename: str, pixels: list[list[int]], width: int, height: int):
    raw = b""
    for row in pixels:
        raw += b"\x00"
        raw += bytes(row)
    compressed = zlib.compress(raw)
    ihdr = _png_chunk(b"IHDR", _png_u32(width) + _png_u32(height) + b"\x08\x00\x00\x00\x00")
    idat = _png_chunk(b"IDAT", compressed)
    iend = _png_chunk(b"IEND", b"")
    Path(filename).write_bytes(b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend)


# ---------------------------------------------------------------------------
# Value Noise
# ---------------------------------------------------------------------------

_hash_seed = 42

def _hash_coord(x: int, y: int, z: int, size: int, seamless: bool = True) -> float:
    if seamless:
        x = x % size
        y = y % size
        z = z % size
    h = _hash_seed ^ (x * 374761393) ^ (y * 668265263) ^ (z * 1274126177)
    h = (h ^ (h >> 13)) * 1103515245
    h = (h ^ (h >> 16))
    return (h & 0x7FFFFFFF) / 0x7FFFFFFF

def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)

def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)

def _sample_value_3d(x: float, y: float, z: float, period: int, seamless: bool) -> float:
    ix, iy, iz = int(x), int(y), int(z)
    fx = _smoothstep(x - ix)
    fy = _smoothstep(y - iy)
    fz = _smoothstep(z - iz)

    if seamless:
        i0x, i1x = ix % period, (ix + 1) % period
        i0y, i1y = iy % period, (iy + 1) % period
        i0z, i1z = iz % period, (iz + 1) % period
    else:
        i0x, i1x = max(0, ix), min(period - 1, ix + 1)
        i0y, i1y = max(0, iy), min(period - 1, iy + 1)
        i0z, i1z = max(0, iz), min(period - 1, iz + 1)

    h000 = _hash_coord(i0x, i0y, i0z, period, True)
    h100 = _hash_coord(i1x, i0y, i0z, period, True)
    h010 = _hash_coord(i0x, i1y, i0z, period, True)
    h110 = _hash_coord(i1x, i1y, i0z, period, True)
    h001 = _hash_coord(i0x, i0y, i1z, period, True)
    h101 = _hash_coord(i1x, i0y, i1z, period, True)
    h011 = _hash_coord(i0x, i1y, i1z, period, True)
    h111 = _hash_coord(i1x, i1y, i1z, period, True)

    v00 = _lerp(h000, h100, fx)
    v10 = _lerp(h010, h110, fx)
    v01 = _lerp(h001, h101, fx)
    v11 = _lerp(h011, h111, fx)
    v0 = _lerp(v00, v10, fy)
    v1 = _lerp(v01, v11, fy)
    return _lerp(v0, v1, fz)


def _precompute_value_table(period: int) -> list[list[list[float]]]:
    table = [[[0.0] * period for _ in range(period)] for _ in range(period)]
    for z in range(period):
        for y in range(period):
            for x in range(period):
                table[z][y][x] = _hash_coord(x, y, z, period, True)
    return table


def _sample_value_table(cx: float, cy: float, cz: float, table: list[list[list[float]]],
                        period: int, seamless: bool) -> float:
    ix, iy, iz = int(cx), int(cy), int(cz)
    fx = _smoothstep(cx - ix)
    fy = _smoothstep(cy - iy)
    fz = _smoothstep(cz - iz)

    if seamless:
        i0x, i1x = ix % period, (ix + 1) % period
        i0y, i1y = iy % period, (iy + 1) % period
        i0z, i1z = iz % period, (iz + 1) % period
    else:
        i0x, i1x = max(0, ix), min(period - 1, ix + 1)
        i0y, i1y = max(0, iy), min(period - 1, iy + 1)
        i0z, i1z = max(0, iz), min(period - 1, iz + 1)

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

def _worley_hash(x: int, y: int, z: int, period: int) -> tuple[float, float, float]:
    hx = x % period if period > 0 else x
    hy = y % period if period > 0 else y
    hz = z % period if period > 0 else z
    h = _hash_seed ^ (hx * 374761393) ^ (hy * 668265263) ^ (hz * 1274126177)
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


def _precompute_worley_table(period: int) -> list[list[tuple[float, float, float]]]:
    table = [[[None] * period for _ in range(period)] for _ in range(period)]
    for z in range(period):
        for y in range(period):
            for x in range(period):
                table[z][y][x] = _worley_hash(x, y, z, period)
    return table


def _sample_worley_table(cx: float, cy: float, cz: float, table: list[list[tuple[float, float, float]]],
                         period: int, seamless: bool) -> float:
    ix, iy, iz = int(cx), int(cy), int(cz)
    min_dist = float("inf")

    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if seamless:
                    cx_i = (ix + dx) % period
                    cy_i = (iy + dy) % period
                    cz_i = (iz + dz) % period
                    fx = cx - ix - table[cz_i][cy_i][cx_i][0]
                    fy = cy - iy - table[cz_i][cy_i][cx_i][1]
                    fz = cz - iz - table[cz_i][cy_i][cx_i][2]
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
                else:
                    cx_i = max(0, min(period - 1, ix + dx))
                    cy_i = max(0, min(period - 1, iy + dy))
                    cz_i = max(0, min(period - 1, iz + dz))
                    fx = max(0, min(1, cx - ix - table[cz_i][cy_i][cx_i][0]))
                    fy = max(0, min(1, cy - iy - table[cz_i][cy_i][cx_i][1]))
                    fz = max(0, min(1, cz - iz - table[cz_i][cy_i][cx_i][2]))

                dist_sq = fx * fx + fy * fy + fz * fz
                if dist_sq < min_dist:
                    min_dist = dist_sq

    return max(0.0, min(1.0, math.sqrt(min_dist) * 2.0))


# ---------------------------------------------------------------------------
# FBM Perlin Noise
# ---------------------------------------------------------------------------

def _perlin_gradient(x: int, y: int, z: int) -> tuple[float, float, float]:
    h1 = _hash_seed ^ (x * 374761393) ^ (y * 668265263) ^ (z * 1274126177)
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


def _precompute_perlin_table(period: int) -> list[list[tuple[float, float, float]]]:
    table = [[[None] * period for _ in range(period)] for _ in range(period)]
    for z in range(period):
        for y in range(period):
            for x in range(period):
                table[z][y][x] = _perlin_gradient(x, y, z)
    return table


def _sample_perlin_table(cx: float, cy: float, cz: float, table: list[list[tuple[float, float, float]]],
                         period: int, seamless: bool) -> float:
    ix, iy, iz = int(cx), int(cy), int(cz)
    dx = _smoothstep(cx - ix)
    dy = _smoothstep(cy - iy)
    dz = _smoothstep(cz - iz)

    if seamless:
        i0x, i1x = ix % period, (ix + 1) % period
        i0y, i1y = iy % period, (iy + 1) % period
        i0z, i1z = iz % period, (iz + 1) % period
    else:
        i0x, i1x = max(0, ix), min(period - 1, ix + 1)
        i0y, i1y = max(0, iy), min(period - 1, iy + 1)
        i0z, i1z = max(0, iz), min(period - 1, iz + 1)

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


# ---------------------------------------------------------------------------
# Volume generation
# ---------------------------------------------------------------------------

def _sample_value_direct(sx: float, sy: float, sz: float, size: int, octave_freq: float,
                         seamless: bool) -> float:
    """Direct value noise sampling without precomputed table."""
    hash_period = max(2, round(size * octave_freq)) if seamless else size
    return _sample_value_3d(sx, sy, sz, hash_period, seamless)


def _sample_perlin_direct(sx: float, sy: float, sz: float, size: int, octave_freq: float,
                          lacunarity: float, octave_idx: int, seamless: bool) -> float:
    """Direct Perlin noise sampling without precomputed table."""
    hash_period = max(2, round(size * octave_freq)) if seamless else size
    ix, iy, iz = int(sx), int(sy), int(sz)
    dx = sx - ix
    dy = sy - iy
    dz = sz - iz
    dx = _smoothstep(dx)
    dy = _smoothstep(dy)
    dz = _smoothstep(dz)

    if seamless:
        i0x, i1x = ix % hash_period, (ix + 1) % hash_period
        i0y, i1y = iy % hash_period, (iy + 1) % hash_period
        i0z, i1z = iz % hash_period, (iz + 1) % hash_period
    else:
        i0x, i1x = max(0, ix), min(hash_period - 1, ix + 1)
        i0y, i1y = max(0, iy), min(hash_period - 1, iy + 1)
        i0z, i1z = max(0, iz), min(hash_period - 1, iz + 1)

    def dot(g, dx_val, dy_val, dz_val):
        return g[0] * dx_val + g[1] * dy_val + g[2] * dz_val

    g000 = _perlin_gradient(i0x, i0y, i0z)
    g100 = _perlin_gradient(i1x, i0y, i0z)
    g010 = _perlin_gradient(i0x, i1y, i0z)
    g110 = _perlin_gradient(i1x, i1y, i0z)
    g001 = _perlin_gradient(i0x, i0y, i1z)
    g101 = _perlin_gradient(i1x, i0y, i1z)
    g011 = _perlin_gradient(i0x, i1y, i1z)
    g111 = _perlin_gradient(i1x, i1y, i1z)

    nx00 = _lerp(dot(g000, sx - ix, sy - iy, sz - iz), dot(g100, sx - ix - 1, sy - iy, sz - iz), dx)
    nx01 = _lerp(dot(g010, sx - ix, sy - iy - 1, sz - iz), dot(g110, sx - ix - 1, sy - iy - 1, sz - iz), dx)
    nx10 = _lerp(dot(g001, sx - ix, sy - iy, sz - iz - 1), dot(g101, sx - ix - 1, sy - iy, sz - iz - 1), dx)
    nx11 = _lerp(dot(g011, sx - ix, sy - iy - 1, sz - iz - 1), dot(g111, sx - ix - 1, sy - iy - 1, sz - iz - 1), dx)
    nx0 = _lerp(nx00, nx01, dy)
    nx1 = _lerp(nx10, nx11, dy)
    return _lerp(nx0, nx1, dz)


def _sample_worley_direct(sx: float, sy: float, sz: float, size: int, octave_freq: float,
                          lacunarity: float, octave_idx: int, seamless: bool) -> float:
    """Direct Worley noise sampling without precomputed table."""
    hash_period = max(2, round(size * octave_freq)) if seamless else size
    ix, iy, iz = int(sx), int(sy), int(sz)

    # Pre-compute neighbor feature points to avoid redundant hashing
    neighbors = {}
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if seamless:
                    cx_i = (ix + dx) % hash_period
                    cy_i = (iy + dy) % hash_period
                    cz_i = (iz + dz) % hash_period
                else:
                    cx_i = max(0, min(hash_period - 1, ix + dx))
                    cy_i = max(0, min(hash_period - 1, iy + dy))
                    cz_i = max(0, min(hash_period - 1, iz + dz))
                neighbors[(dx, dy, dz)] = _worley_hash(cx_i, cy_i, cz_i, hash_period)

    min_dist = float("inf")
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                feat = neighbors[(dx, dy, dz)]
                if seamless:
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
                else:
                    fx = max(0, min(1, sx - ix - feat[0]))
                    fy = max(0, min(1, sy - iy - feat[1]))
                    fz = max(0, min(1, sz - iz - feat[2]))

                dist_sq = fx * fx + fy * fy + fz * fz
                if dist_sq < min_dist:
                    min_dist = dist_sq

    return max(0.0, min(1.0, math.sqrt(min_dist) * 2.0))


def _generate_volume(size: int, seamless: bool, octaves: int, base_freq: float,
                     lacunarity: float, noise_type: str,
                     cancel_event: threading.Event | None = None) -> list[list[list[float]]]:
    """Generate LxLxL volume using pre-computed tables for performance."""
    octave_tables = []
    for octave_idx in range(octaves):
        octave_freq = base_freq * (lacunarity ** octave_idx)
        hash_period = max(2, round(size * octave_freq)) if seamless else size

        if noise_type != "Value Noise":
            # Cap table size to prevent freeze at high frequencies
            if hash_period > MAX_TABLE_PERIOD:
                octave_tables.append((hash_period, None))
                continue

        if noise_type == "Worley Noise":
            table = _precompute_worley_table(hash_period)
        elif noise_type == "FBM Perlin Noise":
            table = _precompute_perlin_table(hash_period)
        else:
            table = _precompute_value_table(hash_period)

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
                                coord_x, coord_y, coord_z, size, octave_freq,
                                lacunarity, octave_idx, seamless)
                        elif noise_type == "FBM Perlin Noise":
                            val += amplitude * _sample_perlin_direct(
                                coord_x, coord_y, coord_z, size, octave_freq,
                                lacunarity, octave_idx, seamless)
                        else:
                            val += amplitude * _sample_value_direct(
                                coord_x, coord_y, coord_z, size, octave_freq,
                                seamless)
                        max_val += amplitude
                        amplitude *= 0.5
                        continue

                    if noise_type == "Worley Noise":
                        val += amplitude * _sample_worley_table(
                            coord_x, coord_y, coord_z, table, hash_period, seamless)
                    elif noise_type == "FBM Perlin Noise":
                        val += amplitude * _sample_perlin_table(
                            coord_x, coord_y, coord_z, table, hash_period, seamless)
                    else:
                        val += amplitude * _sample_value_table(
                            coord_x, coord_y, coord_z, table, hash_period, seamless)

                    max_val += amplitude
                    amplitude *= 0.5
                volume[z][y][x] = val / max_val
    return volume


def _volume_to_grid(volume: list[list[list[float]]], size: int,
                    cols: int, rows: int) -> list[list[int]]:
    total_w = cols * size
    total_h = rows * size
    grid = [[0] * total_w for _ in range(total_h)]

    for s in range(len(volume)):
        col = s % cols
        row = s // cols
        if row >= rows:
            break
        for y in range(size):
            for x in range(size):
                val = int(round(volume[s][y][x] * 255))
                val = max(0, min(255, val))
                grid[row * size + y][col * size + x] = val
    return grid


def _upscale_grid(grid: list[list[int]], scale: int) -> list[list[int]]:
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
# GUI
# ---------------------------------------------------------------------------

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Volumetric Texture Generator")
        self.root.resizable(True, True)

        self.output_path = tk.StringVar(value=str(Path("volumetric_texture.png").resolve()))
        self.size_var = tk.StringVar(value="64")
        self.noise_type_var = tk.StringVar(value="Value Noise")
        self.base_freq_var = tk.DoubleVar(value=1.0)
        self.seed_var = tk.IntVar(value=42)
        self.octaves_var = tk.IntVar(value=4)
        self.lacunarity_var = tk.DoubleVar(value=2.0)
        self.seamless_var = tk.BooleanVar(value=True)
        self.progress = tk.DoubleVar(value=0.0)
        self.generating = False
        self._cancel_event = threading.Event()
        self.preview_image = tk.PhotoImage(width=256, height=256)

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        # --- Left: Controls ---
        ctrl_frame = ttk.LabelFrame(main, text="Controls", padding=8)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        # Reference Size
        ttk.Label(ctrl_frame, text="Reference Size (L):").pack(anchor=tk.W, pady=(4, 2))
        ttk.Combobox(ctrl_frame, textvariable=self.size_var,
                      values=[str(s) for s in VALID_SIZES], state="readonly",
                      width=12).pack(fill=tk.X, pady=(0, 8))

        # Noise Type
        ttk.Label(ctrl_frame, text="Noise Type:").pack(anchor=tk.W, pady=(4, 2))
        ttk.Combobox(ctrl_frame, textvariable=self.noise_type_var,
                      values=NOISE_TYPES, state="readonly", width=12).pack(fill=tk.X, pady=(0, 8))

        # Base Frequency
        ttk.Label(ctrl_frame, text="Base Freq:").pack(anchor=tk.W, pady=(4, 2))
        freq_frame = ttk.Frame(ctrl_frame)
        freq_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Scale(freq_frame, from_=0.0, to=100.0,
                  variable=self.base_freq_var, orient=tk.HORIZONTAL).pack(fill=tk.X, expand=True, side=tk.LEFT)
        ttk.Entry(freq_frame, textvariable=self.base_freq_var, width=7).pack(side=tk.LEFT, padx=(6, 0))

        # Seed
        ttk.Label(ctrl_frame, text="Seed:").pack(anchor=tk.W, pady=(4, 2))
        seed_frame = ttk.Frame(ctrl_frame)
        seed_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Entry(seed_frame, textvariable=self.seed_var, width=7).pack(side=tk.LEFT)
        ttk.Button(seed_frame, text="Randomize", command=self._randomize_seed).pack(side=tk.LEFT, padx=(4, 0))

        # Octaves
        ttk.Label(ctrl_frame, text="Octaves:").pack(anchor=tk.W, pady=(4, 2))
        ttk.Entry(ctrl_frame, textvariable=self.octaves_var, width=12).pack(fill=tk.X, pady=(0, 8))

        # Lacunarity
        ttk.Label(ctrl_frame, text="Lacunarity:").pack(anchor=tk.W, pady=(4, 2))
        lac_frame = ttk.Frame(ctrl_frame)
        lac_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Scale(lac_frame, from_=0.0, to=2.0,
                  variable=self.lacunarity_var, orient=tk.HORIZONTAL).pack(fill=tk.X, expand=True)
        ttk.Label(lac_frame, textvariable=self.lacunarity_var, width=6).pack(side=tk.LEFT, padx=(6, 0))

        # Seamless
        ttk.Checkbutton(ctrl_frame, text="Seamless Tiling",
                        variable=self.seamless_var).pack(anchor=tk.W, pady=(8, 4))

        # Output Path
        ttk.Label(ctrl_frame, text="Output:").pack(anchor=tk.W, pady=(8, 2))
        out_frame = ttk.Frame(ctrl_frame)
        out_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Entry(out_frame, textvariable=self.output_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(out_frame, text="Browse...", command=self._browse_output).pack(side=tk.LEFT, padx=(4, 0))

        # Progress
        progress_frame = ttk.Frame(ctrl_frame)
        progress_frame.pack(fill=tk.X, pady=(8, 4))
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress, maximum=100)
        self.progress_bar.pack(fill=tk.X)
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.pack(pady=(2, 0))

        # Buttons
        btn_frame = ttk.Frame(ctrl_frame)
        btn_frame.pack(fill=tk.X, pady=(8, 0))
        self.btn_preview = ttk.Button(btn_frame, text="Preview", command=self._preview)
        self.btn_preview.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        self.btn_render = ttk.Button(btn_frame, text="Render", command=self._render)
        self.btn_render.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # --- Right: Preview ---
        preview_frame = ttk.LabelFrame(main, text="Preview", padding=8)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.preview_label = ttk.Label(preview_frame, image=self.preview_image, relief=tk.SUNKEN)
        self.preview_label.pack(fill=tk.BOTH, expand=True)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialfile=self.output_path.get()
        )
        if path:
            self.output_path.set(path)

    def _randomize_seed(self):
        import random
        self.seed_var.set(random.randint(0, 2**31 - 1))

    def _update_status(self, text: str, progress: float = None):
        self.status_label.config(text=text)
        if progress is not None:
            self.progress.set(progress)
            self.root.update_idletasks()

    def _toggle_buttons(self):
        state = tk.DISABLED if self.generating else tk.NORMAL
        self.btn_preview.config(state=state)
        self.btn_render.config(state=state)

    def _grid_to_photo(self, grid: list[list[int]]) -> None:
        data = []
        for row in grid:
            row_strs = [f"#{v:02x}{v:02x}{v:02x}" for v in row]
            data.append("{" + " ".join(row_strs) + "}")
        self.preview_image.put(" ".join(data))

    def _preview(self):
        if self.generating:
            return
        self._cancel_event.clear()
        self.generating = True
        self._toggle_buttons()
        self._update_status("Generating preview...", 0)

        seed = self.seed_var.get()
        global _hash_seed
        _hash_seed = seed

        def worker():
            try:
                size = DEFAULT_PREVIEW_SIZE
                seamless = self.seamless_var.get()
                octaves = self.octaves_var.get()
                base_freq = self.base_freq_var.get()
                lacunarity = self.lacunarity_var.get()

                self.root.after(0, self._update_status, "Computing volume...", 20)
                volume = _generate_volume(
                    size, seamless, octaves, base_freq, lacunarity,
                    self.noise_type_var.get(), self._cancel_event
                )
                if self._cancel_event.is_set():
                    self.root.after(0, self._generation_cancelled)
                    return

                self.root.after(0, self._update_status, "Building grid...", 60)
                cols = math.ceil(math.sqrt(size))
                rows = math.ceil(size / cols)
                grid = _volume_to_grid(volume, size, cols, rows)

                # Upscale for display
                target = 256
                scale = max(1, target // (cols * size))
                display_grid = _upscale_grid(grid, scale)

                self.root.after(0, self._preview_ready, display_grid)
            except Exception as e:
                self.root.after(0, self._generation_failed, str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _render(self):
        if self.generating:
            return
        self._cancel_event.clear()
        self.generating = True
        self._toggle_buttons()
        self._update_status("Rendering...", 0)

        seed = self.seed_var.get()
        global _hash_seed
        _hash_seed = seed

        def worker():
            try:
                size = int(self.size_var.get())
                seamless = self.seamless_var.get()
                octaves = self.octaves_var.get()
                base_freq = self.base_freq_var.get()
                lacunarity = self.lacunarity_var.get()
                output = self.output_path.get()

                self.root.after(0, self._update_status, "Computing volume...", 10)
                volume = _generate_volume(
                    size, seamless, octaves, base_freq, lacunarity,
                    self.noise_type_var.get(), self._cancel_event
                )
                if self._cancel_event.is_set():
                    self.root.after(0, self._generation_cancelled)
                    return

                self.root.after(0, self._update_status, "Building grid...", 70)
                cols = math.ceil(math.sqrt(size))
                rows = math.ceil(size / cols)
                grid = _volume_to_grid(volume, size, cols, rows)

                self.root.after(0, self._update_status, "Writing PNG...", 95)
                total_w = cols * size
                total_h = rows * size
                write_png(output, grid, total_w, total_h)

                self.root.after(0, self._generation_complete, output, size, cols, rows)
            except Exception as e:
                self.root.after(0, self._generation_failed, str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _preview_ready(self, grid: list[list[int]]):
        self.generating = False
        self._grid_to_photo(grid)
        self._toggle_buttons()
        self._update_status("Preview ready", 100)

    def _generation_cancelled(self):
        self.generating = False
        self._cancel_event.clear()
        self._toggle_buttons()
        self._update_status("Cancelled", 0)

    def _generation_complete(self, output: str, size: int, cols: int, rows: int):
        self.generating = False
        total_w = cols * size
        total_h = rows * size
        self._toggle_buttons()
        self._update_status(f"Done! ({size}³ → {total_w}x{total_h} px)", 100)
        messagebox.showinfo("Complete",
                            f"Generated {output}\n({size}³ volume, {cols}x{rows} grid, "
                            f"{total_w}x{total_h} px)")

    def _generation_failed(self, error: str):
        self.generating = False
        self._toggle_buttons()
        self._update_status("Error", 0)
        messagebox.showerror("Error", f"Generation failed:\n{error}")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
