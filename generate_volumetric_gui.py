#!/usr/bin/env python3
"""
Volumetric 3D Texture Generator - GUI
A simple Tkinter interface for generate_volumetric.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import sys
import math
import struct
import zlib
from pathlib import Path


# ---------------------------------------------------------------------------
# Minimal PNG writer (inline from generate_volumetric.py)
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

def write_png_bytes(pixels: list[list[int]], width: int, height: int) -> bytes:
    raw = b""
    for row in pixels:
        raw += b"\x00"
        raw += bytes(row)
    compressed = zlib.compress(raw)
    ihdr = _png_chunk(b"IHDR", _png_u32(width) + _png_u32(height) + b"\x08\x00\x00\x00\x00")
    idat = _png_chunk(b"IDAT", compressed)
    iend = _png_chunk(b"IEND", b"")
    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


# ---------------------------------------------------------------------------
# 3D Noise
# ---------------------------------------------------------------------------

_hash_seed = 42

def _hash_coord(x: int, y: int, z: int, size: int = 64, seamless: bool = True) -> float:
    """Deterministic hash -> [0, 1]. Uses modulo for seamless wrapping."""
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

def sample_3d(x: float, y: float, z: float, size: int, seamless: bool,
              octaves: int = 4, base_freq: float = 1.0, hash_period: int = None) -> float:
    if not seamless:
        x = max(0, min(size - 1, x))
        y = max(0, min(size - 1, y))
        z = max(0, min(size - 1, z))

    ix, iy, iz = int(x), int(y), int(z)
    fx, fy, fz = x - ix, y - iy, z - iz

    fx = _smoothstep(fx)
    fy = _smoothstep(fy)
    fz = _smoothstep(fz)

    if seamless and hash_period and hash_period > 0:
        h000 = _hash_coord(ix % hash_period, iy % hash_period, iz % hash_period, hash_period, True)
        h100 = _hash_coord((ix + 1) % hash_period, iy % hash_period, iz % hash_period, hash_period, True)
        h010 = _hash_coord(ix % hash_period, (iy + 1) % hash_period, iz % hash_period, hash_period, True)
        h110 = _hash_coord((ix + 1) % hash_period, (iy + 1) % hash_period, iz % hash_period, hash_period, True)
        h001 = _hash_coord(ix % hash_period, iy % hash_period, (iz + 1) % hash_period, hash_period, True)
        h101 = _hash_coord((ix + 1) % hash_period, iy % hash_period, (iz + 1) % hash_period, hash_period, True)
        h011 = _hash_coord(ix % hash_period, (iy + 1) % hash_period, (iz + 1) % hash_period, hash_period, True)
        h111 = _hash_coord((ix + 1) % hash_period, (iy + 1) % hash_period, (iz + 1) % hash_period, hash_period, True)
    else:
        h000 = _hash_coord(ix, iy, iz, size, seamless)
        h100 = _hash_coord(ix + 1, iy, iz, size, seamless)
        h010 = _hash_coord(ix, iy + 1, iz, size, seamless)
        h110 = _hash_coord(ix + 1, iy + 1, iz, size, seamless)
        h001 = _hash_coord(ix, iy, iz + 1, size, seamless)
        h101 = _hash_coord(ix + 1, iy, iz + 1, size, seamless)
        h011 = _hash_coord(ix, iy + 1, iz + 1, size, seamless)
        h111 = _hash_coord(ix + 1, iy + 1, iz + 1, size, seamless)

    v00 = _lerp(h000, h100, fx)
    v10 = _lerp(h010, h110, fx)
    v01 = _lerp(h001, h101, fx)
    v11 = _lerp(h011, h111, fx)

    v0 = _lerp(v00, v10, fy)
    v1 = _lerp(v01, v11, fy)

    return _lerp(v0, v1, fz)

def _precompute_value_table(period: int) -> list[list[list[float]]]:
    """Pre-compute hash values for a given period."""
    table = [[[0.0] * period for _ in range(period)] for _ in range(period)]
    for z in range(period):
        for y in range(period):
            for x in range(period):
                table[z][y][x] = _hash_coord(x, y, z, period, True)
    return table


def _sample_value_table(cx: float, cy: float, cz: float, table: list[list[list[float]]],
                        period: int, seamless: bool) -> float:
    """Sample pre-computed table with trilinear interpolation."""
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


def _precompute_perlin_table(period: int) -> list[list[tuple[float, float, float]]]:
    """Pre-compute gradient vectors for Perlin noise."""
    table = [[[None] * period for _ in range(period)] for _ in range(period)]
    for z in range(period):
        for y in range(period):
            for x in range(period):
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

                table[z][y][x] = (gx, gy, gz)
    return table


def _sample_perlin_table(cx: float, cy: float, cz: float, table: list[list[tuple[float, float, float]]],
                         period: int, seamless: bool) -> float:
    """Sample pre-computed Perlin table with trilinear interpolation."""
    ix, iy, iz = int(cx), int(cy), int(cz)
    dx = cx - ix
    dy = cy - iy
    dz = cz - iz
    dx = _smoothstep(dx)
    dy = _smoothstep(dy)
    dz = _smoothstep(dz)

    if seamless:
        i0x, i1x = ix % period, (ix + 1) % period
        i0y, i1y = iy % period, (iy + 1) % period
        i0z, i1z = iz % period, (iz + 1) % period
    else:
        i0x, i1x = max(0, ix), min(period - 1, ix + 1)
        i0y, i1y = max(0, iy), min(period - 1, iy + 1)
        i0z, i1z = max(0, iz), min(period - 1, iz + 1)

    g000 = table[i0z][i0y][i0x]
    g100 = table[i0z][i0y][i1x]
    g010 = table[i0z][i1y][i0x]
    g110 = table[i0z][i1y][i1x]
    g001 = table[i1z][i0y][i0x]
    g101 = table[i1z][i0y][i1x]
    g011 = table[i1z][i1y][i0x]
    g111 = table[i1z][i1y][i1x]

    def dot(g, dx_val, dy_val, dz_val):
        return g[0] * dx_val + g[1] * dy_val + g[2] * dz_val

    nx00 = _lerp(dot(g000, dx, dy, dz), dot(g100, dx - 1, dy, dz), dx)
    nx01 = _lerp(dot(g010, dx, dy - 1, dz), dot(g110, dx - 1, dy - 1, dz), dx)
    nx10 = _lerp(dot(g001, dx, dy, dz - 1), dot(g101, dx - 1, dy, dz - 1), dx)
    nx11 = _lerp(dot(g011, dx, dy - 1, dz - 1), dot(g111, dx - 1, dy - 1, dz - 1), dx)
    nx0 = _lerp(nx00, nx01, dy)
    nx1 = _lerp(nx10, nx11, dy)
    return _lerp(nx0, nx1, dz)


def _precompute_worley_table(period: int) -> list[list[tuple[float, float, float]]]:
    """Pre-compute feature points for Worley noise."""
    table = [[[None] * period for _ in range(period)] for _ in range(period)]
    for z in range(period):
        for y in range(period):
            for x in range(period):
                table[z][y][x] = _worley_hash_coord(x, y, z, period)
    return table


def _sample_worley_table(cx: float, cy: float, cz: float, table: list[list[tuple[float, float, float]]],
                         period: int, seamless: bool) -> float:
    """Sample pre-computed Worley table - find distance to nearest feature point."""
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


def generate_volume(size: int, seamless: bool, octaves: int = 4,
                    base_freq: float = 1.0, noise_type: str = "Value Noise",
                    cancel_event: threading.Event = None) -> list[list[list[float]]]:
    """Generate volume with pre-computed hash tables for speed."""
    # Pre-compute tables for each octave
    octave_tables = []
    for octave_idx in range(octaves):
        octave_freq = (base_freq / 100.0) * (2.0 ** octave_idx)
        hash_period = max(2, round(size * octave_freq)) if seamless else size

        if noise_type == "Worley Noise":
            table = _precompute_worley_table(hash_period)
        elif noise_type == "FBM Perlin Noise":
            table = _precompute_perlin_table(hash_period)
        else:
            table = _precompute_value_table(hash_period)

        octave_tables.append((hash_period, table))

    # Generate volume using pre-computed tables
    volume = [[[0.0] * size for _ in range(size)] for _ in range(size)]
    for z in range(size):
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

                    if noise_type == "Worley Noise":
                        val += amplitude * _sample_worley_table(
                            coord_x, coord_y, coord_z, table, hash_period, seamless
                        )
                    elif noise_type == "FBM Perlin Noise":
                        val += amplitude * _sample_perlin_table(
                            coord_x, coord_y, coord_z, table, hash_period, seamless
                        )
                    else:
                        val += amplitude * _sample_value_table(
                            coord_x, coord_y, coord_z, table, hash_period, seamless
                        )

                    max_val += amplitude
                    amplitude *= 0.5
                volume[z][y][x] = val / max_val

        if cancel_event and cancel_event.is_set():
            break

    return volume


# ---------------------------------------------------------------------------
# Worley Noise
# ---------------------------------------------------------------------------

def _worley_hash_coord(x: int, y: int, z: int, period: int) -> tuple[float, float, float]:
    """Generate a deterministic pseudo-random point for Worley noise."""
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


def sample_worley(x: float, y: float, z: float, size: int, seamless: bool,
                  hash_period: int = None) -> float:
    """Worley (cellular) noise - returns distance to nearest feature point."""
    if seamless and hash_period and hash_period > 0:
        period = hash_period
    else:
        period = size

    ix, iy, iz = int(x), int(y), int(z)

    min_dist = float("inf")
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                cx = ix + dx
                cy = iy + dy
                cz = iz + dz

                fx, fy, fz = x - cx, y - cy, z - cz

                if seamless:
                    fx -= _worley_hash_coord(cx, cy, cz, period)[0]
                    fy -= _worley_hash_coord(cx, cy, cz, period)[1]
                    fz -= _worley_hash_coord(cx, cy, cz, period)[2]

                    # Wrap to find closest image
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
                    fx -= _worley_hash_coord(cx, cy, cz, period)[0]
                    fy -= _worley_hash_coord(cx, cy, cz, period)[1]
                    fz -= _worley_hash_coord(cx, cy, cz, period)[2]
                    fx = max(0, min(1, fx))
                    fy = max(0, min(1, fy))
                    fz = max(0, min(1, fz))

                dist_sq = fx * fx + fy * fy + fz * fz
                if dist_sq < min_dist:
                    min_dist = dist_sq

    dist = math.sqrt(min_dist)
    # Normalize: Worley distance ranges from 0 to ~0.5
    return max(0.0, min(1.0, dist * 2.0))


# ---------------------------------------------------------------------------
# FBM Perlin Noise
# ---------------------------------------------------------------------------

def _perlin_hash(x: int, y: int, z: int) -> float:
    """Hash for Perlin noise gradient selection."""
    h = _hash_seed ^ (x * 374761393) ^ (y * 668265263) ^ (z * 1274126177)
    h = (h ^ (h >> 13)) * 1103515245
    h = h ^ (h >> 16)
    return (h & 0x7FFFFFFF) / 0x7FFFFFFF


def _perlin_gradient(x: int, y: int, z: int, dx: float, dy: float, dz: float) -> float:
    """Compute gradient dot product for Perlin noise."""
    # Generate gradient components from separate hash streams
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

    return gx * dx + gy * dy + gz * dz


def sample_perlin(x: float, y: float, z: float, size: int, seamless: bool,
                  hash_period: int = None) -> float:
    """3D Perlin noise with optional seamless wrapping."""
    if seamless and hash_period and hash_period > 0:
        period = hash_period
    else:
        period = size

    ix, iy, iz = int(x), int(y), int(z)
    fx, fy, fz = x - ix, y - iy, z - iz

    fx = _smoothstep(fx)
    fy = _smoothstep(fy)
    fz = _smoothstep(fz)

    if seamless:
        def hash_wrapped(ix_val: int, iy_val: int, iz_val: int) -> float:
            return _perlin_hash(
                ix_val % period, iy_val % period, iz_val % period
            )

        def grad_wrapped(ix_val: int, iy_val: int, iz_val: int,
                         dx_val: float, dy_val: float, dz_val: float) -> float:
            return _perlin_gradient(
                ix_val % period, iy_val % period, iz_val % period,
                dx_val, dy_val, dz_val
            )
    else:
        hash_wrapped = _perlin_hash
        grad_wrapped = _perlin_gradient

    h000 = hash_wrapped(ix, iy, iz)
    h100 = hash_wrapped(ix + 1, iy, iz)
    h010 = hash_wrapped(ix, iy + 1, iz)
    h110 = hash_wrapped(ix + 1, iy + 1, iz)
    h001 = hash_wrapped(ix, iy, iz + 1)
    h101 = hash_wrapped(ix + 1, iy, iz + 1)
    h011 = hash_wrapped(ix, iy + 1, iz + 1)
    h111 = hash_wrapped(ix + 1, iy + 1, iz + 1)

    g000 = grad_wrapped(ix, iy, iz, fx, fy, fz)
    g100 = grad_wrapped(ix + 1, iy, iz, fx - 1, fy, fz)
    g010 = grad_wrapped(ix, iy + 1, iz, fx, fy - 1, fz)
    g110 = grad_wrapped(ix + 1, iy + 1, iz, fx - 1, fy - 1, fz)
    g001 = grad_wrapped(ix, iy, iz + 1, fx, fy, fz - 1)
    g101 = grad_wrapped(ix + 1, iy, iz + 1, fx - 1, fy, fz - 1)
    g011 = grad_wrapped(ix, iy + 1, iz + 1, fx, fy - 1, fz - 1)
    g111 = grad_wrapped(ix + 1, iy + 1, iz + 1, fx - 1, fy - 1, fz - 1)

    nx00 = _lerp(g000, g100, fx)
    nx01 = _lerp(g010, g110, fx)
    nx10 = _lerp(g001, g101, fx)
    nx11 = _lerp(g011, g111, fx)

    nx0 = _lerp(nx00, nx01, fy)
    nx1 = _lerp(nx10, nx11, fy)

    return _lerp(nx0, nx1, fz)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def sample_noise(x: float, y: float, z: float, size: int, seamless: bool,
                 octaves: int = 4, base_freq: float = 1.0,
                 hash_period: int = None, noise_type: str = "Value Noise") -> float:
    """Dispatch to the appropriate noise sampler."""
    if noise_type == "Worley Noise":
        return sample_worley(x, y, z, size, seamless, hash_period)
    elif noise_type == "FBM Perlin Noise":
        return sample_perlin(x, y, z, size, seamless, hash_period)
    else:
        return sample_3d(x, y, z, size, seamless, octaves, base_freq, hash_period)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Volumetric Texture Generator")
        self.root.resizable(False, False)

        self.output_path = tk.StringVar(value=str(Path("volumetric_texture.png").resolve()))
        self.size_var = tk.StringVar(value="64")
        self.seamless_var = tk.BooleanVar(value=True)
        self.noise_type_var = tk.StringVar(value="Value Noise")
        self.octaves_var = tk.IntVar(value=4)
        self.seed_var = tk.IntVar(value=42)
        self.base_freq_var = tk.DoubleVar(value=1.0)
        self.progress = tk.DoubleVar(value=0.0)
        self.generating = False
        self._cancel_event = threading.Event()
        self.preview_image = tk.PhotoImage(width=256, height=256)

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        # --- Controls Frame ---
        ctrl_frame = ttk.Frame(main)
        ctrl_frame.pack(fill=tk.X, pady=(0, 12))

        # --- Row 1: Size ---
        ttk.Label(ctrl_frame, text="Size (LxLxL):").grid(row=0, column=0, sticky=tk.W, pady=4)
        size_frame = ttk.Frame(ctrl_frame)
        size_frame.grid(row=0, column=1, sticky=tk.EW, pady=4)
        ttk.Combobox(size_frame, textvariable=self.size_var, values=["16", "64", "256", "1024"], state="readonly", width=6).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # --- Row 2: Seed ---
        ttk.Label(ctrl_frame, text="Seed:").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Entry(ctrl_frame, textvariable=self.seed_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=4, padx=(8, 0))

        # --- Row 3: Base Frequency ---
        ttk.Label(ctrl_frame, text="Base Freq:").grid(row=2, column=0, sticky=tk.W, pady=4)
        freq_frame = ttk.Frame(ctrl_frame)
        freq_frame.grid(row=2, column=1, sticky=tk.EW, pady=4)
        ttk.Scale(freq_frame, from_=0.01, to=10.0, orient=tk.HORIZONTAL,
                  variable=self.base_freq_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        freq_entry = ttk.Entry(freq_frame, textvariable=self.base_freq_var, width=7)
        freq_entry.pack(side=tk.LEFT, padx=(6, 0))
        freq_entry.bind("<Return>", lambda e: self._validate_freq())

        # --- Row 4: Octaves ---
        ttk.Label(ctrl_frame, text="Octaves:").grid(row=3, column=0, sticky=tk.W, pady=4)
        oct_frame = ttk.Frame(ctrl_frame)
        oct_frame.grid(row=3, column=1, sticky=tk.EW, pady=4)
        ttk.Scale(oct_frame, from_=1, to=8, orient=tk.HORIZONTAL,
                  variable=self.octaves_var, command=lambda v: self.octaves_var.set(int(float(v)))).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(oct_frame, textvariable=self.octaves_var, width=4).pack(side=tk.LEFT, padx=(6, 0))

        # --- Row 5: Seamless ---
        ttk.Checkbutton(ctrl_frame, text="Seamless Tiling", variable=self.seamless_var).grid(
            row=4, column=1, sticky=tk.W, pady=4, padx=(8, 0))

        # --- Row 6: Noise Type ---
        ttk.Label(ctrl_frame, text="Noise Type:").grid(row=5, column=0, sticky=tk.W, pady=4)
        noise_frame = ttk.Frame(ctrl_frame)
        noise_frame.grid(row=5, column=1, sticky=tk.EW, pady=4, padx=(8, 0))
        ttk.Combobox(noise_frame, textvariable=self.noise_type_var, values=["Value Noise", "Worley Noise", "FBM Perlin Noise"], state="readonly", width=20).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # --- Row 7: Output Path ---
        ttk.Label(ctrl_frame, text="Output:").grid(row=7, column=0, sticky=tk.W, pady=4)
        out_frame = ttk.Frame(ctrl_frame)
        out_frame.grid(row=7, column=1, sticky=tk.EW, pady=4, padx=(8, 0))
        ttk.Entry(out_frame, textvariable=self.output_path, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(out_frame, text="Browse...", command=self._browse_output).pack(side=tk.LEFT, padx=(6, 0))

        # --- Row 8: Progress ---
        progress_frame = ttk.Frame(ctrl_frame)
        progress_frame.grid(row=8, column=0, columnspan=2, sticky=tk.EW, pady=8)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress, maximum=100)
        self.progress_bar.pack(fill=tk.X)
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.pack(fill=tk.X, pady=(2, 0))

        # --- Row 9: Buttons ---
        btn_frame = ttk.Frame(ctrl_frame)
        btn_frame.grid(row=9, column=0, columnspan=2, pady=10)
        self.btn_preview = ttk.Button(btn_frame, text="Preview", command=self._show_preview)
        self.btn_preview.pack(side=tk.LEFT, expand=True, padx=(0, 4))
        self.btn_render = ttk.Button(btn_frame, text="Render", command=self._render)
        self.btn_render.pack(side=tk.LEFT, expand=True, padx=(4, 0))
        self.btn_cancel = ttk.Button(btn_frame, text="Cancel", command=self._cancel_generation, state=tk.DISABLED)
        self.btn_cancel.pack(side=tk.LEFT, expand=True, padx=(4, 0))

        ctrl_frame.columnconfigure(1, weight=1)

        # --- Preview Frame ---
        preview_outer = ttk.LabelFrame(main, text="Preview", padding=6)
        preview_outer.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        self.preview_label = ttk.Label(preview_outer, image=self.preview_image, relief=tk.SUNKEN)
        self.preview_label.pack(fill=tk.BOTH, expand=True)

    def _validate_freq(self):
        try:
            v = float(self.base_freq_var.get())
            self.base_freq_var.set(max(0.01, min(10.0, v)))
        except ValueError:
            self.base_freq_var.set(0.1)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialfile=self.output_path.get()
        )
        if path:
            self.output_path.set(path)

    def _update_status(self, text: str, progress: float = None):
        self.status_label.config(text=text)
        if progress is not None:
            self.progress.set(progress)
            self.root.update_idletasks()

    def _cancel_generation(self):
        self._cancel_event.set()

    def _grid_to_photo(self, grid: list[list[int]]) -> None:
        """Convert a 2D grayscale grid to PhotoImage pixels."""
        data = []
        for row in grid:
            row_strs = [f"#{v:02x}{v:02x}{v:02x}" for v in row]
            data.append(" ".join(row_strs))
        self.preview_image.put(" ".join(data))

    def _show_preview(self):
        if self.generating:
            return
        self._cancel_event.clear()
        self.generating = True
        self._update_status("Generating preview...", 0)
        self._toggle_buttons()

        preview_size = 256
        seamless = self.seamless_var.get()
        octaves = self.octaves_var.get()
        seed = self.seed_var.get()
        base_freq = self.base_freq_var.get()

        global _hash_seed
        _hash_seed = seed

        def worker():
            try:
                self._update_status("Computing volume...", 10)
                volume = generate_volume(
                    preview_size, seamless, octaves=octaves,
                    base_freq=base_freq, noise_type=self.noise_type_var.get(),
                    cancel_event=self._cancel_event
                )

                if self._cancel_event.is_set():
                    self.root.after(0, self._generation_cancelled)
                    return

                self._update_status("Building grid...", 70)
                cols = math.ceil(math.sqrt(preview_size))
                rows = math.ceil(preview_size / cols)
                total_w = cols * preview_size
                total_h = rows * preview_size
                grid = [[0] * total_w for _ in range(total_h)]

                for s in range(preview_size):
                    col = s % cols
                    row = s // cols
                    for y in range(preview_size):
                        for x in range(preview_size):
                            val = int(round(volume[s][y][x] * 255))
                            grid[row * preview_size + y][col * preview_size + x] = max(0, min(255, val))

                    pct = 70 + (s / preview_size) * 25
                    self.root.after(0, self._update_status, f"Building grid... ({s+1}/{preview_size})", pct)

                self.root.after(0, self._preview_ready, grid)
            except Exception as e:
                self.root.after(0, self._generation_failed, str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _preview_ready(self, grid: list[list[int]]):
        self.generating = False
        self._grid_to_photo(grid)
        self._toggle_buttons()
        self._update_status("Preview ready", 100)

    def _render(self):
        if self.generating:
            return
        self._cancel_event.clear()
        self.generating = True
        self._update_status("Rendering...", 0)
        self._toggle_buttons()

        size = int(self.size_var.get())
        seamless = self.seamless_var.get()
        octaves = self.octaves_var.get()
        seed = self.seed_var.get()
        base_freq = self.base_freq_var.get()
        output = self.output_path.get()

        global _hash_seed
        _hash_seed = seed

        def worker():
            try:
                self._update_status("Computing volume...", 10)
                volume = generate_volume(
                    size, seamless, octaves=octaves,
                    base_freq=base_freq, noise_type=self.noise_type_var.get(),
                    cancel_event=self._cancel_event
                )

                if self._cancel_event.is_set():
                    self.root.after(0, self._generation_cancelled)
                    return

                self._update_status("Building grid...", 70)
                cols = math.ceil(math.sqrt(size))
                rows = math.ceil(size / cols)
                total_w = cols * size
                total_h = rows * size
                grid = [[0] * total_w for _ in range(total_h)]

                for s in range(size):
                    col = s % cols
                    row = s // cols
                    for y in range(size):
                        for x in range(size):
                            val = int(round(volume[s][y][x] * 255))
                            grid[row * size + y][col * size + x] = max(0, min(255, val))

                    pct = 70 + (s / size) * 25
                    self.root.after(0, self._update_status, f"Building grid... ({s+1}/{size})", pct)

                self._update_status("Writing PNG...", 97)
                png_bytes = write_png_bytes(grid, total_w, total_h)
                Path(output).write_bytes(png_bytes)

                self.root.after(0, self._generation_complete, output, size, cols, rows)
            except Exception as e:
                self.root.after(0, self._generation_failed, str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _toggle_buttons(self):
        state = tk.DISABLED if self.generating else tk.NORMAL
        self.btn_preview.config(state=state)
        self.btn_render.config(state=state)
        self.btn_cancel.config(state=tk.NORMAL if self.generating else tk.DISABLED)

    def _generation_cancelled(self):
        self.generating = False
        self._cancel_event.clear()
        self._toggle_buttons()
        self._update_status("Cancelled", 0)
        self._cancel_event.clear()

    def _generation_complete(self, output: str, size: int, cols: int, rows: int):
        self.generating = False
        self._update_status(f"Done! ({size}³ → {cols}x{rows} grid, {cols*size}x{rows*size} px)", 100)
        messagebox.showinfo("Complete", f"Generated {output}\n({size}x{size}x{size} volume, {cols}x{rows} grid)")

    def _generation_failed(self, error: str):
        self.generating = False
        self._update_status("Error", 0)
        messagebox.showerror("Error", f"Generation failed:\n{error}")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
