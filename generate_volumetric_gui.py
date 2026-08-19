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

def generate_volume(size: int, seamless: bool, octaves: int = 4,
                    base_freq: float = 1.0, noise_type: str = "Value Noise") -> list[list[list[float]]]:
    volume = [[[0.0] * size for _ in range(size)] for _ in range(size)]
    for z in range(size):
        for y in range(size):
            for x in range(size):
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

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        # --- Row 1: Size ---
        ttk.Label(main, text="Size (LxLxL):").grid(row=0, column=0, sticky=tk.W, pady=4)
        size_frame = ttk.Frame(main)
        size_frame.grid(row=0, column=1, sticky=tk.EW, pady=4)
        ttk.Combobox(size_frame, textvariable=self.size_var, values=["16", "64", "256", "1024"], state="readonly", width=6).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # --- Row 2: Seed ---
        ttk.Label(main, text="Seed:").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Entry(main, textvariable=self.seed_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=4, padx=(8, 0))

        # --- Row 3: Base Frequency ---
        ttk.Label(main, text="Base Freq:").grid(row=2, column=0, sticky=tk.W, pady=4)
        freq_frame = ttk.Frame(main)
        freq_frame.grid(row=2, column=1, sticky=tk.EW, pady=4)
        ttk.Scale(freq_frame, from_=0.01, to=1.0, orient=tk.HORIZONTAL,
                  variable=self.base_freq_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        freq_entry = ttk.Entry(freq_frame, textvariable=self.base_freq_var, width=6)
        freq_entry.pack(side=tk.LEFT, padx=(6, 0))
        freq_entry.bind("<Return>", lambda e: self._validate_freq())

        # --- Row 4: Octaves ---
        ttk.Label(main, text="Octaves:").grid(row=3, column=0, sticky=tk.W, pady=4)
        oct_frame = ttk.Frame(main)
        oct_frame.grid(row=3, column=1, sticky=tk.EW, pady=4)
        ttk.Scale(oct_frame, from_=1, to=8, orient=tk.HORIZONTAL,
                  variable=self.octaves_var, command=lambda v: self.octaves_var.set(int(float(v)))).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(oct_frame, textvariable=self.octaves_var, width=4).pack(side=tk.LEFT, padx=(6, 0))

        # --- Row 5: Seamless ---
        ttk.Checkbutton(main, text="Seamless Tiling", variable=self.seamless_var).grid(
            row=4, column=1, sticky=tk.W, pady=4, padx=(8, 0))

        # --- Row 6: Noise Type ---
        ttk.Label(main, text="Noise Type:").grid(row=5, column=0, sticky=tk.W, pady=4)
        noise_frame = ttk.Frame(main)
        noise_frame.grid(row=5, column=1, sticky=tk.EW, pady=4, padx=(8, 0))
        ttk.Combobox(noise_frame, textvariable=self.noise_type_var, values=["Value Noise"], state="readonly", width=20).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # --- Row 7: Output Path ---
        ttk.Label(main, text="Output:").grid(row=7, column=0, sticky=tk.W, pady=4)
        out_frame = ttk.Frame(main)
        out_frame.grid(row=7, column=1, sticky=tk.EW, pady=4, padx=(8, 0))
        ttk.Entry(out_frame, textvariable=self.output_path, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(out_frame, text="Browse...", command=self._browse_output).pack(side=tk.LEFT, padx=(6, 0))

        # --- Row 8: Progress ---
        progress_frame = ttk.Frame(main)
        progress_frame.grid(row=8, column=0, columnspan=2, sticky=tk.EW, pady=8)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress, maximum=100)
        self.progress_bar.pack(fill=tk.X)
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.pack(fill=tk.X, pady=(2, 0))

        # --- Row 9: Buttons ---
        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=9, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="Generate Preview (64³)", command=lambda: self._start_generation(64)).pack(side=tk.LEFT, expand=True, padx=(0, 4))
        ttk.Button(btn_frame, text="Generate Full", command=self._start_generation).pack(side=tk.LEFT, expand=True, padx=(4, 0))

        # --- Grid config ---
        main.columnconfigure(1, weight=1)

    def _validate_freq(self):
        try:
            v = float(self.base_freq_var.get())
            self.base_freq_var.set(max(0.01, min(1.0, v)))
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

    def _start_generation(self, override_size: int = None):
        if self.generating:
            return
        self.generating = True
        self._update_status("Generating...", 0)

        size = override_size if override_size is not None else int(self.size_var.get())
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
                volume = generate_volume(size, seamless, octaves=octaves, base_freq=base_freq, noise_type=self.noise_type_var.get())

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
