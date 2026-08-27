#!/usr/bin/env python3
"""
Volumetric 3D Texture Generator - GUI
Generates LxLxL volumetric textures as greyscale PNG grids.

Imports generation logic from generate_volumetric.py.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import math
import threading
from pathlib import Path

# Import all generation logic from the CLI module
from generate_volumetric import (
    _generate_volume,
    compute_grid_dims,
    volume_to_grid,
    upscale_grid,
    write_png,
)

# Valid cube dimensions where output texture is always Power of Two.
# For cube size L, grid is sqrt(L) x sqrt(L), output = L * sqrt(L).
VALID_SIZES = [4, 16, 64, 256]

NOISE_TYPES = ["Value Noise", "Worley Noise", "FBM Perlin Noise"]
DEFAULT_PREVIEW_SIZE = 16


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
        self.base_freq_var = tk.DoubleVar(value=0.01)
        self.seed_var = tk.IntVar(value=42)
        self.octaves_var = tk.IntVar(value=4)
        self.lacunarity_var = tk.DoubleVar(value=2.0)
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

        def worker():
            try:
                seed = self.seed_var.get()
                size = int(self.size_var.get())
                octaves = self.octaves_var.get()
                base_freq = self.base_freq_var.get()
                lacunarity = self.lacunarity_var.get()
                noise_type = self.noise_type_var.get()

                self.root.after(0, self._update_status, "Computing volume...", 20)
                volume = _generate_volume(
                    size, octaves, base_freq, lacunarity,
                    seed, noise_type, self._cancel_event
                )
                if self._cancel_event.is_set():
                    self.root.after(0, self._generation_cancelled)
                    return

                self.root.after(0, self._update_status, "Building grid...", 60)
                cols = math.ceil(math.sqrt(size))
                rows = math.ceil(size / cols)
                grid = volume_to_grid(volume, size, cols, rows)

                # Upscale for display
                target = 256
                scale = max(1, target // (cols * size))
                display_grid = upscale_grid(grid, scale)

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

        def worker():
            try:
                seed = self.seed_var.get()
                size = int(self.size_var.get())
                octaves = self.octaves_var.get()
                base_freq = self.base_freq_var.get()
                lacunarity = self.lacunarity_var.get()
                output = self.output_path.get()
                noise_type = self.noise_type_var.get()

                self.root.after(0, self._update_status, "Computing volume...", 10)
                volume = _generate_volume(
                    size, octaves, base_freq, lacunarity,
                    seed, noise_type, self._cancel_event
                )
                if self._cancel_event.is_set():
                    self.root.after(0, self._generation_cancelled)
                    return

                self.root.after(0, self._update_status, "Building grid...", 70)
                cols = math.ceil(math.sqrt(size))
                rows = math.ceil(size / cols)
                grid = volume_to_grid(volume, size, cols, rows)

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
