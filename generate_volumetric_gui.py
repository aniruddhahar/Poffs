#!/usr/bin/env python3
"""
Volumetric 3D Texture Generator - GUI

A thin Tkinter wrapper around generate_volumetric.py. All noise algorithms,
volume generation, grid layout, PNG writing, and upscaling are implemented
in generate_volumetric.py and imported here.

This GUI adds only:
- Tkinter UI controls (size, noise type, parameters, output path)
- Threading with cancel support
- Live preview rendering and file completion dialogs
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import queue
import threading
from pathlib import Path

# This GUI is a thin wrapper — all generation logic is imported from the CLI module.
from generate_volumetric import (
    _generate_volume,
    compute_grid_dims,
    volume_to_grid,
    upscale_grid,
    write_png,
    volume_to_flat,
    render_volume_3d,
)

try:
    from generate_volumetric_gpu import (
        generate_volume_gpu as _generate_volume_gpu,
        render_volume_3d_gpu as _render_volume_3d_gpu,
        OPENCL_AVAILABLE as _OPENCL_AVAILABLE,
    )
except ImportError:
    _generate_volume_gpu = None
    _render_volume_3d_gpu = None
    _OPENCL_AVAILABLE = False

# Import Voronoi modes from the main module
try:
    from generate_volumetric import VORONOI_MODES
except ImportError:
    VORONOI_MODES = ["F1"]

# Valid cube dimensions where output texture is always Power of Two.
# For cube size L, grid is sqrt(L) x sqrt(L), output = L * sqrt(L).
VALID_SIZES = [4, 16, 64, 256]

NOISE_TYPES = ["Value Noise", "Worley Noise", "FBM Perlin Noise", "Voronoi Noise"]

# Max cached upscaled preview grids (each entry is (L*scale)^2 ints).
MAX_UPSCALE_CACHE = 12

# Base frequency slider/input bounds.
MIN_BASE_FREQ = 0.001
MAX_BASE_FREQ = 2.0

# --- 3D preview (ray-march) constants ---
# Final ray-march buffer resolution (square). Coarse passes are smaller.
RENDER_IMAGE = 128
# Max cached fully-rendered 3D frames (each is RENDER_IMAGE^2 ints).
MAX_FRAME_CACHE = 12
# Density slider (0-100) maps to an opacity multiplier via / DENSITY_SCALE.
DENSITY_SCALE = 20.0
MIN_DENSITY = 0.0
MAX_DENSITY = 100.0
MIN_STEPS = 8
MAX_STEPS = 32


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Volumetric Texture Generator")
        self.root.resizable(True, True)
        # Keep the window at least large enough that both the Controls panel and
        # the preview (with its 2D/3D sliders) are fully drawn; shrinking below
        # this starves the preview panel and the sliders would not be mapped.
        self.root.minsize(540, 560)

        self.output_path = tk.StringVar(value=str(Path("volumetric_texture.png").resolve()))
        self.size_var = tk.StringVar(value="64")
        self.noise_type_var = tk.StringVar(value="Value Noise")
        self.base_freq_var = tk.DoubleVar(value=0.01)
        self.seed_var = tk.IntVar(value=42)
        self.octaves_var = tk.IntVar(value=4)
        self.lacunarity_var = tk.DoubleVar(value=2.0)
        self.progress = tk.DoubleVar(value=0.0)
        self.use_opencl = tk.BooleanVar(value=_OPENCL_AVAILABLE)
        self.invert_output = tk.BooleanVar(value=False)
        self.generating = False
        self._cancel_event = threading.Event()
        self.preview_image = tk.PhotoImage(width=1, height=1)
        self._cached_volume = None
        self._cached_voronoi_mode = None
        self._quantized_slices = {}
        self._upscaled_cache = {}
        self._last_frame_size = (0, 0)
        self._last_slice_index = -1
        self._ui_queue = queue.Queue()
        self._resize_after_id = None
        self._slice_after_id = None

        # --- 3D preview (ray-march) state ---
        self.preview_mode = tk.StringVar(value="2D")
        self.yaw_var = tk.DoubleVar(value=30.0)
        self.pitch_var = tk.DoubleVar(value=20.0)
        self.density_var = tk.DoubleVar(value=40.0)   # slider 0-100 -> /DENSITY_SCALE
        self.step_count_var = tk.IntVar(value=32)
        self._render_flat = None
        self._render_res = 0
        self._cache_token = 0
        self._frame_cache = {}
        self._upscaled_3d_cache = {}
        self._current_3d_grid = None
        self._current_3d_res = 0
        self._current_frame_key = None
        self._render3d_active = False
        self._render3d_cancel = threading.Event()
        self._render3d_token = 0
        self._render3d_after_id = None

        self._configure_fonts()
        self._build_ui()
        self._poll_ui()

    def _configure_fonts(self):
        """On X11/XWayland, Tk 9 renders the default Helvetica->Nimbus Sans L
        match aliased and ignores Xft resources (option_add has no effect);
        the 'fixed' generic family renders cleanly through Xft/freetype.
        'fixed' is a Tk generic family (Courier/Menlo on Windows/macOS, whose
        native fonts already antialias, so those platforms keep defaults)."""
        if self.root.tk.call("tk", "windowingsystem") != "x11":
            return
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont",
                     "TkCaptionFont", "TkSmallCaptionFont", "TkIconFont",
                     "TkTooltipFont"):
            try:
                self.root.tk.call("font", "configure", name, "-family", "fixed")
            except tk.TclError:
                pass

    def _post_ui(self, callback, *args):
        """Queue a callback to run on the main thread (worker threads must
        never touch Tkinter directly: after() from a non-main thread is
        silently dropped on thread-aware Tcl builds)."""
        self._ui_queue.put((callback, args))

    def _poll_ui(self):
        try:
            while True:
                callback, args = self._ui_queue.get_nowait()
                callback(*args)
        except queue.Empty:
            pass
        self.root.after(50, self._poll_ui)

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

        # Trace noise type changes to show/hide Voronoi mode selector
        self.noise_type_var.trace_add('write', self._on_noise_type_change)

        # Voronoi Mode (shown when Voronoi Noise is selected)
        self.voronoi_mode_var = tk.StringVar(value="F1")
        self.voronoi_mode_label = ttk.Label(ctrl_frame, text="Voronoi Mode:")
        self.voronoi_mode_label.pack_forget()
        self.voronoi_mode_combo = ttk.Combobox(ctrl_frame, textvariable=self.voronoi_mode_var,
                                                values=VORONOI_MODES, state="readonly", width=12)
        self.voronoi_mode_combo.pack_forget()

        # Base Frequency
        ttk.Label(ctrl_frame, text="Base Freq:").pack(anchor=tk.W, pady=(4, 2))
        freq_frame = ttk.Frame(ctrl_frame)
        freq_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Scale(freq_frame, from_=MIN_BASE_FREQ, to=MAX_BASE_FREQ,
                  variable=self.base_freq_var, orient=tk.HORIZONTAL).pack(fill=tk.X, expand=True, side=tk.LEFT)
        freq_entry = ttk.Entry(freq_frame, textvariable=self.base_freq_var, width=7)
        freq_entry.pack(side=tk.LEFT, padx=(6, 0))
        freq_entry.bind("<FocusOut>", self._clamp_base_freq)

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
        ttk.Scale(lac_frame, from_=0, to=10,
                  variable=self.lacunarity_var, orient=tk.HORIZONTAL,
                  command=lambda v: self.lacunarity_var.set(float(round(float(v))))).pack(fill=tk.X, expand=True)
        ttk.Label(lac_frame, textvariable=self.lacunarity_var, width=6).pack(side=tk.LEFT, padx=(6, 0))

        # OpenCL Toggle
        if _OPENCL_AVAILABLE:
            ttk.Checkbutton(ctrl_frame, text="Use OpenCL (GPU)", variable=self.use_opencl).pack(anchor=tk.W, pady=(8, 0))
        else:
            ttk.Label(ctrl_frame, text="OpenCL: not available", foreground="gray").pack(anchor=tk.W, pady=(8, 0))

        # Invert Output
        ttk.Checkbutton(ctrl_frame, text="Invert Output", variable=self.invert_output,
                        command=self._on_invert_toggle).pack(anchor=tk.W, pady=(8, 0))

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
        self.btn_render.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        self.btn_cancel = ttk.Button(btn_frame, text="Cancel", command=self._cancel, state=tk.DISABLED)
        self.btn_cancel.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Preset buttons (below Preview/Render/Cancel)
        preset_frame = ttk.Frame(ctrl_frame)
        preset_frame.pack(fill=tk.X, pady=(4, 0))
        self.btn_save_preset = ttk.Button(preset_frame, text="Save Preset", command=self._save_preset)
        self.btn_save_preset.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        self.btn_load_preset = ttk.Button(preset_frame, text="Load Preset", command=self._load_preset)
        self.btn_load_preset.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # --- Right: Preview ---
        self.preview_frame = ttk.LabelFrame(main, text="Preview", padding=8)
        self.preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.preview_frame.bind('<Configure>', self._on_preview_resize)

        # View mode toggle (top of preview panel)
        self.mode_frame = ttk.Frame(self.preview_frame)
        self.mode_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        ttk.Label(self.mode_frame, text="View:").pack(side=tk.LEFT)
        ttk.Radiobutton(self.mode_frame, text="2D Slices", variable=self.preview_mode,
                        value="2D", command=self._on_mode_change).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Radiobutton(self.mode_frame, text="3D Volume", variable=self.preview_mode,
                        value="3D", command=self._on_mode_change).pack(side=tk.LEFT, padx=(8, 0))

        # Preview image (created here, packed last so the bottom controls are
        # always reserved — see _set_3d_controls_visible).
        self.preview_label = ttk.Label(self.preview_frame, image=self.preview_image, relief=tk.SUNKEN)

        # 2D Slice slider (fixed at bottom)
        self.slice_slider_frame = ttk.Frame(self.preview_frame)
        self.slice_slider_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.slice_label = ttk.Label(self.slice_slider_frame, text="Slice: 0 / 0")
        self.slice_label.pack(side=tk.LEFT)
        self.slice_var = tk.IntVar(value=0)
        self.slice_slider = ttk.Scale(self.slice_slider_frame, from_=0, to=0,
                                       variable=self.slice_var, orient=tk.HORIZONTAL)
        self.slice_slider.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(4, 0))
        self.slice_slider.bind('<B1-Motion>', self._on_slice_slider_change)

        # 3D ray-march controls (bottom; hidden while in 2D mode). One control
        # per row so they stay narrow and map even when the preview panel is thin.
        self.render3d_frame = ttk.Frame(self.preview_frame)
        self.yaw_scale = self._make_3d_scale(self.render3d_frame, "Yaw", self.yaw_var, 0, 360)
        self.pitch_scale = self._make_3d_scale(self.render3d_frame, "Pitch", self.pitch_var, 0, 360)
        self.density_scale = self._make_3d_scale(self.render3d_frame, "Density", self.density_var,
                                                 MIN_DENSITY, MAX_DENSITY)
        self.step_scale = self._make_3d_scale(
            self.render3d_frame, "Steps", self.step_count_var, MIN_STEPS, MAX_STEPS,
            cmd=lambda v: self.step_count_var.set(int(round(float(v)))))
        self._set_3d_controls_visible(False)

        # Initialize Voronoi mode selector visibility based on initial noise type
        self._on_noise_type_change()

    def _make_3d_scale(self, parent, label, var, lo, hi, cmd=None):
        """Build a labelled horizontal scale for the 3D controls (stacked one per
        row); any change schedules a debounced re-render."""
        f = ttk.Frame(parent)
        f.pack(side=tk.TOP, fill=tk.X, pady=(1, 0))
        ttk.Label(f, text=label).pack(side=tk.LEFT)
        s = ttk.Scale(f, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL, command=cmd)
        s.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        ttk.Label(f, textvariable=var, width=5).pack(side=tk.LEFT, padx=(4, 0))
        for seq in ('<B1-Motion>', '<ButtonRelease-1>'):
            s.bind(seq, self._on_3d_param_change)
        return s

    def _set_3d_controls_visible(self, visible: bool):
        # Reserve the bottom control frame first (side=BOTTOM), then (re)pack the
        # expanding image last so it only takes leftover space. Otherwise a large
        # PhotoImage makes the Label claim the whole panel and the sliders below
        # get clipped when the window is short.
        if visible:
            self.slice_slider_frame.pack_forget()
            self.render3d_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(4, 0))
        else:
            self.render3d_frame.pack_forget()
            self.slice_slider_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.preview_label.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

    def _on_noise_type_change(self, *args):
        """Show/hide Voronoi mode selector based on noise type."""
        if self.noise_type_var.get() == "Voronoi Noise":
            self.voronoi_mode_label.pack(anchor=tk.W, pady=(4, 2))
            self.voronoi_mode_combo.pack(fill=tk.X, pady=(0, 8))
        else:
            self.voronoi_mode_label.pack_forget()
            self.voronoi_mode_combo.pack_forget()

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

    def _clamp_base_freq(self, *_args):
        try:
            value = float(self.base_freq_var.get())
        except (tk.TclError, ValueError):
            value = MIN_BASE_FREQ
        value = max(MIN_BASE_FREQ, min(MAX_BASE_FREQ, value))
        self.base_freq_var.set(value)

    def _update_status(self, text: str, progress: float = None):
        self.status_label.config(text=text)
        if progress is not None:
            self.progress.set(progress)
            self.root.update_idletasks()

    def _toggle_buttons(self):
        state = tk.DISABLED if self.generating else tk.NORMAL
        self.btn_preview.config(state=state)
        self.btn_render.config(state=state)
        cancel_state = tk.NORMAL if (self.generating or self._render3d_active) else tk.DISABLED
        self.btn_cancel.config(state=cancel_state)
        self.btn_save_preset.config(state=state)
        self.btn_load_preset.config(state=state)

    def _current_preset(self) -> dict:
        """Snapshot of all generation parameters (raises on invalid entries)."""
        return {
            "app": "volumetric-texture-generator",
            "version": 1,
            "size": int(self.size_var.get()),
            "noise_type": self.noise_type_var.get(),
            "voronoi_mode": self.voronoi_mode_var.get(),
            "base_freq": float(self.base_freq_var.get()),
            "seed": int(self.seed_var.get()),
            "octaves": int(self.octaves_var.get()),
            "lacunarity": float(self.lacunarity_var.get()),
            "invert": bool(self.invert_output.get()),
            "use_opencl": bool(self.use_opencl.get()),
            "output": self.output_path.get(),
        }

    def _apply_preset(self, data: dict):
        """Validate and apply a preset dict; raises ValueError on bad fields."""
        size = int(data.get("size", self.size_var.get()))
        if size not in VALID_SIZES:
            raise ValueError(f"size must be one of {VALID_SIZES}, got {size}")
        noise_type = str(data.get("noise_type", self.noise_type_var.get()))
        if noise_type not in NOISE_TYPES:
            raise ValueError(f"noise_type must be one of {NOISE_TYPES}, got {noise_type!r}")
        voronoi_mode = str(data.get("voronoi_mode", self.voronoi_mode_var.get()))
        if voronoi_mode not in VORONOI_MODES:
            raise ValueError(f"voronoi_mode must be one of {VORONOI_MODES}, got {voronoi_mode!r}")
        base_freq = max(MIN_BASE_FREQ, min(MAX_BASE_FREQ,
                                           float(data.get("base_freq", self.base_freq_var.get()))))
        seed = int(data.get("seed", self.seed_var.get()))
        octaves = int(data.get("octaves", self.octaves_var.get()))
        if octaves < 1:
            raise ValueError(f"octaves must be >= 1, got {octaves}")
        lacunarity = max(0.0, min(10.0, float(data.get("lacunarity", self.lacunarity_var.get()))))

        self.size_var.set(str(size))
        self.noise_type_var.set(noise_type)
        self.voronoi_mode_var.set(voronoi_mode)
        self.base_freq_var.set(base_freq)
        self.seed_var.set(seed)
        self.octaves_var.set(octaves)
        self.lacunarity_var.set(lacunarity)
        self.invert_output.set(bool(data.get("invert", self.invert_output.get())))
        if _OPENCL_AVAILABLE:
            self.use_opencl.set(bool(data.get("use_opencl", self.use_opencl.get())))
        output = data.get("output")
        if isinstance(output, str) and output:
            self.output_path.set(output)

    def _save_preset(self):
        try:
            preset = self._current_preset()
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid values",
                                 "Fix invalid parameter values before saving a preset.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Preset", defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            Path(path).write_text(json.dumps(preset, indent=2) + "\n")
        except OSError as e:
            messagebox.showerror("Save failed", str(e))
            return
        self._update_status(f"Preset saved: {Path(path).name}")

    def _load_preset(self):
        path = filedialog.askopenfilename(
            title="Load Preset",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, ValueError) as e:
            messagebox.showerror("Load failed", f"Could not read preset:\n{e}")
            return
        if not isinstance(data, dict):
            messagebox.showerror("Load failed", "Preset must be a JSON object.")
            return
        try:
            self._apply_preset(data)
        except (ValueError, TypeError) as e:
            messagebox.showerror("Invalid preset", str(e))
            return
        self._update_status(f"Preset loaded: {Path(path).name}")

    def _cancel(self):
        if self.generating:
            self._cancel_event.set()
            self.btn_cancel.config(state=tk.DISABLED)
            self._update_status("Cancelling...")
        elif self._render3d_active:
            self._render3d_cancel.set()
            self.btn_cancel.config(state=tk.DISABLED)
            self._update_status("Cancelling 3D render...")

    def _build_png_data(self, grid: list[list[int]]) -> str:
        """Build PhotoImage data string efficiently using precomputed hex lookup."""
        if not hasattr(self, '_hex_lookup'):
            self._hex_lookup = [f'#{v:02x}{v:02x}{v:02x}' for v in range(256)]
        data = []
        for row in grid:
            data.append('{' + ' '.join(self._hex_lookup[v] for v in row) + '}')
        return ' '.join(data)

    def _slice_to_photo(self, slice_data: list[list[int]], size: int, target_w: int, target_h: int,
                        slice_index: int = 0) -> None:
        """Convert a single Z-slice to a PhotoImage at appropriate scale."""
        target = min(target_w, target_h)
        scale = max(1, target // size)
        display_w = size * scale
        display_h = size * scale

        # Cache upscaled grid by (slice, scale) tuple, with a size cap so
        # large volumes (L=256) cannot grow the cache unbounded
        cache_key = (slice_index, scale)
        if cache_key not in self._upscaled_cache:
            if len(self._upscaled_cache) >= MAX_UPSCALE_CACHE:
                self._upscaled_cache.pop(next(iter(self._upscaled_cache)))
            self._upscaled_cache[cache_key] = upscale_grid(slice_data, scale)
        display_grid = self._upscaled_cache[cache_key]

        # Only recreate PhotoImage if dimensions changed
        cur_w = self.preview_image.width()
        cur_h = self.preview_image.height()
        if cur_w != display_w or cur_h != display_h:
            self.preview_image = tk.PhotoImage(width=display_w, height=display_h)
            self.preview_label.config(image=self.preview_image)

        data = self._build_png_data(display_grid)
        self.preview_image.put(data)

    def _get_volume_generator(self):
        """Return the appropriate volume generation function based on OpenCL setting."""
        if self.use_opencl.get() and _generate_volume_gpu is not None:
            return _generate_volume_gpu
        return _generate_volume

    def _update_2d_preview(self):
        """Update the 2D preview with the current slice from cached volume."""
        if self.preview_mode.get() == "3D":
            return
        if self._cached_volume is None:
            return
        volume = self._cached_volume
        size = len(volume)
        if size == 0:
            return
        
        # Get available display area
        frame_w = self.preview_frame.winfo_width()
        frame_h = self.preview_frame.winfo_height()
        if frame_w < 2 or frame_h < 2:
            return
        
        current_slice = self.slice_var.get()
        
        # Skip only if nothing changed (frame size and slice both same)
        if (frame_w, frame_h) == self._last_frame_size and current_slice == self._last_slice_index:
            return
        
        # Clear upscaled cache when frame size changes (scale may change)
        if (frame_w, frame_h) != self._last_frame_size:
            self._upscaled_cache.clear()
        
        self._last_frame_size = (frame_w, frame_h)
        self._last_slice_index = current_slice
        
        if 0 <= current_slice < size:
            slice_data = self._quantized_slices.get(current_slice)
            if slice_data is not None:
                self._slice_to_photo(slice_data, size, frame_w, frame_h, current_slice)
                self.slice_label.config(text=f"Slice: {current_slice} / {size}")

    def _on_preview_resize(self, event=None):
        """Regenerate the preview image when the preview frame is resized (debounced)."""
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        target = self._blit_3d if self.preview_mode.get() == "3D" else self._update_2d_preview
        self._resize_after_id = self.root.after(50, target)

    def _on_slice_slider_change(self, event=None):
        """Handle slider movement to show different slices (debounced)."""
        if self._slice_after_id is not None:
            self.root.after_cancel(self._slice_after_id)
        self._slice_after_id = self.root.after(30, self._update_2d_preview)

    # --- 3D preview (ray-march) ---

    def _on_mode_change(self):
        """Switch between the 2D slice view and the 3D ray-marched view."""
        mode = self.preview_mode.get()
        self._set_3d_controls_visible(mode == "3D")
        if mode == "3D":
            if self._render_flat is not None:
                self._request_3d_render()
            else:
                self._blit_3d()
        else:
            # Leaving 3D: abandon any in-flight render so it cannot clobber the 2D view.
            self._render3d_cancel.set()
            self._render3d_token += 1
            self._render3d_active = False
            self._toggle_buttons()
            self._last_frame_size = (0, 0)
            self._update_2d_preview()

    def _on_3d_param_change(self, *_args):
        """Debounce 3D control changes into a single re-render."""
        if self._render3d_after_id is not None:
            self.root.after_cancel(self._render3d_after_id)
        self._render3d_after_id = self.root.after(30, self._request_3d_render)

    def _on_invert_toggle(self):
        """Invert is applied at blit time in 3D mode (no re-march needed)."""
        if self.preview_mode.get() == "3D":
            self._blit_3d()

    def _get_3d_renderer(self):
        if self.use_opencl.get() and _render_volume_3d_gpu is not None:
            return _render_volume_3d_gpu
        return render_volume_3d

    def _progressive_passes(self, step_count: int):
        """Coarse -> fine (image_size, steps) passes for progressive refinement."""
        step = max(MIN_STEPS, min(MAX_STEPS, int(step_count)))
        return [
            (48, max(MIN_STEPS, step // 4)),
            (96, max(12, step // 2)),
            (RENDER_IMAGE, step),
        ]

    def _request_3d_render(self):
        """Cancel any in-flight frame and start a single-slot progressive render."""
        if self._render_flat is None or self.preview_mode.get() != "3D":
            return
        yaw = float(self.yaw_var.get())
        pitch = float(self.pitch_var.get())
        density = float(self.density_var.get()) / DENSITY_SCALE
        step = int(self.step_count_var.get())
        key = (self._cache_token, round(yaw, 1), round(pitch, 1), round(density, 3), step)

        cached = self._frame_cache.get(key)
        if cached is not None:
            self._current_3d_grid = cached
            self._current_3d_res = len(cached)
            self._current_frame_key = key
            self._blit_3d()
            return

        # Supersede any in-flight frame; its results are discarded by token check.
        self._render3d_cancel.set()
        self._render3d_token += 1
        token = self._render3d_token
        cancel = threading.Event()
        self._render3d_cancel = cancel
        self._render3d_active = True
        self._toggle_buttons()
        self._update_status("Rendering 3D...", 0)

        def worker():
            try:
                renderer = self._get_3d_renderer()
                grid = None
                passes = self._progressive_passes(step)
                for i, (isize, isteps) in enumerate(passes):
                    if cancel.is_set():
                        self._post_ui(self._3d_frame_cancelled, token)
                        return
                    grid = renderer(self._render_flat, self._render_res, yaw, pitch,
                                    density, isteps, isize, cancel_event=cancel)
                    if grid is None:
                        self._post_ui(self._3d_frame_cancelled, token)
                        return
                    final = (i == len(passes) - 1)
                    frac = 100.0 * (i + 1) / len(passes)
                    self._post_ui(self._3d_frame_ready, grid, isize, token, key, final, frac)
            except Exception as e:
                self._post_ui(self._3d_frame_failed, str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _3d_frame_ready(self, grid, isize, token, key, final, frac):
        """Main-thread handler for a progressive frame (only the latest render wins)."""
        if token != self._render3d_token:
            return
        self._current_3d_grid = grid
        self._current_3d_res = isize
        self._current_frame_key = key
        if final:
            self._frame_cache[key] = grid
            while len(self._frame_cache) > MAX_FRAME_CACHE:
                self._frame_cache.pop(next(iter(self._frame_cache)))
            self._render3d_active = False
            self._toggle_buttons()
            self._update_status("3D preview ready", 100)
        else:
            self._update_status("Rendering 3D...", frac)
        self._blit_3d()

    def _3d_frame_cancelled(self, token):
        if token != self._render3d_token:
            return
        self._render3d_active = False
        self._toggle_buttons()
        self._update_status("3D render cancelled", 0)

    def _3d_frame_failed(self, error):
        self._render3d_active = False
        self._toggle_buttons()
        self._update_status("3D render error", 0)
        messagebox.showerror("Error", f"3D render failed:\n{error}")

    def _blit_3d(self):
        """Composite the current 3D frame into the preview PhotoImage at display
        size (nearest upscale + integer scale, invert applied here)."""
        if self.preview_mode.get() != "3D":
            return
        grid = self._current_3d_grid
        if grid is None:
            return
        res = self._current_3d_res
        frame_w = self.preview_frame.winfo_width()
        frame_h = self.preview_frame.winfo_height()
        if frame_w < 2 or frame_h < 2 or res < 1:
            return
        scale = max(1, min(frame_w, frame_h) // res)
        invert = self.invert_output.get()
        ckey = (self._current_frame_key, scale, invert)
        disp = self._upscaled_3d_cache.get(ckey)
        if disp is None:
            while len(self._upscaled_3d_cache) >= MAX_UPSCALE_CACHE:
                self._upscaled_3d_cache.pop(next(iter(self._upscaled_3d_cache)))
            disp = upscale_grid(grid, scale) if scale > 1 else [row[:] for row in grid]
            if invert:
                disp = [[255 - v for v in row] for row in disp]
            self._upscaled_3d_cache[ckey] = disp
        disp_w = len(disp[0])
        disp_h = len(disp)
        if self.preview_image.width() != disp_w or self.preview_image.height() != disp_h:
            self.preview_image = tk.PhotoImage(width=disp_w, height=disp_h)
            self.preview_label.config(image=self.preview_image)
        self.preview_image.put(self._build_png_data(disp))

    def _preview(self):
        if self.generating:
            return
        # Supersede any in-flight 3D render (avoids OpenCL queue contention and
        # stale frames while the volume is being regenerated).
        self._render3d_cancel.set()
        self._render3d_token += 1
        self._render3d_active = False
        # Read all Tk variables on the main thread; worker threads must not
        # touch Tkinter objects.
        params = dict(
            size=int(self.size_var.get()),
            octaves=self.octaves_var.get(),
            base_freq=self.base_freq_var.get(),
            lacunarity=self.lacunarity_var.get(),
            seed=self.seed_var.get(),
            noise_type=self.noise_type_var.get(),
            voronoi_mode=self.voronoi_mode_var.get() if self.noise_type_var.get() == "Voronoi Noise" else "F1",
        )
        self._cancel_event.clear()
        self.generating = True
        self._toggle_buttons()
        self._update_status("Generating preview...", 0)

        def worker():
            try:
                gen = self._get_volume_generator()
                self._post_ui(self._update_status, "Computing volume...", 0)
                volume = gen(
                    params["size"], params["octaves"], params["base_freq"],
                    params["lacunarity"], params["seed"], params["noise_type"],
                    self._cancel_event, params["voronoi_mode"],
                    progress_cb=lambda f: self._post_ui(
                        self._update_status, "Computing volume...", 90.0 * f),
                )
                if self._cancel_event.is_set():
                    self._post_ui(self._generation_cancelled)
                    return

                # Build the downsampled flat copy for the 3D ray-march off the
                # main thread, then cache everything on the main thread.
                flat, rres = volume_to_flat(volume)
                # Cache the full-size volume for the 2D slice slider
                self._post_ui(self._cache_volume, volume, params["voronoi_mode"], flat, rres)
                self._post_ui(self._update_2d_preview)
            except Exception as e:
                self._post_ui(self._generation_failed, str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _cache_volume(self, volume: list, voronoi_mode: str, flat=None, rres: int = 0):
        """Cache the generated volume, pre-compute quantized slices, and configure the slice slider."""
        self._cached_volume = volume
        self._cached_voronoi_mode = voronoi_mode
        size = len(volume)
        
        # Pre-compute quantized slices (8-bit greyscale) for all Z-slices
        self._quantized_slices = {}
        for i in range(size):
            self._quantized_slices[i] = [
                [255 - max(0, min(255, int(round(v * 255)))) if self.invert_output.get()
                 else max(0, min(255, int(round(v * 255))))
                 for v in row]
                for row in volume[i]
            ]
        
        # Clear upscaling cache
        self._upscaled_cache.clear()
        self._last_frame_size = (0, 0)

        # 3D ray-march: store the flat (downsampled) volume and reset frame caches.
        self._render_flat = flat
        self._render_res = rres
        self._cache_token += 1
        self._frame_cache.clear()
        self._upscaled_3d_cache.clear()
        self._current_3d_grid = None
        self._current_frame_key = None
        
        # Configure slider range
        self.slice_slider.config(from_=0, to=size - 1)
        self.slice_var.set(0)
        
        self.generating = False
        self._toggle_buttons()
        self._update_status("Preview ready", 100)

        if self.preview_mode.get() == "3D":
            self._request_3d_render()

    def _render(self):
        if self.generating:
            return
        # Supersede any in-flight 3D render before starting a full render.
        self._render3d_cancel.set()
        self._render3d_token += 1
        self._render3d_active = False
        params = dict(
            size=int(self.size_var.get()),
            octaves=self.octaves_var.get(),
            base_freq=self.base_freq_var.get(),
            lacunarity=self.lacunarity_var.get(),
            seed=self.seed_var.get(),
            noise_type=self.noise_type_var.get(),
            voronoi_mode=self.voronoi_mode_var.get() if self.noise_type_var.get() == "Voronoi Noise" else "F1",
            output=self.output_path.get(),
            invert=self.invert_output.get(),
        )
        self._cancel_event.clear()
        self.generating = True
        self._toggle_buttons()
        self._update_status("Rendering...", 0)

        def worker():
            try:
                gen = self._get_volume_generator()
                self._post_ui(self._update_status, "Computing volume...", 0)
                volume = gen(
                    params["size"], params["octaves"], params["base_freq"],
                    params["lacunarity"], params["seed"], params["noise_type"],
                    self._cancel_event, params["voronoi_mode"],
                    progress_cb=lambda f: self._post_ui(
                        self._update_status, "Computing volume...", 60.0 * f),
                )
                if self._cancel_event.is_set():
                    self._post_ui(self._generation_cancelled)
                    return

                self._post_ui(self._update_status, "Building grid...", 70)
                cols, rows = compute_grid_dims(params["size"])
                grid = volume_to_grid(volume, params["size"], cols, rows)

                if params["invert"]:
                    grid = [[255 - pixel for pixel in row] for row in grid]

                self._post_ui(self._update_status, "Writing PNG...", 95)
                total_w = cols * params["size"]
                total_h = rows * params["size"]
                write_png(params["output"], grid, total_w, total_h)

                self._post_ui(self._generation_complete, params["output"], params["size"], cols, rows)
            except Exception as e:
                self._post_ui(self._generation_failed, str(e))

        threading.Thread(target=worker, daemon=True).start()

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
