import os, sys

import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import tkinter as tk
import generate_volumetric_gui as G

OUT = "/tmp/vp_gui_smoke_render.png"
calls = []
G.messagebox.showinfo = lambda title, msg: calls.append(("info", msg))
G.messagebox.showerror = lambda title, msg: calls.append(("error", msg))

root = tk.Tk()
app = G.App(root)

fails = []
def check(label, cond):
    print(f"{'PASS' if cond else 'FAIL'}: {label}", flush=True)
    if not cond:
        fails.append(label)

def finish():
    root.destroy()
    print("RESULT:", "ALL PASS" if not fails else f"FAILURES: {fails}", flush=True)

def wait_then(next_fn, tries=0):
    if not app.generating:
        root.after(50, next_fn)
    elif tries > 900:
        check("generation timeout", False)
        finish()
    else:
        root.after(50, lambda: wait_then(next_fn, tries + 1))

# --- 1) Preview: Voronoi Jitter at 64^3 ---
def step1():
    app.size_var.set("64")
    app.noise_type_var.set("Voronoi Noise")
    root.after(20, lambda: (
        check("voronoi mode combo visible", app.voronoi_mode_combo.winfo_manager() == "pack"),
        app.voronoi_mode_var.set("Jitter"),
        app.base_freq_var.set(0.05),
        app._preview(),
        wait_then(step2),
    ))

def step2():
    check("preview ready status", "Preview ready" in app.status_label["text"])
    check("quantized slices cached", len(app._quantized_slices) == 64)
    # --- 2) Cancel mid-generation (slow params) ---
    app.noise_type_var.set("Worley Noise")
    app.base_freq_var.set(0.5)
    app._preview()
    root.after(400, step2b)

def step2b():
    check("cancel button enabled during gen", str(app.btn_cancel["state"]) == "normal")
    check("status advanced via ui queue", "Computing volume" in app.status_label["text"])
    app._cancel()
    wait_then(step2c)

def step2c():
    check("cancelled status", app.status_label["text"] == "Cancelled")
    check("cancel button disabled after", str(app.btn_cancel["state"]) == "disabled")
    # --- 3) Render to /tmp ---
    if os.path.exists(OUT):
        os.remove(OUT)
    app.output_path.set(OUT)
    app.noise_type_var.set("Value Noise")
    app.base_freq_var.set(0.05)
    app._render()
    wait_then(step3)

def step3():
    check("render done status", "Done!" in app.status_label["text"])
    check("output file written", os.path.exists(OUT) and os.path.getsize(OUT) > 100)
    check("completion dialog", bool(calls) and calls[-1][0] == "info")
    # --- 4) Slice slider ---
    app.slice_var.set(40)
    app._last_frame_size = (0, 0)
    app._update_2d_preview()
    root.after(20, step4)

def step4():
    check("slice photo rendered", app.preview_image.width() > 1 and app.preview_image.height() > 1)
    # --- 5) Invert ---
    app.invert_output.set(True)
    app._preview()
    wait_then(step5a)

def step5a():
    inv = app._quantized_slices[0][0][0]
    app.invert_output.set(False)
    app._preview()
    wait_then(lambda: step5b(inv))

def step5b(inv):
    norm = app._quantized_slices[0][0][0]
    check("invert quantization", inv == 255 - norm)
    # --- 6) Preset save/load round-trip ---
    import json
    p = app._current_preset()
    ppath = "/tmp/vp_preset_roundtrip.json"
    with open(ppath, "w") as f:
        f.write(json.dumps(p))
    app.noise_type_var.set("FBM Perlin Noise")
    app.size_var.set("16")
    app.seed_var.set(1)
    try:
        with open(ppath) as f:
            app._apply_preset(json.load(f))
        check("preset round-trip", app._current_preset() == p)
    except Exception as e:
        check(f"preset apply ({e})", False)
    try:
        app._apply_preset(dict(p, size=7, noise_type="nope", octaves=0))
        check("preset rejects invalid fields", False)
    except ValueError:
        check("preset rejects invalid fields", True)
    # --- 7) Upscaled cache growth ---
    for i in range(0, 64, 4):
        app.slice_var.set(i)
        app._last_frame_size = (0, 0)
        app._update_2d_preview()
    print(f"upscaled cache entries after 16 slices: {len(app._upscaled_cache)}", flush=True)
    finish()

root.after(300, step1)
root.mainloop()
