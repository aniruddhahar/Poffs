# GUI Guide

## Launch

```bash
python3 generate_volumetric_gui.py
```

## Layout

- **Left panel:** Controls (Reference Size, Noise Type, parameters, output path, buttons)
- **Right panel:** Live preview of the generated texture

## Controls

### Reference Size (L)

Dropdown with fixed options: `4`, `16`, `64`, `256`.

These sizes are chosen so that `L × √L` is always a power of two, ensuring the output PNG dimensions are always powers of two.

### Noise Type

Dropdown with three options:
- **Value Noise** — Fast, simple, smooth patterns
- **Worley Noise** — Cellular, organic, cell-like patterns
- **FBM Perlin Noise** — Rich, natural multi-octave detail

### Base Frequency (Base Freq)

Controls the base frequency of the noise. Range: 0–100 (via slider), adjustable via text input.

Lower values produce larger-scale features; higher values produce finer detail.

### Seed

Deterministic seed for reproducible results.

- Type a specific integer for consistent output
- Click **Randomize** to generate a random seed (0 to 2³¹−1)

### Octaves

Number of FBM octaves (detail levels). More octaves add finer detail at higher frequencies.

### Lacunarity

Frequency multiplier between octaves. Range: 0–2 (via slider). Default: 2.0.

Higher values increase the frequency jump between octaves, producing coarser detail at higher levels.

### Output

File path for the saved PNG.

- Type a path directly, or click **Browse...** to open the file picker
- Default: `volumetric_texture.png` in the current directory

### Buttons

| Button | Action |
|--------|--------|
| **Preview** | Generates a low-res preview for quick iteration (see Workflow below) |
| **Render** | Generates the full volume and writes it to the output PNG file |

### Progress Bar & Status

- Displays generation progress as a percentage
- Shows status text: `Ready`, `Computing volume...`, `Building grid...`, `Writing PNG...`, `Preview ready`, `Done!`, `Cancelled`, or `Error`
- Buttons are disabled during generation

## Workflow

### Preview

1. Click **Preview**
2. A 16³ volume is generated regardless of the selected Reference Size
3. The result is upscaled to fit a 256px display area
4. No file is written — useful for rapid parameter exploration
5. The preview updates in the right panel when complete

### Render

1. Set parameters (Size, Noise Type, Base Freq, Seed, Octaves, Lacunarity)
2. Set output path (or use default)
3. Click **Render**
4. The full L³ volume is generated at the selected Reference Size
5. The PNG is written to the output path
6. A completion dialog appears with generation stats (output path, volume size, grid dimensions, resolution)

## Threaded Generation

Generation runs in a background thread. The UI remains responsive during generation. The cancel event is supported via the `_cancel_event` threading mechanism (generation checks for cancellation between voxels).

## Limitations

- Requires a display server (won't work in headless environments)
- Large sizes (256³) may be slow and consume significant memory
- GUI uses Tkinter (may have rendering artifacts on some systems)
