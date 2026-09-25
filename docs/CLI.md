# CLI Reference

## Usage

```bash
python3 generate_volumetric.py [OPTIONS]
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--size`, `-s` | int | 64 | Cube dimension L (volume is L×L×L) |
| `--output`, `-o` | str | `volumetric_texture.png` | Output PNG file path |
| `--octaves` | int | 4 | Number of FBM octaves for detail |
| `--seed` | int | 42 | Random seed for reproducible results |
| `--base-freq` | float | 0.01 | Base noise frequency |
| `--lacunarity` | float | 2.0 | Frequency multiplier between octaves |
| `--noise-type` | str | `value` | Noise algorithm: `value`, `worley`, `perlin`, or `voronoi` |
| `--voronoi-mode` | str | `F1` | Voronoi output mode (only with `--noise-type voronoi`): `F1`, `F2`, `F1 - F2`, `Jitter`, `Edge` |

## Noise Type Codes

| Code | Internal Name | Description |
|------|---------------|-------------|
| `value` | Value Noise | Smooth interpolation of random values at grid points |
| `worley` | Worley Noise | Cellular noise based on distance to nearest feature point |
| `perlin` | FBM Perlin Noise | Gradient-based Perlin noise with fractal Brownian motion |
| `voronoi` | Voronoi Noise | Cellular noise with 5 output modes controlled by `--voronoi-mode` |

## Voronoi Mode Codes

| Code | Description | Visual Result |
|------|-------------|---------------|
| `F1` | Distance to nearest feature point | Identical to Worley noise — organic cell patterns |
| `F2` | Distance to second nearest feature point | Broader cell-like regions, different topology |
| `F1 - F2` | Difference between first and second nearest | Thin lines along cell boundaries |
| `Jitter` | Distance from query point to its assigned feature point | Points clustered around feature locations |
| `Edge` | Normalized edge detection: F1 / (F1 + F2) | Clean cell boundary outlines |

## Usage Examples

### Default generation (64³, value noise)

```bash
python3 generate_volumetric.py
```

### Custom size and seed

```bash
python3 generate_volumetric.py --size 16 --seed 123
```

### Worley noise with high detail

```bash
python3 generate_volumetric.py --size 64 --noise-type worley --octaves 6 --base-freq 0.05
```

### FBM Perlin noise with custom lacunarity

```bash
python3 generate_volumetric.py --size 16 --noise-type perlin --lacunarity 2.5 --base-freq 0.02
```

### Custom output path

```bash
python3 generate_volumetric.py --size 64 --output my_texture.png --seed 99
```

### Voronoi noise with edge detection

```bash
python3 generate_volumetric.py --size 64 --noise-type voronoi --voronoi-mode "Edge"
```

### Voronoi noise with cell boundary highlighting

```bash
python3 generate_volumetric.py --size 64 --noise-type voronoi --voronoi-mode "F1 - F2" --octaves 3
```

## Output

- Exports a greyscale PNG where each cell represents a Z-axis slice
- Grid layout: `√L × √L` cells
- Total PNG dimensions: `(L × √L) × (L × √L)` pixels

| Size (L) | Grid | Output Resolution |
|----------|------|-------------------|
| 4 | 2×2 | 8×8 px |
| 16 | 4×4 | 64×64 px |
| 64 | 8×8 | 512×512 px |
| 256 | 16×16 | 4096×4096 px |

## Performance Notes

- Default size is 64³ (262,144 voxels)
- Larger sizes (256³) may be slow and consume significant memory
- Pre-computed hash/value/gradient/feature tables are used for performance
- Table size is capped at `MAX_TABLE_PERIOD = 128` to prevent freezing at high frequencies
- Performance ranking (fastest to slowest): Value Noise → Worley Noise → Voronoi Noise → FBM Perlin Noise
- Voronoi noise requires tracking both nearest and second-nearest distances (slightly more expensive than Worley)
