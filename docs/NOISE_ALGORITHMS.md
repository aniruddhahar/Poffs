# Noise Algorithms

## Overview

The generator supports four noise algorithms, each combined via Fractal Brownian Motion (FBM) to produce multi-scale detail.

---

## Value Noise

### What it is
Value noise is the simplest and fastest of the four algorithms. It places deterministic random scalar values at integer grid points in 3D space and interpolates between them to produce smooth, continuous output.

### How it works
1. **Hash-based value assignment:** For each integer grid point `(x, y, z)`, a deterministic hash function maps the coordinate to a random scalar value in `[0, 1]`. The hash combines the seed with the coordinate multiplied by large prime constants, then applies bit manipulation (XOR shifts and multiplication) to produce a well-distributed pseudo-random value.

2. **Trilinear interpolation:** For a query point at floating-point coordinates `(sx, sy, sz)`, the algorithm identifies the 8 corners of the unit cube containing that point. Each corner's integer coordinates are used to look up its hash-derived value.

3. **Smoothstep easing:** Before interpolation, the fractional part of each coordinate is passed through a smoothstep function: `smoothstep(t) = t²(3 − 2t)`. This is a Hermite interpolation easing curve that produces `C¹`-continuous transitions (the derivative is zero at the grid points), eliminating visible "blocky" seams between cells.

4. **Lerp chain:** The 8 corner values are interpolated along X (pairwise), then along Y, then along Z — a total of 3 lerp passes yielding the final value.

### Seaming
Seamless 3D tiling is achieved by wrapping grid coordinates modulo the hash period. The sampling coordinates are scaled so the volume spans exactly `period` lattice cells, so opposite faces of the volume evaluate the same periodic lattice point and match exactly for any period.

### Performance
- **Fastest algorithm** — no neighbor search, no gradient computation
- Supports both direct per-voxel sampling and pre-computed 3D lookup tables
- Table pre-computation stores `period³` hash values once, then uses table lookups during volume generation (128³ ≈ 2M entries; ≈16 MB as raw float64 payload, roughly twice that once Python's list/float object overhead is counted)
- Direct sampling computes a hash per lattice corner on the fly (8 hashes per voxel)

### Characteristics
- Smooth, amorphous, cloud-like patterns
- Lacks directional structure — features are featureless blobs
- Best for generic texture fills, background noise, or as a base layer in compositing

### Implementation
All in `generate_volumetric.py`:
- Hash: `_hash_coord()`
- Sample (direct): `_sample_value_3d()`
- Sample (table): `_sample_value_table()`
- Precompute: `_precompute_value_table()`

---

## Worley Noise (Cellular)

### What it is
Worley noise (also called cellular noise or F1 noise) produces organic, cell-like, cratered, or stone-like patterns. It is based on the distance from each query point to the nearest "feature point" in space.

### How it works
1. **Feature point generation:** For each integer grid cell `(nx, ny, nz)`, a deterministic hash produces a pseudo-random 3D vector `(fx, fy, fz)` in `[0, 1]³`. This vector represents the position of a "feature point" inside that grid cell.

2. **Neighborhood search:** For a query point at `(sx, sy, sz)`, the algorithm searches the cells in a 3×3×3 neighborhood (including the containing cell) — 5×5×5 when the tiling is only 2 cells wide, which is the smallest window that still covers half a cell diagonal. Each neighbor's feature point is retrieved via hash.

3. **Distance computation:** For each neighbor's feature point, the algorithm computes the offset from the query point to the feature point, measured to that feature's **nearest periodic image of the whole tiling**: the offset is wrapped by the tiling period (if `|offset| > period/2`, subtract/add `period`). Wrapping by a single cell instead would alias a neighbour's feature into the current cell and make the field discontinuous at every cell boundary for any period ≥ 4. The squared Euclidean distance `dx² + dy² + dz²` is computed.

4. **Minimum selection:** The algorithm tracks the minimum distance across all searched neighbors. This is the distance to the nearest feature point (F1).

5. **Normalization:** The square root of the minimum distance is multiplied by `CELLULAR_SCALE` = `2/√3` and clamped to `[0, 1]`. The nearest periodic feature is never farther than half a cell diagonal (`√3/2`), so this maps the full possible range onto `[0, 1]`.

### Seaming
Seamless wrapping needs both halves of the periodic wrap: modulo on the neighbour cell indices (so the feature layout wraps at the volume edges) **and** wrapping each feature offset by the tiling period (so distances are taken to the nearest image of the tiling). Opposite edges then match exactly.

### Performance
- Slightly slower than value noise due to the neighborhood search (27 hash computations per voxel in direct mode)
- Table-based mode stores `period³` feature points (triples), which costs roughly twice the memory of value noise tables (measured; three floats per entry plus tuple overhead)
- Direct mode is used when `hash_period > MAX_TABLE_PERIOD (128)`

### Characteristics
- Organic, cell-like, cratered patterns resembling honeycombs, cellular structures, or cracked earth
- Commonly used for terrain generation, stone/rock textures, water surfaces, and cellular materials
- The output is purely based on nearest-neighbor distance — no directional gradients

### Implementation
All in `generate_volumetric.py`:
- Feature point hash: `_worley_hash_coord()`
- Sample (direct): `_sample_worley_direct()`
- Sample (table): `_sample_worley_table()`
- Precompute: `_precompute_worley_table()`

---

## FBM Perlin Noise

### What it is
Perlin noise is a gradient-based noise algorithm developed by Ken Perlin. It produces the most natural-looking patterns — cloudy, wooden, or marbled — by combining deterministic gradient vectors with trilinear interpolation. When combined via FBM (Fractal Brownian Motion), it produces rich, multi-scale detail.

### How it works
1. **Gradient vector generation:** For each integer grid point `(x, y, z)`, a deterministic hash produces a 3D gradient vector `(gx, gy, gz)` where each component is in `[-1, 1]`. The gradient is built from three chained hash steps (gx seeds h1, h1 seeds h2 for gy, h2 seeds h3 for gz), giving each component its own bit-mixing pass.

2. **Unit cube corner evaluation:** For a query point at `(sx, sy, sz)`, the algorithm identifies the 8 corners of the unit cube. At each corner, it computes the dot product between the corner's gradient vector and the offset vector from the corner to the query point. This dot product represents the "gradient influence" at that corner.

3. **Smoothstep interpolation:** The fractional coordinates are smoothed via `smoothstep(t) = t²(3 − 2t)`. The 8 dot product values are then trilinearly interpolated (X → Y → Z) using these smoothstep values as interpolation weights. The result is the noise value at the query point, which ranges roughly from −1 to +1.

4. **FBM composition:** Multiple octaves of Perlin noise are layered. Each octave `i` uses frequency `base_freq × lacunarity^i` and amplitude `0.5^i`. The values are accumulated and normalized by total max amplitude. This produces self-similar detail at multiple scales — coarse features at low frequency, fine detail at high frequency.

### Seaming
Same modulo-based wrapping as value noise. Grid coordinates are wrapped modulo the hash period, ensuring opposite edges match exactly.

### Performance
- **Slowest algorithm** due to gradient vector computation and dot products at each of 8 corners
- Each corner requires computing a dot product: `gx*dx + gy*dy + gz*dz` (3 multiplies + 2 adds)
- Table-based mode stores `period³` gradient triples (~2× the memory of value noise tables, measured)
- Pre-computation avoids re-hashing gradients, but the dot product chain per voxel is still heavier than value noise

### Characteristics
- Natural, flowing patterns resembling clouds, wood grain, marble, terrain elevation, or water ripples
- Gradient-based structure gives it directional coherence absent in value noise
- FBM composition produces rich, detailed output with detail at every scale
- The sign of Perlin noise (−1 to +1 range) produces both bright and dark features symmetrically

### Implementation
All in `generate_volumetric.py`:
- Gradient: `_perlin_gradient()`
- Sample (direct): `_sample_perlin_direct()`
- Sample (table): `_sample_perlin_table()`
- Precompute: `_precompute_perlin_table()`

---

## Voronoi Noise (Cellular)

### What it is
Voronoi noise is a cellular noise algorithm closely related to Worley noise. Instead of returning only the nearest feature distance, Voronoi noise examines both the nearest (F1) and second-nearest (F2) feature points, enabling 5 distinct output modes with dramatically different visual characteristics.

### How it works
1. **Feature point generation:** Identical to Worley noise — deterministic hash produces feature points per grid cell.

2. **Neighborhood search:** Same periodic neighborhood search as Worley noise (3×3×3, or 5×5×5 for a 2-cell tiling), computing squared distances to every searched feature point.

3. **Dual distance tracking:** As distances are computed, the algorithm tracks both the nearest (F1) and second-nearest (F2) squared distances. This requires a single comparison per neighbor: if `d < F1`, shift F1→F2 and set F1=d; else if `d < F2`, set F2=d.

4. **Mode-specific output:** After the search, F1 and F2 are square-rooted to get actual distances. The output mode determines the final value (`CELLULAR_SCALE` = `2/√3`, the reciprocal of the half cell diagonal):

   | Mode | Formula | Description |
   |------|---------|-------------|
   | **F1** | `F1 × 2/√3` (clamped) | Distance to nearest feature point. Identical to Worley noise output. Produces organic cell patterns. |
   | **F2** | `F2 × 1/√3` (clamped) | Distance to second nearest feature point. Produces broader, more spread-out cell patterns with different topology than F1. |
   | **F1 - F2** | `|F1 - F2| × 2/√3` (clamped) | Difference between nearest and second-nearest distances. On cell boundaries F1 ≈ F2, so the difference → 0 and boundaries appear as thin dark lines; inside a cell F2 grows well beyond F1, so cell interiors appear bright. |
   | **Jitter** | `d_own × 2/√3` (clamped) | Distance from query point to the feature point of the cell containing it (not the nearest feature point). Each cell is darkest at its own (off-center) feature point and brightens with distance toward its edges/corners. |
   | **Edge** | `F1 / (F1 + F2) × 2` (clamped) | Normalized edge detection. Produces clean cell boundary outlines. When F1 ≈ F2 (on a boundary), the ratio ≈ 0.5 and doubling maps it to ≈ 1.0, giving bright boundary lines; deep inside a cell, F1 ≪ F2, driving the value toward 0 (dark). |

### Seaming
Identical periodic wrapping to Worley noise: neighbour cell indices are wrapped modulo the period, and each feature offset is taken to the nearest image of the tiling.

### Performance
- Slightly more expensive than Worley noise because it must track both F1 and F2 distances
- The computational cost is marginal — just one extra comparison and assignment per neighbor
- Same table pre-computation as Worley (feature point tables)
- Supports both table-based and direct sampling modes

### Characteristics
- **F1:** Organic cell patterns (same as Worley)
- **F2:** Broader cell regions, different spatial distribution
- **F1 - F2:** Thin cell boundary lines — useful for tileable texture borders, crack effects, or network patterns
- **Jitter:** Grid-aligned cell gradients — useful for scattered detail, particulate effects, or stippling
- **Edge:** Clean boundary outlines — useful for cell segmentation visualization, honeycomb patterns, or structural grids

### Implementation
All in `generate_volumetric.py`:
- Sample (table): `_voronoi_sample_table()`
- Sample (direct): `_voronoi_sample_direct()`
- Jitter containing-cell lookup: `_jitter_distance_table()`
- Feature-point hash/tables: shared with Worley noise (`_worley_hash_coord()`, `_precompute_worley_table()`)

---

## Fractal Brownian Motion (FBM)

All four noise types are combined via FBM to produce multi-scale detail.

### Process
1. For each octave `i` (0 to octaves−1):
   - Frequency = `base_freq × lacunarity^i`
   - Amplitude = `0.5^i` (halves each octave)
2. Sample the noise at each voxel with the octave's frequency
3. Accumulate: `value += amplitude × noise_sample`
4. Normalize by total max amplitude

### Parameters
- **Octaves:** Number of noise layers. More octaves = finer detail. Each octave doubles the frequency (default lacunarity) and halves the contribution. Typical range: 2–8.
- **Base Freq:** Controls the scale of the coarsest octave. Higher values produce finer features from the start. Typical range: 0.01–1.0 for texture work.
- **Lacunarity:** Frequency multiplier between octaves (default: 2.0). Higher values produce larger frequency jumps, leaving gaps in the frequency spectrum. Lower values (closer to 1.0) produce smoother transitions between octaves.

### Amplitude Decay
Each octave contributes half as much as the previous one:
- Octave 0: amplitude = 1.0
- Octave 1: amplitude = 0.5
- Octave 2: amplitude = 0.25
- Octave 3: amplitude = 0.125

This geometric decay ensures the highest-frequency (finest) octaves add subtle detail rather than dominating the output.

---

## Seamless 3D Tiling

All noise types support seamless wrapping by default.

### How it works
- Grid coordinates are wrapped using modulo operations (`% period`)
- This ensures noise values match exactly at opposite edges of the volume
- The period is derived from `round(size × octave_freq)`
- Maximum period is capped at `MAX_TABLE_PERIOD = 128` to prevent memory issues

### Period Calculation
For each octave, the hash period is computed as:
```
hash_period = max(2, round(size × base_freq × lacunarity^octave))
```
The coordinate mapping spans exactly one full lattice period across the L voxels of the volume, so the two opposite faces of the volume coincide with the same point of the periodic lattice and match **exactly** for any period (`tests/cellular_check.py` asserts a zero wrap mismatch for every Voronoi mode over periods {2, 3, 4, 5, 6, 9, 16}). When `hash_period` does not divide `size` evenly, the lattice spacing is simply a fractional number of voxels — the tiling stays seamless, only the per-voxel lattice alignment changes.

---

## Performance Optimizations

### Pre-computed Tables

For each octave, a 3D lookup table is pre-computed:
- Value noise: table of scalar values `(period)³`
- Worley noise: table of feature points `(period)³` (CPU: one `(rx, ry, rz)` tuple per cell; the GPU module keeps three component arrays for buffer merging)
- Perlin noise: table of gradient vectors `(period)³` (CPU: one `(gx, gy, gz)` tuple per cell; the GPU module keeps three component arrays)
- Voronoi noise: table of feature points `(period)³` (same structure as Worley)

This avoids re-computing hashes/gradients/feature points for every voxel sample.

### Table Size Cap

When `hash_period > MAX_TABLE_PERIOD (128)`, direct sampling is used instead of a pre-computed table. This prevents memory exhaustion at high frequencies while maintaining correctness.

### Data Structures

- Volume: `list[list[list[float]]]` — indexed as `[z][y][x]`
- Grid: `list[list[int]]` — indexed as `[row][col]`, pixel values 0–255
- Output: greyscale PNG (8-bit, no color channel)
