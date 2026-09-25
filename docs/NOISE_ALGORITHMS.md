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
Seamless 3D tiling is achieved by wrapping grid coordinates modulo the hash period. When the volume size is an exact multiple of the period, or when the period is set to match the volume, opposite edges produce identical values, making the texture tile perfectly in all three dimensions.

### Performance
- **Fastest algorithm** — no neighbor search, no gradient computation
- Supports both direct per-voxel sampling and pre-computed 3D lookup tables
- Table pre-computation stores `period³` hash values once, then uses table lookups during volume generation (128³ = ~2M entries, ~16 MB)
- Direct sampling computes a hash per lattice corner on the fly (8 hashes per voxel)

### Characteristics
- Smooth, amorphous, cloud-like patterns
- Lacks directional structure — features are featureless blobs
- Best for generic texture fills, background noise, or as a base layer in compositing

### Implementation
- Hash: `generate_volumetric.py:64`
- Sample (direct): `generate_volumetric.py:109`
- Sample (table): `generate_volumetric.py:145`
- Precompute: `generate_volumetric.py:137`

---

## Worley Noise (Cellular)

### What it is
Worley noise (also called cellular noise or F1 noise) produces organic, cell-like, cratered, or stone-like patterns. It is based on the distance from each query point to the nearest "feature point" in space.

### How it works
1. **Feature point generation:** For each integer grid cell `(nx, ny, nz)`, a deterministic hash produces a pseudo-random 3D vector `(fx, fy, fz)` in `[0, 1]³`. This vector represents the position of a "feature point" inside that grid cell.

2. **Neighborhood search:** For a query point at `(sx, sy, sz)`, the algorithm searches all 27 cells in a 3×3×3 neighborhood (including the containing cell). Each neighbor's feature point is retrieved via hash.

3. **Distance computation:** For each neighbor's feature point, the algorithm computes the offset from the query point to the feature point. The offset is wrapped to `[-0.5, 0.5]` for seamless tiling (if `|offset| > 0.5`, subtract/add 1.0). The squared Euclidean distance `dx² + dy² + dz²` is computed.

4. **Minimum selection:** The algorithm tracks the minimum distance across all 27 neighbors. This is the distance to the nearest feature point (F1).

5. **Normalization:** The square root of the minimum distance is multiplied by 2.0 and clamped to `[0, 1]`. The factor of 2.0 scales the maximum possible distance in a unit cube to approximately 1.0, providing good contrast.

### Seaming
Seamless wrapping is achieved by applying modulo operations to the neighbor cell indices. This ensures that the feature point layout wraps around at volume boundaries, producing identical noise values on opposite edges.

### Performance
- Slightly slower than value noise due to the 27-neighbor search (27 hash computations per voxel in direct mode)
- Table-based mode stores `period³` feature points (triples), which is ~3× more memory than value noise tables
- Direct mode is used when `hash_period > MAX_TABLE_PERIOD (128)`

### Characteristics
- Organic, cell-like, cratered patterns resembling honeycombs, cellular structures, or cracked earth
- Commonly used for terrain generation, stone/rock textures, water surfaces, and cellular materials
- The output is purely based on nearest-neighbor distance — no directional gradients

### Implementation
- Feature point hash: `generate_volumetric.py:77`
- Sample (direct): `generate_volumetric.py:219`
- Sample (table): `generate_volumetric.py:185`
- Precompute: `generate_volumetric.py:177`

---

## FBM Perlin Noise

### What it is
Perlin noise is a gradient-based noise algorithm developed by Ken Perlin. It produces the most natural-looking patterns — cloudy, wooden, or marbled — by combining deterministic gradient vectors with trilinear interpolation. When combined via FBM (Fractal Brownian Motion), it produces rich, multi-scale detail.

### How it works
1. **Gradient vector generation:** For each integer grid point `(x, y, z)`, a deterministic hash produces a 3D gradient vector `(gx, gy, gz)` where each component is in `[-1, 1]`. The gradient is derived from three independent hash chains to ensure good directional distribution.

2. **Unit cube corner evaluation:** For a query point at `(sx, sy, sz)`, the algorithm identifies the 8 corners of the unit cube. At each corner, it computes the dot product between the corner's gradient vector and the offset vector from the corner to the query point. This dot product represents the "gradient influence" at that corner.

3. **Smoothstep interpolation:** The fractional coordinates are smoothed via `smoothstep(t) = t²(3 − 2t)`. The 8 dot product values are then trilinearly interpolated (X → Y → Z) using these smoothstep values as interpolation weights. The result is the noise value at the query point, which ranges roughly from −1 to +1.

4. **FBM composition:** Multiple octaves of Perlin noise are layered. Each octave `i` uses frequency `base_freq × lacunarity^i` and amplitude `0.5^i`. The values are accumulated and normalized by total max amplitude. This produces self-similar detail at multiple scales — coarse features at low frequency, fine detail at high frequency.

### Seaming
Same modulo-based wrapping as value noise. Grid coordinates are wrapped modulo the hash period, ensuring opposite edges match exactly.

### Performance
- **Slowest algorithm** due to gradient vector computation and dot products at each of 8 corners
- Each corner requires computing a dot product: `gx*dx + gy*dy + gz*dz` (3 multiplies + 2 adds)
- Table-based mode stores `period³` gradient triples (~3× the memory of value noise tables)
- Pre-computation avoids re-hashing gradients, but the dot product chain per voxel is still heavier than value noise

### Characteristics
- Natural, flowing patterns resembling clouds, wood grain, marble, terrain elevation, or water ripples
- Gradient-based structure gives it directional coherence absent in value noise
- FBM composition produces rich, detailed output with detail at every scale
- The sign of Perlin noise (−1 to +1 range) produces both bright and dark features symmetrically

### Implementation
- Gradient: `generate_volumetric.py:91`
- Sample (direct): `generate_volumetric.py:433`
- Sample (table): `generate_volumetric.py:402`
- Precompute: `generate_volumetric.py:394`

---

## Voronoi Noise (Cellular)

### What it is
Voronoi noise is a cellular noise algorithm closely related to Worley noise. Instead of returning only the nearest feature distance, Voronoi noise examines both the nearest (F1) and second-nearest (F2) feature points, enabling 5 distinct output modes with dramatically different visual characteristics.

### How it works
1. **Feature point generation:** Identical to Worley noise — deterministic hash produces feature points per grid cell.

2. **Neighborhood search:** Identical 3×3×3 neighborhood search, computing squared distances to all 27 feature points.

3. **Dual distance tracking:** As distances are computed, the algorithm tracks both the nearest (F1) and second-nearest (F2) squared distances. This requires a single comparison per neighbor: if `d < F1`, shift F1→F2 and set F1=d; else if `d < F2`, set F2=d.

4. **Mode-specific output:** After the search, F1 and F2 are square-rooted to get actual distances. The output mode determines the final value:

   | Mode | Formula | Description |
   |------|---------|-------------|
   | **F1** | `F1 × 2` (clamped) | Distance to nearest feature point. Identical to Worley noise output. Produces organic cell patterns. |
   | **F2** | `F2 × 2` (clamped) | Distance to second nearest feature point. Produces broader, more spread-out cell patterns with different topology than F1. |
   | **F1 - F2** | `|F1 - F2| × 2` (clamped) | Difference between nearest and second-nearest distances. Produces thin lines along cell boundaries where F1 and F2 are similar. The cell interiors appear dark (similar distances) while boundaries appear bright. |
   | **Jitter** | `F1 × 2` (clamped) | Distance from query point to its assigned feature point. Produces bright spots at feature point locations with falloff outward. |
   | **Edge** | `F1 / (F1 + F2) × 2` (clamped) | Normalized edge detection. Produces clean cell boundary outlines. When F1 ≈ F2 (on a boundary), the ratio ≈ 0.5, giving mid-grey edges. Deep inside a cell, F1 ≪ F2, giving dark values. |

### Seaming
Identical modulo-based wrapping to Worley noise. Neighbor cell indices are wrapped to ensure seamless tiling.

### Performance
- Slightly more expensive than Worley noise because it must track both F1 and F2 distances
- The computational cost is marginal — just one extra comparison and assignment per neighbor
- Same table pre-computation as Worley (feature point tables)
- Supports both table-based and direct sampling modes

### Characteristics
- **F1:** Organic cell patterns (same as Worley)
- **F2:** Broader cell regions, different spatial distribution
- **F1 - F2:** Thin cell boundary lines — useful for tileable texture borders, crack effects, or network patterns
- **Jitter:** Clustered points — useful for scattered detail, particulate effects, or stippling
- **Edge:** Clean boundary outlines — useful for cell segmentation visualization, honeycomb patterns, or structural grids

### Implementation
- Sample (table): `generate_volumetric.py:266`
- Sample (direct): `generate_volumetric.py:332`

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
When `hash_period` divides evenly into `size`, the tiling is mathematically exact. When it doesn't, the modulo wrapping still produces visually seamless results but with potential minor artifacts at very high frequencies.

---

## Performance Optimizations

### Pre-computed Tables

For each octave, a 3D lookup table is pre-computed:
- Value noise: table of scalar values `(period)³`
- Worley noise: table of feature points `(period)³` (stored as 3 component arrays)
- Perlin noise: table of gradient vectors `(period)³` (stored as 3 component arrays)
- Voronoi noise: table of feature points `(period)³` (stored as 3 component arrays, same as Worley)

This avoids re-computing hashes/gradients/feature points for every voxel sample.

### Table Size Cap

When `hash_period > MAX_TABLE_PERIOD (128)`, direct sampling is used instead of a pre-computed table. This prevents memory exhaustion at high frequencies while maintaining correctness.

### Data Structures

- Volume: `list[list[list[float]]]` — indexed as `[z][y][x]`
- Grid: `list[list[int]]` — indexed as `[row][col]`, pixel values 0–255
- Output: greyscale PNG (8-bit, no color channel)
