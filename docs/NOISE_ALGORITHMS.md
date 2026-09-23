# Noise Algorithms

## Overview

The generator supports three noise algorithms, each combined via Fractal Brownian Motion (FBM) to produce multi-scale detail.

## Value Noise

**Algorithm:** Smooth interpolation of random scalar values placed at grid points in 3D space.

**Implementation:**
- A hash function maps integer coordinates to `[0, 1]` deterministically
- Trilinear interpolation with smoothstep (`t²(3−2t)`) produces smooth transitions
- Supports both direct sampling and pre-computed lookup tables

**Characteristics:**
- Fastest of the three algorithms
- Smooth, amorphous patterns
- Good for generic texture fills

**Code locations:**
- Hash: `generate_volumetric.py:64`
- Sample (direct): `generate_volumetric.py:109`
- Sample (table): `generate_volumetric.py:145`
- Precompute: `generate_volumetric.py:137`

## Worley Noise (Cellular)

**Algorithm:** For each query point, compute the distance to the nearest "feature point" (pseudo-random locations in 3D space).

**Implementation:**
- Feature points are generated per grid cell using a deterministic hash
- The minimum distance across a 3×3×3 neighborhood (including wrapping) is used
- Distance is normalized to `[0, 1]`

**Characteristics:**
- Produces organic, cell-like, or cratered patterns
- Commonly used for terrain, stone, and cellular textures
- Slightly slower than value noise due to neighbor computation

**Code locations:**
- Feature point hash: `generate_volumetric.py:77`
- Sample (direct): `generate_volumetric.py:219`
- Sample (table): `generate_volumetric.py:185`
- Precompute: `generate_volumetric.py:177`

## FBM Perlin Noise

**Algorithm:** Gradient-based noise with trilinear interpolation of dot products between gradient vectors and position offsets. Combined via FBM for multi-octave detail.

**Implementation:**
- Deterministic gradient vectors are generated per grid point
- The dot product between gradient and position offset is computed at each corner of the unit cube
- Smoothstep-based trilinear interpolation produces the final value

**Characteristics:**
- Produces the most natural-looking detail
- Rich, cloudy, or wood-like patterns depending on parameters
- Slowest of the three algorithms due to gradient computation

**Code locations:**
- Gradient: `generate_volumetric.py:91`
- Sample (direct): `generate_volumetric.py:302`
- Sample (table): `generate_volumetric.py:271`
- Precompute: `generate_volumetric.py:263`

## Fractal Brownian Motion (FBM)

All three noise types are combined via FBM to produce multi-scale detail.

**Process:**
1. For each octave `i` (0 to octaves−1):
   - Frequency = `base_freq × lacunarity^i`
   - Amplitude = `0.5^i` (halves each octave)
2. Sample the noise at each voxel with the octave's frequency
3. Accumulate: `value += amplitude × noise_sample`
4. Normalize by total max amplitude

**Parameters:**
- **Octaves:** Number of noise layers. More octaves = finer detail.
- **Base Freq:** Controls the scale of the coarsest octave.
- **Lacunarity:** Frequency multiplier between octaves (default: 2.0).

## Seamless 3D Tiling

All noise types support seamless wrapping by default.

**How it works:**
- Grid coordinates are wrapped using modulo operations (`% period`)
- This ensures noise values match exactly at opposite edges of the volume
- The period is derived from `round(size × octave_freq)`
- Maximum period is capped at `MAX_TABLE_PERIOD = 128` to prevent memory issues

## Performance Optimizations

### Pre-computed Tables

For each octave, a 3D lookup table is pre-computed:
- Value noise: table of scalar values `(period)³`
- Worley noise: table of feature points `(period)³`
- Perlin noise: table of gradient vectors `(period)³`

This avoids re-computing hashes/gradient for every voxel sample.

### Table Size Cap

When `hash_period > MAX_TABLE_PERIOD (128)`, direct sampling is used instead of a pre-computed table. This prevents memory exhaustion at high frequencies while maintaining correctness.

### Data Structures

- Volume: `list[list[list[float]]]` — indexed as `[z][y][x]`
- Grid: `list[list[int]]` — indexed as `[row][col]`, pixel values 0–255
- Output: greyscale PNG (8-bit, no color channel)
