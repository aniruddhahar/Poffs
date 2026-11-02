#!/usr/bin/env python3
"""
Volumetric 3D Texture Generator - OpenCL GPU Backend

Provides GPU-accelerated volume generation via OpenCL.
All noise algorithms, hash functions, and interpolation are identical to the CPU version.

Kernels use IEEE double-precision arithmetic (cl_khr_fp64) matching Python floats,
so GPU output is bit-identical to CPU. On devices without fp64 support the kernel
build fails and generation falls back to CPU automatically.
"""

from typing import Optional

try:
    import pyopencl as cl
    import numpy as np
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    cl = None  # type: ignore
    np = None  # type: ignore


# ---------------------------------------------------------------------------
# OpenCL Kernel Source (one kernel per noise type for correctness)
# ---------------------------------------------------------------------------

_KERNEL_VALUE = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
/* volatile forces materialization of products: prevents the GPU compiler from
   contracting mul+add into FMA, which would break bit-exact parity with the
   Python (float64) reference. */
double sstep(double t) {
    volatile double s = t * t;
    volatile double u = 3.0 - 2.0 * t;
    return s * u;
}
double lerp(double a, double b, double t) {
    volatile double m = t * (b - a);
    return a + m;
}

kernel void generate_value(
    int size, int octaves, int seed,
    constant int* periods, constant int* p2s, constant int* offsets,
    constant double* tables,
    __global double* output
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    if (x >= size || y >= size || z >= size) return;

    int idx = z * size * size + y * size + x;
    double val = 0.0, amp = 1.0, max_v = 0.0;

    for (int o = 0; o < octaves; o++) {
        int period = periods[o];
        int offset = offsets[o];
        int p2 = p2s[o];
        double cx = (double)x / (double)size * (double)period;
        double cy = (double)y / (double)size * (double)period;
        double cz = (double)z / (double)size * (double)period;

        int ix = (int)cx, iy = (int)cy, iz = (int)cz;
        double fx = sstep(cx - (double)ix);
        double fy = sstep(cy - (double)iy);
        double fz = sstep(cz - (double)iz);

        int i0x = ix % period, i1x = (ix+1) % period;
        int i0y = iy % period, i1y = (iy+1) % period;
        int i0z = iz % period, i1z = (iz+1) % period;

        int b000 = offset + i0z*p2 + i0y*period + i0x;
        int b100 = offset + i0z*p2 + i0y*period + i1x;
        int b010 = offset + i0z*p2 + i1y*period + i0x;
        int b110 = offset + i0z*p2 + i1y*period + i1x;
        int b001 = offset + i1z*p2 + i0y*period + i0x;
        int b101 = offset + i1z*p2 + i0y*period + i1x;
        int b011 = offset + i1z*p2 + i1y*period + i0x;
        int b111 = offset + i1z*p2 + i1y*period + i1x;

        double v00 = lerp(tables[b000], tables[b100], fx);
        double v10 = lerp(tables[b010], tables[b110], fx);
        double v01 = lerp(tables[b001], tables[b101], fx);
        double v11 = lerp(tables[b011], tables[b111], fx);
        double v0 = lerp(v00, v10, fy);
        double v1 = lerp(v01, v11, fy);

        volatile double p = amp * lerp(v0, v1, fz);
        val += p;
        max_v += amp;
        amp *= 0.5;
    }
    output[idx] = val / max_v;
}
"""

_KERNEL_WORLEY = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
kernel void generate_worley(
    int size, int octaves, int seed,
    constant int* periods, constant int* p2s, constant int* offsets,
    constant double* tx, constant double* ty, constant double* tz,
    __global double* output
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    if (x >= size || y >= size || z >= size) return;

    int idx = z * size * size + y * size + x;
    double val = 0.0, amp = 1.0, max_v = 0.0;

    for (int o = 0; o < octaves; o++) {
        int period = periods[o];
        int offset = offsets[o];
        int p2 = p2s[o];
        double cx = (double)x / (double)size * (double)period;
        double cy = (double)y / (double)size * (double)period;
        double cz = (double)z / (double)size * (double)period;

        int ix = (int)cx, iy = (int)cy, iz = (int)cz;
        double min_dsq = 1e300;

        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int ci = ((ix+dx) % period + period) % period;
                    int cj = ((iy+dy) % period + period) % period;
                    int ck = ((iz+dz) % period + period) % period;
                    int ti = offset + ck*p2 + cj*period + ci;

                    double fx = cx - (double)ix - tx[ti];
                    double fy = cy - (double)iy - ty[ti];
                    double fz = cz - (double)iz - tz[ti];
                    if (fx > 0.5) fx -= 1.0; else if (fx < -0.5) fx += 1.0;
                    if (fy > 0.5) fy -= 1.0; else if (fy < -0.5) fy += 1.0;
                    if (fz > 0.5) fz -= 1.0; else if (fz < -0.5) fz += 1.0;

                    volatile double qx = fx*fx, qy = fy*fy, qz = fz*fz;
                    double d = qx + qy + qz;
                    if (d < min_dsq) min_dsq = d;
                }
            }
        }
        double s = sqrt(min_dsq) * 2.0;
        if (s > 1.0) s = 1.0; if (s < 0.0) s = 0.0;
        volatile double p = amp * s;
        val += p; max_v += amp; amp *= 0.5;
    }
    output[idx] = val / max_v;
}
"""

_KERNEL_PERLIN = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
double sstep(double t) {
    volatile double s = t * t;
    volatile double u = 3.0 - 2.0 * t;
    return s * u;
}
double lerp(double a, double b, double t) {
    volatile double m = t * (b - a);
    return a + m;
}
double dotg(constant double* gx, constant double* gy, constant double* gz, int b,
            double ux, double uy, double uz) {
    volatile double p = gx[b]*ux;
    volatile double q = gy[b]*uy;
    volatile double r = gz[b]*uz;
    return p + q + r;
}

kernel void generate_perlin(
    int size, int octaves, int seed,
    constant int* periods, constant int* p2s, constant int* offsets,
    constant double* gx, constant double* gy, constant double* gz,
    __global double* output
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    if (x >= size || y >= size || z >= size) return;

    int idx = z * size * size + y * size + x;
    double val = 0.0, amp = 1.0, max_v = 0.0;

    for (int o = 0; o < octaves; o++) {
        int period = periods[o];
        int offset = offsets[o];
        int p2 = p2s[o];
        double cx = (double)x / (double)size * (double)period;
        double cy = (double)y / (double)size * (double)period;
        double cz = (double)z / (double)size * (double)period;

        int ix = (int)cx, iy = (int)cy, iz = (int)cz;
        double dx = sstep(cx - (double)ix);
        double dy = sstep(cy - (double)iy);
        double dz = sstep(cz - (double)iz);

        double ux = cx - (double)ix, uy = cy - (double)iy, uz = cz - (double)iz;
        double wx = ux - 1.0, wy = uy - 1.0, wz = uz - 1.0;

        int i0x = ix % period, i1x = (ix+1) % period;
        int i0y = iy % period, i1y = (iy+1) % period;
        int i0z = iz % period, i1z = (iz+1) % period;

        int b000 = offset + i0z*p2 + i0y*period + i0x;
        int b100 = offset + i0z*p2 + i0y*period + i1x;
        int b010 = offset + i0z*p2 + i1y*period + i0x;
        int b110 = offset + i0z*p2 + i1y*period + i1x;
        int b001 = offset + i1z*p2 + i0y*period + i0x;
        int b101 = offset + i1z*p2 + i0y*period + i1x;
        int b011 = offset + i1z*p2 + i1y*period + i0x;
        int b111 = offset + i1z*p2 + i1y*period + i1x;

        double d00 = lerp(dotg(gx, gy, gz, b000, ux, uy, uz),
                          dotg(gx, gy, gz, b100, wx, uy, uz), dx);
        double d01 = lerp(dotg(gx, gy, gz, b010, ux, wy, uz),
                          dotg(gx, gy, gz, b110, wx, wy, uz), dx);
        double d10 = lerp(dotg(gx, gy, gz, b001, ux, uy, wz),
                          dotg(gx, gy, gz, b101, wx, uy, wz), dx);
        double d11 = lerp(dotg(gx, gy, gz, b011, ux, wy, wz),
                          dotg(gx, gy, gz, b111, wx, wy, wz), dx);

        double s = lerp(lerp(d00, d01, dy), lerp(d10, d11, dy), dz);
        volatile double p = amp * s;
        val += p; max_v += amp; amp *= 0.5;
    }
    output[idx] = val / max_v;
}
"""

_KERNEL_VORONOI = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
/* Voronoi noise kernel */
kernel void generate_voronoi(
    int size, int octaves, int seed, int voronoi_mode,
    constant int* periods, constant int* p2s, constant int* offsets,
    constant double* tx, constant double* ty, constant double* tz,
    __global double* output
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    if (x >= size || y >= size || z >= size) return;

    int idx = z * size * size + y * size + x;
    double val = 0.0, amp = 1.0, max_v = 0.0;

    for (int o = 0; o < octaves; o++) {
        int period = periods[o];
        int offset = offsets[o];
        int p2 = p2s[o];
        double cx = (double)x / (double)size * (double)period;
        double cy = (double)y / (double)size * (double)period;
        double cz = (double)z / (double)size * (double)period;

        int ix = (int)cx, iy = (int)cy, iz = (int)cz;
        double f1 = 1e300, f2 = 1e300;

        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int ci = ((ix+dx) % period + period) % period;
                    int cj = ((iy+dy) % period + period) % period;
                    int ck = ((iz+dz) % period + period) % period;
                    int ti = offset + ck*p2 + cj*period + ci;

                    double fx = cx - (double)ix - tx[ti];
                    double fy = cy - (double)iy - ty[ti];
                    double fz = cz - (double)iz - tz[ti];
                    if (fx > 0.5) fx -= 1.0; else if (fx < -0.5) fx += 1.0;
                    if (fy > 0.5) fy -= 1.0; else if (fy < -0.5) fy += 1.0;
                    if (fz > 0.5) fz -= 1.0; else if (fz < -0.5) fz += 1.0;

                    volatile double qx = fx*fx, qy = fy*fy, qz = fz*fz;
                    double d = qx + qy + qz;
                    if (d < f1) { f2 = f1; f1 = d; }
                    else if (d < f2) { f2 = d; }
                }
            }
        }

        double s1 = sqrt(f1);
        double s2 = sqrt(f2);
        double sample = 0.0;

        if (voronoi_mode == 0) sample = s1 * 2.0;
        else if (voronoi_mode == 1) sample = s2 * 2.0;
        else if (voronoi_mode == 2) sample = (s2 - s1) * 2.0; /* s1 <= s2 by construction */
        else if (voronoi_mode == 3) {
            int ci = ((ix % period) + period) % period;
            int cj = ((iy % period) + period) % period;
            int ck = ((iz % period) + period) % period;
            int ti0 = offset + ck*p2 + cj*period + ci;
            double jx = cx - (double)ix - tx[ti0];
            double jy = cy - (double)iy - ty[ti0];
            double jz = cz - (double)iz - tz[ti0];
            if (jx > 0.5) jx -= 1.0; else if (jx < -0.5) jx += 1.0;
            if (jy > 0.5) jy -= 1.0; else if (jy < -0.5) jy += 1.0;
            if (jz > 0.5) jz -= 1.0; else if (jz < -0.5) jz += 1.0;
            volatile double jqx = jx*jx, jqy = jy*jy, jqz = jz*jz;
            sample = sqrt(jqx + jqy + jqz) * 2.0 / 1.7320508075688772;
        }
        else if (voronoi_mode == 4) {
            double denom = s1 + s2;
            sample = (denom < 1e-10) ? 0.5 : (s1 / denom) * 2.0;
        }

        if (sample > 1.0) sample = 1.0;
        if (sample < 0.0) sample = 0.0;
        volatile double p = amp * sample;
        val += p; max_v += amp; amp *= 0.5;
    }
    output[idx] = val / max_v;
}
"""


# ---------------------------------------------------------------------------
# OpenCL Context Manager
# ---------------------------------------------------------------------------

class _OpenCLManager:
    """Manages OpenCL context, queues, and compiled programs."""

    def __init__(self):
        self.ctx = None  # Optional[cl.Context] when pyopencl present
        self.queue = None  # Optional[cl.CommandQueue]
        self.programs = {}  # dict[str, cl.Program]
        self.kernels = {}  # dict[str, cl.Kernel]
        self._ready = False

    def get_or_create_queue(self):
        if self.queue:
            return self.queue
        try:
            platforms = cl.get_platforms()
            for p in platforms:
                devs = p.get_devices(cl.device_type.GPU)
                if not devs:
                    devs = p.get_devices()
                if devs:
                    self.ctx = cl.Context(devs)
                    self.queue = cl.CommandQueue(self.ctx)
                    break
        except Exception:
            pass
        return self.queue

    def get_kernel(self, name: str, kernel_name: str, source: str):
        if name in self.kernels:
            return self.kernels[name]

        queue = self.get_or_create_queue()
        if not queue:
            return None

        if name not in self.programs:
            try:
                self.programs[name] = cl.Program(self.ctx, source).build()
            except Exception:
                return None

        prog = self.programs[name]
        try:
            self.kernels[name] = cl.Kernel(prog, kernel_name)
        except Exception:
            return None
        return self.kernels[name]

    def is_ready(self) -> bool:
        return self.get_or_create_queue() is not None


_mgr_instance: Optional[_OpenCLManager] = None


def _get_mgr() -> _OpenCLManager:
    global _mgr_instance
    if _mgr_instance is None:
        _mgr_instance = _OpenCLManager()
    return _mgr_instance


# ---------------------------------------------------------------------------
# Helper: merge octave tables into flat arrays
# ---------------------------------------------------------------------------

# Noise types whose tables carry 3 float components per cell (x/y/z channels).
_THREE_CHANNEL_NOISE = ("Worley Noise", "Voronoi Noise", "FBM Perlin Noise")


def _merge_octave_tables(octaves_info: list[tuple[int, Optional[list]]],
                         noise_type: str):
    """Merge per-octave table data into flat GPU arrays.

    Table structures (as produced by the _precompute_* helpers in this module):
        Value Noise      -> [z][y][x] float
        Worley/Voronoi   -> (tx, ty, tz), each [z][y][x] float
        FBM Perlin Noise -> (gx, gy, gz), each [z][y][x] float

    Returns:
        (periods_arr, p2s_arr, offsets_arr, (buf0, buf1, buf2))
        - periods_arr: int32[N] — hash period per octave
        - p2s_arr: int32[N] — period^2 per octave
        - offsets_arr: int32[N] — element offset into table buffers (-1 if capped)
        - buf0/buf1/buf2: flat float32 device buffers; buf1/buf2 are None for
          single-channel (Value) noise.
    """
    import numpy as np

    n_octaves = len(octaves_info)
    periods_arr = np.zeros(n_octaves, dtype=np.int32)
    p2s_arr = np.zeros(n_octaves, dtype=np.int32)
    offsets_arr = np.zeros(n_octaves, dtype=np.int32)

    three_channel = noise_type in _THREE_CHANNEL_NOISE

    total = 0
    for i, (period, table) in enumerate(octaves_info):
        periods_arr[i] = period
        p2s_arr[i] = period * period
        if table is None:
            offsets_arr[i] = -1
            continue
        offsets_arr[i] = total
        total += period * period * period

    if total == 0:
        return periods_arr, p2s_arr, offsets_arr, (None, None, None)

    n_channels = 3 if three_channel else 1
    merged = [np.zeros(total, dtype=np.float64) for _ in range(n_channels)]

    for i, (period, table) in enumerate(octaves_info):
        if table is None:
            continue
        start = int(offsets_arr[i])
        p2 = period * period
        if three_channel:
            g0, g1, g2 = table
            for z in range(period):
                zbase = start + z * p2
                for y in range(period):
                    row = zbase + y * period
                    for x in range(period):
                        merged[0][row + x] = g0[z][y][x]
                        merged[1][row + x] = g1[z][y][x]
                        merged[2][row + x] = g2[z][y][x]
        else:
            for z in range(period):
                zbase = start + z * p2
                for y in range(period):
                    row = zbase + y * period
                    for x in range(period):
                        merged[0][row + x] = table[z][y][x]

    ctx = _get_mgr().ctx
    buffers = tuple(
        cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=arr)
        for arr in merged
    )
    return periods_arr, p2s_arr, offsets_arr, (buffers[0],
                                               buffers[1] if n_channels == 3 else None,
                                               buffers[2] if n_channels == 3 else None)


# ---------------------------------------------------------------------------
# Main GPU entry point
# ---------------------------------------------------------------------------

_KERNEL_SPECS = {
    "Value Noise": ("generate_value", _KERNEL_VALUE),
    "Worley Noise": ("generate_worley", _KERNEL_WORLEY),
    "FBM Perlin Noise": ("generate_perlin", _KERNEL_PERLIN),
    "Voronoi Noise": ("generate_voronoi", _KERNEL_VORONOI),
}


def _any_octave_capped(size: int, octaves: int, base_freq: float, lacunarity: float) -> bool:
    """True if any octave's hash period exceeds MAX_TABLE_PERIOD.

    GPU kernels only support precomputed table sampling; the CPU falls back
    to direct hashing for capped octaves. To keep CPU/GPU output identical,
    such generations must run on CPU.
    """
    from generate_volumetric import MAX_TABLE_PERIOD
    for i in range(octaves):
        if max(2, round(size * base_freq * (lacunarity ** i))) > MAX_TABLE_PERIOD:
            return True
    return False


def generate_volume_gpu(
    size: int,
    octaves: int,
    base_freq: float,
    lacunarity: float,
    seed: int,
    noise_type: str,
    cancel_event=None,
    voronoi_mode: str = "F1",
    progress_cb=None,
) -> list[list[list[float]]]:
    """Generate L×L×L volume using OpenCL GPU, falling back to CPU if needed.

    Args:
        voronoi_mode: Only used when noise_type is "Voronoi Noise".
            One of: "F1", "F2", "F1 - F2", "Jitter", "Edge"
        progress_cb: Optional callable(fraction 0.0-1.0); CPU fallback reports
            per-slice progress, GPU reports per-slice launch progress.
    """
    if not OPENCL_AVAILABLE or not _get_mgr().is_ready():
        return _generate_volume_cpu(size, octaves, base_freq, lacunarity, seed, noise_type, cancel_event, voronoi_mode, progress_cb)

    if noise_type not in _KERNEL_SPECS:
        return _generate_volume_cpu(size, octaves, base_freq, lacunarity, seed, noise_type, cancel_event, voronoi_mode, progress_cb)

    if _any_octave_capped(size, octaves, base_freq, lacunarity):
        return _generate_volume_cpu(size, octaves, base_freq, lacunarity, seed, noise_type, cancel_event, voronoi_mode, progress_cb)

    # Kernels receive the seed as np.int32; seeds outside that range cannot
    # be reproduced bit-exactly on GPU, so route them to CPU explicitly.
    if not 0 <= seed <= 2**31 - 1:
        return _generate_volume_cpu(size, octaves, base_freq, lacunarity, seed, noise_type, cancel_event, voronoi_mode, progress_cb)

    try:
        return _generate_volume_opencl(size, octaves, base_freq, lacunarity, seed, noise_type, cancel_event, voronoi_mode, progress_cb)
    except Exception:
        return _generate_volume_cpu(size, octaves, base_freq, lacunarity, seed, noise_type, cancel_event, voronoi_mode, progress_cb)


def _generate_volume_opencl(
    size: int, octaves: int, base_freq: float, lacunarity: float,
    seed: int, noise_type: str, cancel_event=None, voronoi_mode: str = "F1",
    progress_cb=None,
) -> list[list[list[float]]]:
    """Run volume generation on GPU via OpenCL."""
    import numpy as np
    import pyopencl.array as cl_array
    from generate_volumetric import MAX_TABLE_PERIOD

    mgr = _get_mgr()
    queue = mgr.queue

    # Build octave info (same as CPU; capped octaves are rejected earlier
    # by _any_octave_capped so tables always exist here)
    octaves_info = []
    for i in range(octaves):
        octave_freq = base_freq * (lacunarity ** i)
        hash_period = max(2, round(size * octave_freq))
        if hash_period > MAX_TABLE_PERIOD:
            octaves_info.append((hash_period, None))
            continue

        if noise_type in ("Worley Noise", "Voronoi Noise"):
            table = _precompute_worley_table(hash_period, seed)
        elif noise_type == "Value Noise":
            table = _precompute_value_table(hash_period, seed)
        else:
            table = _precompute_perlin_table(hash_period, seed)
        octaves_info.append((hash_period, table))

    # Prepare GPU buffers
    periods_arr, p2s_arr, offsets_arr, (t0, t1, t2) = _merge_octave_tables(octaves_info, noise_type)

    total_voxels = size * size * size
    output_dev = cl_array.Array(queue, (total_voxels,), np.float64)

    # Kernel args must be device buffers, not host numpy arrays
    def _int_buf(arr):
        return cl.Buffer(mgr.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=arr)

    periods_buf, p2s_buf, offsets_buf = _int_buf(periods_arr), _int_buf(p2s_arr), _int_buf(offsets_arr)

    # Map voronoi mode to integer
    voronoi_mode_map = {"F1": 0, "F2": 1, "F1 - F2": 2, "Jitter": 3, "Edge": 4}
    vm = voronoi_mode_map.get(voronoi_mode, 0)

    kernel_name, source = _KERNEL_SPECS[noise_type]
    kernel = mgr.get_kernel(noise_type, kernel_name, source)

    if kernel is None:
        for buf in [periods_buf, p2s_buf, offsets_buf, t0, t1, t2]:
            if isinstance(buf, cl.Buffer):
                buf.release()
        raise RuntimeError("Failed to compile OpenCL kernel")

    empty = cl.Buffer(mgr.ctx, cl.mem_flags.READ_ONLY, 4)
    if t0 is None:
        t0 = empty
    if t1 is None:
        t1 = empty
    if t2 is None:
        t2 = empty

    if noise_type == "Value Noise":
        kernel.set_args(
            np.int32(size), np.int32(octaves), np.int32(seed),
            periods_buf, p2s_buf, offsets_buf,
            t0, output_dev.data,
        )
    elif noise_type == "Worley Noise":
        kernel.set_args(
            np.int32(size), np.int32(octaves), np.int32(seed),
            periods_buf, p2s_buf, offsets_buf,
            t0, t1, t2, output_dev.data,
        )
    elif noise_type == "FBM Perlin Noise":
        kernel.set_args(
            np.int32(size), np.int32(octaves), np.int32(seed),
            periods_buf, p2s_buf, offsets_buf,
            t0, t1, t2, output_dev.data,
        )
    elif noise_type == "Voronoi Noise":
        kernel.set_args(
            np.int32(size), np.int32(octaves), np.int32(seed), np.int32(vm),
            periods_buf, p2s_buf, offsets_buf,
            t0, t1, t2, output_dev.data,
        )

    # Launch one Z-slice at a time so cancel_event can interrupt between
    # slices and progress_cb can report real progress.
    try:
        global_size = [size, size, 1]
        cancelled = False
        for z in range(size):
            if cancel_event and cancel_event.is_set():
                cancelled = True
                break
            cl.enqueue_nd_range_kernel(queue, kernel, global_size, None,
                                       global_work_offset=[0, 0, z])
            queue.finish()
            if progress_cb:
                progress_cb((z + 1) / size)

        # Read back
        output_host = None if cancelled else output_dev.get()
    finally:
        # Clean up buffers (deduplicated: t1/t2 may alias the empty placeholder)
        released = set()
        for buf in [periods_buf, p2s_buf, offsets_buf, t0, t1, t2, empty]:
            if isinstance(buf, cl.Buffer) and id(buf) not in released:
                released.add(id(buf))
                buf.release()

    if cancelled:
        # Caller checks cancel_event and discards; return a zeroed volume.
        return [[[0.0] * size for _ in range(size)] for _ in range(size)]

    # Convert to 3D list [z][y][x]
    volume = [[[0.0] * size for _ in range(size)] for _ in range(size)]
    for z in range(size):
        for y in range(size):
            for x in range(size):
                volume[z][y][x] = float(output_host[z * size * size + y * size + x])

    return volume


# ---------------------------------------------------------------------------
# Table builders for GPU buffers + CPU fallback (delegates to generate_volumetric.py)
# ---------------------------------------------------------------------------

def _precompute_value_table(period: int, seed: int) -> list:
    """Precompute value noise table (same as generate_volumetric.py)."""
    from generate_volumetric import _hash_coord
    table = [[[0.0] * period for _ in range(period)] for _ in range(period)]
    for z in range(period):
        for y in range(period):
            for x in range(period):
                table[z][y][x] = _hash_coord(x, y, z, seed)
    return table


def _precompute_worley_table(period: int, seed: int) -> tuple:
    """Precompute Worley noise table — returns (tx, ty, tz) flat lists.

    Same as generate_volumetric.py but returns separate x/y/z arrays.
    """
    from generate_volumetric import _worley_hash_coord
    tx = [[[0.0] * period for _ in range(period)] for _ in range(period)]
    ty = [[[0.0] * period for _ in range(period)] for _ in range(period)]
    tz = [[[0.0] * period for _ in range(period)] for _ in range(period)]
    for z in range(period):
        for y in range(period):
            for x in range(period):
                fx, fy, fz = _worley_hash_coord(x, y, z, seed)
                tx[z][y][x] = fx
                ty[z][y][x] = fy
                tz[z][y][x] = fz
    return (tx, ty, tz)


def _precompute_perlin_table(period: int, seed: int) -> tuple:
    """Precompute Perlin noise table — returns (gx, gy, gz) flat lists.

    Same as generate_volumetric.py but returns separate component arrays.
    """
    from generate_volumetric import _perlin_gradient
    gx = [[[0.0] * period for _ in range(period)] for _ in range(period)]
    gy = [[[0.0] * period for _ in range(period)] for _ in range(period)]
    gz = [[[0.0] * period for _ in range(period)] for _ in range(period)]
    for z in range(period):
        for y in range(period):
            for x in range(period):
                gxi, gyi, gzi = _perlin_gradient(x, y, z, seed)
                gx[z][y][x] = gxi
                gy[z][y][x] = gyi
                gz[z][y][x] = gzi
    return (gx, gy, gz)




def _generate_volume_cpu(size, octaves, base_freq, lacunarity, seed, noise_type, cancel_event=None, voronoi_mode: str = "F1", progress_cb=None):
    """CPU fallback — delegates to the canonical CPU generator in
    generate_volumetric.py (single source of truth for noise sampling)."""
    from generate_volumetric import _generate_volume
    return _generate_volume(size, octaves, base_freq, lacunarity, seed,
                            noise_type, cancel_event, voronoi_mode, progress_cb)
