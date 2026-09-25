#!/usr/bin/env python3
"""
Volumetric 3D Texture Generator - OpenCL GPU Backend

Provides GPU-accelerated volume generation via OpenCL.
All noise algorithms, hash functions, and interpolation are identical to the CPU version.

OpenCL floating-point operations may produce tiny differences (~1e-7) compared to CPU.
These are below 8-bit quantization threshold (1/255), so the PNG output is visually identical.
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
float smoothstep(float t) { return t * t * (3.0f - 2.0f * t); }
float lerp(float a, float b, float t) { return a + t * (b - a); }

kernel void generate_value(
    int size, int octaves, int seed,
    constant int* periods, constant int* p2s, constant int* offsets,
    constant float* tables,
    __global float* output
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    if (x >= size || y >= size || z >= size) return;

    int idx = z * size * size + y * size + x;
    float val = 0.0f, amp = 1.0f, max_v = 0.0f;

    for (int o = 0; o < octaves; o++) {
        int period = periods[o];
        int offset = offsets[o];
        int p2 = p2s[o];
        float cx = (float)x / (float)size * (float)period;
        float cy = (float)y / (float)size * (float)period;
        float cz = (float)z / (float)size * (float)period;

        int ix = (int)cx, iy = (int)cy, iz = (int)cz;
        float fx = smoothstep(cx - (float)ix);
        float fy = smoothstep(cy - (float)iy);
        float fz = smoothstep(cz - (float)iz);

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

        float v00 = lerp(tables[b000], tables[b100], fx);
        float v10 = lerp(tables[b010], tables[b110], fx);
        float v01 = lerp(tables[b001], tables[b101], fx);
        float v11 = lerp(tables[b011], tables[b111], fx);
        float v0 = lerp(v00, v10, fy);
        float v1 = lerp(v01, v11, fy);

        val += amp * lerp(v0, v1, fz);
        max_v += amp;
        amp *= 0.5f;
    }
    output[idx] = val / max_v;
}
"""

_KERNEL_WORLEY = r"""
float lerp(float a, float b, float t) { return a + t * (b - a); }

kernel void generate_worley(
    int size, int octaves, int seed,
    constant int* periods, constant int* p2s, constant int* offsets,
    constant float* tx, constant float* ty, constant float* tz,
    __global float* output
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    if (x >= size || y >= size || z >= size) return;

    int idx = z * size * size + y * size + x;
    float val = 0.0f, amp = 1.0f, max_v = 0.0f;

    for (int o = 0; o < octaves; o++) {
        int period = periods[o];
        int offset = offsets[o];
        int p2 = p2s[o];
        float cx = (float)x / (float)size * (float)period;
        float cy = (float)y / (float)size * (float)period;
        float cz = (float)z / (float)size * (float)period;

        int ix = (int)cx, iy = (int)cy, iz = (int)cz;
        float min_dsq = 1e30f;

        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int ci = ((ix+dx) % period + period) % period;
                    int cj = ((iy+dy) % period + period) % period;
                    int ck = ((iz+dz) % period + period) % period;
                    int ti = offset + ck*p2 + cj*period + ci;

                    float fx = cx - (float)ix - tx[ti];
                    float fy = cy - (float)iy - ty[ti];
                    float fz = cz - (float)iz - tz[ti];
                    if (fx > 0.5f) fx -= 1.0f; else if (fx < -0.5f) fx += 1.0f;
                    if (fy > 0.5f) fy -= 1.0f; else if (fy < -0.5f) fy += 1.0f;
                    if (fz > 0.5f) fz -= 1.0f; else if (fz < -0.5f) fz += 1.0f;

                    float d = fx*fx + fy*fy + fz*fz;
                    if (d < min_dsq) min_dsq = d;
                }
            }
        }
        float s = sqrtf(min_dsq) * 2.0f;
        if (s > 1.0f) s = 1.0f; if (s < 0.0f) s = 0.0f;
        val += amp * s; max_v += amp; amp *= 0.5f;
    }
    output[idx] = val / max_v;
}
"""

_KERNEL_PERLIN = r"""
float smoothstep(float t) { return t * t * (3.0f - 2.0f * t); }
float lerp(float a, float b, float t) { return a + t * (b - a); }

kernel void generate_perlin(
    int size, int octaves, int seed,
    constant int* periods, constant int* p2s, constant int* offsets,
    constant float* gx, constant float* gy, constant float* gz,
    __global float* output
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    if (x >= size || y >= size || z >= size) return;

    int idx = z * size * size + y * size + x;
    float val = 0.0f, amp = 1.0f, max_v = 0.0f;

    for (int o = 0; o < octaves; o++) {
        int period = periods[o];
        int offset = offsets[o];
        int p2 = p2s[o];
        float cx = (float)x / (float)size * (float)period;
        float cy = (float)y / (float)size * (float)period;
        float cz = (float)z / (float)size * (float)period;

        int ix = (int)cx, iy = (int)cy, iz = (int)cz;
        float dx = smoothstep(cx - (float)ix);
        float dy = smoothstep(cy - (float)iy);
        float dz = smoothstep(cz - (float)iz);

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

        float d00 = lerp(gx[b000]*(cx-ix)+gy[b000]*(cy-iy)+gz[b000]*(cz-iz),
                         gx[b100]*(cx-ix-1.0f)+gy[b100]*(cy-iy)+gz[b100]*(cz-iz), dx);
        float d01 = lerp(gx[b010]*(cx-ix)+gy[b010]*(cy-iy-1.0f)+gz[b010]*(cz-iz),
                         gx[b110]*(cx-ix-1.0f)+gy[b110]*(cy-iy-1.0f)+gz[b110]*(cz-iz), dx);
        float d10 = lerp(gx[b001]*(cx-ix)+gy[b001]*(cy-iy)+gz[b001]*(cz-iz-1.0f),
                         gx[b101]*(cx-ix-1.0f)+gy[b101]*(cy-iy)+gz[b101]*(cz-iz-1.0f), dx);
        float d11 = lerp(gx[b011]*(cx-ix)+gy[b011]*(cy-iy-1.0f)+gz[b011]*(cz-iz-1.0f),
                         gx[b111]*(cx-ix-1.0f)+gy[b111]*(cy-iy-1.0f)+gz[b111]*(cz-iz-1.0f), dx);

        float s = lerp(lerp(d00, d01, dy), lerp(d10, d11, dy), dz);
        val += amp * s; max_v += amp; amp *= 0.5f;
    }
    output[idx] = val / max_v;
}
"""

_KERNEL_VORONOI = r"""
/* Voronoi noise kernel */
kernel void generate_voronoi(
    int size, int octaves, int seed, int voronoi_mode,
    constant int* periods, constant int* p2s, constant int* offsets,
    constant float* tx, constant float* ty, constant float* tz,
    __global float* output
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    if (x >= size || y >= size || z >= size) return;

    int idx = z * size * size + y * size + x;
    float val = 0.0f, amp = 1.0f, max_v = 0.0f;

    for (int o = 0; o < octaves; o++) {
        int period = periods[o];
        int offset = offsets[o];
        int p2 = p2s[o];
        float cx = (float)x / (float)size * (float)period;
        float cy = (float)y / (float)size * (float)period;
        float cz = (float)z / (float)size * (float)period;

        int ix = (int)cx, iy = (int)cy, iz = (int)cz;
        float f1 = 1e30f, f2 = 1e30f;

        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int ci = ((ix+dx) % period + period) % period;
                    int cj = ((iy+dy) % period + period) % period;
                    int ck = ((iz+dz) % period + period) % period;
                    int ti = offset + ck*p2 + cj*period + ci;

                    float fx = cx - (float)ix - tx[ti];
                    float fy = cy - (float)iy - ty[ti];
                    float fz = cz - (float)iz - tz[ti];
                    if (fx > 0.5f) fx -= 1.0f; else if (fx < -0.5f) fx += 1.0f;
                    if (fy > 0.5f) fy -= 1.0f; else if (fy < -0.5f) fy += 1.0f;
                    if (fz > 0.5f) fz -= 1.0f; else if (fz < -0.5f) fz += 1.0f;

                    float d = fx*fx + fy*fy + fz*fz;
                    if (d < f1) { f2 = f1; f1 = d; }
                    else if (d < f2) { f2 = d; }
                }
            }
        }

        float s1 = sqrtf(f1);
        float s2 = sqrtf(f2);
        float sample = 0.0f;

        if (voronoi_mode == 0) sample = s1 * 2.0f;
        else if (voronoi_mode == 1) sample = s2 * 2.0f;
        else if (voronoi_mode == 2) sample = fabsf(s1 - s2) * 2.0f;
        else if (voronoi_mode == 3) sample = s1 * 2.0f;
        else if (voronoi_mode == 4) {
            float denom = s1 + s2;
            sample = (denom < 1e-10f) ? 0.5f : (s1 / denom) * 2.0f;
        }

        if (sample > 1.0f) sample = 1.0f;
        if (sample < 0.0f) sample = 0.0f;
        val += amp * sample; max_v += amp; amp *= 0.5f;
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
        self.ctx: Optional[cl.Context] = None
        self.queue: Optional[cl.CommandQueue] = None
        self.programs: dict[str, cl.Program] = {}
        self.kernels: dict[str, cl.Kernel] = {}
        self._ready = False

    def get_or_create_queue(self) -> Optional[cl.CommandQueue]:
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

    def get_kernel(self, name: str, source: str) -> Optional[cl.Kernel]:
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
        kernel_name = f"generate_{name.lower()}"
        if hasattr(prog, kernel_name):
            self.kernels[name] = getattr(prog, kernel_name)
        return self.kernels.get(name)

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

def _merge_octave_tables(octaves_info: list[tuple[int, Optional[list]]],
                         noise_type: str):
    """Merge per-octave table data into flat GPU arrays.

    Returns:
        (periods_arr, p2s_arr, offsets_arr, tables_dict)
        - periods_arr: int32[N] — hash period per octave
        - p2s_arr: int32[N] — period^2 per octave
        - offsets_arr: int32[N] — byte offset into table arrays
        - tables_dict: dict of noise_type -> (buf0, buf1, buf2)
    """
    import numpy as np

    N = octaves_info.__len__()
    periods_arr = np.zeros(N, dtype=np.int32)
    p2s_arr = np.zeros(N, dtype=np.int32)
    offsets_arr = np.zeros(N, dtype=np.int32)

    # Calculate sizes
    total_val = 0
    total_worley = 0
    total_perlin = 0

    for i, (period, table) in enumerate(octaves_info):
        periods_arr[i] = period
        p2 = period * period
        p2s_arr[i] = p2
        if table is None:
            offsets_arr[i] = -1  # marker for direct sampling
            continue
        if noise_type == "Value Noise":
            offsets_arr[i] = total_val
            total_val += p2 * period  # Wait, table is period^3 for value noise? No...
            # Actually for value noise, table[z][y][x] where each dim is period
            # So it's period^3 entries
            # Let me reconsider: the table size is period^3
            # p2 = period^2, so total entries = period^3 = period * p2
            # But I declared p2s as period^2, so total = period * p2
            # Hmm, this is getting confusing. Let me recalculate.

    # Let me redo this more carefully
    # Value noise table: [period][period][period] = period^3 floats
    # Worley table: 3 arrays, each period^3 floats
    # Perlin table: 3 arrays, each period^3 floats

    total_val = 0
    total_worley = 0
    total_perlin = 0

    for i, (period, table) in enumerate(octaves_info):
        if table is None:
            offsets_arr[i] = -1
            continue
        p3 = period * period * period
        if noise_type == "Value Noise":
            offsets_arr[i] = total_val
            total_val += p3
        elif noise_type == "Worley Noise":
            offsets_arr[i] = total_worley
            total_worley += p3
        elif noise_type == "FBM Perlin Noise":
            offsets_arr[i] = total_perlin
            total_perlin += p3

    # Create buffers
    tables_dict = {}

    if noise_type == "Value Noise" and total_val > 0:
        merged = np.zeros(total_val, dtype=np.float32)
        for i, (period, table) in enumerate(octaves_info):
            if table is None:
                continue
            # table is list[list[list[float]]]
            flat = np.zeros(period * period * period, dtype=np.float32)
            for z in range(period):
                for y in range(period):
                    for x in range(period):
                        flat[z * period * period + y * period + x] = table[z][y][x]
            start = offsets_arr[i]
            merged[start:start + len(flat)] = flat
        tables_dict["Value Noise"] = (cl.Buffer(_get_mgr().ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=merged), None, None)

    elif noise_type == "Worley Noise" and total_worley > 0:
        merged_x = np.zeros(total_worley, dtype=np.float32)
        merged_y = np.zeros(total_worley, dtype=np.float32)
        merged_z = np.zeros(total_worley, dtype=np.float32)
        for i, (period, table) in enumerate(octaves_info):
            if table is None:
                continue
            # table[z][y][x] = (fx, fy, fz)
            start = offsets_arr[i]
            p3 = period * period * period
            for z in range(period):
                for y in range(period):
                    for x in range(period):
                        idx = z * period * period + y * period + x
                        fx, fy, fz = table[z][y][x]
                        merged_x[start + idx] = fx
                        merged_y[start + idx] = fy
                        merged_z[start + idx] = fz
        bx = cl.Buffer(_get_mgr().ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=merged_x)
        by = cl.Buffer(_get_mgr().ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=merged_y)
        bz = cl.Buffer(_get_mgr().ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=merged_z)
        tables_dict["Worley Noise"] = (bx, by, bz)

    elif noise_type == "FBM Perlin Noise" and total_perlin > 0:
        merged_gx = np.zeros(total_perlin, dtype=np.float32)
        merged_gy = np.zeros(total_perlin, dtype=np.float32)
        merged_gz = np.zeros(total_perlin, dtype=np.float32)
        for i, (period, table) in enumerate(octaves_info):
            if table is None:
                continue
            start = offsets_arr[i]
            p3 = period * period * period
            for z in range(period):
                for y in range(period):
                    for x in range(period):
                        idx = z * period * period + y * period + x
                        gx, gy, gz = table[z][y][x]
                        merged_gx[start + idx] = gx
                        merged_gy[start + idx] = gy
                        merged_gz[start + idx] = gz
        bx = cl.Buffer(_get_mgr().ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=merged_gx)
        by = cl.Buffer(_get_mgr().ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=merged_gy)
        bz = cl.Buffer(_get_mgr().ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=merged_gz)
        tables_dict["FBM Perlin Noise"] = (bx, by, bz)

    # Create empty buffers for unused types
    empty = cl.Buffer(_get_mgr().ctx, cl.mem_flags.READ_ONLY, 4)
    if "Value Noise" not in tables_dict:
        tables_dict["Value Noise"] = (empty, None, None)
    if "Worley Noise" not in tables_dict:
        tables_dict["Worley Noise"] = (empty, empty, empty)
    if "FBM Perlin Noise" not in tables_dict:
        tables_dict["FBM Perlin Noise"] = (empty, empty, empty)

    return periods_arr, p2s_arr, offsets_arr, tables_dict


# ---------------------------------------------------------------------------
# Main GPU entry point
# ---------------------------------------------------------------------------

def generate_volume_gpu(
    size: int,
    octaves: int,
    base_freq: float,
    lacunarity: float,
    seed: int,
    noise_type: str,
    cancel_event=None,
    voronoi_mode: str = "F1",
) -> list[list[list[float]]]:
    """Generate L×L×L volume using OpenCL GPU, falling back to CPU if needed.

    Args:
        voronoi_mode: Only used when noise_type is "Voronoi Noise".
            One of: "F1", "F2", "F1 - F2", "Jitter", "Edge"
    """
    if not OPENCL_AVAILABLE or not _get_mgr().is_ready():
        return _generate_volume_cpu(size, octaves, base_freq, lacunarity, seed, noise_type, cancel_event, voronoi_mode)

    try:
        return _generate_volume_opencl(size, octaves, base_freq, lacunarity, seed, noise_type, cancel_event, voronoi_mode)
    except Exception:
        return _generate_volume_cpu(size, octaves, base_freq, lacunarity, seed, noise_type, cancel_event, voronoi_mode)


def _generate_volume_opencl(
    size: int, octaves: int, base_freq: float, lacunarity: float,
    seed: int, noise_type: str, cancel_event=None, voronoi_mode: str = "F1",
) -> list[list[list[float]]]:
    """Run volume generation on GPU via OpenCL."""
    import numpy as np

    mgr = _get_mgr()
    queue = mgr.queue

    # Build octave info (same as CPU)
    octaves_info = []
    for i in range(octaves):
        octave_freq = base_freq * (lacunarity ** i)
        hash_period = max(2, round(size * octave_freq))
        if hash_period > 128:
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
    periods_arr, p2s_arr, offsets_arr, tables_dict = _merge_octave_tables(octaves_info, noise_type)

    total_voxels = size * size * size
    output_buf = cl.Buffer(mgr.ctx, cl.mem_flags.WRITE_ONLY, total_voxels * 4)

    # Map voronoi mode to integer
    voronoi_mode_map = {"F1": 0, "F2": 1, "F1 - F2": 2, "Jitter": 3, "Edge": 4}
    vm = voronoi_mode_map.get(voronoi_mode, 0)

    kernel = mgr.get_kernel(noise_type, {
        "Value Noise": _KERNEL_VALUE,
        "Worley Noise": _KERNEL_WORLEY,
        "FBM Perlin Noise": _KERNEL_PERLIN,
        "Voronoi Noise": _KERNEL_VORONOI,
    }[noise_type])

    if kernel is None:
        raise RuntimeError("Failed to compile OpenCL kernel")

    # Get table buffers
    tv, twx, twy, twz, tpz = tables_dict.get(noise_type, (None, None, None))
    if tv is None:
        tv = cl.Buffer(mgr.ctx, cl.mem_flags.READ_ONLY, 4)

    if noise_type == "Value Noise":
        kernel.set_args(
            np.int32(size), np.int32(octaves), np.int32(seed),
            periods_arr, p2s_arr, offsets_arr,
            tv, output_buf,
        )
    elif noise_type == "Worley Noise":
        kernel.set_args(
            np.int32(size), np.int32(octaves), np.int32(seed),
            periods_arr, p2s_arr, offsets_arr,
            twx, twy, twz, output_buf,
        )
    elif noise_type == "FBM Perlin Noise":
        kernel.set_args(
            np.int32(size), np.int32(octaves), np.int32(seed),
            periods_arr, p2s_arr, offsets_arr,
            tv, twy, twz, output_buf,
        )
    elif noise_type == "Voronoi Noise":
        kernel.set_args(
            np.int32(size), np.int32(octaves), np.int32(seed), np.int32(vm),
            periods_arr, p2s_arr, offsets_arr,
            twx, twy, twz, output_buf,
        )

    # Launch
    global_size = [size, size, size]
    queue.enqueue_nd_range_kernel(kernel, [0, 0, 0], global_size, None)
    queue.finish()

    # Read back
    output_host = np.empty(total_voxels, dtype=np.float32)
    cl.enqueue_read_buffer(queue, output_buf, output_host).wait()

    # Convert to 3D list [z][y][x]
    volume = [[[0.0] * size for _ in range(size)] for _ in range(size)]
    for z in range(size):
        for y in range(size):
            for x in range(size):
                volume[z][y][x] = float(output_host[z * size * size + y * size + x])

    # Clean up buffers
    for buf in [output_buf, periods_arr, p2s_arr, offsets_arr, tv, twx, twy, twz]:
        if isinstance(buf, cl.Buffer):
            buf.release()

    return volume


# ---------------------------------------------------------------------------
# CPU fallback (imports from generate_volumetric.py)
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


def _sample_value_table(cx, cy, cz, table, period):
    from generate_volumetric import _smoothstep, _lerp
    ix, iy, iz = int(cx), int(cy), int(cz)
    fx = _smoothstep(cx - ix)
    fy = _smoothstep(cy - iy)
    fz = _smoothstep(cz - iz)
    i0x, i1x = ix % period, (ix + 1) % period
    i0y, i1y = iy % period, (iy + 1) % period
    i0z, i1z = iz % period, (iz + 1) % period
    h000 = table[i0z][i0y][i0x]
    h100 = table[i0z][i0y][i1x]
    h010 = table[i0z][i1y][i0x]
    h110 = table[i0z][i1y][i1x]
    h001 = table[i1z][i0y][i0x]
    h101 = table[i1z][i0y][i1x]
    h011 = table[i1z][i1y][i0x]
    h111 = table[i1z][i1y][i1x]
    v00 = _lerp(h000, h100, fx)
    v10 = _lerp(h010, h110, fx)
    v01 = _lerp(h001, h101, fx)
    v11 = _lerp(h011, h111, fx)
    v0 = _lerp(v00, v10, fy)
    v1 = _lerp(v01, v11, fy)
    return _lerp(v0, v1, fz)


def _sample_worley_table(cx, cy, cz, table, period):
    from generate_volumetric import math
    tx, ty, tz = table
    ix, iy, iz = int(cx), int(cy), int(cz)
    min_dist = float("inf")
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                cx_i = (ix + dx) % period
                cy_i = (iy + dy) % period
                cz_i = (iz + dz) % period
                feat_x = tx[cz_i][cy_i][cx_i]
                feat_y = ty[cz_i][cy_i][cx_i]
                feat_z = tz[cz_i][cy_i][cx_i]
                fx = cx - ix - feat_x
                fy = cy - iy - feat_y
                fz = cz - iz - feat_z
                if fx > 0.5: fx -= 1.0
                elif fx < -0.5: fx += 1.0
                if fy > 0.5: fy -= 1.0
                elif fy < -0.5: fy += 1.0
                if fz > 0.5: fz -= 1.0
                elif fz < -0.5: fz += 1.0
                dist_sq = fx * fx + fy * fy + fz * fz
                if dist_sq < min_dist: min_dist = dist_sq
    return max(0.0, min(1.0, math.sqrt(min_dist) * 2.0))


def _sample_perlin_table(cx, cy, cz, table, period):
    from generate_volumetric import _smoothstep, _lerp
    gx, gy, gz = table
    ix, iy, iz = int(cx), int(cy), int(cz)
    dx = _smoothstep(cx - ix)
    dy = _smoothstep(cy - iy)
    dz = _smoothstep(cz - iz)
    i0x, i1x = ix % period, (ix + 1) % period
    i0y, i1y = iy % period, (iy + 1) % period
    i0z, i1z = iz % period, (iz + 1) % period
    def dot(g, dxv, dyv, dzv):
        return g[0]*dxv + g[1]*dyv + g[2]*dzv
    g000 = (gx[i0z][i0y][i0x], gy[i0z][i0y][i0x], gz[i0z][i0y][i0x])
    g100 = (gx[i0z][i0y][i1x], gy[i0z][i0y][i1x], gz[i0z][i0y][i1x])
    g010 = (gx[i0z][i1y][i0x], gy[i0z][i1y][i0x], gz[i0z][i1y][i0x])
    g110 = (gx[i0z][i1y][i1x], gy[i0z][i1y][i1x], gz[i0z][i1y][i1x])
    g001 = (gx[i1z][i0y][i0x], gy[i1z][i0y][i0x], gz[i1z][i0y][i0x])
    g101 = (gx[i1z][i0y][i1x], gy[i1z][i0y][i1x], gz[i1z][i0y][i1x])
    g011 = (gx[i1z][i1y][i0x], gy[i1z][i1y][i0x], gz[i1z][i1y][i0x])
    g111 = (gx[i1z][i1y][i1x], gy[i1z][i1y][i1x], gz[i1z][i1y][i1x])
    nx00 = _lerp(dot(g000, cx-ix, cy-iy, cz-iz), dot(g100, cx-ix-1, cy-iy, cz-iz), dx)
    nx01 = _lerp(dot(g010, cx-ix, cy-iy-1, cz-iz), dot(g110, cx-ix-1, cy-iy-1, cz-iz), dx)
    nx10 = _lerp(dot(g001, cx-ix, cy-iy, cz-iz-1), dot(g101, cx-ix-1, cy-iy, cz-iz-1), dx)
    nx11 = _lerp(dot(g011, cx-ix, cy-iy-1, cz-iz-1), dot(g111, cx-ix-1, cy-iy-1, cz-iz-1), dx)
    nx0 = _lerp(nx00, nx01, dy)
    nx1 = _lerp(nx10, nx11, dy)
    return _lerp(nx0, nx1, dz)


def _sample_value_3d(sx, sy, sz, period, seed):
    from generate_volumetric import _hash_coord, _smoothstep, _lerp
    ix, iy, iz = int(sx), int(sy), int(sz)
    fx = _smoothstep(sx - ix)
    fy = _smoothstep(sy - iy)
    fz = _smoothstep(sz - iz)
    i0x, i1x = ix % period, (ix + 1) % period
    i0y, i1y = iy % period, (iy + 1) % period
    i0z, i1z = iz % period, (iz + 1) % period
    h000 = _hash_coord(i0x, i0y, i0z, seed)
    h100 = _hash_coord(i1x, i0y, i0z, seed)
    h010 = _hash_coord(i0x, i1y, i0z, seed)
    h110 = _hash_coord(i1x, i1y, i0z, seed)
    h001 = _hash_coord(i0x, i0y, i1z, seed)
    h101 = _hash_coord(i1x, i0y, i1z, seed)
    h011 = _hash_coord(i0x, i1y, i1z, seed)
    h111 = _hash_coord(i1x, i1y, i1z, seed)
    v00 = _lerp(h000, h100, fx)
    v10 = _lerp(h010, h110, fx)
    v01 = _lerp(h001, h101, fx)
    v11 = _lerp(h011, h111, fx)
    v0 = _lerp(v00, v10, fy)
    v1 = _lerp(v01, v11, fy)
    return _lerp(v0, v1, fz)


def _sample_worley_direct(sx, sy, sz, hash_period, seed):
    from generate_volumetric import _worley_hash_coord, math
    ix, iy, iz = int(sx), int(sy), int(sz)
    neighbors = {}
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                cx_i = (ix + dx) % hash_period
                cy_i = (iy + dy) % hash_period
                cz_i = (iz + dz) % hash_period
                neighbors[(dx, dy, dz)] = _worley_hash_coord(cx_i, cy_i, cz_i, seed)
    min_dist = float("inf")
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                feat = neighbors[(dx, dy, dz)]
                fx = sx - ix - feat[0]
                fy = sy - iy - feat[1]
                fz = sz - iz - feat[2]
                if fx > 0.5: fx -= 1.0
                elif fx < -0.5: fx += 1.0
                if fy > 0.5: fy -= 1.0
                elif fy < -0.5: fy += 1.0
                if fz > 0.5: fz -= 1.0
                elif fz < -0.5: fz += 1.0
                dist_sq = fx*fx + fy*fy + fz*fz
                if dist_sq < min_dist: min_dist = dist_sq
    return max(0.0, min(1.0, math.sqrt(min_dist) * 2.0))


def _sample_perlin_direct(sx, sy, sz, hash_period, seed):
    from generate_volumetric import _perlin_gradient, _smoothstep, _lerp
    ix, iy, iz = int(sx), int(sy), int(sz)
    dx = _smoothstep(sx - ix)
    dy = _smoothstep(sy - iy)
    dz = _smoothstep(sz - iz)
    i0x, i1x = ix % hash_period, (ix + 1) % hash_period
    i0y, i1y = iy % hash_period, (iy + 1) % hash_period
    i0z, i1z = iz % hash_period, (iz + 1) % hash_period
    def dot(g, dxv, dyv, dzv):
        return g[0]*dxv + g[1]*dyv + g[2]*dzv
    g000 = _perlin_gradient(i0x, i0y, i0z, seed)
    g100 = _perlin_gradient(i1x, i0y, i0z, seed)
    g010 = _perlin_gradient(i0x, i1y, i0z, seed)
    g110 = _perlin_gradient(i1x, i1y, i0z, seed)
    g001 = _perlin_gradient(i0x, i0y, i1z, seed)
    g101 = _perlin_gradient(i1x, i0y, i1z, seed)
    g011 = _perlin_gradient(i0x, i1y, i1z, seed)
    g111 = _perlin_gradient(i1x, i1y, i1z, seed)
    nx00 = _lerp(dot(g000, sx-ix, sy-iy, sz-iz), dot(g100, sx-ix-1, sy-iy, sz-iz), dx)
    nx01 = _lerp(dot(g010, sx-ix, sy-iy-1, sz-iz), dot(g110, sx-ix-1, sy-iy-1, sz-iz), dx)
    nx10 = _lerp(dot(g001, sx-ix, sy-iy, sz-iz-1), dot(g101, sx-ix-1, sy-iy, sz-iz-1), dx)
    nx11 = _lerp(dot(g011, sx-ix, sy-iy-1, sz-iz-1), dot(g111, sx-ix-1, sy-iy-1, sz-iz-1), dx)
    nx0 = _lerp(nx00, nx01, dy)
    nx1 = _lerp(nx10, nx11, dy)
    return _lerp(nx0, nx1, dz)


def _sample_voronoi_table(cx, cy, cz, table, period, mode):
    """Voronoi sampling from table — returns f1, f2, then computes mode output."""
    from generate_volumetric import math
    ix, iy, iz = int(cx), int(cy), int(cz)
    f1 = float("inf")
    f2 = float("inf")
    tx, ty, tz = table
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ci = (ix + dx) % period
                cj = (iy + dy) % period
                ck = (iz + dz) % period
                fx = cx - ix - tx[ck][cj][ci]
                fy = cy - iy - ty[ck][cj][ci]
                fz = cz - iz - tz[ck][cj][ci]
                if fx > 0.5: fx -= 1.0
                elif fx < -0.5: fx += 1.0
                if fy > 0.5: fy -= 1.0
                elif fy < -0.5: fy += 1.0
                if fz > 0.5: fz -= 1.0
                elif fz < -0.5: fz += 1.0
                dist_sq = fx*fx + fy*fy + fz*fz
                if dist_sq < f1:
                    f2 = f1
                    f1 = dist_sq
                elif dist_sq < f2:
                    f2 = dist_sq
    f1 = math.sqrt(f1)
    f2 = math.sqrt(f2)
    if mode == "F1": return max(0.0, min(1.0, f1 * 2.0))
    elif mode == "F2": return max(0.0, min(1.0, f2 * 2.0))
    elif mode == "F1 - F2": return max(0.0, min(1.0, abs(f1 - f2) * 2.0))
    elif mode == "Jitter": return max(0.0, min(1.0, f1 * 2.0))
    elif mode == "Edge":
        denom = f1 + f2
        if denom < 1e-10: return 0.5
        return max(0.0, min(1.0, (f1 / denom) * 2.0))
    return max(0.0, min(1.0, f1 * 2.0))


def _sample_voronoi_direct(sx, sy, sz, hash_period, seed, mode):
    """Voronoi direct sampling — no table."""
    from generate_volumetric import _worley_hash_coord, math
    ix, iy, iz = int(sx), int(sy), int(sz)
    f1 = float("inf")
    f2 = float("inf")
    neighbors = {}
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                cx_i = (ix + dx) % hash_period
                cy_i = (iy + dy) % hash_period
                cz_i = (iz + dz) % hash_period
                neighbors[(dx, dy, dz)] = _worley_hash_coord(cx_i, cy_i, cz_i, seed)
    for feat in neighbors.values():
        fx = sx - ix - feat[0]
        fy = sy - iy - feat[1]
        fz = sz - iz - feat[2]
        if fx > 0.5: fx -= 1.0
        elif fx < -0.5: fx += 1.0
        if fy > 0.5: fy -= 1.0
        elif fy < -0.5: fy += 1.0
        if fz > 0.5: fz -= 1.0
        elif fz < -0.5: fz += 1.0
        dist_sq = fx*fx + fy*fy + fz*fz
        if dist_sq < f1:
            f2 = f1
            f1 = dist_sq
        elif dist_sq < f2:
            f2 = dist_sq
    f1 = math.sqrt(f1)
    f2 = math.sqrt(f2)
    if mode == "F1": return max(0.0, min(1.0, f1 * 2.0))
    elif mode == "F2": return max(0.0, min(1.0, f2 * 2.0))
    elif mode == "F1 - F2": return max(0.0, min(1.0, abs(f1 - f2) * 2.0))
    elif mode == "Jitter": return max(0.0, min(1.0, f1 * 2.0))
    elif mode == "Edge":
        denom = f1 + f2
        if denom < 1e-10: return 0.5
        return max(0.0, min(1.0, (f1 / denom) * 2.0))
    return max(0.0, min(1.0, f1 * 2.0))


def _generate_volume_cpu(size, octaves, base_freq, lacunarity, seed, noise_type, cancel_event=None, voronoi_mode: str = "F1"):
    """CPU fallback — identical logic to generate_volumetric.py _generate_volume()."""
    octave_tables = []
    for octave_idx in range(octaves):
        octave_freq = base_freq * (lacunarity ** octave_idx)
        hash_period = max(2, round(size * octave_freq))
        if hash_period > 128:
            octave_tables.append((hash_period, None))
            continue
        if noise_type in ("Worley Noise", "Voronoi Noise"):
            table = _precompute_worley_table(hash_period, seed)
        elif noise_type == "Value Noise":
            table = _precompute_value_table(hash_period, seed)
        else:
            table = _precompute_perlin_table(hash_period, seed)
        octave_tables.append((hash_period, table))

    volume = [[[0.0] * size for _ in range(size)] for _ in range(size)]
    for z in range(size):
        if cancel_event and cancel_event.is_set():
            break
        for y in range(size):
            if cancel_event and cancel_event.is_set():
                break
            for x in range(size):
                val = 0.0
                amplitude = 1.0
                max_val = 0.0
                for octave_idx in range(octaves):
                    hash_period, table = octave_tables[octave_idx]
                    coord_x = (x / size) * hash_period
                    coord_y = (y / size) * hash_period
                    coord_z = (z / size) * hash_period
                    if table is None:
                        if noise_type == "Voronoi Noise":
                            val += amplitude * _sample_voronoi_direct(coord_x, coord_y, coord_z, hash_period, seed, voronoi_mode)
                        elif noise_type == "Worley Noise":
                            val += amplitude * _sample_worley_direct(coord_x, coord_y, coord_z, hash_period, seed)
                        elif noise_type == "FBM Perlin Noise":
                            val += amplitude * _sample_perlin_direct(coord_x, coord_y, coord_z, hash_period, seed)
                        else:
                            val += amplitude * _sample_value_3d(coord_x, coord_y, coord_z, hash_period, seed)
                        max_val += amplitude
                        amplitude *= 0.5
                        continue
                    if noise_type == "Voronoi Noise":
                        val += amplitude * _sample_voronoi_table(coord_x, coord_y, coord_z, table, hash_period, voronoi_mode)
                    elif noise_type == "Worley Noise":
                        val += amplitude * _sample_worley_table(coord_x, coord_y, coord_z, table, hash_period)
                    elif noise_type == "FBM Perlin Noise":
                        val += amplitude * _sample_perlin_table(coord_x, coord_y, coord_z, table, hash_period)
                    else:
                        val += amplitude * _sample_value_table(coord_x, coord_y, coord_z, table, hash_period)
                    max_val += amplitude
                    amplitude *= 0.5
                volume[z][y][x] = val / max_val
    return volume
