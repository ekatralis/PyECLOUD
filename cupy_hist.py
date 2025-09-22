 
import cupy as cp
from line_profiler import profile

def compute_hist_gpu(x_mp, wei_mp, bias_x, Dx, Nxg, hist = None):

    # Compute real-valued bin indices
    fi = (x_mp - bias_x) / Dx
    i = cp.floor(fi).astype(cp.int32)  # i = int(fi)
    hx = fi - i                        # remainder

    # Allocate histogram
    if hist is None:
        hist = cp.zeros(Nxg, dtype=cp.float64)

    # Valid indices (avoid overflow at Nxg)
    valid = i < (Nxg - 1)

    # Deposit to i (hist[i] += w * (1 - hx))
    cp.jit.atomic_add(hist, i[valid], wei_mp[valid] * (1 - hx[valid]))

    # Deposit to i+1 (hist[i+1] += w * hx)
    cp.jit.atomic_add(hist, i[valid] + 1, wei_mp[valid] * hx[valid])

    # Handle edge case where i == Nxg (like your else clause)
    edge = i >= (Nxg - 1)
    cp.jit.atomic_add(hist, Nxg - 1, cp.sum(wei_mp[edge]))

    return hist

cuda_src = r'''
extern "C" {

__device__ inline double atomicAdd_double(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    // Fallback for pre-sm_60: CAS loop
    unsigned long long int* addr_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *addr_as_ull, assumed;
    do {
        assumed = old;
        double sum = __longlong_as_double(assumed) + val;
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(sum));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

__global__ void compute_hist_kernel(
    const long long N_mp,
    const double* __restrict__ x_mp,
    const double* __restrict__ wei_mp,
    const double bias_x,
    const double Dx,
    const int Nxg,
    double* __restrict__ hist
){
    // grid-stride loop
    for (long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         p < N_mp;
         p += (long long)blockDim.x * gridDim.x)
    {
        double fi = 1.0 + (x_mp[p] - bias_x) / Dx;
        // Fortran INT: trunc toward zero; assume fi >= 0 in typical use.
        long long i_f = (long long)fi;      // 1-based index
        double hx = fi - (double)i_f;

        long long i0 = i_f - 1;             // 0-based index for hist(i)

        if (i_f < (long long)Nxg) {
            // Add to hist[i0] and hist[i0+1], guarding bounds just in case
            if (i0 >= 0 && i0 < Nxg) {
                atomicAdd_double(&hist[i0], wei_mp[p] * (1.0 - hx));
            }
            long long i1 = i0 + 1;
            if (i1 >= 0 && i1 < Nxg) {
                atomicAdd_double(&hist[i1], wei_mp[p] * hx);
            }
        } else {
            // Spill into last bin hist(Nxg) -> 0-based Nxg-1
            atomicAdd_double(&hist[Nxg - 1], wei_mp[p]);
        }
    }
}

} // extern "C"
'''

compute_hist_kernel = cp.RawKernel(cuda_src, 'compute_hist_kernel')

def compute_hist_rawkernel(x_mp, wei_mp, bias_x, Dx, Nxg, hist=None,
                           threads_per_block=256):
    """
    RawKernel version of compute_hist.
    """
    assert x_mp.dtype == cp.float64 and wei_mp.dtype == cp.float64
    N_mp = x_mp.size

    if hist is None:
        hist = cp.zeros(Nxg, dtype=cp.float64)

    blocks = (N_mp + threads_per_block - 1) // threads_per_block
    # Heuristic cap to avoid oversubscribing tiny workloads
    blocks = int(min(blocks, 65535))

    compute_hist_kernel((blocks,), (threads_per_block,),
                        (cp.int64(N_mp), x_mp, wei_mp,
                         float(bias_x), float(Dx), int(Nxg), hist))
    return hist

# Example:
# x = cp.asarray([...], dtype=cp.float64)
# w = cp.asarray([...], dtype=cp.float64)
# h = compute_hist_rawkernel(x, w, bias_x=..., Dx=..., Nxg=..., hist=None)


def compute_hist_cupy2(x_mp, wei_mp, bias_x, Dx, Nxg, hist=None):
    """
    Vectorized CuPy equivalent of compute_hist, using cp.add.at
    for backward compatibility (no scatter_add needed).
    """
    assert x_mp.dtype == cp.float64 and wei_mp.dtype == cp.float64

    if hist is None:
        hist = cp.zeros(Nxg, dtype=cp.float64)

    fi = 1.0 + (x_mp - bias_x) / Dx
    i_f = fi.astype(cp.int64)          # Fortran int()
    hx  = fi - i_f.astype(cp.float64)
    i0 = i_f - 1                       # 0-based

    # Case: i < Nxg
    m = i_f < Nxg

    # hist[i0] += wei*(1-hx)
    idx0 = i0[m]
    val0 = wei_mp[m] * (1.0 - hx[m])
    keep0 = (idx0 >= 0) & (idx0 < Nxg)
    cp.add.at(hist, idx0[keep0], val0[keep0])

    # hist[i0+1] += wei*hx
    idx1 = (i0 + 1)[m]
    val1 = wei_mp[m] * hx[m]
    keep1 = (idx1 >= 0) & (idx1 < Nxg)
    cp.add.at(hist, idx1[keep1], val1[keep1])

    # Else branch: dump to last bin
    m_tail = ~m
    if m_tail.any():
        hist[-1] += wei_mp[m_tail].sum()

    return hist


def compute_hist_bincount(x_mp, wei_mp, bias_x, Dx, Nxg):
    """
    Fast CuPy version without atomics: reduce-by-key via cp.bincount.
    Matches the Fortran behavior including 'spill into last bin' when i >= Nxg.
    All arrays are float64 to match real*8.
    """
    x = cp.asarray(x_mp,  dtype=cp.float64).ravel()
    w = cp.asarray(wei_mp, dtype=cp.float64).ravel()
    assert x.size == w.size

    fi = 1.0 + (x - bias_x) / Dx
    i_f = fi.astype(cp.int64)                 # Fortran INT (trunc toward zero)
    hx  = fi - i_f.astype(cp.float64)
    i0  = i_f - 1                              # 0-based

    # Main branch: i_f < Nxg  → contribute to i0 and i0+1
    m = i_f < Nxg

    idx0 = i0[m]
    val0 = w[m] * (1.0 - hx[m])

    idx1 = idx0 + 1
    val1 = w[m] * hx[m]

    # Guard lower/upper bounds (Fortran assumes >=1; we clip safely)
    keep0 = (idx0 >= 0) & (idx0 < Nxg)
    keep1 = (idx1 >= 0) & (idx1 < Nxg)

    # Build keys/values for reduction
    keys   = cp.concatenate([idx0[keep0], idx1[keep1]], axis=0)
    values = cp.concatenate([val0[keep0], val1[keep1]], axis=0)

    # Parallel histogram reduce; produces length >= max(keys)+1
    hist = cp.bincount(keys, weights=values, minlength=Nxg).astype(cp.float64, copy=False)

    # Tail: i_f >= Nxg → dump to last bin
    tail = (~m)
    if tail.any():
        hist[-1] += w[tail].sum()

    return hist

# Latestattempt

cuda_src_opt = r'''
extern "C" {

__device__ inline double atomicAdd_double(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    // Fallback for pre-sm_60: CAS loop (works for global or shared memory)
    unsigned long long int* addr_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *addr_as_ull, assumed;
    do {
        assumed = old;
        double sum = __longlong_as_double(assumed) + val;
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(sum));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

__global__ void compute_hist_kernel_shmem(
    const long long N_mp,
    const double* __restrict__ x_mp,
    const double* __restrict__ wei_mp,
    const double bias_x,
    const double invDx,          // precompute 1/Dx on host
    const int Nxg,
    double* __restrict__ hist)
{
    extern __shared__ double sh_hist[]; // size Nxg

    // Zero shared histogram cooperatively
    for (int k = threadIdx.x; k < Nxg; k += blockDim.x) {
        sh_hist[k] = 0.0;
    }
    __syncthreads();

    // Grid-stride loop over particles
    for (long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         p < N_mp;
         p += (long long)blockDim.x * gridDim.x)
    {
        double fi = 1.0 + (x_mp[p] - bias_x) * invDx;
        long long i_f = (long long)fi;      // Fortran INT -> trunc toward zero
        double hx = fi - (double)i_f;
        long long i0 = i_f - 1;             // 0-based index for hist(i)

        if (i_f < (long long)Nxg) {
            // Fast path: if your data never yields negative i0, you can
            // define FAST_ASSUME_IN_RANGE to drop the lower-bound checks.
#ifndef FAST_ASSUME_IN_RANGE
            if (i0 >= 0 && i0 < Nxg) {
                atomicAdd_double(&sh_hist[i0], wei_mp[p] * (1.0 - hx));
            }
            long long i1 = i0 + 1;
            if (i1 >= 0 && i1 < Nxg) {
                atomicAdd_double(&sh_hist[i1], wei_mp[p] * hx);
            }
#else
            atomicAdd_double(&sh_hist[i0],     wei_mp[p] * (1.0 - hx));
            atomicAdd_double(&sh_hist[i0 + 1], wei_mp[p] * hx);
#endif
        } else {
            // Spill into last bin (Fortran hist(Nxg) -> 0-based Nxg-1)
            atomicAdd_double(&sh_hist[Nxg - 1], wei_mp[p]);
        }
    }

    __syncthreads();

    // One global atomic per bin per block
    for (int k = threadIdx.x; k < Nxg; k += blockDim.x) {
        double v = sh_hist[k];
        if (v != 0.0) {
            atomicAdd_double(&hist[k], v);
        }
    }
}

} // extern "C"
'''

compute_hist_kernel_shmem = cp.RawKernel(
    cuda_src_opt, "compute_hist_kernel_shmem"
)

# @profile
def compute_hist_rawkernel_fast(x_mp, wei_mp, bias_x, Dx, Nxg, hist=None,
                                threads_per_block=256, blocks=None,
                                assume_in_range=False):
    """
    Faster RawKernel version with per-block shared-memory histogram.
    - Keeps float64 everywhere (real*8).
    - Great when Nxg is small (e.g., 32–256) and N_mp is large.
    """

    # --- Coerce to device, dtype, shape ---
    # x  = cp.asarray(x_mp,  dtype=cp.float64).ravel(order="C")
    # w  = cp.asarray(wei_mp, dtype=cp.float64).ravel(order="C")
    x = x_mp
    w = wei_mp
    if x.size != w.size:
        raise ValueError(f"x and w must have same length, got {x.size} vs {w.size}")
    N = int(x.size)

    if hist is None:
        hist = cp.zeros(int(Nxg), dtype=cp.float64)
    else:
        hist = cp.asarray(hist, dtype=cp.float64).ravel(order="C")
        if hist.size != int(Nxg):
            raise ValueError(f"hist length ({hist.size}) must equal Nxg ({Nxg})")

    # --- Launch config ---
    tpb = int(threads_per_block)

    if blocks is None:
        # Heuristic: a few blocks per SM, bounded by work size
        try:
            props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
            sms = int(props["multiProcessorCount"])
            # 4 blocks/SM is a reasonable start; adjust if needed
            max_blocks = max(1, 4 * sms)
        except Exception:
            max_blocks = 2048
        blocks = min(max_blocks, (N + tpb - 1) // tpb)
        blocks = max(1, int(min(blocks, 65535)))

    # --- Dynamic shared memory: Nxg * sizeof(double) ---
    shared_bytes = int(Nxg) * 8

    # Optional: compile with fast-path (drop lower-bound guards) if safe for your data
    # (No negatives for fi → i0 >= 0 always.)
    if assume_in_range:
        # Rebuild with FAST_ASSUME_IN_RANGE only once and cache; simple way:
        compute_hist_kernel = cp.RawKernel(
            cuda_src_opt.replace("// define FAST_ASSUME_IN_RANGE", ""),
            "compute_hist_kernel_shmem",
            options=("--std=c++11", "-DFAST_ASSUME_IN_RANGE=1"),
        )
    else:
        compute_hist_kernel = compute_hist_kernel_shmem

    # --- Launch ---
    invDx = float(1.0 / Dx)
    compute_hist_kernel(
        (blocks,), (tpb,),
        (cp.int64(N), x, w, float(bias_x), invDx, int(Nxg), hist),
        shared_mem=shared_bytes
    )
    return hist

cuda_src_warpagg = r'''
extern "C" {

__device__ inline double atomicAdd_double(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    // CAS fallback for pre-sm_60
    unsigned long long int* addr_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *addr_as_ull, assumed;
    do {
        assumed = old;
        double sum = __longlong_as_double(assumed) + val;
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(sum));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

#if __CUDA_ARCH__ >= 700
// Warp-aggregate: group lanes with the same key; only the leader issues one atomic.
__device__ inline void warp_atomic_add_by_key_double(
    int key, double val, double* __restrict__ hist, int Nxg)
{
    unsigned mask = __activemask();
    // Group of lanes in this warp having the same key
    unsigned same = __match_any_sync(mask, key);
    int leader = __ffs(same) - 1;              // first lane in the group

    int lane = threadIdx.x & 31;
    if (lane == leader) {
        // Reduce values from all lanes in the group
        double sum = 0.0;
        unsigned m = same;
        while (m) {
            int src = __ffs(m) - 1;
            sum += __shfl_sync(same, val, src);
            m &= m - 1;
        }
        if (key >= 0 && key < Nxg) {
            atomicAdd_double(&hist[key], sum);
        }
    }
}
#endif

__global__ void compute_hist_kernel_warpagg(
    const long long N_mp,
    const double* __restrict__ x_mp,
    const double* __restrict__ wei_mp,
    const double bias_x,
    const double invDx,
    const int Nxg,
    double* __restrict__ hist)
{
    for (long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         p < N_mp;
         p += (long long)blockDim.x * gridDim.x)
    {
        double fi = 1.0 + (x_mp[p] - bias_x) * invDx;
        long long i_f = (long long)fi;     // Fortran INT (trunc toward zero)
        double hx = fi - (double)i_f;
        int i0 = (int)(i_f - 1);           // 0-based

#if __CUDA_ARCH__ >= 700
        if (i_f < (long long)Nxg) {
            // two updates: (i0, wei*(1-hx)) and (i0+1, wei*hx)
            double w = wei_mp[p];
            warp_atomic_add_by_key_double(i0,     w * (1.0 - hx), hist, Nxg);
            warp_atomic_add_by_key_double(i0 + 1, w * hx,         hist, Nxg);
        } else {
            // spill to last bin
            warp_atomic_add_by_key_double(Nxg - 1, wei_mp[p], hist, Nxg);
        }
#else
        // Fallback on pre-Volta: plain global atomics
        if (i_f < (long long)Nxg) {
            if (i0 >= 0 && i0 < Nxg) atomicAdd_double(&hist[i0],     wei_mp[p] * (1.0 - hx));
            int i1 = i0 + 1;
            if (i1 >= 0 && i1 < Nxg) atomicAdd_double(&hist[i1],     wei_mp[p] * hx);
        } else {
            atomicAdd_double(&hist[Nxg - 1], wei_mp[p]);
        }
#endif
    }
}
} // extern "C"
'''

compute_hist_kernel_warpagg = cp.RawKernel(cuda_src_warpagg, "compute_hist_kernel_warpagg")

def compute_hist_rawkernel_warpagg(x_mp, wei_mp, bias_x, Dx, Nxg,
                                   hist=None, threads_per_block=256, blocks=None):
    # x = cp.asarray(x_mp, dtype=cp.float64).ravel()
    # w = cp.asarray(wei_mp, dtype=cp.float64).ravel()
    x = x_mp
    w = wei_mp
    assert x.size == w.size
    N = int(x.size)

    if hist is None:
        hist = cp.zeros(int(Nxg), dtype=cp.float64)
    # else:
    #     hist = cp.asarray(hist, dtype=cp.float64).ravel()
    #     assert hist.size == int(Nxg)

    if blocks is None:
        # ~4 blocks per SM is a good start
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        sms = int(props["multiProcessorCount"])
        blocks = min(max(1, 4 * sms), (N + threads_per_block - 1)//threads_per_block)
        blocks = max(1, int(min(blocks, 65535)))

    invDx = float(1.0 / Dx)
    compute_hist_kernel_warpagg((blocks,), (int(threads_per_block),),
                                (cp.int64(N), x, w, float(bias_x), invDx, int(Nxg), hist))
    return hist


cuda_src_warppriv = r'''
extern "C" {

__device__ inline double atomicAdd_double(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    unsigned long long int* addr_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *addr_as_ull, assumed;
    do {
        assumed = old;
        double sum = __longlong_as_double(assumed) + val;
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(sum));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

__global__ void compute_hist_kernel_warppriv(
    const long long N_mp,
    const double* __restrict__ x_mp,
    const double* __restrict__ wei_mp,
    const double bias_x,
    const double invDx,
    const int Nxg,
    double* __restrict__ hist)
{
    const int WARP = 32;
    int lane = threadIdx.x & (WARP - 1);
    int warp_id = threadIdx.x >> 5;                            // warp index in block
    int warps_per_block = blockDim.x >> 5;

    extern __shared__ double shmem[];                          // size: warps_per_block * Nxg
    double* sh = shmem + warp_id * Nxg;                        // this warp's private histogram

    // Zero this warp's histogram
    for (int k = lane; k < Nxg; k += WARP) sh[k] = 0.0;
    __syncwarp();

    // Accumulate into warp-private histogram
    for (long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         p < N_mp;
         p += (long long)blockDim.x * gridDim.x)
    {
        double fi = 1.0 + (x_mp[p] - bias_x) * invDx;
        long long i_f = (long long)fi;
        double hx = fi - (double)i_f;
        int i0 = (int)(i_f - 1);

        if (i_f < (long long)Nxg) {
            // If negatives can't happen in your data, drop the bounds for speed.
            if (i0 >= 0 && i0 < Nxg) atomicAdd_double(&sh[i0],     wei_mp[p] * (1.0 - hx));
            int i1 = i0 + 1;
            if (i1 >= 0 && i1 < Nxg) atomicAdd_double(&sh[i1],     wei_mp[p] * hx);
        } else {
            atomicAdd_double(&sh[Nxg - 1], wei_mp[p]);
        }
    }

    __syncthreads(); // make sure all warps finished writing their private slices

    // Reduce warp histograms to global: each thread handles some bins
    for (int k = threadIdx.x; k < Nxg; k += blockDim.x) {
        double sum = 0.0;
        // sum across warps in this block
        for (int w = 0; w < warps_per_block; ++w) {
            sum += shmem[w * Nxg + k];
        }
        if (sum != 0.0) atomicAdd_double(&hist[k], sum);
    }
}
} // extern "C"
'''

compute_hist_kernel_warppriv = cp.RawKernel(cuda_src_warppriv, "compute_hist_kernel_warppriv")

def compute_hist_rawkernel_warppriv(x_mp, wei_mp, bias_x, Dx, Nxg,
                                    hist=None, threads_per_block=256, blocks=None):
    # x = cp.asarray(x_mp, dtype=cp.float64).ravel()
    # w = cp.asarray(wei_mp, dtype=cp.float64).ravel()
    x = x_mp
    w = wei_mp
    # assert x.size == w.size
    N = int(x.size)

    if hist is None:
        hist = cp.zeros(int(Nxg), dtype=cp.float64)
    # else:
    #     hist = cp.asarray(hist, dtype=cp.float64).ravel()
    #     assert hist.size == int(Nxg)

    if blocks is None:
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        sms = int(props["multiProcessorCount"])
        blocks = min(max(1, 4 * sms), (N + threads_per_block - 1)//threads_per_block)
        blocks = max(1, int(min(blocks, 65535)))

    warps_per_block = int(threads_per_block // 32)
    shared_bytes = int(Nxg) * warps_per_block * 8  # doubles

    invDx = float(1.0 / Dx)
    compute_hist_kernel_warppriv((blocks,), (int(threads_per_block),),
                                 (cp.int64(N), x, w, float(bias_x), invDx, int(Nxg), hist),
                                 shared_mem=shared_bytes)
    return hist


# Split into two histograms
# -------------------------------
# CUDA sources
# -------------------------------
cuda_src_pass1 = r'''
extern "C" {

__device__ inline double atomicAdd_double(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    // CAS fallback for pre-sm_60; works in shared or global
    unsigned long long int* addr_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *addr_as_ull, assumed;
    do {
        assumed = old;
        double sum = __longlong_as_double(assumed) + val;
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(sum));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

__global__ void hist_accumulate_blocks(
    const long long N_mp,
    const double* __restrict__ x_mp,
    const double* __restrict__ wei_mp,
    const double bias_x,
    const double invDx,
    const int Nxg,
    double* __restrict__ tmp   // shape: [gridDim.x, Nxg]
){
    extern __shared__ double sh[]; // Nxg doubles

    // zero shared histogram
    for (int k = threadIdx.x; k < Nxg; k += blockDim.x) {
        sh[k] = 0.0;
    }
    __syncthreads();

    // grid-stride over particles
    for (long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         p < N_mp;
         p += (long long)blockDim.x * gridDim.x)
    {
        double fi = 1.0 + (x_mp[p] - bias_x) * invDx;
        long long i_f = (long long)fi;   // Fortran INT (trunc toward zero)
        double hx = fi - (double)i_f;
        long long i0 = i_f - 1;          // 0-based

        if (i_f < (long long)Nxg) {
            // (optional) bounds guards; remove if you know i0>=0
            if (i0 >= 0 && i0 < Nxg) {
                atomicAdd_double(&sh[i0],     wei_mp[p] * (1.0 - hx));
            }
            long long i1 = i0 + 1;
            if (i1 >= 0 && i1 < Nxg) {
                atomicAdd_double(&sh[i1],     wei_mp[p] * hx);
            }
        } else {
            atomicAdd_double(&sh[Nxg - 1], wei_mp[p]);
        }
    }
    __syncthreads();

    // write this block's histogram to tmp[blockIdx.x, :]
    double* out = tmp + (long long)blockIdx.x * Nxg;
    for (int k = threadIdx.x; k < Nxg; k += blockDim.x) {
        out[k] = sh[k];
    }
}
} // extern "C"
'''

cuda_src_pass2 = r'''
extern "C" __global__
void hist_merge_blocks(
    const double* __restrict__ tmp, // [numBlocks, Nxg]
    const int numBlocks,
    const int Nxg,
    double* __restrict__ hist       // [Nxg]
){
    // each thread reduces one or more bins across all blocks
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < Nxg; k += blockDim.x * gridDim.x) {
        double sum = 0.0;
        for (int b = 0; b < numBlocks; ++b) {
            sum += tmp[(long long)b * Nxg + k];
        }
        hist[k] += sum;
    }
}
'''

hist_accumulate_blocks = cp.RawKernel(cuda_src_pass1, "hist_accumulate_blocks")
hist_merge_blocks      = cp.RawKernel(cuda_src_pass2, "hist_merge_blocks")

# -------------------------------
# Python wrapper
# -------------------------------
def compute_hist_rawkernel_split(x_mp, wei_mp, bias_x, Dx, Nxg,
                                 hist=None,
                                 threads_per_block=256,
                                 blocks=None,
                                 max_blocks_for_tmp=4096):
    """
    Two-pass histogram:
      Pass 1: per-block shared histogram -> tmp[block, bin]
      Pass 2: merge tmp over blocks -> hist
    All float64; exact same binning/tail as your Fortran.
    """
    # --- inputs ---
    x = cp.asarray(x_mp, dtype=cp.float64).ravel(order="C")
    w = cp.asarray(wei_mp, dtype=cp.float64).ravel(order="C")
    if x.size != w.size:
        raise ValueError(f"x and w must have same length (got {x.size} vs {w.size})")
    N = int(x.size)
    Nxg = int(Nxg)

    if hist is None:
        hist = cp.zeros(Nxg, dtype=cp.float64)
    else:
        hist = cp.asarray(hist, dtype=cp.float64).ravel(order="C")
        if hist.size != Nxg:
            raise ValueError(f"hist length ({hist.size}) must equal Nxg ({Nxg})")

    # --- launch config for pass 1 ---
    tpb = int(threads_per_block)
    # plenty of blocks, but cap to limit tmp memory
    calc_blocks = max(1, (N + tpb - 1) // tpb)
    if blocks is None:
        # heuristic: min(calc_blocks, 4*SMs, cap)
        try:
            props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
            sms = int(props["multiProcessorCount"])
            blk_cap = 4 * sms
        except Exception:
            blk_cap = 2048
        blocks = min(calc_blocks, blk_cap, max_blocks_for_tmp)
    else:
        blocks = int(min(blocks, calc_blocks))

    # tmp size: blocks * Nxg * 8 bytes
    tmp = cp.empty((blocks, Nxg), dtype=cp.float64)

    # dynamic shared memory per block
    shared_bytes = Nxg * 8

    invDx = float(1.0 / Dx)

    # --- pass 1: build per-block histograms into tmp ---
    hist_accumulate_blocks(
        (blocks,), (tpb,),
        (cp.int64(N), x, w, float(bias_x), invDx, Nxg, tmp),
        shared_mem=shared_bytes
    )

    # --- pass 2: merge tmp into hist ---
    # each thread reduces a subset of bins across all blocks
    tpb2 = 256
    blk2 = max(1, (Nxg + tpb2 - 1) // tpb2)
    hist_merge_blocks(
        (blk2,), (tpb2,),
        (tmp, int(blocks), Nxg, hist)
    )

    return hist

cuda_src_opt = r'''
extern "C" {

__device__ __forceinline__ double atomicAdd_double(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    // CAS fallback for pre-sm_60 (works in global/shared)
    unsigned long long* addr_as_ull = (unsigned long long*)address;
    unsigned long long old = *addr_as_ull, assumed;
    do {
        assumed = old;
        double sum = __longlong_as_double(assumed) + val;
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(sum));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

// Define to drop lower-bound guards if i0>=0 is guaranteed by your data
// #define ASSUME_IN_RANGE 1

// Define to use warp-aggregated flush for the tail bin (one atomic per warp)
// (Requires Volta+ for __shfl_down_sync)
#define WARP_AGGREGATE_TAIL 1

__global__ void compute_hist_kernel_opt(
    const int N_mp,
    const double* __restrict__ x_mp,
    const double* __restrict__ wei_mp,
    const double bias_x,
    const double invDx,      // 1.0 / Dx
    const int Nxg,
    double* __restrict__ hist)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Accumulate tail-bin contributions locally to reduce atomics.
    double tail_local = 0.0;

    for (int p = tid; p < N_mp; p += stride)
    {
        // Compute real index and interpolation fraction
        double fi = 1.0 + (x_mp[p] - bias_x) * invDx;

        // Fortran INT truncates toward zero; for nonnegative fi it matches floor
        int i_f = (int)fi;           // 1-based
        double hx = fi - (double)i_f;
        int i0 = i_f - 1;            // 0-based

        double w = wei_mp[p];

        if (i_f < Nxg) {
#if ASSUME_IN_RANGE
            atomicAdd_double(&hist[i0    ], w * (1.0 - hx));
            atomicAdd_double(&hist[i0 + 1], w * hx);
#else
            // branchless bounds check via unsigned cast
            if ((unsigned)i0 < (unsigned)Nxg)
                atomicAdd_double(&hist[i0], w * (1.0 - hx));
            int i1 = i0 + 1;
            if ((unsigned)i1 < (unsigned)Nxg)
                atomicAdd_double(&hist[i1], w * hx);
#endif
        } else {
            // Defer tail updates; flush after the loop
            tail_local += w;
        }
    }

    // Flush tail: either one atomic per thread, or one per warp (fewer atomics)
#if WARP_AGGREGATE_TAIL && (__CUDA_ARCH__ >= 700)
    unsigned mask = __activemask();
    // reduce tail_local within the warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        tail_local += __shfl_down_sync(mask, tail_local, offset);
    }
    if ((threadIdx.x & 31) == 0 && tail_local != 0.0) {
        atomicAdd_double(&hist[Nxg - 1], tail_local);
    }
#else
    if (tail_local != 0.0) {
        atomicAdd_double(&hist[Nxg - 1], tail_local);
    }
#endif
}

} // extern "C"
'''.strip()

# kernel_opt = cp.RawKernel(
#     cuda_src_opt, "compute_hist_kernel_opt",
#     options=("--std=c++14", "-O3", "-Xptxas=-O3,-dlcm=ca")
# )

mod = cp.RawModule(
    code=cuda_src_opt,
    backend='nvcc',
    options=(
        "-O3",
        "--std=c++14",
        "-Xptxas=-O3,-dlcm=ca",   # optional: cache hint to ptxas
    ),
)
kernel_opt = mod.get_function("compute_hist_kernel_opt")


def compute_hist_rawkernel_opt(x_mp, wei_mp, bias_x, Dx, Nxg,
                               hist=None, threads_per_block=256, blocks=None):
    # x = cp.asarray(x_mp,  dtype=cp.float64).ravel(order="C")
    # w = cp.asarray(wei_mp, dtype=cp.float64).ravel(order="C")
    x = x_mp
    w = wei_mp
    # if x.size != w.size:
    #     raise ValueError(f"x and w must have same length (got {x.size} vs {w.size})")
    N = x.size
    # Nxg = int(Nxg)

    if hist is None:
        hist = cp.zeros(Nxg, dtype=cp.float64)
    # else:
    #     hist = cp.asarray(hist, dtype=cp.float64).ravel(order="C")
    #     if hist.size != Nxg:
    #         raise ValueError(f"hist length ({hist.size}) must equal Nxg ({Nxg})")

    tpb = int(threads_per_block)

    if blocks is None:
        # a few blocks per SM; cap by available work
        try:
            props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
            sms = int(props["multiProcessorCount"])
            max_blocks = 4 * sms
        except Exception:
            max_blocks = 2048
        blocks = min(max_blocks, (N + tpb - 1) // tpb)
        blocks = max(1, int(min(blocks, 65535)))

    invDx = float(1.0 / Dx)

    kernel_opt((blocks,), (tpb,),
               (int(N), x, w, float(bias_x), invDx, int(Nxg), hist))
    return hist


# def compute_hist_rawkernel2(x_mp, wei_mp, bias_x, Dx, Nxg, hist=None,
#                            threads_per_block=256):
#     """
#     RawKernel version of compute_hist.
#     """
#     assert x_mp.dtype == cp.float64 and wei_mp.dtype == cp.float64
#     N_mp = x_mp.size

#     if hist is None:
#         hist = cp.zeros(Nxg, dtype=cp.float64)

#     blocks = (N_mp + threads_per_block - 1) // threads_per_block
#     # Heuristic cap to avoid oversubscribing tiny workloads
#     blocks = int(min(blocks, 65535))

#     kernel_opt((blocks,), (threads_per_block,),
#                         (cp.int64(N_mp), x_mp, wei_mp,
#                          float(bias_x), float(1/Dx), int(Nxg), hist))
#     return hist

cuda_src_invDx = r'''
extern "C" {

__device__ inline double atomicAdd_double(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    // Fallback for pre-sm_60: CAS loop
    unsigned long long int* addr_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *addr_as_ull, assumed;
    do {
        assumed = old;
        double sum = __longlong_as_double(assumed) + val;
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(sum));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

__global__ void compute_hist_kernel(
    const long long N_mp,
    const double* __restrict__ x_mp,
    const double* __restrict__ wei_mp,
    const double bias_x,
    const double invDx,
    const int Nxg,
    double* __restrict__ hist
){
    // grid-stride loop
    for (long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         p < N_mp;
         p += (long long)blockDim.x * gridDim.x)
    {
        double fi = 1.0 + (x_mp[p] - bias_x) * invDx;
        // Fortran INT: trunc toward zero; assume fi >= 0 in typical use.
        long long i_f = (long long)fi;      // 1-based index
        double hx = fi - (double)i_f;

        long long i0 = i_f - 1;             // 0-based index for hist(i)

        if (i_f < (long long)Nxg) {
            // Add to hist[i0] and hist[i0+1], guarding bounds just in case
            if (i0 >= 0 && i0 < Nxg) {
                atomicAdd_double(&hist[i0], wei_mp[p] * (1.0 - hx));
            }
            long long i1 = i0 + 1;
            if (i1 >= 0 && i1 < Nxg) {
                atomicAdd_double(&hist[i1], wei_mp[p] * hx);
            }
        } else {
            // Spill into last bin hist(Nxg) -> 0-based Nxg-1
            atomicAdd_double(&hist[Nxg - 1], wei_mp[p]);
        }
    }
}

} // extern "C"
'''

compute_hist_kernel_invDx = cp.RawKernel(cuda_src_invDx, 'compute_hist_kernel')

def compute_hist_rawkernel_invDx(x_mp, wei_mp, bias_x, Dx, Nxg, hist=None,
                           threads_per_block=256):
    """
    RawKernel version of compute_hist.
    """
    assert x_mp.dtype == cp.float64 and wei_mp.dtype == cp.float64
    N_mp = x_mp.size

    if hist is None:
        hist = cp.zeros(Nxg, dtype=cp.float64)

    blocks = (N_mp + threads_per_block - 1) // threads_per_block
    # Heuristic cap to avoid oversubscribing tiny workloads
    blocks = int(min(blocks, 65535))

    compute_hist_kernel_invDx((blocks,), (threads_per_block,),
                        (cp.int64(N_mp), x_mp, wei_mp,
                         float(bias_x), float(1/Dx), int(Nxg), hist))
    return hist

cuda_src_GPU_only = r'''
extern "C" {

__device__ inline double atomicAdd_double(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    // Fallback for pre-sm_60: CAS loop
    unsigned long long int* addr_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *addr_as_ull, assumed;
    do {
        assumed = old;
        double sum = __longlong_as_double(assumed) + val;
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(sum));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

__global__ void compute_hist_kernel(
    const long long N_mp,
    const double* __restrict__ x_mp,
    const double* __restrict__ wei_mp,
    const double* __restrict__ bias_x_p,  // <-- device scalar
    const double* __restrict__ Dx_p,      // <-- device scalar
    const int*    __restrict__ Nxg_p,     // <-- device scalar
    double* __restrict__ hist
){  
    const double bias_x = *bias_x_p;
    const double Dx     = *Dx_p;
    const int    Nxg    = *Nxg_p;
    // grid-stride loop
    for (long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         p < N_mp;
         p += (long long)blockDim.x * gridDim.x)
    {
        double fi = 1.0 + (x_mp[p] - bias_x) / Dx;
        // Fortran INT: trunc toward zero; assume fi >= 0 in typical use.
        long long i_f = (long long)fi;      // 1-based index
        double hx = fi - (double)i_f;

        long long i0 = i_f - 1;             // 0-based index for hist(i)

        if (i_f < (long long)Nxg) {
            // Add to hist[i0] and hist[i0+1], guarding bounds just in case
            if (i0 >= 0 && i0 < Nxg) {
                atomicAdd_double(&hist[i0], wei_mp[p] * (1.0 - hx));
            }
            long long i1 = i0 + 1;
            if (i1 >= 0 && i1 < Nxg) {
                atomicAdd_double(&hist[i1], wei_mp[p] * hx);
            }
        } else {
            // Spill into last bin hist(Nxg) -> 0-based Nxg-1
            atomicAdd_double(&hist[Nxg - 1], wei_mp[p]);
        }
    }
}

} // extern "C"
'''

compute_hist_kernel_2 = cp.RawKernel(cuda_src_GPU_only, 'compute_hist_kernel')

def compute_hist_rawkernel_notrans(x_mp, wei_mp, bias_x, Dx, Nxg, hist=None,
                           threads_per_block=256):
    """
    RawKernel version of compute_hist.
    """
    assert x_mp.dtype == cp.float64 and wei_mp.dtype == cp.float64
    N_mp = x_mp.size
    # Nxg = hist.size

    if hist is None:
        hist = cp.zeros(Nxg, dtype=cp.float64)

    blocks = (N_mp + threads_per_block - 1) // threads_per_block
    # Heuristic cap to avoid oversubscribing tiny workloads
    blocks = int(min(blocks, 65535))
    # print(N_mp.dtype)
    # print(x_mp.dtype)
    # print(bias_x)
    # print(bias_x.dtype)
    # print(Dx)
    # print(Dx.dtype)
    # print(Nxg)
    # print(Nxg.dtype)
    compute_hist_kernel((blocks,), (threads_per_block,),
                        (cp.int64(N_mp), x_mp, wei_mp,
                         bias_x, Dx, Nxg, hist))
    return hist

def compute_hist_rawkernel_notrans2(x_mp, wei_mp, bias_x, Dx, Nxg, hist=None,
                           threads_per_block=256):
    """
    RawKernel version of compute_hist.
    """
    assert x_mp.dtype == cp.float64 and wei_mp.dtype == cp.float64
    N_mp = x_mp.size
    # Nxg = hist.size

    if hist is None:
        hist = cp.zeros(Nxg, dtype=cp.float64)

    blocks = (N_mp + threads_per_block - 1) // threads_per_block
    # Heuristic cap to avoid oversubscribing tiny workloads
    blocks = int(min(blocks, 65535))
    # print(N_mp.dtype)
    # print(x_mp.dtype)
    # print(bias_x)
    # print(bias_x.dtype)
    # print(Dx)
    # print(Dx.dtype)
    # print(Nxg)
    # print(Nxg.dtype)
    compute_hist_kernel_2((blocks,), (threads_per_block,),
                        (cp.int64(N_mp), x_mp, wei_mp,
                         bias_x, Dx, Nxg, hist))
    return hist
