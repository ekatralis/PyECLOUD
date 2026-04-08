import cupy as cp
import numpy as np


_UPDATE_SEG_IMPACT_SRC = r"""
extern "C" {

__device__ inline double atomicAdd_double(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    unsigned long long int* addr_as_ull =
        (unsigned long long int*)address;
    unsigned long long int old = *addr_as_ull, assumed;
    do {
        assumed = old;
        const double sum = __longlong_as_double(assumed) + val;
        old = atomicCAS(
            addr_as_ull,
            assumed,
            __double_as_longlong(sum)
        );
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

__global__ void update_seg_impact_kernel(
    const long long N_mp,
    const int* __restrict__ i_seg_mp,
    const double* __restrict__ wei_mp,
    const int N_seg,
    double* __restrict__ hist
){
    for (long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         p < N_mp;
         p += (long long)blockDim.x * gridDim.x)
    {
        const int i = i_seg_mp[p];
        if (i >= 0 && i < N_seg) {
            atomicAdd_double(&hist[i], wei_mp[p]);
        }
    }
}

} // extern "C"
"""


_update_seg_impact_kernel = cp.RawKernel(
    _UPDATE_SEG_IMPACT_SRC,
    "update_seg_impact_kernel",
)


# def _as_1d_device_array(arr, dtype, name):
#     out = cp.asarray(arr, dtype=dtype)
#     if out.ndim != 1:
#         raise ValueError(f"{name} must be a 1D array.")
#     return out


def update_seg_impact(
    i_seg_mp,
    wei_mp,
    hist=None,
    N_seg=None,
    *,
    threads_per_block=256,
    stream=None,
):
    """
    GPU port of the Fortran ``update_seg_impact`` routine.

    Parameters
    ----------
    i_seg_mp : array_like, shape (N_mp,)
        Segment indices for each macro-particle. Valid entries are in
        ``[0, N_seg - 1]``.
    wei_mp : array_like, shape (N_mp,)
        Weights to accumulate into ``hist``.
    hist : cupy.ndarray, optional
        Output histogram. If omitted, a new device histogram is allocated.
    N_seg : int, optional
        Histogram length. Required when ``hist`` is not provided.

    Returns
    -------
    cupy.ndarray
        The updated histogram on the GPU.
    """
    # i_seg_mp = _as_1d_device_array(i_seg_mp, cp.int32, "i_seg_mp")
    # wei_mp = _as_1d_device_array(wei_mp, cp.float64, "wei_mp")

    if i_seg_mp.size != wei_mp.size:
        raise ValueError("i_seg_mp and wei_mp must have the same length.")

    if hist is None:
        if N_seg is None:
            raise ValueError("N_seg must be provided when hist is None.")
        hist = cp.zeros(int(N_seg), dtype=cp.float64)
    else:
        # hist = _as_1d_device_array(hist, cp.float64, "hist")
        if N_seg is not None and int(N_seg) != int(hist.size):
            raise ValueError("N_seg must match hist.size when both are given.")

    n_seg = int(hist.size)
    n_mp = int(i_seg_mp.size)

    if n_mp == 0:
        return hist

    blocks = (n_mp + threads_per_block - 1) // threads_per_block
    blocks = max(1, min(int(blocks), 65535))

    args = (
        np.int64(n_mp),
        i_seg_mp,
        wei_mp,
        np.int32(n_seg),
        hist,
    )

    if stream is None:
        _update_seg_impact_kernel((blocks,), (threads_per_block,), args)
    else:
        with stream:
            _update_seg_impact_kernel(
                (blocks,),
                (threads_per_block,),
                args,
                stream=stream,
            )

    return hist
