import cupy as cp
import numpy as np


_IMPACT_POINT_AND_NORMAL_SRC = r"""
extern "C" __global__
void impact_point_and_normal_kernel(
    const double* __restrict__ x_in,
    const double* __restrict__ y_in,
    const double* __restrict__ x_out,
    const double* __restrict__ y_out,
    const double* __restrict__ Vx,
    const double* __restrict__ Vy,
    const double* __restrict__ Nx,
    const double* __restrict__ Ny,
    const int N_impacts,
    const int N_edg,
    const double resc_fac,
    double* __restrict__ x_int,
    double* __restrict__ y_int,
    double* __restrict__ z_int,
    double* __restrict__ Nx_int,
    double* __restrict__ Ny_int,
    int* __restrict__ i_found
){
    for (int i_imp = blockDim.x * blockIdx.x + threadIdx.x;
         i_imp < N_impacts;
         i_imp += blockDim.x * gridDim.x)
    {
        double t_min_curr = 1.0;
        int i_found_curr = -1;

        const double x_in_curr = x_in[i_imp];
        const double y_in_curr = y_in[i_imp];
        const double x_out_curr = x_out[i_imp];
        const double y_out_curr = y_out[i_imp];

        for (int ii = 0; ii < N_edg; ++ii) {
            const double den =
                (y_out_curr - y_in_curr) * (Vx[ii + 1] - Vx[ii]) +
                (x_in_curr - x_out_curr) * (Vy[ii + 1] - Vy[ii]);

            double t_border;
            if (den == 0.0) {
                t_border = -2.0;
            } else {
                t_border =
                    ((y_out_curr - y_in_curr) * (x_in_curr - Vx[ii]) +
                     (x_in_curr - x_out_curr) * (y_in_curr - Vy[ii])) / den;
            }

            if (t_border >= 0.0 && t_border <= 1.0) {
                const double t_denom =
                    Nx[ii] * (x_out_curr - x_in_curr) +
                    Ny[ii] * (y_out_curr - y_in_curr);

                if (t_denom != 0.0) {
                    const double t_ii =
                        (Nx[ii] * (Vx[ii] - x_in_curr) +
                         Ny[ii] * (Vy[ii] - y_in_curr)) / t_denom;

                    if (t_ii >= 0.0 && t_ii < t_min_curr) {
                        t_min_curr = t_ii;
                        i_found_curr = ii;
                    }
                }
            }
        }

        t_min_curr = resc_fac * t_min_curr;
        x_int[i_imp] = t_min_curr * x_out_curr + (1.0 - t_min_curr) * x_in_curr;
        y_int[i_imp] = t_min_curr * y_out_curr + (1.0 - t_min_curr) * y_in_curr;
        z_int[i_imp] = 0.0;

        if (i_found_curr >= 0) {
            Nx_int[i_imp] = Nx[i_found_curr];
            Ny_int[i_imp] = Ny[i_found_curr];
            i_found[i_imp] = i_found_curr;
        } else {
            Nx_int[i_imp] = 0.0;
            Ny_int[i_imp] = 0.0;
        }
    }
}
"""


_impact_point_and_normal_kernel = cp.RawKernel(
    _IMPACT_POINT_AND_NORMAL_SRC,
    "impact_point_and_normal_kernel",
    options=("--std=c++11", ), #, "--fmad=false" for more accurate floating-point results, but may reduce performance
    backend="nvrtc",
)


# def _as_f64_device_array(arr, name):
#     out = cp.asarray(arr, dtype=cp.float64)
#     if out.ndim != 1:
#         raise ValueError(f"{name} must be a 1D array.")
#     return out


def impact_point_and_normal(
    x_in,
    y_in,
    z_in,
    x_out,
    y_out,
    z_out,
    Vx,
    Vy,
    Nx,
    Ny,
    N_edg=None,
    resc_fac=0.99,
    *,
    threads_per_block=256,
    stream=None,
):
    # x_in = _as_f64_device_array(x_in, "x_in")
    # y_in = _as_f64_device_array(y_in, "y_in")
    # z_in = _as_f64_device_array(z_in, "z_in")
    # x_out = _as_f64_device_array(x_out, "x_out")
    # y_out = _as_f64_device_array(y_out, "y_out")
    # z_out = _as_f64_device_array(z_out, "z_out")
    # Vx = _as_f64_device_array(Vx, "Vx")
    # Vy = _as_f64_device_array(Vy, "Vy")
    # Nx = _as_f64_device_array(Nx, "Nx")
    # Ny = _as_f64_device_array(Ny, "Ny")

    n_impacts = int(x_in.size)
    if any(arr.size != n_impacts for arr in (y_in, z_in, x_out, y_out, z_out)):
        raise ValueError("Input trajectory arrays must all have the same length.")

    if Vx.size != Vy.size:
        raise ValueError("Vx and Vy must have the same length.")
    if Nx.size != Ny.size:
        raise ValueError("Nx and Ny must have the same length.")

    if N_edg is None:
        N_edg = int(Nx.size)
    else:
        N_edg = int(N_edg)

    if N_edg < 0:
        raise ValueError("N_edg must be non-negative.")
    if Vx.size < N_edg + 1 or Vy.size < N_edg + 1:
        raise ValueError("Vx and Vy must contain at least N_edg + 1 entries.")
    if Nx.size < N_edg or Ny.size < N_edg:
        raise ValueError("Nx and Ny must contain at least N_edg entries.")

    x_int = cp.zeros(n_impacts, dtype=cp.float64)
    y_int = cp.zeros(n_impacts, dtype=cp.float64)
    z_int = cp.zeros(n_impacts, dtype=cp.float64)
    Nx_int = cp.zeros(n_impacts, dtype=cp.float64)
    Ny_int = cp.zeros(n_impacts, dtype=cp.float64)
    i_found = cp.zeros(n_impacts, dtype=cp.int32)

    if n_impacts == 0:
        return x_int, y_int, z_int, Nx_int, Ny_int, i_found

    blocks = (n_impacts + threads_per_block - 1) // threads_per_block
    blocks = max(1, min(blocks, 65535))

    args = (
        x_in,
        y_in,
        x_out,
        y_out,
        Vx,
        Vy,
        Nx,
        Ny,
        np.int32(n_impacts),
        np.int32(N_edg),
        np.float64(resc_fac),
        x_int,
        y_int,
        z_int,
        Nx_int,
        Ny_int,
        i_found,
    )

    if stream is None:
        _impact_point_and_normal_kernel((blocks,), (threads_per_block,), args)
    else:
        with stream:
            _impact_point_and_normal_kernel(
                (blocks,),
                (threads_per_block,),
                args,
                stream=stream,
            )

    return x_int, y_int, z_int, Nx_int, Ny_int, i_found


# CUDA C kernel
_is_outside_convex_src = r'''
extern "C" __global__
void is_outside_convex_kernel(
    const double* __restrict__ x_mp,
    const double* __restrict__ y_mp,
    const int N_mp,
    const double* __restrict__ Vx,
    const double* __restrict__ Vy,
    const int N_edg,         // number of edges; Vx/Vy must have N_edg+1 entries (last==first)
    const double cx,
    const double cy,
    unsigned char* __restrict__ out_mask  // 0 = inside, 1 = outside
){
    // grid-stride loop for arbitrary N_mp
    for (int idx = blockDim.x * blockIdx.x + threadIdx.x;
         idx < N_mp;
         idx += blockDim.x * gridDim.x)
    {
        const double x = x_mp[idx];
        const double y = y_mp[idx];

        // Elliptical early-inclusion test (same as original)
        int inside = (((x/cx)*(x/cx) + (y/cy)*(y/cy)) <= 1.0) ? 1 : 0;

        if (!inside) {
            inside = 1;
            int ii = 0;
            while (inside == 1 && ii < N_edg) {
                const double vx0 = Vx[ii];
                const double vy0 = Vy[ii];
                const double vx1 = Vx[ii+1];
                const double vy1 = Vy[ii+1];

                // Cross product > 0 means point is left of edge (for CCW polygon)
                const double cross =
                    ( (y - vy0) * (vx1 - vx0) ) - ( (x - vx0) * (vy1 - vy0) );
                inside = (cross > 0.0) ? 1 : 0;
                ++ii;
            }
        }

        out_mask[idx] = (unsigned char)(!inside); // 1 = outside, 0 = inside
    }
}
''';

_is_outside_convex_kernel = cp.RawKernel(
    _is_outside_convex_src, "is_outside_convex_kernel"
)

# @profile
def is_outside_convex(x_mp, y_mp, Vx, Vy, cx, cy, N_edg=None, *,
                          threads_per_block=256, stream=None):
    """
    GPU version of is_outside_convex.
    Parameters
    ----------
    x_mp, y_mp : array_like (N,), float64
        Query point coordinates.
    Vx, Vy     : array_like (M,), float64
        Polygon vertices; must have length N_edg+1 with last vertex==first.
    cx, cy     : float
        Ellipse radii used for the early-inclusion test.
    N_edg      : int, optional
        Number of edges (defaults to len(Vx)-1).
    Returns
    -------
    out : cupy.ndarray (N,), bool
        True for points outside; False for inside.
    """
    # Move data to device in expected dtypes
    # x_d  = cp.asarray(x_mp, dtype=cp.float64)
    # y_d  = cp.asarray(y_mp, dtype=cp.float64)
    # Vx_d = cp.asarray(Vx,   dtype=cp.float64)
    # Vy_d = cp.asarray(Vy,   dtype=cp.float64)
    x_d = x_mp
    y_d = y_mp
    Vx_d = Vx
    Vy_d = Vy

    if N_edg is None:
        N_edg = int(Vx_d.size) - 1
    else:
        N_edg = int(N_edg)

    if Vx_d.size != Vy_d.size:
        raise ValueError("Vx and Vy must have the same length.")
    if Vx_d.size < 2 or N_edg + 1 > Vx_d.size:
        raise ValueError("Vx/Vy must have at least N_edg+1 vertices (with last==first).")

    N_mp = int(x_d.size)
    if y_d.size != N_mp:
        raise ValueError("x_mp and y_mp must have the same length.")

    out_u8 = cp.empty(N_mp, dtype=cp.uint8)

    blocks = (N_mp + threads_per_block - 1) // threads_per_block
    # a modest cap to avoid excessive empty blocks on tiny inputs
    blocks = max(1, min(blocks, 65535))

    args = (
        x_d, y_d, np.int32(N_mp),
        Vx_d, Vy_d, np.int32(N_edg),
        np.float64(cx), np.float64(cy),
        out_u8,
    )

    if stream is None:
        _is_outside_convex_kernel((blocks,), (threads_per_block,), args)
    else:
        with stream:
            _is_outside_convex_kernel((blocks,), (threads_per_block,), args, stream=stream)

    return out_u8.view(cp.bool_)  # boolean mask: True = outside, False = inside