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
            i_found[i_imp] = -1;
        }
    }
}
"""


_impact_point_and_normal_kernel = cp.RawKernel(
    _IMPACT_POINT_AND_NORMAL_SRC,
    "impact_point_and_normal_kernel",
    options=("--std=c++11",),
    backend="nvrtc",
)


def _as_f64_device_array(arr, name):
    out = cp.asarray(arr, dtype=cp.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    return out


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
    x_in = _as_f64_device_array(x_in, "x_in")
    y_in = _as_f64_device_array(y_in, "y_in")
    z_in = _as_f64_device_array(z_in, "z_in")
    x_out = _as_f64_device_array(x_out, "x_out")
    y_out = _as_f64_device_array(y_out, "y_out")
    z_out = _as_f64_device_array(z_out, "z_out")
    Vx = _as_f64_device_array(Vx, "Vx")
    Vy = _as_f64_device_array(Vy, "Vy")
    Nx = _as_f64_device_array(Nx, "Nx")
    Ny = _as_f64_device_array(Ny, "Ny")

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
    i_found = cp.full(n_impacts, -1, dtype=cp.int32)

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


def is_outside_convex(x_mp, y_mp, Vx, Vy, cx, cy):
    """
    Vectorized GPU implementation of convex polygon point-outside test using CuPy.
    """
    x_mp = cp.asarray(x_mp)
    y_mp = cp.asarray(y_mp)
    Vx = cp.asarray(Vx)
    Vy = cp.asarray(Vy)

    is_inside_ellipse = (x_mp / cx) ** 2 + (y_mp / cy) ** 2 <= 1.0

    Vx0 = Vx[:-1]
    Vy0 = Vy[:-1]
    Vx1 = Vx[1:]
    Vy1 = Vy[1:]

    edge_dx = Vx1 - Vx0
    edge_dy = Vy1 - Vy0

    px = x_mp[:, None]
    py = y_mp[:, None]

    dx = px - Vx0
    dy = py - Vy0

    cross = dy * edge_dx - dx * edge_dy
    is_inside_poly = cp.all(cross > 0.0, axis=1)

    return ~(is_inside_poly | is_inside_ellipse)
