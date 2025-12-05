#-Begin-preamble-------------------------------------------------------
#
#                           CERN
#
#     European Organization for Nuclear Research
#
#
#     This file is part of the code:
#
#                   PyECLOUD Version 8.7.1
#
#
#     Main author:          Giovanni IADAROLA
#                           BE-ABP Group
#                           CERN
#                           CH-1211 GENEVA 23
#                           SWITZERLAND
#                           giovanni.iadarola@cern.ch
#
#     Contributors:         Eleonora Belli
#                           Philipp Dijkstal
#                           Lorenzo Giacomel
#                           Lotta Mether
#                           Annalisa Romano
#                           Giovanni Rumolo
#                           Eric Wulff
#
#
#     Copyright  CERN,  Geneva  2011  -  Copyright  and  any   other
#     appropriate  legal  protection  of  this  computer program and
#     associated documentation reserved  in  all  countries  of  the
#     world.
#
#     Organizations collaborating with CERN may receive this program
#     and documentation freely and without charge.
#
#     CERN undertakes no obligation  for  the  maintenance  of  this
#     program,  nor responsibility for its correctness,  and accepts
#     no liability whatsoever resulting from its use.
#
#     Program  and documentation are provided solely for the use  of
#     the organization to which they are distributed.
#
#     This program  may  not  be  copied  or  otherwise  distributed
#     without  permission. This message must be retained on this and
#     any other authorized copies.
#
#     The material cannot be sold. CERN should be  given  credit  in
#     all references.
#
#-End-preamble---------------------------------------------------------

import math
import numpy as np
from .boris_cython import boris_step_multipole
from .boris_cupy_function import boris_c_cupy
from line_profiler import profile
import cupy as cp

# Cache compiled kernels so we don’t recompile on every call
_BORIS_KERNELS = {}

_F32_SRC = r'''
extern "C" __global__
void boris_kernel_f32(
    const int    N_sub_steps,
    const int    N_multipoles,
    const float  half_qm_dt,   // 0.5 * (charge/mass) * Dtt
    const float  Dtt,
    const float* __restrict__ B_field,   // [N_multipoles]
    const float* __restrict__ B_skew,    // [N_multipoles]
    float*       __restrict__ xn1,       // [N_mp]
    float*       __restrict__ yn1,
    float*       __restrict__ zn1,
    float*       __restrict__ vxn1,
    float*       __restrict__ vyn1,
    float*       __restrict__ vzn1,
    const float* __restrict__ Ex_n,      // [N_mp]
    const float* __restrict__ Ey_n,      // [N_mp]
    const float* __restrict__ Bx_n_custom, // [N_mp] (safe even if unused)
    const float* __restrict__ By_n_custom,
    const float* __restrict__ Bz_n_custom,
    const int    custom_B,               // 0 or 1
    const int    N_mp)
{
    int p = blockDim.x * blockIdx.x + threadIdx.x;
    if (p >= N_mp) return;

    // Load particle state to registers
    float Ex_np = Ex_n[p];
    float Ey_np = Ey_n[p];

    float x  = xn1[p];
    float y  = yn1[p];
    float z  = zn1[p];
    float vx = vxn1[p];
    float vy = vyn1[p];
    float vz = vzn1[p];

    // Time integration
    for (int isub = 0; isub < N_sub_steps; ++isub) {
        // Multipole field evaluation at (x, y)
        float By_n = (N_multipoles > 0) ? B_field[0] : 0.0f;
        float Bx_n = (N_multipoles > 0) ? B_skew[0]  : 0.0f;
        float Bz_n = 0.0f;

        float rexy = 1.0f;
        float imxy = 0.0f;

        for (int order = 1; order < N_multipoles; ++order) {
            float rexy0 = rexy;
            rexy = rexy0 * x - imxy * y;
            imxy = imxy * x + rexy0 * y;

            By_n += B_field[order] * rexy - B_skew[order] * imxy;
            Bx_n += B_field[order] * imxy + B_skew[order] * rexy;
        }

        if (custom_B) {
            Bx_n += Bx_n_custom[p];
            By_n += By_n_custom[p];
            Bz_n += Bz_n_custom[p];
        }

        // Boris rotation (t = half_qm_dt * B, s = 2t/(1+t^2))
        float tBx  = half_qm_dt * Bx_n;
        float tBy  = half_qm_dt * By_n;
        float tBz  = half_qm_dt * Bz_n;
        float tBsq = tBx*tBx + tBy*tBy + tBz*tBz;
        float inv  = 1.0f / (1.0f + tBsq);
        float sBx  = 2.0f * tBx * inv;
        float sBy  = 2.0f * tBy * inv;
        float sBz  = 2.0f * tBz * inv;

        // v_minus
        float vx_min = vx + half_qm_dt * Ex_np;
        float vy_min = vy + half_qm_dt * Ey_np;
        float vz_min = vz;

        // v' = v_min + v_min x tB
        float vx_prime =  vy_min*tBz - vz_min*tBy + vx_min;
        float vy_prime =  vz_min*tBx - vx_min*tBz + vy_min;
        float vz_prime =  vx_min*tBy - vy_min*tBx + vz_min;

        // v+ = v_min + v' x sB
        float vx_plus =  vy_prime*sBz - vz_prime*sBy + vx_min;
        float vy_plus =  vz_prime*sBx - vx_prime*sBz + vy_min;
        float vz_plus =  vx_prime*sBy - vy_prime*sBx + vz_min;

        // second half E-kick
        vx = vx_plus + half_qm_dt * Ex_np;
        vy = vy_plus + half_qm_dt * Ey_np;
        vz = vz_plus;

        // drift
        x += vx * Dtt;
        y += vy * Dtt;
        z += vz * Dtt;
    }

    // Write back once
    xn1[p]  = x;   yn1[p]  = y;   zn1[p]  = z;
    vxn1[p] = vx;  vyn1[p] = vy;  vzn1[p] = vz;
}
'''

_F64_SRC = r'''
extern "C" __global__
void boris_kernel_f64(
    const int     N_sub_steps,
    const int     N_multipoles,
    const double  half_qm_dt,
    const double  Dtt,
    const double* __restrict__ B_field,
    const double* __restrict__ B_skew,
    double*       __restrict__ xn1,
    double*       __restrict__ yn1,
    double*       __restrict__ zn1,
    double*       __restrict__ vxn1,
    double*       __restrict__ vyn1,
    double*       __restrict__ vzn1,
    const double* __restrict__ Ex_n,
    const double* __restrict__ Ey_n,
    const double* __restrict__ Bx_n_custom,
    const double* __restrict__ By_n_custom,
    const double* __restrict__ Bz_n_custom,
    const int     custom_B,
    const int     N_mp)
{
    int p = blockDim.x * blockIdx.x + threadIdx.x;
    if (p >= N_mp) return;

    double Ex_np = Ex_n[p];
    double Ey_np = Ey_n[p];

    double x  = xn1[p];
    double y  = yn1[p];
    double z  = zn1[p];
    double vx = vxn1[p];
    double vy = vyn1[p];
    double vz = vzn1[p];

    for (int isub = 0; isub < N_sub_steps; ++isub) {
        double By_n = (N_multipoles > 0) ? B_field[0] : 0.0;
        double Bx_n = (N_multipoles > 0) ? B_skew[0]  : 0.0;
        double Bz_n = 0.0;

        double rexy = 1.0;
        double imxy = 0.0;

        for (int order = 1; order < N_multipoles; ++order) {
            double rexy0 = rexy;
            rexy = rexy0 * x - imxy * y;
            imxy = imxy * x + rexy0 * y;

            By_n += B_field[order] * rexy - B_skew[order] * imxy;
            Bx_n += B_field[order] * imxy + B_skew[order] * rexy;
        }

        if (custom_B) {
            Bx_n += Bx_n_custom[p];
            By_n += By_n_custom[p];
            Bz_n += Bz_n_custom[p];
        }

        double tBx  = half_qm_dt * Bx_n;
        double tBy  = half_qm_dt * By_n;
        double tBz  = half_qm_dt * Bz_n;
        double tBsq = tBx*tBx + tBy*tBy + tBz*tBz;
        double inv  = 1.0 / (1.0 + tBsq);
        double sBx  = 2.0 * tBx * inv;
        double sBy  = 2.0 * tBy * inv;
        double sBz  = 2.0 * tBz * inv;

        double vx_min = vx + half_qm_dt * Ex_np;
        double vy_min = vy + half_qm_dt * Ey_np;
        double vz_min = vz;

        double vx_prime =  vy_min*tBz - vz_min*tBy + vx_min;
        double vy_prime =  vz_min*tBx - vx_min*tBz + vy_min;
        double vz_prime =  vx_min*tBy - vy_min*tBx + vz_min;

        double vx_plus =  vy_prime*sBz - vz_prime*sBy + vx_min;
        double vy_plus =  vz_prime*sBx - vx_prime*sBz + vy_min;
        double vz_plus =  vx_prime*sBy - vy_prime*sBx + vz_min;

        vx = vx_plus + half_qm_dt * Ex_np;
        vy = vy_plus + half_qm_dt * Ey_np;
        vz = vz_plus;

        x += vx * Dtt;
        y += vy * Dtt;
        z += vz * Dtt;
    }

    xn1[p]  = x;   yn1[p]  = y;   zn1[p]  = z;
    vxn1[p] = vx;  vyn1[p] = vy;  vzn1[p] = vz;
}
'''

def _get_boris_kernel(dtype):
    """Compile (once) and return the RawKernel matching dtype."""
    if dtype == cp.float32:
        name, src = "boris_kernel_f32", _F32_SRC
    elif dtype == cp.float64:
        name, src = "boris_kernel_f64", _F64_SRC
    else:
        raise TypeError(f"Unsupported dtype {dtype}; use float32 or float64.")

    if name not in _BORIS_KERNELS:
        _BORIS_KERNELS[name] = cp.RawKernel(src, name, options=('--std=c++11',), backend='nvrtc')
    return _BORIS_KERNELS[name]

@profile
def boris_c_gpu(
    N_sub_steps, Dtt,
    B_field, B_skew,
    xn1, yn1, zn1,
    vxn1, vyn1, vzn1,
    Ex_n, Ey_n,
    charge, mass,
    Bx_n_custom=None, By_n_custom=None, Bz_n_custom=None,
    custom_B=False,
    threads_per_block=256
):
    """
    CuPy accelerated Boris pusher with multipoles.
    - Launches one kernel; maps one thread per particle.
    - Updates arrays IN PLACE and also returns them.

    All array arguments should be CuPy arrays on device and have the same dtype (float32/float64).
    Shapes:
      * scalars per particle: xn1, yn1, zn1, vxn1, vyn1, vzn1, Ex_n, Ey_n, (Bx/By/Bz)_custom  -> (N_mp,)
      * field multipoles: B_field, B_skew -> (N_multipoles,)
    """
    # Ensure CuPy arrays and contiguity; infer dtype from state arrays
    for arr in (xn1, yn1, zn1, vxn1, vyn1, vzn1, Ex_n, Ey_n, B_field, B_skew):
        if not isinstance(arr, cp.ndarray):
            raise TypeError("All inputs must be CuPy arrays already on device.")
    dtype = xn1.dtype
    if any(a.dtype != dtype for a in (yn1, zn1, vxn1, vyn1, vzn1, Ex_n, Ey_n, B_field, B_skew)):
        raise TypeError("All arrays must have the same dtype (float32 or float64).")

    xn1  = cp.ascontiguousarray(xn1)
    yn1  = cp.ascontiguousarray(yn1)
    zn1  = cp.ascontiguousarray(zn1)
    vxn1 = cp.ascontiguousarray(vxn1)
    vyn1 = cp.ascontiguousarray(vyn1)
    vzn1 = cp.ascontiguousarray(vzn1)
    Ex_n = cp.ascontiguousarray(Ex_n)
    Ey_n = cp.ascontiguousarray(Ey_n)
    B_field = cp.ascontiguousarray(B_field)
    B_skew  = cp.ascontiguousarray(B_skew)

    N_mp = xn1.size
    N_multipoles = B_field.size
    if B_skew.size != N_multipoles:
        raise ValueError("B_field and B_skew must have the same length (N_multipoles).")

    # Provide safe placeholders for custom_B pointers
    if not custom_B:
        Bx_n_custom = cp.zeros_like(xn1)
        By_n_custom = cp.zeros_like(xn1)
        Bz_n_custom = cp.zeros_like(xn1)
    else:
        for arr in (Bx_n_custom, By_n_custom, Bz_n_custom):
            if not isinstance(arr, cp.ndarray):
                raise TypeError("Custom B arrays must be CuPy arrays when custom_B=True.")
            if arr.dtype != dtype:
                raise TypeError("Custom B arrays must have the same dtype as the state arrays.")
        Bx_n_custom = cp.ascontiguousarray(Bx_n_custom)
        By_n_custom = cp.ascontiguousarray(By_n_custom)
        Bz_n_custom = cp.ascontiguousarray(Bz_n_custom)

    # Precompute constants on host (cast to dtype)
    qm = (charge / mass)
    half_qm_dt = dtype.type(0.5) * dtype.type(qm) * dtype.type(Dtt)
    Dtt_typed = dtype.type(Dtt)

    # Launch
    blocks = (N_mp + threads_per_block - 1) // threads_per_block
    ker = _get_boris_kernel(dtype)
    ker((blocks,), (threads_per_block,),
        (int(N_sub_steps), int(N_multipoles), half_qm_dt, Dtt_typed,
         B_field, B_skew,
         xn1, yn1, zn1,
         vxn1, vyn1, vzn1,
         Ex_n, Ey_n,
         Bx_n_custom, By_n_custom, Bz_n_custom,
         int(bool(custom_B)), int(N_mp))
    )
    # Optional: cp.cuda.runtime.deviceSynchronize()

    return xn1, yn1, zn1, vxn1, vyn1, vzn1

@profile
def boris_c_gpu_cleanedup(
    N_sub_steps, Dtt,
    B_field, B_skew,
    xn1, yn1, zn1,
    vxn1, vyn1, vzn1,
    Ex_n, Ey_n,
    charge, mass,
    Bx_n_custom=None, By_n_custom=None, Bz_n_custom=None,
    custom_B=False,
    threads_per_block=512
):
    """
    CuPy accelerated Boris pusher with multipoles.
    - Launches one kernel; maps one thread per particle.
    - Updates arrays IN PLACE and also returns them.

    All array arguments should be CuPy arrays on device and have the same dtype (float32/float64).
    Shapes:
      * scalars per particle: xn1, yn1, zn1, vxn1, vyn1, vzn1, Ex_n, Ey_n, (Bx/By/Bz)_custom  -> (N_mp,)
      * field multipoles: B_field, B_skew -> (N_multipoles,)
    """

    xn1  = cp.ascontiguousarray(xn1)
    yn1  = cp.ascontiguousarray(yn1)
    zn1  = cp.ascontiguousarray(zn1)
    vxn1 = cp.ascontiguousarray(vxn1)
    vyn1 = cp.ascontiguousarray(vyn1)
    vzn1 = cp.ascontiguousarray(vzn1)
    Ex_n = cp.ascontiguousarray(Ex_n)
    Ey_n = cp.ascontiguousarray(Ey_n)
    B_field = cp.ascontiguousarray(B_field)
    B_skew  = cp.ascontiguousarray(B_skew)

    N_mp = xn1.size
    N_multipoles = B_field.size

    # Provide safe placeholders for custom_B pointers
    if not custom_B:
        Bx_n_custom = cp.zeros_like(xn1)
        By_n_custom = cp.zeros_like(xn1)
        Bz_n_custom = cp.zeros_like(xn1)
    else:
        Bx_n_custom = cp.ascontiguousarray(Bx_n_custom)
        By_n_custom = cp.ascontiguousarray(By_n_custom)
        Bz_n_custom = cp.ascontiguousarray(Bz_n_custom)

    # Precompute constants on host (cast to dtype)
    qm = (charge / mass)
    half_qm_dt = 0.5 * qm * Dtt

    # Launch
    blocks = (N_mp + threads_per_block - 1) // threads_per_block
    ker = cp.RawKernel(_F64_SRC,"boris_kernel_f64", options=('--std=c++11',), backend='nvrtc')
    ker((blocks,), (threads_per_block,),
        (int(N_sub_steps), int(N_multipoles), half_qm_dt, Dtt,
         B_field, B_skew,
         xn1, yn1, zn1,
         vxn1, vyn1, vzn1,
         Ex_n, Ey_n,
         Bx_n_custom, By_n_custom, Bz_n_custom,
         int(bool(custom_B)), int(N_mp))
    )
    # Optional: cp.cuda.runtime.deviceSynchronize()

    return xn1, yn1, zn1, vxn1, vyn1, vzn1

class pusher_Boris_multipole():

    def __init__(self, Dt, N_sub_steps=1, B_multip=None, B_skew=None,
        B0x=None, B0y=None, B0z=None):

        self.N_sub_steps = N_sub_steps
        self.Dt = Dt

        if Dt is None or N_sub_steps is None:
            self.Dtt = None
        else:
            self.Dtt = Dt / float(N_sub_steps)

        if B_multip is None or len(B_multip) == 0:
            B_multip = np.array([0.], dtype=float)

        # B_multip are derivatives of B_field
        # B_field are field strengths at x=1 m, y=0
        factorial = np.array([math.factorial(ii) for ii in range(len(B_multip))], dtype=float)
        self.B_field = np.array(B_multip, dtype=float) / factorial
        if B_skew is None:
            self.B_field_skew = np.zeros_like(self.B_field, dtype=float)
        else:
            self.B_field_skew = np.array(B_skew, dtype=float) / factorial

        self.B0x = B0x
        self.B0y = B0y
        self.B0z = B0z
        print("Tracker: Boris multipole")

        print("N_subst_init=%d" % self.N_sub_steps)

    # @profile
    def step(self, MP_e, Ex_n, Ey_n, Ez_n=0., Bx_n=None, By_n=None, Bz_n=None):
        MP_e = self.stepcustomDt(MP_e, Ex_n, Ey_n, Ez_n,
            Bx_n, By_n, Bz_n,
            Dt_substep=self.Dtt, N_sub_steps=self.N_sub_steps)
        return MP_e
    @profile
    def stepcustomDt(self, MP_e, Ex_n, Ey_n, Ez_n=0.,
        Bx_n=None, By_n=None, Bz_n=None,
        Dt_substep=None, N_sub_steps=None):

        custom_B = 0
        Bx_arr = np.zeros(MP_e.N_mp)
        By_arr = np.zeros(MP_e.N_mp)
        Bz_arr = np.zeros(MP_e.N_mp)

        if self.B0x is not None:
            Bx_arr += self.B0x
            custom_B = 1
        if self.B0y is not None:
            By_arr += self.B0y
            custom_B = 1
        if self.B0z is not None:
            Bz_arr += self.B0z
            custom_B = 1

        if Bx_n is not None:
            Bx_arr += Bx_n
            custom_B = 1
        if By_n is not None:
            By_arr += By_n
            custom_B = 1
        if Bz_n is not None:
            Bz_arr += Bz_n
            custom_B = 1

        if MP_e.N_mp > 0:

            nar = lambda x: cp.asnumpy(x)
            car = lambda x: cp.asarray(x)

            xn1 = MP_e.x_mp[0:MP_e.N_mp]
            yn1 = MP_e.y_mp[0:MP_e.N_mp]
            zn1 = MP_e.z_mp[0:MP_e.N_mp]
            vxn1 = MP_e.vx_mp[0:MP_e.N_mp]
            vyn1 = MP_e.vy_mp[0:MP_e.N_mp]
            vzn1 = MP_e.vz_mp[0:MP_e.N_mp]

            cu_Bfield = car(self.B_field)
            cu_Bfieldskew = car(self.B_field_skew)
            cu_xn1 = car(xn1)
            cu_yn1 = car(yn1)
            cu_zn1 = car(zn1)
            cu_vxn1 = car(vxn1)
            cu_vyn1 = car(vyn1)
            cu_vzn1 = car(vzn1)
            cu_Ex_n = car(Ex_n)
            cu_Ey_n = car(Ey_n)
            cu_Bx_arr = car(Bx_arr)
            cu_By_arr = car(By_arr)
            cu_Bz_arr = car(Bz_arr)
            # cu_customB = car(custom_B)

            if Ez_n != 0.:
                raise ValueError('Oooops! Not implemented....')

            boris_step_multipole(N_sub_steps, Dt_substep, self.B_field, self.B_field_skew,
                         xn1, yn1, zn1, vxn1, vyn1, vzn1,
                         Ex_n, Ey_n, Bx_arr, By_arr, Bz_arr, custom_B, MP_e.charge, MP_e.mass)


            xxn1, xyn1, xzn1, xvxn1, xvyn1, xvzn1 = boris_c_gpu_cleanedup(N_sub_steps, Dt_substep, cu_Bfield, cu_Bfieldskew,
                         cu_xn1, cu_yn1, cu_zn1, cu_vxn1, cu_vyn1, cu_vzn1,
                         cu_Ex_n, cu_Ey_n, MP_e.charge, MP_e.mass, cu_Bx_arr, cu_By_arr, cu_Bz_arr, bool(custom_B))

            np.testing.assert_allclose(nar(xxn1),xn1,atol=1e-7,rtol = 1e-4)
            np.testing.assert_allclose(nar(xyn1),yn1,atol=1e-7,rtol = 1e-4)
            np.testing.assert_allclose(nar(xzn1),zn1,atol=1e-7,rtol = 1e-4)
            np.testing.assert_allclose(nar(xvxn1),vxn1,atol=1e-7,rtol = 1e-4)
            np.testing.assert_allclose(nar(xvyn1),vyn1,atol=1e-7,rtol = 1e-4)
            np.testing.assert_allclose(nar(xvzn1),vzn1,atol=1e-7,rtol = 1e-4)

            cp._default_memory_pool.free_all_blocks()
        return MP_e
