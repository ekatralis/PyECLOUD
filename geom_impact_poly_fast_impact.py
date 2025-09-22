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



from numpy import sum, arctan2, sin, cos
import scipy.io as sio
import numpy as np
import numpy.random as random

from . import geom_impact_poly_cython as gipc
# from . import geom_impact_poly_cupy as gipcu
# import cupy as cp
from line_profiler import profile

import cupy as cp

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

@profile
def is_outside_convex_gpu(x_mp, y_mp, Vx, Vy, cx, cy, N_edg=None, *,
                          threads_per_block=256, stream=None):
    """
    GPU version of your is_outside_convex.
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



class PyECLOUD_ChamberException(ValueError):
    pass


class polyg_cham_geom_object(object):

    chamb_type = 'polyg'

    def __init__(self, filename_chm, flag_non_unif_sey, flag_verbose_file=False, flag_verbose_stdout=False,
                 flag_assume_convex=True):

        print('Polygonal chamber - cython implementation')

        if type(filename_chm) == str:
            dict_chm = sio.loadmat(filename_chm)
        else:
            dict_chm = filename_chm
        self.dict_chm = dict_chm

        Vx = np.squeeze(dict_chm['Vx'])
        Vy = np.squeeze(dict_chm['Vy'])
        cx = float(np.squeeze(dict_chm['x_sem_ellip_insc']))
        cy = float(np.squeeze(dict_chm['y_sem_ellip_insc']))

        if flag_non_unif_sey == 1:
            self.del_max_segments = np.squeeze(dict_chm['del_max_segments'])
            self.R0_segments = np.squeeze(dict_chm['R0_segments'])
            self.Emax_segments = np.squeeze(dict_chm['Emax_segments'])

            if 'flag_charging' in list(dict_chm.keys()):
                self.flag_charging = np.squeeze(dict_chm['flag_charging'])
                self.Q_max_segments = np.squeeze(dict_chm['Q_max_segments'])
                self.EQ_segments = np.squeeze(dict_chm['EQ_segments'])
                self.tau_segments = np.squeeze(dict_chm['tau_segments'])


        if np.any(np.sqrt(np.diff(Vx)**2 + np.diff(Vy)**2) < 1e-9):
            raise PyECLOUD_ChamberException('There is a zero length segment!')

        self.N_vert = len(Vx)

        N_edg = len(Vx)

        Vx = list(Vx)
        Vy = list(Vy)

        Vx.append(Vx[0])
        Vy.append(Vy[0])

        Vx = np.array(Vx, float)
        Vy = np.array(Vy, float)

        self.area = -0.5 * np.sum((Vy[1:] + Vy[:-1]) * (Vx[1:] - Vx[:-1]))

        print("The area of the chamber is %.3e m^2"%self.area)

        if self.area < 0:
            raise PyECLOUD_ChamberException("The area of the chamber is negative!\nVerteces must be provided with counter-clockwise order!")

        Nx = -np.diff(Vy, 1)
        Ny = np.diff(Vx, 1)

        norm_N = np.sqrt(Nx**2 + Ny**2)
        Nx = Nx / norm_N
        Ny = Ny / norm_N

        self.x_aper = np.max(np.abs(Vx))
        self.y_aper = np.max(np.abs(Vy))
        self.Vx = Vx
        self.Vy = Vy
        self.Nx = Nx
        self.Ny = Ny
        self.N_edg = N_edg
        self.cx = cx
        self.cy = cy

        self.L_edg = norm_N

        self.N_mp_impact = 0
        self.N_mp_corrected = 0

        self.flag_verbose_stdout = flag_verbose_stdout
        self.flag_verbose_file = flag_verbose_file

        self.flag_assume_convex = flag_assume_convex

        if self.flag_verbose_file:
            fbckt = open('bcktr_errors.txt', 'w')
            fbckt.write('kind,x_in,y_in,x_out, y_out\n')
            fbckt.close()

        if self.flag_assume_convex:
            if not(self.is_convex()):
                raise PyECLOUD_ChamberException(
                    'The polygon looks not convex!!!!\nIn this case you can use the general algorithm (probably slower) by setting:\nflag_assume_convex = False')
            self.cythonisoutside = gipc.is_outside_convex
            print('Assuming convex polygon')
        else:
            self.cythonisoutside = gipc.is_outside_nonconvex
            print('No assumption on the convexity of the polygon')

    @profile
    def is_outside(self, x_mp, y_mp):
        nar = lambda x: cp.asnumpy(x)
        car = lambda x: cp.asarray(x)
        ret = self.cythonisoutside(x_mp, y_mp, self.Vx, self.Vy, self.cx, self.cy, self.N_edg)
        x_mp_gpu = car(x_mp)
        y_mp_gpu = car(y_mp)
        Vx_gpu = car(self.Vx)
        Vy_gpu = car(self.Vy)
        ret_gpu = is_outside_convex_gpu(x_mp_gpu,y_mp_gpu,Vx_gpu,Vy_gpu,self.cx,self.cy,self.N_edg)
        np.testing.assert_allclose(nar(ret_gpu),ret,atol=1e-7,rtol = 1e-4)
        return ret
    # @profile
    def impact_point_and_normal(self, x_in, y_in, z_in, x_out, y_out, z_out, resc_fac=0.99, flag_robust=True):

        N_impacts = len(x_in)
        self.N_mp_impact = self.N_mp_impact + N_impacts

        x_int, y_int, z_int, Nx_int, Ny_int, i_found = gipc.impact_point_and_normal(x_in, y_in, z_in, x_out, y_out, z_out,
                                                                                    self.Vx, self.Vy, self.Nx, self.Ny, self.N_edg, resc_fac)
        # print(x_in, y_in, z_in, x_out, y_out, z_out,self.Vx, self.Vy, self.Nx, self.Ny, self.N_edg, resc_fac)
        # print(x_int,y_int,z_int,Nx_int,Ny_int,i_found)
        # print(self.N_edg)
        #CUPY--------------------------
        # cu_x_in = cp.array(x_in)
        # cu_y_in = cp.array(y_in)
        # cu_z_in = cp.array(z_in)
        # cu_x_out = cp.array(x_out)
        # cu_y_out = cp.array(y_out)
        # cu_z_out = cp.array(z_out)
        # cu_Vx = cp.array(self.Vx)
        # cu_Vy = cp.array(self.Vy)
        # cu_Nx = cp.array(self.Nx)
        # cu_Ny = cp.array(self.Ny)
        # cu_resc_fac = cp.array(resc_fac)
        # cu_x_int, cu_y_int, cu_z_int, cu_Nx_int, cu_Ny_int, cu_i_found = gipcu.impact_point_and_normal(cu_x_in, cu_y_in, cu_z_in, cu_x_out, cu_y_out, cu_z_out,
        #                                                                             cu_Vx, cu_Vy, cu_Nx, cu_Ny, cu_resc_fac)
        # nar = lambda x: cp.asnumpy(x)
        # np.testing.assert_allclose(nar(cu_x_int),x_int)
        # np.testing.assert_allclose(nar(cu_y_int),y_int)
        # np.testing.assert_allclose(nar(cu_z_int),z_int)
        # np.testing.assert_allclose(nar(cu_Nx_int),Nx_int)
        # np.testing.assert_allclose(nar(cu_Ny_int),Ny_int)
        # np.testing.assert_allclose(nar(cu_i_found),i_found)

        mask_found = i_found >= 0

        if sum(mask_found) < N_impacts:
            mask_not_found = ~mask_found

            x_int[mask_not_found] = x_in[mask_not_found]
            y_int[mask_not_found] = y_in[mask_not_found]

            #compute some kind of normal ....
            par_cross = arctan2(self.cx * y_in[mask_not_found], self.cy * x_int[mask_not_found])

            Dx = -self.cx * sin(par_cross)
            Dy = self.cy * cos(par_cross)

            Nx_corr = -Dy
            Ny_corr = Dx

            neg_flag = ((Nx_corr * x_int[mask_not_found] + Ny_corr * y_int[mask_not_found]) > 0)

            Nx_corr[neg_flag] = -Nx_corr[neg_flag]
            Ny_corr[neg_flag] = -Ny_corr[neg_flag]

            Nx_int[mask_not_found] = Nx_corr
            Ny_int[mask_not_found] = Ny_corr

            x_in_error = x_in[mask_not_found]
            y_in_error = y_in[mask_not_found]
            x_out_error = x_out[mask_not_found]
            y_out_error = y_out[mask_not_found]
            N_errors = len(x_in_error)
            self.N_mp_corrected = self.N_mp_corrected + N_errors

            if self.flag_verbose_stdout:
                print('Reporting backtrack error of kind 1: no impact found')
                print('x_in, y_in, x_out, y_out')
                for i_err in range(N_errors):
                    lcurr = '%.10e,%.10e,%.10e,%.10e' % (x_in_error[i_err], y_in_error[i_err], x_out_error[i_err], y_out_error[i_err])
                    print(lcurr)
                print('End reporting backtrack error of kind 1')

            if self.flag_verbose_file:
                with open('bcktr_errors.txt', 'a') as fbckt:
                    for i_err in range(N_errors):
                        lcurr = '%.10e,%.10e,%.10e,%.10e' % (x_in_error[i_err], y_in_error[i_err], x_out_error[i_err], y_out_error[i_err])
                        fbckt.write('1,' + lcurr + '\n')

        if flag_robust:
            flag_impact = self.is_outside(x_int, y_int)
            if flag_impact.any():
                self.N_mp_corrected = self.N_mp_corrected + sum(flag_impact)
                x_int[flag_impact] = x_in[flag_impact]
                y_int[flag_impact] = y_in[flag_impact]
                x_in_error = x_in[flag_impact]
                y_in_error = y_in[flag_impact]
                x_out_error = x_out[flag_impact]
                y_out_error = y_out[flag_impact]
                N_errors = len(x_in_error)

                if self.flag_verbose_stdout:
                    print('Reporting backtrack error of kind 2: outside after backtracking')
                    print('x_in, y_in, x_out, y_out')
                    for i_err in range(N_errors):
                        lcurr = '%.10e,%.10e,%.10e,%.10e' % (x_in_error[i_err], y_in_error[i_err], x_out_error[i_err], y_out_error[i_err])
                        print(lcurr)
                    print('End reporting backtrack error of kind 2')

                if self.flag_verbose_file:
                    with open('bcktr_errors.txt', 'a') as fbckt:
                        for i_err in range(N_errors):
                            lcurr = '%.10e,%.10e,%.10e,%.10e' % (x_in_error[i_err], y_in_error[i_err], x_out_error[i_err], y_out_error[i_err])
                            fbckt.write('2,' + lcurr + '\n')

            flag_impact = self.is_outside(x_int, y_int)
            if sum(flag_impact) > 0:
                #~ import pylab as pl
                #~ pl.close('all')
                #~ pl.plot(self.Vx, self.Vy)
                #~ pl.plot(x_in, y_in,'.b')
                #~ pl.plot(x_out, y_out,'.k')
                #~ pl.plot(x_int, y_int,'.g')
                #~ pl.plot(x_int[flag_impact], y_int[flag_impact],'.r')
                #~ pl.show()
                if self.flag_verbose_stdout:
                    print('Reporting backtrack error of kind 3: outside after correction')
                    print('x_in, y_in, x_out, y_out')
                x_in_error = x_in[flag_impact]
                y_in_error = y_in[flag_impact]
                x_out_error = x_out[flag_impact]
                y_out_error = y_out[flag_impact]
                N_errors = len(x_in_error)

                if self.flag_verbose_stdout:
                    print('Reporting backtrack error of kind 3: outside after correction')
                    print('x_in, y_in, x_out, y_out')
                    for i_err in range(N_errors):
                        lcurr = '%.10e,%.10e,%.10e,%.10e' % (x_in_error[i_err], y_in_error[i_err], x_out_error[i_err], y_out_error[i_err])
                        print(lcurr)
                    print('End reporting backtrack error of kind 3')

                if self.flag_verbose_file:
                    with open('bcktr_errors.txt', 'a') as fbckt:
                        for i_err in range(N_errors):
                            lcurr = '%.10e,%.10e,%.10e,%.10e' % (x_in_error[i_err], y_in_error[i_err], x_out_error[i_err], y_out_error[i_err])
                            fbckt.write('3,' + lcurr + '\n')

                raise PyECLOUD_ChamberException('Outside after backtracking!!!!')

        return x_int, y_int, z_int, Nx_int, Ny_int, i_found

    def is_convex(self):
            # From:
            # http://csharphelper.com/blog/2014/07/determine-whether-a-polygon-is-convex-in-c/

            # For each set of three adjacent points A, B, C,
            # find the cross product AB x BC. If the sign of
            # all the cross products is the same, the angles
            # are all positive or negative (depending on the
            # order in which we visit them) so the polygon
            # is convex.
            got_negative = False
            got_positive = False
            num_points = self.N_edg
            for A in range(num_points):

                B = np.mod((A + 1), num_points)
                C = np.mod((B + 1), num_points)

                BAx = self.Vx[A] - self.Vx[B]
                BAy = self.Vy[A] - self.Vy[B]
                BCx = self.Vx[C] - self.Vx[B]
                BCy = self.Vy[C] - self.Vy[B]

                cross_product = (BAx * BCy - BAy * BCx)

                if (cross_product < 0):
                    got_negative = True
                elif (cross_product > 0):
                    got_positive = True
                if (got_negative and got_positive):
                    return False

            # If we got this far, the polygon is convex.
            return True

    def vertex_is_on_edge(self, x, y):
        """
        Tests if one point is on one of the chamber edges.
        """
        for diff_x, diff_y, vx, vy in zip(np.diff(self.Vx), np.diff(self.Vy), self.Vx, self.Vy):

            if x == vx and y == vy:
                return True
            elif diff_x != 0 and diff_y != 0:
                a = (x - vx) / diff_x
                b = (y - vy) / diff_y
                if a == b and 0 <= a <= 1:
                    return True
            elif diff_x == 0 and x == vx:
                b = (y - vy) / diff_y
                if 0 <= b <= 1:
                    return True
            elif diff_y == 0 and y == vy:
                a = (x - vx) / diff_x
                if 0 <= a <= 1:
                    return True

        return False

    def vertexes_are_subset(self, other_chamb):
        """
        Tests if all points of this chamber are on the edge of the other chamber, and vice-versa.
        """
        if not all(self.vertex_is_on_edge(x, y) for x, y in zip(other_chamb.Vx, other_chamb.Vy)):
            return False
        if not all(other_chamb.vertex_is_on_edge(x, y) for x, y in zip(self.Vx, self.Vy)):
            return False
        return True


class polyg_cham_photoemission(polyg_cham_geom_object):

    """
    The same requirements for filename_chm as in polyg_cham_geom_object also hold for this class.
    The only addition is the 'phem_cdf' property, which controls the number of photoelectrons per segment.
    It must be monotonically increasing and end on 1.
    A fraction of phem_cdf[0] photoelectrons is generated on the first edge, but slightly shifted inside the chamber.
    A fraction of phem_cdf[1] - phem_cdf[0] photoelectrons is generated on the second edge, etc.
    """

    # Distance of generated photoelecron MP relative to edge
    distance_new_phem = 1e-14

    def __init__(self, filename_chm):

        if isinstance(filename_chm, dict):
            dict_chm = filename_chm
        else:
            dict_chm = sio.loadmat(filename_chm)
        phem_cdf = np.squeeze(dict_chm['phem_cdf'])

        # Make sure phem_cdf has correct shape
        if phem_cdf[-1] != 1:
            raise PyECLOUD_ChamberException('phem_cdf of chamb_dict does not end with 1.')
        if np.any(np.diff(phem_cdf) < 0):
            raise PyECLOUD_ChamberException('phem_cdf of chamb_dict is not monotonically increasing.')

        # Optionally use distinct photoemission chamber segments
        # This allows for a finer resolution of photoemission per segment, without increasing the computational
        # burden on the is_outside and impact_point_and_normalfunctions.
        orig_Vx = np.squeeze(dict_chm['Vx'])
        orig_Vy = np.squeeze(dict_chm['Vy'])

        # Needed for cythonisoutside
        self.N_edg = len(orig_Vx)
        self.cx = float(np.squeeze(dict_chm['x_sem_ellip_insc']))
        self.cy = float(np.squeeze(dict_chm['y_sem_ellip_insc']))

        # Needed to calculate histograms and positions later
        self.Vx = Vx = np.append(orig_Vx, orig_Vx[0])
        self.Vy = Vy = np.append(orig_Vy, orig_Vy[0])

        self.area = -0.5 * np.sum((Vy[1:] + Vy[:-1]) * (Vx[1:] - Vx[:-1]))
        print("The area of the chamber is %.3e m^2"%self.area)
        if self.area < 0:
            raise PyECLOUD_ChamberException("The area of the chamber is negative!\nVerteces must be provided with counter-clockwise order!")

        self.seg_diff_x = seg_diff_x = np.diff(Vx)
        self.seg_diff_y = seg_diff_y = np.diff(Vy)
        self.cdf_bins = np.append(0, phem_cdf)

        len_segments = np.sqrt(seg_diff_x**2 + seg_diff_y**2)

        if np.any(len_segments < 1e-9):
            raise PyECLOUD_ChamberException('Some segments have length 0!')

        self.normal_vect_x = -seg_diff_y / len_segments
        self.normal_vect_y = seg_diff_x / len_segments

        self.phem_x0 = orig_Vx + self.distance_new_phem * self.normal_vect_x
        self.phem_y0 = orig_Vy + self.distance_new_phem * self.normal_vect_y

        if self.is_convex():
            self.cythonisoutside = gipc.is_outside_convex
            print('Assuming convex polygon')
        else:
            self.cythonisoutside = gipc.is_outside_nonconvex
            print('No assumption on the convexity of the polygon')

    def get_photoelectron_positions(self, N_mp_gen):
        """
        input: N_mp_gen - Number of MPs for which positions and normal vectors are to be calculated
        The cdf that is part of the chamber definition is used to generate the output
        output: positions and normal vectors for every generated macroparticle

        The MP positions are shifted towards the inner part of the chamber by a length defined in
        self.distance_new_phem
        """
        # Only meaningful for photoemission from segment
        x_new_mp = np.zeros(N_mp_gen)
        y_new_mp = np.zeros(N_mp_gen)
        norm_x_new_mp = np.empty(N_mp_gen)
        norm_y_new_mp = np.empty(N_mp_gen)

        # Distributing new MPs into segments according to the cdf
        N_mp_segment, _ = np.histogram(random.rand(N_mp_gen), self.cdf_bins)

        # The MPs in each segment are distributed evenly along the segment
        N_mp_curr = 0
        for i_seg, N_mp_seg in enumerate(N_mp_segment):
            if N_mp_seg != 0:
                N_mp_after = N_mp_curr + N_mp_seg
                # The output of this call to _get_photoelectron_segment does not have to be assigned
                self._get_photoelectron_position_segment(N_mp_seg, x_new_mp[N_mp_curr:N_mp_after], y_new_mp[N_mp_curr:N_mp_after], i_seg)
                norm_x_new_mp[N_mp_curr:N_mp_after] = self.normal_vect_x[i_seg]
                norm_y_new_mp[N_mp_curr:N_mp_after] = self.normal_vect_y[i_seg]
                N_mp_curr = N_mp_after

        return x_new_mp, y_new_mp, norm_x_new_mp, norm_y_new_mp

    def _get_photoelectron_position_segment(self, N_mp, x_new_mp, y_new_mp, i_seg):
        """
        Potentially recursive function.
        It makes sure that no MPs are generated outside of the chamber.
        In unsuitable chamber designs, it may run into the recursion limit and crash.
        But that would be due to unsuitable chamber designs.
        """
        rr = random.rand(N_mp)
        x_new_mp[:] = self.phem_x0[i_seg] + rr * self.seg_diff_x[i_seg]
        y_new_mp[:] = self.phem_y0[i_seg] + rr * self.seg_diff_y[i_seg]

        flag_outside = self.is_outside(x_new_mp, y_new_mp)
        n_mp_outside = sum(flag_outside)
        if n_mp_outside != 0:
            # This code branch should practically never be reached.
            # This depends on the choice of self.distance_new_phem.
            # With distance_new_phem set to 1e-12, this method has been tested for 1e10 particles
            # withot ever reaching this part of the code.
            print('%i out of %i MPs were generated outside! -> recursion' % (n_mp_outside, N_mp))

            # Because of advanced numpy array indexing, the output of this call has to be explicitly assigned.
            x_new_mp[flag_outside], y_new_mp[flag_outside] = self._get_photoelectron_position_segment(n_mp_outside, x_new_mp[flag_outside], y_new_mp[flag_outside], i_seg)

        return x_new_mp, y_new_mp


