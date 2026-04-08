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


import numpy as np

import scipy.io as sio
from scipy.constants import c

from . import electron_emission
from .backend_context import build_backend_context


class PyECLOUD_PhotoemissionException(ValueError):
    pass


class photoemission_base(object):
    def _init_backend(self, use_gpu):
        self.use_gpu = bool(use_gpu)
        self.backend_context = build_backend_context(self.use_gpu)
        self.array_backend = self.backend_context.array_backend
        self.random_backend = self.backend_context.random_backend

    def _to_backend_float_array(self, values):
        return self.array_backend.array(values, dtype=float)

    def get_number_new_mps(self, k_pe_st, lambda_t, Dt, nel_mp_ref):

        if self.flag_continuous_emission:
            lambda_t = self.mean_lambda

        DNel = k_pe_st * c * lambda_t * Dt
        N_new_MP = self.backend_context.scalar_to_float(DNel / nel_mp_ref)
        Nint_new_MP = int(N_new_MP)
        rest = N_new_MP - Nint_new_MP
        return int(Nint_new_MP + int(self.backend_context.scalar_to_float(self.random_backend.rand()) < rest))

    def gen_energy_and_set_MPs(self, Nint_new_MP, x_in, y_in, x_out, y_out, MP_e):
        # Assumes convex_chamber

        #generate points and normals
        z_in = self.array_backend.zeros_like(x_out, dtype=float)
        z_out = self.array_backend.zeros_like(x_out, dtype=float)
        x_int, y_int, _, Norm_x, Norm_y, i_found = self.chamb.impact_point_and_normal(
            x_in, y_in, z_in, x_out, y_out, z_out, resc_fac=self.resc_fac)

        #generate energies (the same distr. for all photoelectr.)
        En_gen = self.get_energy(Nint_new_MP)  # in eV

        # generate velocities like in impact managment
        vx_gen, vy_gen, vz_gen = self.angle_dist_func(Nint_new_MP, En_gen, Norm_x, Norm_y, MP_e.mass)

        t_last_impact = -1
        MP_e.add_new_MPs(Nint_new_MP, MP_e.nel_mp_ref, x_int, y_int, z_in, vx_gen, vy_gen, vz_gen, t_last_impact)


class photoemission(photoemission_base):

    def __init__(self, inv_CDF_refl_photoem_file, k_pe_st, refl_frac, e_pe_sigma, e_pe_max, alimit, x0_refl,
                 y0_refl, out_radius, chamb, resc_fac, energy_distribution, photoelectron_angle_distribution,
                 beamtim=None, flag_continuous_emission=False, use_gpu=False):

        print('Start photoemission init.')
        self._init_backend(use_gpu)

        if not chamb.is_convex():
            print('Warning! This photoemission module is not suited for a non-convex chamber!')

        if inv_CDF_refl_photoem_file == 'unif_no_file':
            self.flag_unif = True

        else:
            self.flag_unif = False
            dict_psi_inv_CDF = sio.loadmat(inv_CDF_refl_photoem_file)
            self.inv_CDF_refl = self._to_backend_float_array(np.squeeze(dict_psi_inv_CDF['inv_CDF'].real))
            self.u_sam_CDF_refl = self._to_backend_float_array(np.squeeze(dict_psi_inv_CDF['u_sam'].real))

        self.k_pe_st = k_pe_st
        self.refl_frac = refl_frac
        self.e_pe_sigma = e_pe_sigma
        self.e_pe_max = e_pe_max
        self.alimit = alimit
        self.y0_refl = y0_refl
        self.out_radius = out_radius
        self.chamb = chamb
        self.resc_fac = resc_fac
        self.angle_dist_func = electron_emission.get_angle_dist_func(
            photoelectron_angle_distribution, self.backend_context)
        self.flag_continuous_emission = flag_continuous_emission

        if flag_continuous_emission:
            self.mean_lambda = self.backend_context.scalar_to_float(
                self.array_backend.mean(beamtim.lam_t_array))

        if y0_refl != 0.:
            raise PyECLOUD_PhotoemissionException('The case y0_refl!=0 is NOT IMPLEMETED yet!!!!')

        if x0_refl == 'left' or x0_refl=='right':
            if x0_refl == 'left':
                xout = -self.out_radius
            elif x0_refl == 'right':
                xout = self.out_radius
            origin = self.array_backend.array([0.], dtype=float)
            xout_arr = 3. * self.array_backend.array([xout], dtype=float)
            x_int, _, _, _, _, _ = self.chamb.impact_point_and_normal(
                origin, origin, origin,
                xout_arr, origin, origin, resc_fac=0.99999)
            self.x0_refl = self.backend_context.scalar_to_float(x_int[0])
        else:
            self.x0_refl = float(x0_refl)

        x0_refl_arr = self.array_backend.array([self.x0_refl], dtype=float)
        y0_refl_arr = self.array_backend.array([float(self.y0_refl)], dtype=float)
        if self.backend_context.any(self.chamb.is_outside(x0_refl_arr, y0_refl_arr)):
            raise PyECLOUD_PhotoemissionException('x0_refl, y0_refl is outside of the chamber!')

        self.get_energy = electron_emission.get_energy_distribution_func(
            energy_distribution, e_pe_sigma, e_pe_max, self.backend_context)

        # Check that outer circle is correct
        psi_gen = self.array_backend.linspace(0., 2. * np.pi, 10000)
        x_out = -2. * self.out_radius * self.array_backend.cos(psi_gen) + self.x0_refl
        y_out = 2. * self.out_radius * self.array_backend.sin(psi_gen)
        if self.backend_context.any(~self.chamb.is_outside(x_out, y_out)):
            raise ValueError('Photoemission circle points are inside the chamber, check your settings!')
        psi_gen = self.array_backend.linspace(0., 2. * np.pi, 10000)
        x_out = self.out_radius * self.array_backend.cos(psi_gen)
        y_out = self.out_radius * self.array_backend.sin(psi_gen)
        if self.backend_context.any(~self.chamb.is_outside(x_out, y_out)):
            raise ValueError('Photoemission circle points are inside the chamber, check your settings!')

        print('Done photoemission init. Energy distribution: %s' % energy_distribution)

    def generate(self, MP_e, lambda_t, Dt):

        Nint_new_MP = self.get_number_new_mps(self.k_pe_st, lambda_t, Dt, MP_e.nel_mp_ref)

        if Nint_new_MP > 0:
            #generate appo x_in and x_out
            x_in = self.array_backend.zeros(Nint_new_MP, dtype=float)
            y_in = self.array_backend.zeros(Nint_new_MP, dtype=float)
            x_out = self.array_backend.zeros(Nint_new_MP, dtype=float)
            y_out = self.array_backend.zeros(Nint_new_MP, dtype=float)

            #for each one generate flag refl
            refl_flag = (self.random_backend.rand(Nint_new_MP) < self.refl_frac)
            gauss_flag = ~refl_flag

            #generate psi for refl. photons generation
            N_refl = self.backend_context.count_nonzero(refl_flag)
            if N_refl > 0:
                u_gen = self.random_backend.rand(N_refl)
                if self.flag_unif:
                    psi_gen = 2. * np.pi * u_gen
                    x_out[refl_flag] = self.out_radius * self.array_backend.cos(psi_gen)
                    y_out[refl_flag] = self.out_radius * self.array_backend.sin(psi_gen)
                else:
                    psi_gen = self.array_backend.interp(u_gen, self.u_sam_CDF_refl, self.inv_CDF_refl)
                    x_in[refl_flag] = self.x0_refl
                    x_out[refl_flag] = -2. * self.out_radius * self.array_backend.cos(psi_gen) + self.x0_refl
                    y_out[refl_flag] = 2. * self.out_radius * self.array_backend.sin(psi_gen)

            #generate theta for nonreflected photon generation
            N_gauss = self.backend_context.count_nonzero(gauss_flag)
            if N_gauss > 0:
                theta_gen = self.random_backend.normal(0, self.alimit, N_gauss)
                x_out[gauss_flag] = self.out_radius * self.array_backend.cos(theta_gen)
                y_out[gauss_flag] = self.out_radius * self.array_backend.sin(theta_gen)

            self.gen_energy_and_set_MPs(Nint_new_MP, x_in, y_in, x_out, y_out, MP_e)

        return MP_e


class photoemission_from_file(photoemission_base):

    def __init__(self, inv_CDF_all_photoem_file, chamb, resc_fac, energy_distribution, e_pe_sigma, e_pe_max,
                 k_pe_st, out_radius, photoelectron_angle_distribution, beamtim=None,
                 flag_continuous_emission=False, use_gpu=False):
        self._init_backend(use_gpu)
        if isinstance(inv_CDF_all_photoem_file, str):
            print('Start photoemission init from file %s.' % inv_CDF_all_photoem_file)
        elif isinstance(inv_CDF_all_photoem_file, dict):
            print('Start photoemission init from dict.')

        if not chamb.is_convex():
            print('Warning! This photoemission module is not suited for a non-convex chamber!')

        self.flag_unif = (inv_CDF_all_photoem_file == 'unif_no_file')
        if not self.flag_unif:
            if isinstance(inv_CDF_all_photoem_file, dict):
                mat = inv_CDF_all_photoem_file
            else:
                mat = sio.loadmat(inv_CDF_all_photoem_file)
            self.u_sam = self._to_backend_float_array(np.squeeze(mat['u_sam']))
            self.angles = self._to_backend_float_array(np.squeeze(mat['angles']))

        self.k_pe_st = k_pe_st
        self.out_radius = out_radius
        self.chamb = chamb
        self.resc_fac = resc_fac
        self.flag_continuous_emission = flag_continuous_emission

        if flag_continuous_emission:
            self.mean_lambda = self.backend_context.scalar_to_float(
                self.array_backend.mean(beamtim.lam_t_array))

        self.get_energy = electron_emission.get_energy_distribution_func(
            energy_distribution, e_pe_sigma, e_pe_max, self.backend_context)
        self.angle_dist_func = electron_emission.get_angle_dist_func(
            photoelectron_angle_distribution, self.backend_context)
        print('Done photoemission init')

    def generate(self, MP_e, lambda_t, Dt):

        Nint_new_MP = self.get_number_new_mps(self.k_pe_st, lambda_t, Dt, MP_e.nel_mp_ref)
        if Nint_new_MP > 0:

            if self.flag_unif:
                theta_gen = self.random_backend.rand(Nint_new_MP) * 2 * np.pi
            else:
                cdf_gen = self.random_backend.rand(Nint_new_MP)
                theta_gen = self.array_backend.interp(cdf_gen, self.u_sam, self.angles)

            x_out = self.out_radius * self.array_backend.cos(theta_gen)
            y_out = self.out_radius * self.array_backend.sin(theta_gen)

            x_in = self.array_backend.zeros(Nint_new_MP, dtype=float)
            y_in = self.array_backend.zeros(Nint_new_MP, dtype=float)
            self.gen_energy_and_set_MPs(Nint_new_MP, x_in, y_in, x_out, y_out, MP_e)

        return MP_e


class photoemission_per_segment(photoemission_base):

    def __init__(self, chamb, energy_distribution, e_pe_sigma, e_pe_max, k_pe_st,
                 photoelectron_angle_distribution, beamtim=None, flag_continuous_emission=False,
                 use_gpu=False):
        print('Start photoemission per segment init')
        self._init_backend(use_gpu)
        self.k_pe_st = k_pe_st
        self.chamb = chamb
        self.flag_continuous_emission = flag_continuous_emission
        self.get_energy = electron_emission.get_energy_distribution_func(
            energy_distribution, e_pe_sigma, e_pe_max, self.backend_context)
        self.angle_dist_func = electron_emission.get_angle_dist_func(
            photoelectron_angle_distribution, self.backend_context)
        if self.flag_continuous_emission:
            self.mean_lambda = self.backend_context.scalar_to_float(
                self.array_backend.mean(beamtim.lam_t_array))
        print('Done photoemission init')

    def generate(self, MP_e, lambda_t, Dt):

        Nint_new_MP = self.get_number_new_mps(self.k_pe_st, lambda_t, Dt, MP_e.nel_mp_ref)
        if Nint_new_MP > 0:
            x_new_mp, y_new_mp, Norm_x, Norm_y = self.chamb.get_photoelectron_positions(Nint_new_MP)
            En_gen = self.get_energy(Nint_new_MP)  # in eV
            vx_gen, vy_gen, vz_gen = self.angle_dist_func(Nint_new_MP, En_gen, Norm_x, Norm_y, MP_e.mass)
            t_last_impact = -1
            MP_e.add_new_MPs(x_new_mp.size, MP_e.nel_mp_ref, x_new_mp, y_new_mp, 0., vx_gen, vy_gen, vz_gen, t_last_impact)

        return MP_e
