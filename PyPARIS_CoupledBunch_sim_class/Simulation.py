import types
import os

import numpy as np
from scipy.constants import c

import PyPARIS.communication_helpers as ch
import PyPARIS.share_segments as shs
import PyPARIS.slicing_tool as sl
import PyECLOUD.myfilemanager as mfm

from PyHEADTAIL.particles.slicing import UniformBinSlicer

N_turns_target = 18

sigma_z_bunch = 10e-2

machine_configuration = 'HLLHC-injection'
n_segments = 2 #8

octupole_knob = 0.0
Qp_x = 0.
Qp_y = 0.

flag_aperture = True

enable_transverse_damper = True
dampingrate_x = 10000.
dampingrate_y = 20.

# Beam properties
non_linear_long_matching = False

bunch_intensity = 1e11
epsn_x = 2.5e-6
epsn_y = 2.5e-6
sigma_z = sigma_z_bunch

#Filling pattern: here head is left and tail is right
b_spac_s = 25e-9/5
filling_pattern = 5*([1.]+4*[0.])#(0*(72*([1.]+4*[0.]) + 7*5*[0.]) + 72*([1.]+4*[0.]))

load_beam_from_folder = None #'bunch_states_turn0'

macroparticlenumber = 1000 #1000000
min_inten_slice4EC = 1e7

x_kick_in_sigmas = 0.25
y_kick_in_sigmas = 0.25

target_size_internal_grid_sigma = 10.

enable_ecloud = False #True

enable_kick_x = True
enable_kick_y = False

L_ecloud_tot = 20e3


class Simulation(object):
    def __init__(self):
        self.N_turns = 30
        self.N_buffer_float_size = 10000000
        self.N_buffer_int_size = 20
        self.N_parellel_rings = 3
        
        self.n_slices_per_bunch = 200
        self.z_cut_slicing = 3*sigma_z_bunch
        self.N_pieces_per_transfer = 300
        self.verbose = False
        self.mpi_verbose = True
        self.enable_barriers = True
        self.save_beam_at_turns = [5]

    def init_all(self):
        
        print('Exec init...')
        
        from LHC_custom import LHC
        self.machine = LHC(n_segments = n_segments, machine_configuration = machine_configuration,
                        Qp_x=Qp_x, Qp_y=Qp_y,
                        octupole_knob=octupole_knob)
        self.n_non_parallelizable = 1 #RF

        inj_optics = self.machine.transverse_map.get_injection_optics()
        sigma_x_smooth = np.sqrt(inj_optics['beta_x']*epsn_x/self.machine.betagamma)
        sigma_y_smooth = np.sqrt(inj_optics['beta_y']*epsn_y/self.machine.betagamma)

        if flag_aperture:
            # setup transverse losses (to "protect" the ecloud)
            import PyHEADTAIL.aperture.aperture as aperture
            apt_xy = aperture.EllipticalApertureXY(x_aper=target_size_internal_grid_sigma*sigma_x_smooth, 
                                                   y_aper=target_size_internal_grid_sigma*sigma_x_smooth)
            self.machine.one_turn_map.append(apt_xy)
            self.n_non_parallelizable +=1 

        if enable_transverse_damper:
            # setup transverse damper
            from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
            damper = TransverseDamper(dampingrate_x=dampingrate_x, dampingrate_y=dampingrate_y)
            self.machine.one_turn_map.append(damper)
            self.n_non_parallelizable +=1
            
        if enable_ecloud:
            print('Build ecloud...')
            import PyECLOUD.PyEC4PyHT as PyEC4PyHT
            ecloud = PyEC4PyHT.Ecloud(
                    L_ecloud=L_ecloud_tot/n_segments, slicer=None, slice_by_slice_mode=True,
                    Dt_ref=5e-12, pyecl_input_folder='./pyecloud_config',
                    chamb_type = 'polyg' ,
                    filename_chm= 'LHC_chm_ver.mat', 
                    #init_unif_edens_flag=1,
                    #init_unif_edens=1e7,
                    #N_mp_max = 3000000,
                    #nel_mp_ref_0 = 1e7/(0.7*3000000),
                    #B_multip = [0.],
                    #~ PyPICmode = 'ShortleyWeller_WithTelescopicGrids',
                    #~ f_telescope = 0.3,
                    target_grid = {'x_min_target':-target_size_internal_grid_sigma*sigma_x_smooth, 'x_max_target':target_size_internal_grid_sigma*sigma_x_smooth,
                                   'y_min_target':-target_size_internal_grid_sigma*sigma_y_smooth,'y_max_target':target_size_internal_grid_sigma*sigma_y_smooth,
                                   'Dh_target':.2*sigma_x_smooth},
                    #~ N_nodes_discard = 10.,
                    #~ N_min_Dh_main = 10,
                    #x_beam_offset = x_beam_offset,
                    #y_beam_offset = y_beam_offset,
                    #probes_position = probes_position,
                    save_pyecl_outp_as = 'cloud_evol_ring%d'%self.ring_of_CPUs.myring,
                    save_only = ['lam_t_array', 'nel_hist', 'Nel_timep', 't', 't_hist', 'xg_hist'],
                    sparse_solver = 'PyKLU', enable_kick_x=enable_kick_x, enable_kick_y=enable_kick_y)
            print('Done.')



        # split the machine
        i_end_parallel = len(self.machine.one_turn_map)-self.n_non_parallelizable
        sharing = shs.ShareSegments(i_end_parallel, self.ring_of_CPUs.N_nodes_per_ring)
        i_start_part, i_end_part = sharing.my_part(self.ring_of_CPUs.myid_in_ring)
        self.mypart = self.machine.one_turn_map[i_start_part:i_end_part]

        if self.ring_of_CPUs.I_am_at_end_ring:
            self.non_parallel_part = self.machine.one_turn_map[i_end_parallel:]
            

        #install eclouds in my part
        if enable_ecloud:
            my_new_part = []
            self.my_list_eclouds = []
            for ele in self.mypart:
                if ele in self.machine.transverse_map:
                    ecloud_new = ecloud.generate_twin_ecloud_with_shared_space_charge()
                    
                    # we save buildup info only for the first cloud in each ring
                    if self.ring_of_CPUs.myid_in_ring>0 or len(self.my_list_eclouds)>0:
                        ecloud_new.remove_savers()
                    
                    my_new_part.append(ecloud_new)
                    self.my_list_eclouds.append(ecloud_new)
                my_new_part.append(ele)

            self.mypart = my_new_part
            
            print('Hello, I am %d.%d, my part looks like: %s. Saver status: %s'%(
                self.ring_of_CPUs.myring, self.ring_of_CPUs.myid_in_ring, self.mypart, 
                [(ec.cloudsim.cloud_list[0].pyeclsaver is not None) for ec in self.my_list_eclouds]))
            
       
    def init_master(self):
        
        import PyPARIS.gen_multibunch_beam as gmb
        from scipy.constants import c as clight, e as qe
        from PyHEADTAIL.particles.slicing import UniformBinSlicer
        
        # Manage multi-run operation
        import PyPARIS_sim_class.Save_Load_Status as SLS
        SimSt = SLS.SimulationStatus(N_turns_per_run=self.N_turns,
                check_for_resubmit = False, N_turns_target=N_turns_target)
        SimSt.before_simulation()
        self.SimSt = SimSt

        if SimSt.first_run:
            if load_beam_from_folder is None:
                print('Building the beam!')
                list_bunches = gmb.gen_matched_multibunch_beam(self.machine, macroparticlenumber, filling_pattern, b_spac_s, 
                    bunch_intensity, epsn_x, epsn_y, sigma_z, non_linear_long_matching, min_inten_slice4EC)
                # compute and apply initial displacements
                inj_opt = self.machine.transverse_map.get_injection_optics()
                sigma_x = np.sqrt(inj_opt['beta_x']*epsn_x/self.machine.betagamma)
                sigma_y = np.sqrt(inj_opt['beta_y']*epsn_y/self.machine.betagamma)
                x_kick = x_kick_in_sigmas*sigma_x
                y_kick = y_kick_in_sigmas*sigma_y
                for bunch in list_bunches:
                    bunch.x += x_kick
                    bunch.y += y_kick
            else:
                # Load based on input
                list_bunches = gmb.load_multibunch_beam(load_beam_from_folder)
        else:
            # Load from previous run
            print 'Loading beam from file...'
            dirname = 'beam_status_part%02d'%(SimSt.present_simulation_part-1)
            list_bunches = gmb.load_multibunch_beam(dirname) 
            print 'Loaded beam from file.'
        
        for bb in list_bunches:
            bb.slice_info['simstate_part'] = self.SimSt.present_simulation_part

        return list_bunches


    def init_start_ring(self):
        self.bunch_monitor = None
    
    def perform_bunch_operations_at_start_ring(self, bunch):
        
        if self.bunch_monitor is None:
            
            simstate_part = bunch.slice_info['simstate_part']
            stats_to_store = [
             'mean_x', 'mean_xp', 'mean_y', 'mean_yp', 'mean_z', 'mean_dp',
             'sigma_x', 'sigma_y', 'sigma_z','sigma_dp', 'epsn_x', 'epsn_y',
             'epsn_z', 'macroparticlenumber',
             'i_bunch', 'i_turn']

            n_stored_turns = np.sum(np.array(filling_pattern)>0)*(\
                self.ring_of_CPUs.N_turns/self.ring_of_CPUs.N_parellel_rings\
                + self.ring_of_CPUs.N_parellel_rings)

            from PyHEADTAIL.monitors.monitors import BunchMonitor
            self.bunch_monitor = BunchMonitor(
                                'bunch_monitor_part%03d_ring%03d'%(
                                    simstate_part, self.ring_of_CPUs.myring),
                                n_stored_turns, 
                                {'Comment':'PyHDTL simulation'}, 
                                write_buffer_every = 1,
                                stats_to_store = stats_to_store)

            # define a slice monitor 
            z_left = bunch.slice_info['z_bin_left'] - bunch.slice_info['z_bin_center']
            z_right = bunch.slice_info['z_bin_right'] - bunch.slice_info['z_bin_center']
            slicer = UniformBinSlicer(n_slices = self.n_slices_per_bunch, z_cuts=(z_left, z_right))
            from PyHEADTAIL.monitors.monitors import SliceMonitor
            self.slice_monitor = SliceMonitor('slice_monitor_part%03d_ring%03d'%(
                simstate_part, self.ring_of_CPUs.myring),
                n_stored_turns, slicer,  {'Comment':'PyHDTL simulation'}, 
                write_buffer_every = 1, bunch_stats_to_store=stats_to_store,
                slice_stats_to_store='mean_x mean_y mean_z n_macroparticles_per_slice'.split())
        
    

        # Save bunch properties
        if bunch.macroparticlenumber > 0 and bunch.slice_info['i_turn'] < self.N_turns:
            # Attach bound methods to monitor i_bunch and i_turns 
            bunch.i_bunch = types.MethodType(lambda ss: ss.slice_info['i_bunch'], bunch)
            bunch.i_turn = types.MethodType(lambda ss: ss.slice_info['i_turn'], bunch)
            self.bunch_monitor.dump(bunch)

            # Monitor slice wrt bunch center
            bunch.z -= bunch.slice_info['z_bin_center']
            self.slice_monitor.dump(bunch)
            bunch.z += bunch.slice_info['z_bin_center']

        # Save full beam at user-defined positions
        if bunch.slice_info['i_turn'] in self.save_beam_at_turns:
            dirname = 'bunch_states_turn%d'%bunch.slice_info['i_turn']
            import PyPARIS.gen_multibunch_beam as gmb
            gmb.save_bunch_to_folder(bunch, dirname)

        # Save full beam at end simulation 
        if bunch.slice_info['i_turn'] == self.N_turns:
            # PyPARIS wants N_turns to be a multiple of N_parellel_rings
            assert(self.ring_of_CPUs.I_am_the_master) 
            dirname = 'beam_status_part%02d'%(self.SimSt.present_simulation_part)
            import PyPARIS.gen_multibunch_beam as gmb
            gmb.save_bunch_to_folder(bunch, dirname)
            if bunch.slice_info['i_bunch'] == bunch.slice_info['N_bunches_tot_beam'] - 1:
                if not self.SimSt.first_run:
                    os.system('rm -r beam_status_part%02d' % (self.SimSt.present_simulation_part - 1))
                self.SimSt.after_simulation()

    def slice_bunch_at_start_ring(self, bunch):
        list_slices = sl.slice_a_bunch(bunch, self.z_cut_slicing, self.n_slices_per_bunch)
        return list_slices

    def treat_piece(self, piece):
        for ele in self.mypart: 
            ele.track(piece)
        
    def merge_slices_at_end_ring(self, list_slices):
        bunch = sl.merge_slices_into_bunch(list_slices)
        return bunch

    def perform_bunch_operations_at_end_ring(self, bunch):
        #finalize present turn (with non parallel part, e.g. synchrotron motion)
        if bunch.macroparticlenumber>0:
            for ele in self.non_parallel_part:
                ele.track(bunch)

    def piece_to_buffer(self, piece):
        buf = ch.beam_2_buffer(piece)
        return buf

    def buffer_to_piece(self, buf):
        piece = ch.buffer_2_beam(buf)
        return piece


class DummyComm(object):

    def __init__(self, N_cores_pretend, pretend_proc_id):
        self.N_cores_pretend = N_cores_pretend
        self.pretend_proc_id = pretend_proc_id

    def Get_size(self):
        return self.N_cores_pretend

    def Get_rank(self):
        return self.pretend_proc_id

    def Barrier(self):
        pass


def get_sim_instance(N_cores_pretend, id_pretend, 
                     init_sim_objects_auto=True):

    from PyPARIS.ring_of_CPUs_multiturn import RingOfCPUs_multiturn
    myCPUring = RingOfCPUs_multiturn(Simulation(),
            comm=DummyComm(N_cores_pretend, id_pretend))
    return myCPUring.sim_content


