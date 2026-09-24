from pathlib import Path
import shutil
import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.mark.parametrize('track_method, magnetic', [('BorisMultipole', 0.), ('BorisMultipole', .01), ('Boris', .01)])
def test_short_buildup(tmp_path, monkeypatch, track_method, magnetic):
    from PyECLOUD.buildup_simulation import BuildupSimulation
    source = Path(__file__).parent / 'fixtures/drift'
    folder = tmp_path / 'input'
    shutil.copytree(source, folder)
    beam = folder / 'beam.beam'
    beam.write_text(beam.read_text().replace('Dh_beam_field = 0.0001', 'Dh_beam_field = 0.003'))
    monkeypatch.chdir(tmp_path)
    np.random.seed(1729)
    sim = BuildupSimulation(pyecl_input_folder=str(folder),
        Dh_sc=.003, sparse_solver='scipy_slu',
        N_mp_max=2000, N_mp_regen=1500, N_mp_after_regen=500,
        N_mp_soft_regen=1500, N_mp_after_soft_regen=500,
        init_unif_flag=1, Nel_init_unif=1.e5, nel_mp_ref_0=1000.,
        photoem_flag=0, extract_sey=False, track_method=track_method,
        B_multip=[magnetic], B0x=0., B0y=magnetic, B0z=0.,
        filen_main_outp=str(tmp_path / 'output.mat'))
    sim.run(t_end_sim=1.e-10)
    cloud = sim.cloud_list[0]
    assert cloud.MP_e.N_mp > 0
    for values in [cloud.MP_e.x_mp, cloud.MP_e.y_mp, cloud.MP_e.vx_mp]:
        assert np.isfinite(values[:cloud.MP_e.N_mp]).all()
    assert sim.beamtim.tt_curr >= 1.e-10

    # Exercise the real simulation state writer and loader, including LU rebuild.
    state = tmp_path / 'state.pkl'
    saver = cloud.pyeclsaver
    saver._sim_state_single_save(sim.beamtim, sim.spacech_ele, sim.t_sc_ON,
        sim.flag_presence_sec_beams, sim.sec_beams_list, sim.flag_multiple_clouds,
        sim.cloud_list, str(state))
    saved_x = cloud.MP_e.x_mp.copy()
    sim.load_state(state.name, load_from_folder=str(tmp_path) + '/', filen_main_outp=None)
    assert_allclose(sim.cloud_list[0].MP_e.x_mp, saved_x)
    sim.run(t_end_sim=2.e-10)
    assert sim.beamtim.tt_curr >= 2.e-10
