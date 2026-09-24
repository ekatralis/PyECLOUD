import importlib
import numpy as np
import pytest
from numpy.testing import assert_allclose

PACKAGE = "PyECLOUD"


def test_charge_deposition_and_interpolation():
    rho_mod = importlib.import_module(PACKAGE + '.rhocompute')
    field_mod = importlib.import_module(PACKAGE + '.int_field_for')
    rho = rho_mod.compute_sc_rho([0.25], [0.5], [8.], 0., 0., 1., 3, 3)
    assert_allclose(rho[:2, :2], [[3., 3.], [1., 1.]])
    assert_allclose(rho.sum(), 8.)
    ex, ey = field_mod.int_field([0.25], [0.5], 0., 0., 1., 1., np.ones((3, 3)), np.full((3, 3), 2.))
    assert_allclose(ex, [1.])
    assert_allclose(ey, [2.])


def test_complex_error_function():
    from scipy.special import wofz
    mod = importlib.import_module(PACKAGE + '.errffor')
    real, imag = mod.errf(0.3, 0.4)
    assert_allclose(real + 1j * imag, wofz(0.3 + 0.4j), rtol=1e-6)


def test_version_and_installed_location():
    from importlib.metadata import version
    package = importlib.import_module(PACKAGE)
    assert package.__version__ == version('PyECLOUD')


def test_histogram_segments_and_sum():
    from PyECLOUD import hist_for, seg_impact, vectsum
    hist = np.zeros(3)
    hist_for.compute_hist([0.25, 1.5], [4., 2.], 0., 1., hist)
    assert_allclose(hist, [3., 2., 1.])
    segments = np.zeros(3)
    seg_impact.update_seg_impact([0, 2, 2], [1., 2., 3.], segments)
    assert_allclose(segments, [1., 0., 5.])
    assert_allclose(vectsum.vectsum(hist), 6.)


@pytest.mark.parametrize('magnetic', [0., 0.2])
def test_boris_fortran_and_cython(magnetic):
    from PyECLOUD.boris_step import boris_step
    from PyECLOUD.boris_cython import boris_step_multipole
    initial = [np.array([v], dtype=float) for v in [0., 0., 0., 1., 2., 3.]]
    fort = [a.copy() for a in initial]
    cyth = [a.copy() for a in initial]
    zero = np.zeros(1)
    bz = np.array([magnetic])
    boris_step(0.01, *fort, zero, zero, zero, zero, zero, bz, 1., 1.)
    boris_step_multipole(1, 0.01, zero, zero, *cyth, zero, zero, zero, zero, bz, True, 1., 1.)
    assert_allclose(cyth, fort, atol=1e-14)
    assert_allclose(sum(v[0]**2 for v in fort[3:]), 14., rtol=1e-14)
    if not magnetic:
        assert_allclose(fort[:3], [[.01], [.02], [.03]])


def test_polygon_extension():
    from PyECLOUD.geom_impact_poly_cython import is_outside_nonconvex
    vx = np.array([-1., 1., 1., -1., -1.])
    vy = np.array([-1., -1., 1., 1., -1.])
    flags = is_outside_nonconvex(np.array([0., 2.]), np.zeros(2), vx, vy, 0.1, 0.1, 4)
    np.testing.assert_array_equal(flags, [False, True])


def test_saver_without_git(tmp_path, monkeypatch):
    from PyECLOUD.pyecloud_saver import pyecloud_saver
    from PyECLOUD import __version__
    monkeypatch.setenv('PATH', '')
    path = tmp_path / 'simulation.log'
    pyecloud_saver(str(path))
    assert __version__ in path.read_text()


def test_save_reload(tmp_path):
    from PyECLOUD.myfilemanager import myloadmat
    from scipy.io import savemat
    path = str(tmp_path / 'data.mat')
    savemat(path, {'particles': np.arange(6.).reshape(2, 3)})
    assert_allclose(myloadmat(path)['particles'], np.arange(6.).reshape(2, 3))


def test_simulation_import():
    from PyECLOUD.buildup_simulation import BuildupSimulation
    assert callable(BuildupSimulation)


def test_public_module_imports():
    import pkgutil
    package = importlib.import_module(PACKAGE)
    for module in pkgutil.iter_modules(package.__path__):
        if module.name in {'Transverse_Efield_map_for_frozen_cloud'}:
            continue  # optional PyHEADTAIL integration is tested separately
        importlib.import_module(PACKAGE + '.' + module.name)
