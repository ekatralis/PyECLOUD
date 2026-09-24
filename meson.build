project('PyECLOUD', 'c', 'fortran', version: '8.7.2', meson_version: '>=1.3')
py = import('python').find_installation(pure: false)
np_inc = run_command(py, ['-c', 'import numpy; print(numpy.get_include())'], check: true).stdout().strip()
f2py_inc = run_command(py, ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'], check: true).stdout().strip()
incs = include_directories(np_inc, f2py_inc, '.')
py.install_sources(['_provenance.py', 'BassErsk.py', 'MP_system.py', 'PyEC4PyHT.py', 'Transverse_Efield_map_for_frozen_cloud.py', '__init__.py', '_version.py', 'beam_and_timing.py', 'buildup_simulation.py', 'cloud_manager.py', 'cross_ionization.py', 'default_input_parameters.py', 'dynamics_Boris_f2py.py', 'dynamics_Boris_multipole.py', 'dynamics_dipole.py', 'dynamics_strong_B_generalized.py', 'electron_emission.py', 'gas_ionization_class.py', 'gen_photoemission_class.py', 'geom_impact_ellip.py', 'geom_impact_poly.py', 'geom_impact_poly_fast_impact.py', 'geom_impact_rect_fast_impact.py', 'impact_management_class.py', 'init.py', 'input_parameters_format_specification.py', 'myfilemanager.py', 'myloadmat_to_obj.py', 'mystyle.py', 'parse_beam_file.py', 'parse_cloud_file.py', 'pyecloud_saver.py', 'sec_emission_model_ECLOUD.py', 'sec_emission_model_ECLOUD_nunif.py', 'sec_emission_model_accurate_low_ene.py', 'sec_emission_model_cos_low_ener.py', 'sec_emission_model_flat_low_ener.py', 'sec_emission_model_from_file.py', 'sec_emission_model_furman_pivi.py', 'sec_emission_model_perfect_absorber.py', 'sincc_cosincc_cubsincc.py', 'space_charge_class.py', 'space_charge_class_electromagnetic.py'], subdir: 'PyECLOUD')

rhocompute_src = custom_target('rhocompute_wrappers',
  input: 'compute_rho.f', output: ['rhocomputemodule.c', 'rhocompute-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'rhocompute', '--lower'])
py.extension_module('rhocompute', ['compute_rho.f', rhocompute_src, f2py_inc / 'fortranobject.c'],
  include_directories: incs, install: true, subdir: 'PyECLOUD')

int_field_for_src = custom_target('int_field_for_wrappers',
  input: 'interp_field_for.f', output: ['int_field_formodule.c', 'int_field_for-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'int_field_for', '--lower'])
py.extension_module('int_field_for', ['interp_field_for.f', int_field_for_src, f2py_inc / 'fortranobject.c'],
  include_directories: incs, install: true, subdir: 'PyECLOUD')

errffor_src = custom_target('errffor_wrappers',
  input: 'errfff.f', output: ['errfformodule.c', 'errffor-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'errffor', '--lower'])
py.extension_module('errffor', ['errfff.f', errffor_src, f2py_inc / 'fortranobject.c'],
  include_directories: incs, install: true, subdir: 'PyECLOUD')

hist_for_src = custom_target('hist_for_wrappers',
  input: 'compute_hist.f', output: ['hist_formodule.c', 'hist_for-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'hist_for', '--lower'])
py.extension_module('hist_for', ['compute_hist.f', hist_for_src, f2py_inc / 'fortranobject.c'],
  include_directories: incs, install: true, subdir: 'PyECLOUD')

seg_impact_src = custom_target('seg_impact_wrappers',
  input: 'update_seg_impact.f', output: ['seg_impactmodule.c', 'seg_impact-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'seg_impact', '--lower'])
py.extension_module('seg_impact', ['update_seg_impact.f', seg_impact_src, f2py_inc / 'fortranobject.c'],
  include_directories: incs, install: true, subdir: 'PyECLOUD')

boris_step_src = custom_target('boris_step_wrappers',
  input: 'boris_step.f', output: ['boris_stepmodule.c', 'boris_step-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'boris_step', '--lower'])
py.extension_module('boris_step', ['boris_step.f', boris_step_src, f2py_inc / 'fortranobject.c'],
  include_directories: incs, install: true, subdir: 'PyECLOUD')

vectsum_src = custom_target('vectsum_wrappers',
  input: 'vectsum.f', output: ['vectsummodule.c', 'vectsum-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'vectsum', '--lower'])
py.extension_module('vectsum', ['vectsum.f', vectsum_src, f2py_inc / 'fortranobject.c'],
  include_directories: incs, install: true, subdir: 'PyECLOUD')

boris_cython_src = custom_target('boris_cython_c', input: 'boris_cython.pyx', output: 'boris_cython.c',
  depend_files: ['boris_cython.pxd', 'boris_c_function.h'],
  command: [py, '-m', 'cython', '-3', '--module-name', 'PyECLOUD.boris_cython', '-I', meson.current_source_dir(), '@INPUT@', '-o', '@OUTPUT@'])
py.extension_module('boris_cython', [boris_cython_src, 'boris_c_function.c'],
  include_directories: incs, install: true, subdir: 'PyECLOUD')

geom_impact_poly_cython_src = custom_target('geom_impact_poly_cython_c', input: 'geom_impact_poly_cython.pyx', output: 'geom_impact_poly_cython.c',
  depend_files: ['boris_cython.pxd', 'boris_c_function.h'],
  command: [py, '-m', 'cython', '-3', '--module-name', 'PyECLOUD.geom_impact_poly_cython', '-I', meson.current_source_dir(), '@INPUT@', '-o', '@OUTPUT@'])
py.extension_module('geom_impact_poly_cython', [geom_impact_poly_cython_src],
  include_directories: incs, install: true, subdir: 'PyECLOUD')

meson.add_dist_script(py, files('tools/prepare_sdist.py'))
