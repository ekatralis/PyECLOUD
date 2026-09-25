project('PyECLOUD', 'c', 'fortran', meson_version: '>=1.3')
py = import('python').find_installation(pure: false)
np_inc = run_command(py, ['-c', 'import numpy; print(numpy.get_include())'], check: true).stdout().strip()
f2py_inc = run_command(py, ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'], check: true).stdout().strip()
incs = include_directories(np_inc, f2py_inc, 'PyECLOUD/_native')
# Install the runtime package as a directory; native inputs stay in the sdist.
install_subdir('PyECLOUD', install_dir: py.get_install_dir(),
  exclude_directories: ['_native', '__pycache__'])

rhocompute_src = custom_target('rhocompute_wrappers',
  input: 'PyECLOUD/_native/compute_rho.f', output: ['rhocomputemodule.c', 'rhocompute-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'rhocompute', '--lower'])
py.extension_module('rhocompute', ['PyECLOUD/_native/compute_rho.f', rhocompute_src, f2py_inc / 'fortranobject.c'],
  include_directories: incs, install: true, subdir: 'PyECLOUD')

int_field_for_src = custom_target('int_field_for_wrappers',
  input: 'PyECLOUD/_native/interp_field_for.f', output: ['int_field_formodule.c', 'int_field_for-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'int_field_for', '--lower'])
py.extension_module('int_field_for', ['PyECLOUD/_native/interp_field_for.f', int_field_for_src, f2py_inc / 'fortranobject.c'],
  include_directories: incs, install: true, subdir: 'PyECLOUD')

errffor_src = custom_target('errffor_wrappers',
  input: 'PyECLOUD/_native/errfff.f', output: ['errfformodule.c', 'errffor-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'errffor', '--lower'])
py.extension_module('errffor', ['PyECLOUD/_native/errfff.f', errffor_src, f2py_inc / 'fortranobject.c'],
  include_directories: incs, install: true, subdir: 'PyECLOUD')

hist_for_src = custom_target('hist_for_wrappers',
  input: 'PyECLOUD/_native/compute_hist.f', output: ['hist_formodule.c', 'hist_for-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'hist_for', '--lower'])
py.extension_module('hist_for', ['PyECLOUD/_native/compute_hist.f', hist_for_src, f2py_inc / 'fortranobject.c'],
  include_directories: incs, install: true, subdir: 'PyECLOUD')

seg_impact_src = custom_target('seg_impact_wrappers',
  input: 'PyECLOUD/_native/update_seg_impact.f', output: ['seg_impactmodule.c', 'seg_impact-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'seg_impact', '--lower'])
py.extension_module('seg_impact', ['PyECLOUD/_native/update_seg_impact.f', seg_impact_src, f2py_inc / 'fortranobject.c'],
  include_directories: incs, install: true, subdir: 'PyECLOUD')

boris_step_src = custom_target('boris_step_wrappers',
  input: 'PyECLOUD/_native/boris_step.f', output: ['boris_stepmodule.c', 'boris_step-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'boris_step', '--lower'])
py.extension_module('boris_step', ['PyECLOUD/_native/boris_step.f', boris_step_src, f2py_inc / 'fortranobject.c'],
  include_directories: incs, install: true, subdir: 'PyECLOUD')

vectsum_src = custom_target('vectsum_wrappers',
  input: 'PyECLOUD/_native/vectsum.f', output: ['vectsummodule.c', 'vectsum-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'vectsum', '--lower'])
py.extension_module('vectsum', ['PyECLOUD/_native/vectsum.f', vectsum_src, f2py_inc / 'fortranobject.c'],
  include_directories: incs, install: true, subdir: 'PyECLOUD')

boris_cython_src = custom_target('boris_cython_c', input: 'PyECLOUD/_native/boris_cython.pyx', output: 'boris_cython.c',
  depend_files: ['PyECLOUD/_native/boris_cython.pxd', 'PyECLOUD/_native/boris_c_function.h'],
  command: [py, '-m', 'cython', '-3', '--module-name', 'PyECLOUD.boris_cython', '-I', meson.current_source_dir() / 'PyECLOUD/_native', '@INPUT@', '-o', '@OUTPUT@'])
py.extension_module('boris_cython', [boris_cython_src, 'PyECLOUD/_native/boris_c_function.c'],
  include_directories: incs, install: true, subdir: 'PyECLOUD')

geom_impact_poly_cython_src = custom_target('geom_impact_poly_cython_c', input: 'PyECLOUD/_native/geom_impact_poly_cython.pyx', output: 'geom_impact_poly_cython.c',
  command: [py, '-m', 'cython', '-3', '--module-name', 'PyECLOUD.geom_impact_poly_cython', '-I', meson.current_source_dir() / 'PyECLOUD/_native', '@INPUT@', '-o', '@OUTPUT@'])
py.extension_module('geom_impact_poly_cython', [geom_impact_poly_cython_src],
  include_directories: incs, install: true, subdir: 'PyECLOUD')

meson.add_dist_script(py, files('tools/prepare_sdist.py'))
