# Packaging and release notes

## Source baseline

This branch starts at PyCOMPLETE/PyECLOUD revision
`3dca0fd`. Existing fork checkouts are preserved.
PyECLOUD fork commits `b23ca6f`, `3bc4d09`, `730e806`, `80c1d2f`, and
`dc2c97d` were cherry-picked in order: boolean indexing, cleanup, setuptools,
Meson legacy builds, and loading MATLAB particle counts. Packaging adds
NumPy 2 Cython types, current SciPy integration calls, and installed provenance.

## Supported build targets

CI builds CPython 3.10, 3.11, 3.12, 3.13, and 3.14 for manylinux_2_28 x86_64,
macOS 14+ x86_64, and macOS 14+ arm64. Free-threaded interpreters, Windows,
Linux ARM, GPU backends and the polar FFTW solver are excluded. cibuildwheel
repairs platform wheels with auditwheel/delocate, including needed Fortran
runtime libraries. No compiler is required to install matching wheels.
NumPy 2.x is required. pip resolves Python-compatible releases of dependencies.
A CI configuration is not evidence of a passing platform: inspect all jobs
before advertising a release matrix.

## Local development and validation

Use `python -m pip install .` for an isolated source build. Install PyPIC first
when testing unpublished local PyECLOUD changes. Install build dependencies
before `python -m pip install --no-build-isolation -e .` for editable builds.
Run `python tools/test_installed.py` to copy tests to a temporary directory and
avoid importing this flat source tree accidentally. Optional KLU tests skip
when PyKLU is absent; wheel CI installs it and asserts the real solver is used.

`python -m build` creates an sdist then builds its wheel in isolation. Meson
archives committed files only: commit changes before validating the source
archive. `python -m twine check dist/*` validates distribution metadata.
The source archive includes small packaging test fixtures, native sources and
headers. Large legacy simulation/reference datasets remain in the Git repo,
not release archives. Neither tests nor examples are installed in wheels.

## Release sequence

1. Resolve upstream publication rights and project ownership. PyPIC's existing
   source notices explicitly require redistribution permission; COPYING retains
   the notice rather than assigning a new license.
2. Confirm availability/ownership of `PyCOMPLETE-PyPIC` and `PyECLOUD` on both
   PyPI and TestPyPI. The unrelated `pypic` distribution must not be installed
   alongside PyCOMPLETE-PyPIC, as both may own the same import namespace.
3. Set the release version consistently in pyproject.toml, meson.build and
   _version.py. Configure trusted publishers for `.github/workflows/release.yml`
   and protected `testpypi`/`pypi` environments in each upstream repository.
4. Run the build matrix and review tests. Stage PyPIC first; then PyECLOUD.
   PyECLOUD CI requires its PyPIC dependency to be available to pip. For local
   staging use a wheelhouse containing both packages (`--find-links`).
5. Use the manual Release workflow targeting TestPyPI. Download the exact
   staged versions with `pip download --index-url https://test.pypi.org/simple/
   --no-deps`, then install those files in a fresh environment, resolving normal
   third-party dependencies from PyPI. Exercise base and KLU installations.
6. Verify optional tracking independently using the manual tracking workflow.
   Published PyHEADTAIL currently builds from source; do not claim successful
   tracking on a Python/platform combination until that job passes.
7. Publish PyPIC to PyPI before PyECLOUD, using the same reviewed commit and
   version. Verify fresh `pip install PyECLOUD` and `pip install 'PyECLOUD[klu]'`.

The manual release workflow does not run merely because a branch is pushed.
No artifacts have been uploaded by the local packaging implementation.

## Physics regression coverage

Installed tests exercise all native modules, both Boris implementations,
short drift/dipole simulations, and saving/reloading simulation state. The
short simulation inputs derive from the upstream LHC drift case.
The legacy `001_comparison_against_reference.py` creates plots, not assertions,
and defines no acceptance tolerances. Full-duration reference comparisons
therefore remain a scientific release review, not an automated passing claim.
Do not update reference files merely to accommodate a packaging change.
