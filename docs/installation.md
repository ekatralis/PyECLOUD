# Installation

If Conda is not installed, follow the
[official Miniforge instructions](https://github.com/conda-forge/miniforge#install)
for your platform, then open a shell where `conda` is available.

## Released packages from PyPI

With Conda available, create an environment with your preferred Python version, install the
compilers and numerical libraries, then install PyECLOUD (and optionally PyHEADTAIL for beam tracking):

```bash
conda create -n ecloud -c conda-forge python=3.13 pip
conda activate ecloud
conda install -c conda-forge c-compiler cxx-compiler fortran-compiler
conda install -c conda-forge numpy scipy matplotlib
pip install --upgrade pyheadtail
pip install --upgrade pyecloud
```

Keep the `ecloud` environment active when installing packages and running
simulations. The compilers are used when pip builds native extensions from
source. PyHEADTAIL provides the beam dynamics functionality used for coupled
PyECLOUD–PyHEADTAIL simulations.

PyECLOUD declares its Python dependencies, including the Poisson solver package
`pypic-poisson` and the sparse solver `PyKLU`, so pip installs these dependencies as needed. The Python import
names are `PyECLOUD`, `PyHEADTAIL`, `PyPIC` and `PyKLU`.

For later sessions, activate the existing environment with:

```bash
conda activate ecloud
```

## Optional packages

[PyKLU](https://pypi.org/project/PyKLU/) supplies the KLU sparse solver, and
[nafflib](https://pypi.org/project/nafflib/) provides frequency analysis. Both
are available from PyPI:

```bash
pip install --upgrade PyKLU nafflib
```

PyKLU is needed when your input explicitly selects the KLU solver. The default
installation also supports the SciPy solver through PyPIC. NAFFlib is useful
for tune and footprint analysis and is not required for a basic buildup run.
If PyKLU needs to build from source, consult its PyPI installation instructions
for the additional CMake and BLAS requirements.

MPI execution requires a separate MPI runtime and `mpi4py`; see
[parallel execution](tutorials/parallel.md). Local multiprocessing does not
require these MPI dependencies.

## Editable installation from Git

For development, use an editable installation so imports use the files in your
checkout. First prepare the environment with the same Conda commands above and
install PyHEADTAIL. Replace the final `pip install --upgrade pyecloud` command
with the following Git checkout and editable install.

With Git installed, clone the main PyECLOUD repository:

```bash
git clone https://github.com/PyCOMPLETE/PyECLOUD.git
cd PyECLOUD
pip install -e .
```

For an existing checkout, activate `ecloud`, change to the repository root (the folder
containing `pyproject.toml`), and run:

```bash
pip install -e .
```

Pip installs the build dependencies and compiles the native extensions. Python
source changes take effect in subsequent runs without reinstalling. Restart
running Python sessions or notebook kernels to load your changes. Rerun
`pip install -e .` after changing C, Cython, or Fortran sources, or package
dependencies.

To install development test dependencies as well:

```bash
pip install -e '.[tests]'
```

The documentation has its own lightweight environment; see
[Working on the documentation](documentation.md) for build and preview commands.
