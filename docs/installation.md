# Installation

## Released packages from PyPI

With Conda available, create an environment with Python 3.13, install the
compilers and numerical libraries, then install PyHEADTAIL and PyECLOUD:

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
`pypic-poisson`, so pip installs these dependencies as needed. The Python import
names are `PyECLOUD`, `PyHEADTAIL`, and `PyPIC`.

For later sessions, activate the existing environment with:

```bash
conda activate ecloud
```

## Editable installation from Git

For development, use an editable installation so imports use the files in your
checkout. First prepare the environment with the same Conda commands above and
install PyHEADTAIL. Replace the final `pip install --upgrade pyecloud` command
with the following Git checkout and editable install.

The packaging and documentation refactor currently lives on the `main-rc1`
branch of `ekatralis/PyECLOUD`. To work on that branch, with Git installed:

```bash
git clone --branch main-rc1 https://github.com/ekatralis/PyECLOUD.git
cd PyECLOUD
pip install -e .
```

Use the repository URL and branch you intend to develop if they differ. For an
existing checkout, activate `ecloud`, change to the repository root (the folder
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
