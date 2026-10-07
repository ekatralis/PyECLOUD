# Single-bunch beam dynamics

PyECLOUD's `PyEC4PyHT` module supplies electron cloud elements for PyHEADTAIL.
The `PyPARIS_sim_class` package assembles a single-bunch simulation from
`Simulation_parameters.py`; PyPARIS provides serial, multiprocessing, and MPI
launchers. These PyPARIS packages are included with PyECLOUD.

## Prepare a working directory

Complete the [installation](../installation.md), including PyHEADTAIL, and obtain
the source checkout as described in the [buildup tutorial](buildup.md). Start
from the checkout root:

```bash
PYECL_SOURCE="$PWD"
mkdir ../ecloud-single-bunch
cp -r "$PYECL_SOURCE/PyPARIS_sim_class/examples/001b_simulation_with_eclouds_synchrotron/." ../ecloud-single-bunch/
cd ../ecloud-single-bunch
```

Review `Simulation_parameters.py` and the files under `pyecloud_config/`.
The [single-bunch parameter reference](../parameters/single-bunch.md) describes
the machine, bunch, electron cloud, and output settings.

For an initial manually launched run, edit the existing settings or add the
following at the end of `Simulation_parameters.py`:

```python
check_for_resubmit = False
N_turns_target = N_turns
```

This runs one job's worth of turns without requesting scheduler resubmission.
The example's particle count and turn count can require substantial computing
time. Reducing them changes the numerical resolution and physics statistics.

## Choose an execution mode

Run **one** of the following commands from the working directory.

### Serial

```bash
python -m PyPARIS.serialexec sim_class=PyPARIS_sim_class.Simulation.Simulation
```

### Multiprocessing on one machine

```bash
python -m PyPARIS.multiprocexec -n 3 sim_class=PyPARIS_sim_class.Simulation.Simulation
```

`-n` is the total process count, including the master. Multiprocessing does not
require `mpi4py`.

### MPI

With a compatible MPI runtime and `mpi4py` installed:

```bash
mpiexec -n 3 python -m PyPARIS.withmpi sim_class=PyPARIS_sim_class.Simulation.Simulation
```

See [parallel execution](parallel.md) for environment setup. On a cluster, run
inside a scheduler allocation and use the site's supported launch command.

## Plot the first part

The first simulation part writes `bunch_evolution_00.h5`. Save this snippet as
`plot_bunch.py` and execute `python plot_bunch.py`:

```python
import matplotlib.pyplot as plt
from PyPARIS.myfilemanager import monitorh5_to_obj

bunch = monitorh5_to_obj("bunch_evolution_00.h5")
fig, ax = plt.subplots()
ax.plot(bunch.mean_x, label="Horizontal")
ax.plot(bunch.mean_y, label="Vertical")
ax.set_xlabel("Recorded turn index")
ax.set_ylabel("Bunch centroid [m]")
ax.legend()
fig.tight_layout()
fig.savefig("bunch-centroid.png", dpi=150)
```

The historical `004_some_checks.py` expects three completed parts. The example
shell wrappers also delete the status file and run three parts; use the explicit
launch commands above for this one-part tutorial.

## Continue or start another simulation

Keep `simulation_status.sta` and the saved bunch state to continue a completed
part. Another invocation advances to the next part; `N_turns_target` controls
automatic resubmission, not whether a manually launched invocation is allowed.
For an independent simulation, prepare a new working directory.

See [restart guidance](restart.md#single-bunch-multijob-runs) before recovering an
interrupted part or enabling automatic resubmission.

The bundled `PyPARIS_CoupledBunch_sim_class` has its own examples and configuration.
Use those examples for its ring layout, turn constraints, and launch commands.

Adapted from the [original PyECLOUD–PyHEADTAIL tutorial](https://github.com/PyCOMPLETE/PyECLOUD/wiki/How-to-perform-PyECLOUD-PyHEADTAIL-simulations).
