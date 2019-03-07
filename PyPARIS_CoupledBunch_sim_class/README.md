# Instructions for running PyECLOUD-PyHEADTAIL coupled bunch simulations on CERN HPC cluster

* We create a workspace folder for software and simulations:
```
mkdir /hpcscratch/user/lmether/sim_workspace
```

* We load MPI: 
```
module load mpi/mvapich2/2.2
```

* We build our software stack in this folder by following the instructions here: https://github.com/PyCOMPLETE/PyECLOUD/wiki/Setup-python-%28including-mpi4py%29-without-admin-rights

* We clone this example into the simulation workspace
```
cd /hpcscratch/user/lmether/sim_workspace
git clone https://github.com/lmether/PyECLOUD_coupled_bunch_example
```

* Change PYT in 000s...
