#!/usr/bin/bash

set -euo pipefail

# Run MPI
mpiexec -n 4 python -m PyPARIS/withmpi.py sim_class=Simulation_with_eclouds.Simulation
