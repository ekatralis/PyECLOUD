#!/usr/bin/bash

set -euo pipefail

# Run Serial
python -m PyPARIS/serialexec.py sim_class=Simulation_with_eclouds.Simulation
