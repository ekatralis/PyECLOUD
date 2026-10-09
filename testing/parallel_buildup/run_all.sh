#!/usr/bin/env bash

rm -r outputs
mpiexec -n 4 python simulation_check.py

mkdir -p outputs
mv output.png outputs/output.png