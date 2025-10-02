#!/usr/bin/env bash

# Exit on any error
set -e

x_coupling_values=( 0.3 0.5 )
n_values=( 15 30 )
states=("GS_phys" "GS_cat" "CatSum" "Phys")

for B in "${x_coupling_values[@]}"; do
  for N in "${n_values[@]}"; do
    python ./quantum_fisher_information_simulation_mpmath.py \
           --system-size "$N" --x-coupling "$B" --plot
  done
done
