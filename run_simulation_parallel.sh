#!/usr/bin/env bash

# Exit on any error
set -e

x_coupling_values=( 0.4 )
n_values=( 10 )
states=("GS_phys" "GS_cat" "CatSum" "Phys")

trap 'echo "🛑 Ctrl-C caught, killing children..."; kill 0' INT

for B in "${x_coupling_values[@]}"; do
  for N in "${n_values[@]}"; do
    for s in "${states[@]}"; do
        uv run python quantum_fisher_information_simulation_mpmath.py \
               --system-size "$N" --x-coupling "$B" --init-state "$s" &
    done
  done
done

wait

echo "✅ Simulation finished."

for B in "${x_coupling_values[@]}"; do
  for N in "${n_values[@]}"; do
    uv run python ./quantum_fisher_information_simulation_mpmath.py \
           --system-size "$N" --x-coupling "$B" --plot
  done
done
