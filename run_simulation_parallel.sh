#!/usr/bin/env bash

# Exit on any error
set -e

# Default parameter pairs: N x_coupling
# Override by passing pairs as arguments, e.g.:
#   ./run_simulation_parallel.sh 5 0.4 10 0.2 20 0.1
default_pairs=("5 0.2")

if [[ $# -gt 0 ]]; then
  if (( $# % 2 != 0 )); then
    echo "Error: arguments must be pairs of N and x_coupling, e.g.: 5 0.4 10 0.2"
    exit 1
  fi
  pairs=()
  while [[ $# -gt 0 ]]; do
    N="$1"
    B="$2"
    if ! [[ "$N" =~ ^[0-9]+$ ]]; then
      echo "Error: N must be a positive integer, got '$N'"
      exit 1
    fi
    if ! [[ "$B" =~ ^[0-9]*\.?[0-9]+$ ]]; then
      echo "Error: x_coupling must be a positive number, got '$B'"
      exit 1
    fi
    pairs+=("$N $B")
    shift 2
  done
else
  pairs=("${default_pairs[@]}")
fi

states=("GS_phys" "GS_cat" "CatSum" "Phys")

trap 'echo "🛑 Ctrl-C caught, killing children..."; kill 0' INT

for pair in "${pairs[@]}"; do
  read -r N B <<< "$pair"
  for s in "${states[@]}"; do
      uv run python quantum_fisher_information_simulation_mpmath.py \
             --system-size "$N" --x-coupling "$B" --init-state "$s" &
  done
done

wait

echo "✅ Simulation finished."

for pair in "${pairs[@]}"; do
  read -r N B <<< "$pair"
  uv run python ./quantum_fisher_information_simulation_mpmath.py \
         --system-size "$N" --x-coupling "$B" --plot
done
