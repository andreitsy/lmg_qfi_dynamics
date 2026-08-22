#!/usr/bin/env bash

# Exit on any error
set -e

# Estimated parameter of the AC field, passed through to the solver:
#   PARAMETER=amplitude  (default) — QFI over the amplitude h
#   PARAMETER=frequency            — QFI over the drive frequency omega
#                                    (requires a nonzero AMPLITUDE)
# Optional knobs:
#   SOLVER=quspin|mpmath    — numerical backend (default: parameters.ini)
#   AMPLITUDE=<h>           — override h from parameters.ini
#   OMEGA=<value>           — probe frequency (default: resonant 2*pi/(nu*T))
#   MAX_TIME_DEGREE=<d>     — cap the simulated time at 10^d kick periods
#                             (frequency mode cost is linear in the max time)
#
# Example (frequency estimation with the mpmath backend):
#   SOLVER=mpmath PARAMETER=frequency AMPLITUDE=0.1 MAX_TIME_DEGREE=3 ./run_simulation_parallel.sh
PARAMETER="${PARAMETER:-amplitude}"
SOLVER="${SOLVER:-}"
AMPLITUDE="${AMPLITUDE:-}"
OMEGA="${OMEGA:-}"
MAX_TIME_DEGREE="${MAX_TIME_DEGREE:-}"

x_coupling_values=( 0.4 )
n_values=( 10 )
states=("GS_phys" "GS_cat" "CatSum" "Phys")

extra_args=( --parameter "$PARAMETER" )
[[ -n "$SOLVER" ]] && extra_args+=( --solver "$SOLVER" )
[[ -n "$AMPLITUDE" ]] && extra_args+=( --amplitude "$AMPLITUDE" )
[[ -n "$OMEGA" ]] && extra_args+=( --omega "$OMEGA" )
[[ -n "$MAX_TIME_DEGREE" ]] && extra_args+=( --max-time-degree "$MAX_TIME_DEGREE" )

plot_args=( --parameter "$PARAMETER" )
[[ -n "$SOLVER" ]] && plot_args+=( --solver "$SOLVER" )

trap 'echo "🛑 Ctrl-C caught, killing children..."; kill 0' INT

for B in "${x_coupling_values[@]}"; do
  for N in "${n_values[@]}"; do
    for s in "${states[@]}"; do
        uv run python quantum_fisher_information_simulation.py \
               --system-size "$N" --x-coupling "$B" --init-state "$s" \
               "${extra_args[@]}" &
    done
  done
done

wait

echo "✅ Simulation finished."

for B in "${x_coupling_values[@]}"; do
  for N in "${n_values[@]}"; do
    uv run python ./quantum_fisher_information_simulation.py \
           --system-size "$N" --x-coupling "$B" --plot "${plot_args[@]}"
  done
done
