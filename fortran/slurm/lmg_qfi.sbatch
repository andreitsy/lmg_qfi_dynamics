#!/bin/bash
# One run of the Fortran QFI solver on a Slurm cluster (OpenMP, one node).
#
# Submit from a run directory <project>/tmp/<experiment>/<run>/; all options
# after the script name go to the solver. Choose the partition on the command
# line if the cluster has no default one:
#
#   mkdir -p logs
#   sbatch --parsable --partition=<partition> lmg_qfi.sbatch --system-size 50 --steps 250 --num-points 300
#
# A parameters.ini in the run directory is picked up automatically. Results
# are written to ./results, logs to ./logs. The executable is taken from
# $LMG_QFI or from <project>/fortran/build/lmg_qfi (built beforehand with
# make on the login node). Adjust --cpus-per-task and --time to the run; the
# solver uses all CPUs of the allocation.
#
#SBATCH --job-name=lmg-qfi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p tmp results
export TMPDIR="$PWD/tmp"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"
# One thread per allocated hardware thread, pinned inside the job's cpuset.
export OMP_PLACES=threads
export OMP_PROC_BIND=close

exe="${LMG_QFI:-$PWD/../../../fortran/build/lmg_qfi}"
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" "$exe" --output-dir results "$@"
