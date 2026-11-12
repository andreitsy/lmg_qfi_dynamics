#!/bin/bash
# Parameter sweep with the Fortran QFI solver as a Slurm job array.
#
# The run directory holds a file `cases.txt` with one set of solver options
# per line (empty lines and lines starting with # are ignored), e.g.
#
#   --system-size 20 --steps 250 --num-points 300
#   --system-size 40 --steps 250 --num-points 300
#   --system-size 50 --steps 250 --num-points 300
#
# Submit one array task per case; %K limits the number of tasks running at
# once (set it to what your account allows):
#
#   mkdir -p logs
#   ncases=$(grep -cvE '^\s*(#|$)' cases.txt)
#   sbatch --parsable --partition=<partition> --array=1-"$ncases"%4 lmg_qfi_array.sbatch
#
# Options given after the script name are appended to every case.
#
#SBATCH --job-name=lmg-qfi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p tmp results
export TMPDIR="$PWD/tmp"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"
export OMP_PLACES=threads
export OMP_PROC_BIND=close

case_line=$(grep -vE '^\s*(#|$)' cases.txt | sed -n "${SLURM_ARRAY_TASK_ID:?}p")
if [ -z "$case_line" ]; then
    echo "cases.txt has no case number $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi
read -r -a case_args <<< "$case_line"
echo "case $SLURM_ARRAY_TASK_ID: ${case_args[*]} $*"

exe="${LMG_QFI:-$PWD/../../../fortran/build/lmg_qfi}"
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" "$exe" --output-dir results "${case_args[@]}" "$@"
