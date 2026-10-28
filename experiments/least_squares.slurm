#!/bin/bash
# Submit through submit_least_squares.sh so log directories exist before launch.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=tsgd-ls

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit using submit_least_squares.sh}"
profile="${1:-full}"
if [[ $# -gt 0 ]]; then shift; fi
case "$profile" in
    smoke) default_runs=200; default_steps=50 ;;
    full) default_runs=4000; default_steps=400 ;;
    *) echo "Unknown profile: $profile" >&2; exit 2 ;;
esac

# Same environment/bootstrap convention as FL_EF21_Second_Order.
# No installs or upgrades are performed by the job.
if [[ -n "${CONDA_BASE:-}" ]]; then
    tsgd_conda_base="$CONDA_BASE"
elif command -v conda >/dev/null 2>&1; then
    tsgd_conda_base="$(conda info --base)"
else
    tsgd_conda_base="/ibex/user/maziany/miniforge"
fi
if [[ ! -f "$tsgd_conda_base/etc/profile.d/conda.sh" ]]; then
    echo "Cannot find Conda at $tsgd_conda_base. Set CONDA_BASE to your installation." >&2
    exit 1
fi
source "$tsgd_conda_base/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-FL_EF21_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

output_dir="${PWD}/results/cluster/${profile}_${SLURM_JOB_ID}"
mkdir -p "$output_dir"
export MPLCONFIGDIR="${output_dir}/matplotlib-cache"
python_bin="${PYTHON_BIN:-python}"
printf 'Conda environment: %s\n' "${CONDA_DEFAULT_ENV:-unknown}"
printf 'Job: %s; CPUs: %s; output: %s\n' "$SLURM_JOB_ID" "$OMP_NUM_THREADS" "$output_dir"
"$python_bin" -c 'import sys, numpy, matplotlib; print("Python:", sys.executable); print("NumPy:", numpy.__version__); print("Matplotlib:", matplotlib.__version__)'
"$python_bin" -B -m unittest discover -s . -p 'test_*.py' -v
"$python_bin" -B -u least_squares.py \
    --runs "$default_runs" --steps "$default_steps" \
    --output "$output_dir" "$@"
