#!/bin/bash
# Run with bash; executable permissions are not required after a Windows checkout.
set -euo pipefail

usage() {
    echo "Usage: bash $0 [smoke|full] [--dry-run] [experiment Python arguments]"
    echo "Examples:"
    echo "  bash $0 smoke"
    echo "  bash $0 full --runs 20000"
    echo "  bash $0 full --dry-run"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
profile="${1:-full}"
if [[ $# -gt 0 ]]; then shift; fi
case "$profile" in
    smoke) time_limit="00:05:00" ;;
    full) time_limit="00:30:00" ;;
    *) usage >&2; exit 2 ;;
esac
dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then dry_run=1; shift; fi

experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# SLURM_SUBMIT_DIR records the submission cwd, independently of --chdir.
cd "$experiment_dir"
log_dir="${experiment_dir}/results/cluster/logs"
submit=(sbatch --parsable --chdir="$experiment_dir"
    --job-name="tsgd-${profile}"
    --cpus-per-task="${TSGD_CPUS:-1}" --mem="${TSGD_MEM:-4G}"
    --time="${TSGD_TIME:-$time_limit}"
    --output="${log_dir}/${profile}_%j.out"
    --error="${log_dir}/${profile}_%j.err"
    "${experiment_dir}/least_squares.slurm" "$profile" "$@")

if [[ "$dry_run" == 1 ]]; then
    printf 'Conda environment: %s\n' "${CONDA_ENV_NAME:-FL_EF21_ENV}"
    printf 'Submission command (not executed): '
    printf '%q ' "${submit[@]}"
    printf '\n'
    exit 0
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is unavailable. Run this script on the Slurm cluster, or use --dry-run." >&2
    exit 1
fi
# Slurm opens log files before the batch script starts: create this directory NOW.
mkdir -p "$log_dir"
job_reply="$("${submit[@]}")"
# --parsable may return jobid;cluster on a federated cluster.
job_id="${job_reply%%;*}"
printf 'Submitted %s job: %s\n' "$profile" "$job_reply"
printf 'Logs: %s/%s_%s.{out,err}\n' "$log_dir" "$profile" "$job_id"
printf 'Results: %s/results/cluster/%s_%s/\n' "$experiment_dir" "$profile" "$job_id"
printf 'Monitor: squeue -j %s\n' "$job_id"
