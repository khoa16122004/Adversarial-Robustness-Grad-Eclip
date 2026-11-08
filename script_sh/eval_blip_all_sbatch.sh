#!/bin/bash
#SBATCH --job-name=BLIP_eval_all
#SBATCH --output=blip_eval_all_%j.out
#SBATCH --error=blip_eval_all_%j.err
#SBATCH --account=uit
#SBATCH --partition=large
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -euo pipefail

repo_root="${REPO_ROOT:-/datastore/khangtd/Adversarial-Robustness-Grad-Eclip}"
conda_init="${CONDA_INIT:-/datastore/khangtd/miniconda3/etc/profile.d/conda.sh}"
source "$conda_init"
conda activate "${CONDA_ENV:-Sparse_Attack}"
cd "$repo_root"

python_bin="${PYTHON:-python}"
clean_root="${CLEAN_ROOT:-$repo_root/BLIP_result_clean}"
output_dir="${OUTPUT_DIR:-$repo_root/BLIP_evaluation_tables}"
epsilon="${EPSILON:-16}"
attack_roots=()
overwrite=()
checkpoint=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean-root|--attack-root|--output-dir|--epsilon|--checkpoint)
      [[ $# -ge 2 ]] || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
      case "$1" in
        --clean-root) clean_root="$2" ;;
        --attack-root) attack_roots+=("$2") ;;
        --output-dir) output_dir="$2" ;;
        --epsilon) epsilon="$2" ;;
        --checkpoint) checkpoint=(--checkpoint "$2") ;;
      esac
      shift 2 ;;
    --overwrite) overwrite=(--overwrite); shift ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ ${#attack_roots[@]} -eq 0 ]]; then
  attack_roots=("${RESULT_ROOT:-$repo_root/BLIP_result_rerun}" "${XSHIFT_ROOT:-$repo_root/BLIP_result_xshift}")
fi
for root in "$clean_root" "${attack_roots[@]}"; do
  [[ -d "$root" ]] || { printf 'Missing results folder: %s\n' "$root" >&2; exit 1; }
done
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
nvidia-smi -L
common=(--epsilon "$epsilon" --blip-root "$repo_root/BLIP" --device cuda
        --noise "${ROAD_NOISE:-0.01}" --seed "${SEED:-42}")

# Evaluate existing clean artifacts, without regenerating explanations or attacks.
"$python_bin" road_score_BLIP.py --input-root "$clean_root" --clean-only \
  "${common[@]}" "${checkpoint[@]}" "${overwrite[@]}"
for root in "${attack_roots[@]}"; do
  "$python_bin" road_score_BLIP.py --input-root "$root" \
    "${common[@]}" "${checkpoint[@]}" "${overwrite[@]}"
done
"$python_bin" script_eval/export_blip_average.py \
  --clean-root "$clean_root" --attack-roots "${attack_roots[@]}" \
  --epsilon "$epsilon" --output-dir "$output_dir"
printf '\nFinished BLIP evaluation and averaged tables: %s\n' "$output_dir"
