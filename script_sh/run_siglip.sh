#!/bin/bash
#SBATCH --job-name=SigLIP_attack
#SBATCH --output=SigLIP_attack_%j.out
#SBATCH --error=SigLIP_attack_%j.err
#SBATCH --account=uit
#SBATCH --partition=large
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -eo pipefail

nvidia-smi -L
conda_sh="${CONDA_SH:-/datastore/khangtd/miniconda3/etc/profile.d/conda.sh}"
source "$conda_sh"
conda activate "${CONDA_ENV:-SigLIP}"

repo_root="${REPO_ROOT:-/datastore/khangtd/Adversarial-Robustness-Grad-Eclip}"
cd "$repo_root"

img_dir="${IMG_DIR:?Set IMG_DIR to the ImageNet validation image directory}"
sample_path="${SAMPLE_PATH:?Set SAMPLE_PATH to a SigLIP sample JSON}"
output_root="${OUTPUT_DIR:-$repo_root/SIGLIP_result/ImageNet}"
model="${MODEL:-google/siglip-base-patch16-224}"
mode="${MODE:-ins}"
process_batch_size="${PROCESS_BATCH_SIZE:-32}"
rise_masks="${RISE_MASKS:-2000}"
rise_batch_size="${RISE_BATCH_SIZE:-50}"
pgd_steps="${PGD_STEPS:-50}"
default_prompt='a photo of {}'
prompt_template="${PROMPT_TEMPLATE:-$default_prompt}"
read -r -a hm_types <<< "${HM_TYPES:-eclip game gradcam maskclip rise}"
read -r -a epsilons <<< "${EPSILONS:-2 4 8 16}"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
if python -c 'import certifi' >/dev/null 2>&1; then
  export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
  export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
fi

python -c 'from transformers import SiglipModel, SiglipProcessor'
explainer_version="$(python -c 'from test_siglip_explainers import EXPLAINER_VERSION; print(EXPLAINER_VERSION)')"
output_root="$output_root/$explainer_version"
extra_args=()
if [[ -n "${CLASSNAMES_PATH:-}" ]]; then
  extra_args+=(--classnames-path "$CLASSNAMES_PATH")
fi
mkdir -p "$output_root"
output_root="$(cd -- "$output_root" && pwd)"
printf 'SigLIP explainer: %s\nOutput root: %s\n' "$explainer_version" "$output_root"

for hm in "${hm_types[@]}"; do
  for epsilon in "${epsilons[@]}"; do
    alpha="${ALPHA:-$(awk -v eps="$epsilon" 'BEGIN {print eps / 4}')}"
    printf '\n========== Running SigLIP %s | eps=%s | alpha=%s ==========\n\n' \
      "$hm" "$epsilon" "$alpha"
    python adv_score_SIGLIP.py \
      --model "$model" \
      --device cuda \
      --hm-type "$hm" \
      --mode "$mode" \
      --img-dir "$img_dir" \
      --sample-path "$sample_path" \
      --output-dir "$output_root/eps_$epsilon" \
      --prompt-template "$prompt_template" \
      --eps "$epsilon" \
      --alpha "$alpha" \
      --pgd-steps "$pgd_steps" \
      --process-batch-size "$process_batch_size" \
      --rise-masks "$rise_masks" \
      --rise-batch-size "$rise_batch_size" \
      "${extra_args[@]}"
  done
done

printf '\nSigLIP sweep complete. Results: %s\n' "$output_root"
