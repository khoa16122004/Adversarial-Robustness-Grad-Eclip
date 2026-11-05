#!/bin/bash
set -eo pipefail
nvidia-smi -L
source "${CONDA_SH:-/datastore/khangtd/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-SigLIP}"
repo_root="${REPO_ROOT:-/datastore/khangtd/Adversarial-Robustness-Grad-Eclip}"
cd "$repo_root"

output_root="${OUTPUT_DIR:-$repo_root/SIGLIP_result_rerun}"
mode="${SIGLIP_MODE:-ins}"
case "$mode" in
  ins|del|del+ins) ;;
  *) printf 'Unsupported SIGLIP_MODE: %s\n' "$mode" >&2; exit 1 ;;
esac
datasets=(ImageNet StandfordPet cub)
img_dirs=(
  "${IMAGENET_IMG_DIR:-$repo_root/imagenetval_1k}"
  "${PET_IMG_DIR:-$repo_root/pet_dataset}"
  "${CUB_IMG_DIR:-$repo_root/cub_dataset}"
)
sample_paths=(
  "${IMAGENET_SAMPLE_PATH:-$repo_root/classification_reuslt/imagenet/siglip_100.json}"
  "${PET_SAMPLE_PATH:-$repo_root/classification_reuslt/pet/siglip_100.json}"
  "${CUB_SAMPLE_PATH:-$repo_root/classification_reuslt/cub/siglip_100.json}"
)
classnames_paths=(
  "${IMAGENET_CLASSNAMES_PATH:-$repo_root/imgnet1k_label.json}"
  "${PET_CLASSNAMES_PATH:-$repo_root/pet.json}"
  "${CUB_CLASSNAMES_PATH:-$repo_root/cub.json}"
)
model="${MODEL:-google/siglip-base-patch16-224}"
process_batch_size="${PROCESS_BATCH_SIZE:-32}"
rise_masks="${RISE_MASKS:-2000}"
rise_batch_size="${RISE_BATCH_SIZE:-50}"
pgd_steps="${PGD_STEPS:-50}"
default_prompt='a photo of {}'
prompt_template="${PROMPT_TEMPLATE:-$default_prompt}"
hm_types=(eclip game gradcam maskclip rise)

for index in "${!datasets[@]}"; do
  [[ -d "${img_dirs[$index]}" ]] || { printf 'Missing image directory: %s\n' "${img_dirs[$index]}" >&2; exit 1; }
  [[ -f "${sample_paths[$index]}" ]] || { printf 'Missing sample JSON: %s\n' "${sample_paths[$index]}" >&2; exit 1; }
  [[ -f "${classnames_paths[$index]}" ]] || { printf 'Missing class names file: %s\n' "${classnames_paths[$index]}" >&2; exit 1; }
done

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
python -c 'from transformers import SiglipModel, SiglipProcessor'
explainer_version="$(python -c 'from test_siglip_explainers import EXPLAINER_VERSION; print(EXPLAINER_VERSION)')"

for index in "${!datasets[@]}"; do
  dataset="${datasets[$index]}"
  dataset_output="$output_root/$dataset/$explainer_version/eps_16"
  for hm in "${hm_types[@]}"; do
    printf '\n========== SigLIP %s | %s | mode=%s | eps=16 ==========\n\n' "$dataset" "$hm" "$mode"
    python adv_score_SIGLIP.py \
      --model "$model" \
      --device cuda \
      --hm-type "$hm" \
      --mode "$mode" \
      --img-dir "${img_dirs[$index]}" \
      --sample-path "${sample_paths[$index]}" \
      --classnames-path "${classnames_paths[$index]}" \
      --output-dir "$dataset_output" \
      --prompt-template "$prompt_template" \
      --eps 16 --alpha 4 \
      --pgd-steps "$pgd_steps" \
      --process-batch-size "$process_batch_size" \
      --rise-masks "$rise_masks" \
      --rise-batch-size "$rise_batch_size"
  done
done
printf '\nSigLIP mode=%s complete for three datasets. Results: %s\n' "$mode" "$output_root"
