#!/usr/bin/env bash
set -eo pipefail

repo_root="${REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"

output_root="${OUTPUT_DIR:-$repo_root/SIGLIP_result_xshift}"
model="${MODEL:-google/siglip-base-patch16-224}"
device="${DEVICE:-cuda}"
python_bin="${PYTHON:-python}"
datasets=(ImageNet StandfordPet cub)
img_dirs=(
  "${IMAGENET_IMG_DIR:-imagenetval_1k}"
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
read -r -a hm_types <<< "${HM_TYPES:-eclip game gradcam maskclip rise}"
eps="${EPS:-16}"
alpha="${ALPHA:-4}"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
if "$python_bin" -c 'import certifi' >/dev/null 2>&1; then
  export SSL_CERT_FILE="$("$python_bin" -c 'import certifi; print(certifi.where())')"
  export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
fi
"$python_bin" -c 'from transformers import SiglipModel, SiglipProcessor'

for index in "${!datasets[@]}"; do
  [[ -d "${img_dirs[$index]}" ]] || { printf 'Missing image directory: %s\n' "${img_dirs[$index]}" >&2; exit 1; }
  [[ -f "${sample_paths[$index]}" ]] || { printf 'Missing sample JSON: %s\n' "${sample_paths[$index]}" >&2; exit 1; }
  [[ -f "${classnames_paths[$index]}" ]] || { printf 'Missing class names file: %s\n' "${classnames_paths[$index]}" >&2; exit 1; }
done

extra_args=()
[[ -z "${PROMPT_TEMPLATE:-}" ]] || extra_args+=(--prompt-template "$PROMPT_TEMPLATE")
for index in "${!datasets[@]}"; do
  dataset="${datasets[$index]}"
  dataset_output="$output_root/$dataset"
  mkdir -p "$dataset_output"
  printf '\n========== SigLIP X-Shift | %s | eps=%s | alpha=%s ==========\n' "$dataset" "$eps" "$alpha"
  "$python_bin" XShift_score_SIGLIP.py \
    --model "$model" --device "$device" \
    --img-dir "${img_dirs[$index]}" \
    --sample-path "${sample_paths[$index]}" \
    --classnames-path "${classnames_paths[$index]}" \
    --output-dir "$dataset_output" \
    --hm-types "${hm_types[@]}" \
    --eps "$eps" --alpha "$alpha" \
    --xshift-steps "${XSHIFT_STEPS:-50}" \
    --rise-masks "${RISE_MASKS:-2000}" \
    --rise-batch-size "${RISE_BATCH_SIZE:-50}" \
    "${extra_args[@]}" "$@"
done
printf '\nSigLIP X-Shift complete. Results: %s\n' "$output_root"
