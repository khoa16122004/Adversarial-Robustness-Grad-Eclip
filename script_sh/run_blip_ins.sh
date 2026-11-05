#!/bin/bash
#SBATCH --job-name=BLIP_attack
#SBATCH --output=BLIP_new/attack_%j.out
#SBATCH --error=BLIP_new/attack_%j.err
#SBATCH --account=uit
#SBATCH --partition=large
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -eo pipefail

nvidia-smi -L
source /datastore/khangtd/miniconda3/etc/profile.d/conda.sh
conda activate Sparse_Attack
cd /datastore/khangtd/Adversarial-Robustness-Grad-Eclip || exit 1
repo_root="$PWD"

checkpoint="${CHECKPOINT:-$repo_root/blip_checkpoint.pth}"
output_root="${OUTPUT_DIR:-$repo_root/BLIP_result_rerun}"
mode="${BLIP_MODE:-ins}"
case "$mode" in
  ins|del|del+ins) ;;
  *) printf 'Unsupported BLIP_MODE: %s\n' "$mode" >&2; exit 1 ;;
esac
datasets=(ImageNet StandfordPet cub)
img_dirs=(
  "${IMAGENET_IMG_DIR:-/datastore/quanphm/datasets/ImageNet/val}"
  "${PET_IMG_DIR:-$repo_root/StandfordPet_Dataset}"
  "${CUB_IMG_DIR:-$repo_root/cub_dataset}"
)
sample_paths=(
  "${IMAGENET_SAMPLE_PATH:-$repo_root/classification_reuslt/imagenet/blip_100.json}"
  "${PET_SAMPLE_PATH:-$repo_root/classification_reuslt/pet/blip_100.json}"
  "${CUB_SAMPLE_PATH:-$repo_root/classification_reuslt/cub/blip_100.json}"
)
classnames_paths=(
  "${IMAGENET_CLASSNAMES_PATH:-$repo_root/imgnet1k_label.json}"
  "${PET_CLASSNAMES_PATH:-$repo_root/pet.json}"
  "${CUB_CLASSNAMES_PATH:-$repo_root/cub.json}"
)
process_batch_size="${PROCESS_BATCH_SIZE:-32}"
rise_masks="${RISE_MASKS:-2000}"
rise_batch_size="${RISE_BATCH_SIZE:-50}"
last_heads="${LAST_HEADS:-1}"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1


export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

hm_types=(eclip game gradcam maskclip rise)
epsilon=16
alpha=4

for index in "${!datasets[@]}"; do
  [[ -d "${img_dirs[$index]}" ]] || { printf 'Missing image directory: %s\n' "${img_dirs[$index]}" >&2; exit 1; }
  [[ -f "${sample_paths[$index]}" ]] || { printf 'Missing sample JSON: %s\n' "${sample_paths[$index]}" >&2; exit 1; }
  if [[ -n "${classnames_paths[$index]}" && ! -f "${classnames_paths[$index]}" ]]; then
    printf 'Missing class names file: %s\n' "${classnames_paths[$index]}" >&2
    exit 1
  fi
done

mkdir -p "$output_root"
output_root="$(cd -- "$output_root" && pwd)"
for index in "${!datasets[@]}"; do
  dataset="${datasets[$index]}"
  extra_args=()
  if [[ -n "${classnames_paths[$index]}" ]]; then
    extra_args+=(--classnames-path "${classnames_paths[$index]}")
  fi
  for hm in "${hm_types[@]}"; do
    printf '\n========== BLIP %s | %s | mode=%s | eps=16 ==========\n\n' "$dataset" "$hm" "$mode"

    # Python appends <hm-type>/<mode>; do not duplicate hm in --output-dir.
    python adv_score_BLIP.py \
      --blip-root "$repo_root/BLIP" \
      --blip-checkpoint "$checkpoint" \
      --device cuda \
      --last-heads "$last_heads" \
      --hm-type "$hm" \
      --mode "$mode" \
      --img-dir "${img_dirs[$index]}" \
      --sample-path "${sample_paths[$index]}" \
      --output-dir "$output_root/$dataset/eps_$epsilon" \
      --eps "$epsilon" \
      --alpha "$alpha" \
      --pgd-steps 50 \
      --process-batch-size "$process_batch_size" \
      --rise-masks "$rise_masks" \
      --rise-batch-size "$rise_batch_size" \
      "${extra_args[@]}"
  done
done

printf '\nBLIP mode=%s complete for three datasets. Results: %s\n' "$mode" "$output_root"
