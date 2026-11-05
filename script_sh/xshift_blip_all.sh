#!/bin/bash
#SBATCH --job-name=BLIP_XShift_16
#SBATCH --output=BLIP_faithfulness/xshift_16_%j.out
#SBATCH --error=BLIP_faithfulness/xshift_16_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -eo pipefail
source /datastore/khangtd/miniconda3/etc/profile.d/conda.sh
conda activate Sparse_Attack
cd /datastore/khangtd/Adversarial-Robustness-Grad-Eclip || exit 1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

repo_root="$PWD"
output_root="${OUTPUT_ROOT:-$repo_root/BLIP_result_xshift}"
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
mkdir -p "$output_root"
for index in "${!datasets[@]}"; do
  python XShift_score_BLIP.py \
    --output-dir "$output_root/${datasets[$index]}/eps_16" \
    --blip-root "$repo_root/BLIP" \
    --checkpoint "${CHECKPOINT:-$repo_root/blip_checkpoint.pth}" \
    --img-dir "${img_dirs[$index]}" \
    --sample-path "${sample_paths[$index]}" \
    --classnames-path "${classnames_paths[$index]}" \
    --last-heads "${LAST_HEADS:-1}" \
    --rise-masks "${RISE_MASKS:-2000}" \
    --rise-batch-size "${RISE_BATCH_SIZE:-50}" \
    --device cuda --eps 16 --alpha 4 --xshift-steps 50 \
    --methods eclip game gradcam maskclip rise "$@"
done
printf '\nCompleted BLIP X-Shift eps=16 for three datasets. Output: %s\n' "$output_root"
