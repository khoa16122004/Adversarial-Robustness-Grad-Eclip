#!/bin/bash
#SBATCH --job-name=BLIP_attack
#SBATCH --output=BLIP_faithfulness/attack_%j.out
#SBATCH --error=BLIP_faithfulness/attack_%j.err
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
img_dir="${IMG_DIR:-/datastore/quanphm/datasets/ImageNet/val}"
sample_path="${SAMPLE_PATH:-$repo_root/classification_reuslt/imagenet/blip_100.json}"
output_root="${OUTPUT_DIR:-$repo_root/BLIP_result/ImageNet"
process_batch_size="${PROCESS_BATCH_SIZE:-32}"
rise_masks="${RISE_MASKS:-2000}"
rise_batch_size="${RISE_BATCH_SIZE:-50}"
last_heads="${LAST_HEADS:-1}"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1


export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

hm_types=(eclip game gradcam maskclip rise)
epsilons=(2 4 8 16)

mkdir -p "$output_root"
output_root="$(cd -- "$output_root" && pwd)"
for hm in "${hm_types[@]}"; do
  for epsilon in "${epsilons[@]}"; do
    alpha="$(awk -v eps="$epsilon" 'BEGIN {print eps / 4}')"
    printf '\n========== Running BLIP %s | eps=%s | alpha=%s ==========\n\n' "$hm" "$epsilon" "$alpha"

    # Python appends <hm-type>/<mode>; do not duplicate hm in --output-dir.
    python adv_score_BLIP.py \
      --blip-root "$repo_root/BLIP" \
      --blip-checkpoint "$checkpoint" \
      --device cuda \
      --last-heads "$last_heads" \
      --hm-type "$hm" \
      --mode ins \
      --img-dir "$img_dir" \
      --sample-path "$sample_path" \
      --output-dir "$output_root/eps_$epsilon" \
      --eps "$epsilon" \
      --alpha "$alpha" \
      --pgd-steps 50 \
      --process-batch-size "$process_batch_size" \
      --rise-masks "$rise_masks" \
      --rise-batch-size "$rise_batch_size"
  done
done

# Archive only after all 15 runs succeed; keep paths relative inside the zip.
archive="${output_root}.zip"
(
  cd -- "$(dirname -- "$output_root")"
  zip -r "$archive" "$(basename -- "$output_root")"
)
printf '\nBLIP sweep complete. Results: %s\nArchive: %s\n' "$output_root" "$archive"
