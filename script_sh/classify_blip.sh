#!/bin/bash
#SBATCH --job-name=BLIP_classify
#SBATCH --output=BLIP_faithfulness/classify_%j.out
#SBATCH --error=BLIP_faithfulness/classify_%j.err
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
index_json="${INDEX_JSON:-$repo_root/imgnet1k_label.json}"
num_samples="${NUM_SAMPLES:-100}"
batch_size="${BATCH_SIZE:-32}"
last_heads="${LAST_HEADS:-native}"
output_dir="${OUTPUT_DIR:-$repo_root/classification_reuslt/imagenet}"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

extra_args=()
if [[ -n "${CLASSNAMES_PATH:-}" ]]; then
  extra_args+=(--classnames-path "$CLASSNAMES_PATH")
fi

python classify.py \
  --model blip \
  --data-path "$img_dir" \
  --index-json "$index_json" \
  --blip-root "$repo_root/BLIP" \
  --blip-checkpoint "$checkpoint" \
  --blip-last-heads "$last_heads" \
  --max-correct-samples "$num_samples" \
  --batch-size "$batch_size" \
  --output "$output_dir/blip_${num_samples}_summary.json" \
  --sample-output "$output_dir/blip_${num_samples}.json" \
  --device cuda \
  "${extra_args[@]}"

python - "$output_dir/blip_${num_samples}.json" "$num_samples" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    samples = json.load(stream)
expected = int(sys.argv[2])
if len(samples) != expected:
    raise SystemExit(
        f"Only {len(samples)} BLIP-correct samples found; requested {expected}. "
        f"Partial results saved to {sys.argv[1]}."
    )
print(f"BLIP classification complete: {len(samples)} correct samples in {sys.argv[1]}")
PY
