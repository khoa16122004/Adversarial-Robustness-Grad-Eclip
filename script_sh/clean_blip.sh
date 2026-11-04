#!/bin/bash
#SBATCH --job-name=BLIP_clean
#SBATCH --output=BLIP_faithfulness/clean_%j.out
#SBATCH --error=BLIP_faithfulness/clean_%j.err
#SBATCH --account=uit
#SBATCH --partition=large
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -eo pipefail
source /datastore/khangtd/miniconda3/etc/profile.d/conda.sh
conda activate Sparse_Attack
cd /datastore/khangtd/Adversarial-Robustness-Grad-Eclip || exit 1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

python clean_score_BLIP.py \
  --blip-root "$PWD/BLIP" \
  --checkpoint "${CHECKPOINT:-$PWD/blip_checkpoint.pth}" \
  --dataset "${DATASET:-ImageNet}" \
  --img-dir "${IMG_DIR:-/datastore/quanphm/datasets/ImageNet/val}" \
  --sample-path "${SAMPLE_PATH:-$PWD/classification_reuslt/imagenet/blip_100.json}" \
  --output-root "${OUTPUT_ROOT:-$PWD/BLIP_result}" \
  --last-heads "${LAST_HEADS:-1}" \
  --device cuda \
  "$@"
