#!/bin/bash
#SBATCH --job-name=BLIP_ROAD
#SBATCH --output=BLIP_faithfulness/road_%j.out
#SBATCH --error=BLIP_faithfulness/road_%j.err
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

python road_score_BLIP.py \
  --input-root "${RESULT_ROOT:-$PWD/BLIP_result}" \
  --epsilon "${EPSILON:-2}" \
  --blip-root "$PWD/BLIP" \
  --checkpoint "${CHECKPOINT:-$PWD/blip_checkpoint.pth}" \
  --device cuda \
  "$@"
