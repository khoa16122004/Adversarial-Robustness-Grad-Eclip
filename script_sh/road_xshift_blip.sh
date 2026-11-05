#!/bin/bash
#SBATCH --job-name=BLIP_XShift_ROAD
#SBATCH --output=BLIP_faithfulness/road_xshift_%j.out
#SBATCH --error=BLIP_faithfulness/road_xshift_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -eo pipefail
export RESULT_ROOT="${RESULT_ROOT:-/datastore/khangtd/Adversarial-Robustness-Grad-Eclip/BLIP_result_xshift}"
export EPSILON="${EPSILON:-16}"
bash /datastore/khangtd/Adversarial-Robustness-Grad-Eclip/script_sh/road_blip.sh --modes xshift "$@"
