#!/bin/bash
#SBATCH --job-name=BLIP_del_16
#SBATCH --output=BLIP_faithfulness/del_16_%j.out
#SBATCH --error=BLIP_faithfulness/del_16_%j.err
#SBATCH --account=uit
#SBATCH --partition=large
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -eo pipefail
export BLIP_MODE=del
bash /datastore/khangtd/Adversarial-Robustness-Grad-Eclip/script_sh/run_blip_ins.sh
