#!/bin/bash
#SBATCH --job-name=BLIP_faithfulness
#SBATCH --output=BLIP_faithfulness/normal_%j.out
#SBATCH --error=BLIP_faithfulness/normal_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      # Tối đa 16 CPU
#SBATCH --mem=4G               # Tối đa 72 GB RAM
#SBATCH --time=72:00:00        # Tối đa 3 ngày (72:00:00)
#SBATCH --gres=gpu:2g.35gb:1


nvidia-smi -L


source /datastore/khangtd/miniconda3/etc/profile.d/conda.sh
conda activate Sparse_Attack
cd /datastore/khangtd/Adversarial-Robustness-Grad-Eclip || exit 1
python - <<'PY' || exit 1
from importlib.metadata import PackageNotFoundError, version

try:
    installed = version("transformers")
except PackageNotFoundError:
    installed = "not installed"
if installed != "4.26.1":
    raise SystemExit(
        f"BLIP requires transformers==4.26.1; found {installed}.\n"
        "In the active conda environment, run:\n"
        "  python -m pip install -r requirements_blip_compat.txt\n"
        "Then resubmit this job."
    )
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
print(f"BLIP Transformers helper check passed ({installed}).")
PY
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID

rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY


checkpoint="${CHECKPOINT:-$PWD/blip_checkpoint.pth}"
output="${OUTPUT_DIR:-$PWD/test_BLIP/results_${SLURM_JOB_ID:-local}}"

if [[ ! -f "$checkpoint" ]]; then
  printf 'BLIP checkpoint not found: %s\nSet CHECKPOINT to the checkpoint path on this server.\n' "$checkpoint" >&2
  exit 1
fi

python test_blip_explainers.py \
  --blip-root "$PWD/BLIP" \
  --checkpoint "$checkpoint" \
  --output "$output" \
  --image "$PWD/images/ostrich.png" \
  --text "an ostrich" \
  --rise-masks 32 --batch-size 4
