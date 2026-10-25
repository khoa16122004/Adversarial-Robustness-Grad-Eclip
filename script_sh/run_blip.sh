#!/bin/bash
#SBATCH --job-name=BLIP_attack
#SBATCH --output=BLIP_faithfulness/attack_%j.out
#SBATCH --error=BLIP_faithfulness/attack_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
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
img_dir="${IMG_DIR:?Set IMG_DIR to the Stanford Dogs Images directory on this server}"
sample_path="${SAMPLE_PATH:-$repo_root/classification_reuslt/pet/vit_b_16_100_random_50.json}"
output_root="${OUTPUT_DIR:-$repo_root/BLIP_faithfulness/attack_${SLURM_JOB_ID:-local}}"
process_batch_size="${PROCESS_BATCH_SIZE:-100}"
rise_masks="${RISE_MASKS:-2000}"
rise_batch_size="${RISE_BATCH_SIZE:-50}"
last_heads="${LAST_HEADS:-1}"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

for file in "$checkpoint" "$sample_path" "$repo_root/adv_score_BLIP.py" "$repo_root/BLIP/models/blip_itm.py"; do
  if [[ ! -f "$file" ]]; then
    printf 'Required file not found: %s\nSet CHECKPOINT or SAMPLE_PATH to the server path as needed.\n' "$file" >&2
    exit 1
  fi
done
if [[ ! -d "$img_dir" ]]; then
  printf 'Image directory not found: %s\nSet IMG_DIR to the server dataset path.\n' "$img_dir" >&2
  exit 1
fi
command -v zip >/dev/null || { printf 'zip is required to archive the results.\n' >&2; exit 1; }

python - <<'PY'
from importlib.metadata import version

installed = version("transformers")
if installed != "4.26.1":
    raise SystemExit(
        f"BLIP requires transformers==4.26.1; found {installed}.\n"
        "Run: python -m pip install -r requirements_blip_compat.txt"
    )
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; check the Slurm GPU allocation and PyTorch environment.")
print(f"BLIP preflight OK: transformers={installed}, GPU={torch.cuda.get_device_name(0)}")
PY
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

hm_types=(eclip game gradcam maskclip rise)
epsilons=(2 4 8)

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
      --mode del+ins \
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
