#!/bin/bash
#SBATCH --job-name=SigLIP_classify
#SBATCH --output=SigLIP_classify_%j.out
#SBATCH --error=SigLIP_classify_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -eo pipefail

nvidia-smi -L
conda_sh="${CONDA_SH:-/datastore/khangtd/miniconda3/etc/profile.d/conda.sh}"
source "$conda_sh"
conda activate "${CONDA_ENV:-SigLIP}"

repo_root="${REPO_ROOT:-/datastore/khangtd/Adversarial-Robustness-Grad-Eclip}"
cd "$repo_root"

img_dir="${IMG_DIR:?Set IMG_DIR to the ImageNet validation image directory}"
index_json="${INDEX_JSON:-$repo_root/imgnet1k_label.json}"
num_samples="${NUM_SAMPLES:-100}"
batch_size="${BATCH_SIZE:-32}"
model="${MODEL:-google/siglip-base-patch16-224}"
output_dir="${OUTPUT_DIR:-$repo_root/classification_reuslt/imagenet}"
prompt_template="${PROMPT_TEMPLATE:-a photo of {}}"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
if python -c 'import certifi' >/dev/null 2>&1; then
  export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
  export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
fi

python -c 'from transformers import SiglipModel, SiglipProcessor'
extra_args=()
if [[ -n "${CLASSNAMES_PATH:-}" ]]; then
  extra_args+=(--classnames-path "$CLASSNAMES_PATH")
fi

python classify.py \
  --model siglip \
  --siglip-model "$model" \
  --data-path "$img_dir" \
  --index-json "$index_json" \
  --max-correct-samples "$num_samples" \
  --batch-size "$batch_size" \
  --prompt-template "$prompt_template" \
  --output "$output_dir/siglip_${num_samples}_summary.json" \
  --sample-output "$output_dir/siglip_${num_samples}.json" \
  --device cuda \
  "${extra_args[@]}"

python - "$output_dir/siglip_${num_samples}.json" "$num_samples" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    samples = json.load(stream)
expected = int(sys.argv[2])
if len(samples) != expected:
    raise SystemExit(
        f"Only {len(samples)} SigLIP-correct samples found; requested {expected}. "
        f"Partial results saved to {sys.argv[1]}."
    )
print(f"SigLIP classification complete: {len(samples)} correct samples in {sys.argv[1]}")
PY
