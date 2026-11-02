#!/bin/bash
#SBATCH --job-name=SigLIP_explainers
#SBATCH --output=SigLIP_explainers_%j.out
#SBATCH --error=SigLIP_explainers_%j.err
#SBATCH --account=uit
#SBATCH --partition=large
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -eo pipefail

nvidia-smi -L
conda_sh="${CONDA_SH:-/datastore/khangtd/miniconda3/etc/profile.d/conda.sh}"
source "$conda_sh"
conda activate "${CONDA_ENV:-SigLIP}"

repo_root="${REPO_ROOT:-/datastore/khangtd/Adversarial-Robustness-Grad-Eclip}"
cd "$repo_root"
image="${IMAGE_PATH:?Set IMAGE_PATH to one input image}"
text="${TEXT:?Set TEXT to the target text, e.g. 'a photo of a dog'}"
model="${MODEL:-google/siglip-base-patch16-224}"
output="${OUTPUT_DIR:-$repo_root/SIGLIP_explainer_results}"
read -r -a methods <<< "${METHODS:-gradeclip game gradcam maskclip rise}"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
if python -c 'import certifi' >/dev/null 2>&1; then
  export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
  export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
fi

python -c 'from transformers import SiglipModel, SiglipProcessor'
python test_siglip_explainers.py \
  --model "$model" \
  --image "$image" \
  --text "$text" \
  --output "$output" \
  --methods "${methods[@]}" \
  --device cuda \
  --rise-masks "${RISE_MASKS:-2000}" \
  --batch-size "${RISE_BATCH_SIZE:-50}"
