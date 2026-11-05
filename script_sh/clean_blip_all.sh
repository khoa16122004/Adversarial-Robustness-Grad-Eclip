#!/bin/bash
#SBATCH --job-name=BLIP_clean_all
#SBATCH --output=BLIP_faithfulness/clean_all_%j.out
#SBATCH --error=BLIP_faithfulness/clean_all_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:2g.35gb:1

set -eo pipefail
source /datastore/khangtd/miniconda3/etc/profile.d/conda.sh
conda activate Sparse_Attack
cd /datastore/khangtd/Adversarial-Robustness-Grad-Eclip || exit 1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

python - "${RESULT_ROOT:-$PWD/BLIP_result_rerun}" "${OUTPUT_ROOT:-$PWD/BLIP_result_clean}" "$@" <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
if not root.is_dir():
    raise FileNotFoundError(f"BLIP result root not found: {root}")
output_root = Path(sys.argv[2]).resolve()
output_root.mkdir(parents=True, exist_ok=True)

datasets = (
    ("ImageNet", {"imagenet"}),
    ("StanfordPets", {"stanfordpet", "stanfordpets", "standfordpet", "standfordpets"}),
    ("CUB", {"cub"}),
)
methods = ("eclip", "game", "gradcam", "maskclip", "rise")
options = (
    "img_dir", "sample_path", "vit", "last_heads", "image_size", "step",
    "kernel_size", "kernel_sigma", "classnames_path", "prompt_template",
    "game_start_layer", "rise_masks", "rise_grid", "rise_prob",
    "rise_batch_size", "seed",
)
jobs = []
for display, aliases in datasets:
    folders = [p for p in root.iterdir() if p.is_dir() and p.name.lower() in aliases]
    if len(folders) != 1:
        raise ValueError(f"Expected exactly one {display} directory under {root}: {folders}")
    folder = folders[0]
    configs = []
    for path in sorted(folder.glob("eps_*/**/run_config.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        if config.get("model") == "BLIP_ITC":
            configs.append((path, config))
    if not configs:
        raise FileNotFoundError(f"No BLIP attack run_config.json under {folder}/eps_*")
    reference_path, reference = configs[0]
    for path, config in configs:
        for key in options + ("checkpoint",):
            if config.get(key) != reference.get(key):
                raise ValueError(
                    f"Attack setting {key!r} differs between {reference_path} and {path}. "
                    "One shared clean baseline cannot match both configurations."
                )
        if config.get("target_source", "pred") != "pred":
            raise ValueError(f"GT-target attacks need a matching GT clean runner: {path}")
    for key in ("img_dir", "sample_path", "checkpoint"):
        value = os.environ.get("CHECKPOINT") if key == "checkpoint" else None
        path = Path(value or reference[key])
        if not path.exists():
            raise FileNotFoundError(f"{key} from {reference_path} does not exist: {path}")
    command = [
        sys.executable, "clean_score_BLIP.py",
        "--dataset", folder.name,
        "--output-root", str(output_root),
        "--blip-root", str(Path("BLIP").resolve()),
        "--checkpoint", os.environ.get("CHECKPOINT") or reference["checkpoint"],
        "--device", "cuda", "--methods", *methods,
    ]
    for key in options:
        if reference.get(key) is not None:
            command.extend(["--" + key.replace("_", "-"), str(reference[key])])
    command.extend(sys.argv[3:])
    jobs.append(command)

for command in jobs:
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
print(f"Completed all clean BLIP datasets, five explainers each. Output: {output_root}", flush=True)
PY
