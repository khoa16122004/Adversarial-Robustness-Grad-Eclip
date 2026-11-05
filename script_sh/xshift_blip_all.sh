#!/bin/bash
#SBATCH --job-name=BLIP_XShift_16
#SBATCH --output=BLIP_faithfulness/xshift_16_%j.out
#SBATCH --error=BLIP_faithfulness/xshift_16_%j.err
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

python - "${RESULT_ROOT:-$PWD/BLIP_result_rerun}" "${OUTPUT_ROOT:-$PWD/BLIP_result_xshift}" "$@" <<'PY'
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
    ("StandfordPet", {"stanfordpet", "stanfordpets", "standfordpet", "standfordpets"}),
    ("cub", {"cub"}),
)
settings = (
    "img_dir", "sample_path", "checkpoint", "vit", "last_heads", "image_size", "step",
    "kernel_size", "kernel_sigma", "classnames_path", "prompt_template",
    "game_start_layer", "rise_masks", "rise_grid", "rise_prob", "rise_batch_size",
    "seed", "target_source",
)
jobs = []
for dataset, aliases in datasets:
    folders = [p for p in root.iterdir() if p.is_dir() and p.name.lower() in aliases]
    if len(folders) != 1:
        raise ValueError(f"Expected exactly one {dataset} directory: {folders}")
    folder = folders[0]
    configs = []
    for path in sorted((folder / "eps_16").glob("**/run_config.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        if config.get("model") == "BLIP_ITC" and config.get("mode") in ("ins", "del", "del+ins"):
            configs.append((path, config))
    if not configs:
        raise FileNotFoundError(f"No BLIP causal attack config under {folder}/eps_16")
    reference_path, reference = configs[0]
    for path, config in configs:
        for key in settings:
            if config.get(key) != reference.get(key):
                raise ValueError(f"{key} differs between {reference_path} and {path}")
        if config.get("target_source", "pred") != "pred":
            raise ValueError(f"X-Shift launcher requires predicted-class targets: {path}")
    for key in ("img_dir", "sample_path", "checkpoint"):
        override = os.environ.get("CHECKPOINT") if key == "checkpoint" else None
        path = Path(override or reference[key])
        if not path.exists():
            raise FileNotFoundError(f"{key} from {reference_path} does not exist: {path}")
    command = [
        sys.executable, "XShift_score_BLIP.py",
        "--run-config", str(reference_path),
        "--output-dir", str(output_root / folder.name / "eps_16"),
        "--blip-root", str(Path("BLIP").resolve()),
        "--device", "cuda", "--eps", "16", "--alpha", "4", "--xshift-steps", "50",
        "--methods", "eclip", "game", "gradcam", "maskclip", "rise",
    ]
    if os.environ.get("CHECKPOINT"):
        command.extend(["--checkpoint", os.environ["CHECKPOINT"]])
    command.extend(sys.argv[3:])
    jobs.append(command)

for command in jobs:
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
print(f"Completed BLIP X-Shift eps=16 for all three datasets: {output_root}", flush=True)
PY
