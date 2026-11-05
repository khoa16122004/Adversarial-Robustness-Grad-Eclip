#!/bin/bash
set -eo pipefail
export SIGLIP_MODE=del
bash "${REPO_ROOT:-/datastore/khangtd/Adversarial-Robustness-Grad-Eclip}/script_sh/run_siglip_ins.sh"
