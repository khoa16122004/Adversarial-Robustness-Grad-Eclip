#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

"${PYTHON}" "${REPO_ROOT}/adv_score_CLIP_fixed_masked.py" \
  --mask-source "${MASK_SOURCE:-clean}" \
  --seed "${SEED:-0}" \
  --mode "${MODE:-del+ins}" \
  "$@"
