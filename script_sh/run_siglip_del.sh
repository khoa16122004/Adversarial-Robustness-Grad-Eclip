#!/bin/bash
set -eo pipefail
export SIGLIP_MODE=del
bash "$PWD/script_sh/run_siglip_ins.sh"
