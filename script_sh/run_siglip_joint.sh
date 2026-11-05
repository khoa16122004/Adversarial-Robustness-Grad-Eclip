#!/bin/bash
set -eo pipefail
export SIGLIP_MODE=del+ins
bash "$PWD/script_sh/run_siglip_ins.sh"
