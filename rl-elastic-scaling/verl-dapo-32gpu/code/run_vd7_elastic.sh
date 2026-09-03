#!/usr/bin/env bash
# elastic-vd RECIPE #4 ELASTIC arm: identical config, elastic driver fork.
# Controller launched separately: python3 /vd-code/vd_controller.py
#   --auto-live --hard-collect 112 (of 128 required samples/micro-step).
set -euo pipefail
export VD_EXP_NAME="${VD_EXP_NAME:-vd7-elastic}"
export VD_DRIVER="python3 /vd-code/vd_main_elastic.py"
export VD_ELASTIC=1
export VD_SPARE_GPU_UTIL="${VD_SPARE_GPU_UTIL:-0.80}"
exec bash "$(dirname "$0")/run_vd7_common.sh"
