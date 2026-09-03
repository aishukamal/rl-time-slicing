#!/usr/bin/env bash
# elastic-vd ELASTIC arm: identical config via run_vd_common.sh, driver
# swapped for the elastic fork (spare bootstrap + handles actor). The policy
# controller is started SEPARATELY (nohup python3 /vd-code/vd_controller.py
# --auto-live ...) once the driver reaches steady state — see VD-STATE.md
# runbook section.
set -euo pipefail
export VD_EXP_NAME="${VD_EXP_NAME:-vd-elastic}"
export VD_DRIVER="python3 /vd-code/vd_main_elastic.py"
export VD_CONFIG_PATH_ARG=""
export VD_ELASTIC=1
export VD_SPARE_GPU_UTIL="${VD_SPARE_GPU_UTIL:-0.70}"
exec bash "$(dirname "$0")/run_vd_common.sh"
