#!/usr/bin/env bash
# elastic-vd RECIPE #4 BASELINE arm: stock driver module end-to-end.
set -euo pipefail
export VD_EXP_NAME="${VD_EXP_NAME:-vd7-baseline}"
export VD_DRIVER="python3 -m verl.experimental.fully_async_policy.fully_async_main"
exec bash "$(dirname "$0")/run_vd7_common.sh"
