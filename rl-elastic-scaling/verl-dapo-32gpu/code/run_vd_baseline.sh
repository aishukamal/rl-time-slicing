#!/usr/bin/env bash
# elastic-vd BASELINE arm: the pinned STOCK driver module end-to-end.
# Only registered deltas VD-D1..D5 (see run_vd_common.sh header + VD-STATE.md).
set -euo pipefail
export VD_EXP_NAME="${VD_EXP_NAME:-vd-baseline}"
export VD_DRIVER="python3 -m verl.experimental.fully_async_policy.fully_async_main"
export VD_CONFIG_PATH_ARG="--config-path=config"
exec bash "$(dirname "$0")/run_vd_common.sh"
