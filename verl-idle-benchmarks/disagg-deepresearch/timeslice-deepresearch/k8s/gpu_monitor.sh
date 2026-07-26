#!/usr/bin/env bash
set -euo pipefail
OUTPUT="${1:?Usage: $0 <output.csv> [interval_ms]}"
INTERVAL_MS="${2:-100}"
echo "timestamp_ms,gpu_index,gpu_util_pct,mem_util_pct,power_w" > "$OUTPUT"
trap 'exit 0' SIGINT SIGTERM
SLEEP_S=$(python3 -c "print($INTERVAL_MS/1000)")
while true; do
    ts=$(date +%s%3N)
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,power.draw \
               --format=csv,noheader,nounits 2>/dev/null | \
    while IFS=', ' read -r idx gpu mem pwr; do
        echo "${ts},${idx},${gpu},${mem},${pwr}"
    done >> "$OUTPUT"
    sleep "$SLEEP_S"
done
