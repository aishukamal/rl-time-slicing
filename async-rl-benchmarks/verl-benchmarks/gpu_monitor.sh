#!/usr/bin/env bash
# Samples GPU utilization at high frequency via nvidia-smi.
# Produces a CSV: timestamp_ms, gpu_index, utilization_gpu%, utilization_mem%, power_w
#
# Usage: ./gpu_monitor.sh <output_csv> [interval_ms]
#   interval_ms defaults to 100 (10 samples/sec)

set -euo pipefail

OUTPUT="${1:?Usage: $0 <output.csv> [interval_ms]}"
INTERVAL_MS="${2:-100}"

echo "timestamp_ms,gpu_index,gpu_util_pct,mem_util_pct,power_w" > "$OUTPUT"

cleanup() { exit 0; }
trap cleanup SIGINT SIGTERM

while true; do
    ts=$(date +%s%3N)
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,power.draw \
               --format=csv,noheader,nounits 2>/dev/null | \
    while IFS=', ' read -r idx gpu mem pwr; do
        echo "${ts},${idx},${gpu},${mem},${pwr}"
    done >> "$OUTPUT"
    sleep "$(python3 -c "print($INTERVAL_MS/1000)")"
done
