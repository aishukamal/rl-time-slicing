#!/bin/bash
# export_logs.sh — Export metrics and logs after RL runs
#
# Usage:
#   Direct Export immediately:
#     ./export_logs.sh
#
#   Watch Mode Export (block execution dynamically until ALL remote log files natively print [RL_JOB_COMPLETED]):
#     ./export_logs.sh --watch "/data/rl_logs/rl_job-a.log,/data/rl_logs/rl_job-b.log"

NAMESPACE=${NAMESPACE:-rl-demo}
MODE=${MODE:-timeslice}

WATCH_FILES=""
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --watch) WATCH_FILES="$2"; shift 2 ;;
    *) shift ;;
  esac
done

if [ -n "$WATCH_FILES" ]; then
    HEAD_POD=$(kubectl get pods -n $NAMESPACE -l ray.io/node-type=head -o custom-columns=NAME:.metadata.name --no-headers | head -n 1)
    if [ -z "$HEAD_POD" ]; then
        echo "Error: Could not find Ray head pod"
        exit 1
    fi
    echo "========================================"
    echo "Watching for [RL_JOB_COMPLETED] across logs:"
    IFS=',' read -ra FILES <<< "$WATCH_FILES"
    for FILE in "${FILES[@]}"; do
        echo "  - $FILE on $HEAD_POD"
    done
    echo "========================================"
    
    for FILE in "${FILES[@]}"; do
        while true; do
            if kubectl exec "$HEAD_POD" -n "$NAMESPACE" -- cat "$FILE" 2>/dev/null | grep -q '\[RL_JOB_COMPLETED\]'; then
                echo "--> Detected completion uniquely inside $FILE!"
                break
            fi
            sleep 15
        done
    done
    echo "All monitored jobs successfully completed natively. Triggering multi-job export pipeline..."
    echo ""
fi

RUN_TS=$(date +%Y%m%d_%H%M%S)
EXPORT_DIR="./rl_logs_export_${MODE}_${RUN_TS}"
mkdir -p "$EXPORT_DIR"
echo ""
echo "Exporting logs and metrics from cluster to ${EXPORT_DIR}..."
echo "  -> dump gpu-scraper"
kubectl logs gpu-scraper -n $NAMESPACE > "$EXPORT_DIR/gpu-scraper.log" 2>/dev/null || true

echo "Shutting down gpu-scraper to generate plots..."
kubectl delete pod gpu-scraper -n $NAMESPACE --grace-period=60

echo "Waiting for metrics files to flush..."
sleep 10

kubectl cp $NAMESPACE/gpu-orchestrator:/data/rl_logs "$EXPORT_DIR"
kubectl cp $NAMESPACE/gpu-orchestrator:/data/rl_metrics.jsonl "$EXPORT_DIR/rl_metrics.jsonl"

echo "Exporting individual pod logs..."
for POD in $(kubectl get pods -n $NAMESPACE -o custom-columns=NAME:.metadata.name --no-headers); do
    echo "  -> dump $POD"
    kubectl logs $POD -n $NAMESPACE > "$EXPORT_DIR/${POD}.log" 2>/dev/null || true
done

echo "========================================"
echo "Exported files to $EXPORT_DIR:"
ls -la "$EXPORT_DIR"
