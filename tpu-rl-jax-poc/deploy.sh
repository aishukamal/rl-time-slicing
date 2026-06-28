#!/usr/bin/env bash
set -euo pipefail

# TPU Time-Slicing PoC — Automated Deploy + Run + Collect
#
# Usage:
#   MODE=baseline bash deploy.sh          # single job, no C/R
#   MODE=timeslice bash deploy.sh         # two jobs, time-sliced with C/R
#
# Full comparison:
#   RUN_ID=run1 MODE=baseline N_RL_STEPS=5 bash deploy.sh
#   RUN_ID=run1 MODE=timeslice N_RL_STEPS=5 bash deploy.sh
#   python3 telemetry/dashboard_generator.py \
#     --baseline-dir runs/run1/baseline --timeslice-dir runs/run1/timeslice \
#     --output runs/run1/dashboard.html

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KUBECTL="${KUBECTL:-/opt/homebrew/bin/kubectl}"

# Run ID groups baseline + timeslice together
RUN_ID="${RUN_ID:-$(date +%Y%m%d)_$(head -c4 /dev/urandom | xxd -p)}"
MODE="${MODE:-timeslice}"

# Config
N_RL_STEPS="${N_RL_STEPS:-5}"
PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-330}"
GROUP_SIZE="${GROUP_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
WEIGHT_SYNC_INTERVAL="${WEIGHT_SYNC_INTERVAL:-0}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-3}"

RUN_DIR="${SCRIPT_DIR}/runs/${RUN_ID}/${MODE}"
mkdir -p "$RUN_DIR"
RUN_START_UTC=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

echo "============================================================"
echo "  TPU Time-Slicing Deploy | MODE=${MODE}"
echo "  Run ID: ${RUN_ID}"
echo "  Output: ${RUN_DIR}"
echo "  Steps: ${N_RL_STEPS}, Prompts: ${PROMPTS_PER_STEP}, G: ${GROUP_SIZE}"
echo "  Max tokens: ${MAX_NEW_TOKENS}, Weight sync: ${WEIGHT_SYNC_INTERVAL}"
echo "============================================================"

# -- Helpers -------------------------------------------------------------------

poll_health() {
    local label="$1" pod="$2" container="$3" port="$4" field="$5" want="$6"
    local max_wait="${7:-600}"
    echo -n "  Waiting for ${label}..."
    for i in $(seq 1 $((max_wait / 5))); do
        val=$($KUBECTL exec "$pod" -c "$container" -- \
            curl -s "http://localhost:${port}/health" 2>/dev/null \
            | python3 -c "import sys,json; print(json.load(sys.stdin).get('${field}',''))" 2>/dev/null || echo "")
        if [ "$val" = "$want" ]; then
            echo " ready (${i}x5s)"
            return 0
        fi
        sleep 5
    done
    echo " TIMEOUT"
    return 1
}

JOBS="a"
if [ "$MODE" = "timeslice" ]; then
    JOBS="a b"
fi

# -- 1. Clean slate ------------------------------------------------------------

echo ""
echo "=== Clean Slate ==="
$KUBECTL delete pod rl-loop-a rl-loop-b tpu-orchestrator tpu-samplers tpu-trainers --ignore-not-found=true 2>/dev/null || true
echo "  Waiting for pods to terminate..."
$KUBECTL wait --for=delete pod/tpu-samplers pod/tpu-trainers --timeout=60s 2>/dev/null || true

# -- 2. Deploy infrastructure -------------------------------------------------

echo ""
echo "=== Deploy Infrastructure ==="

$KUBECTL apply -f "${SCRIPT_DIR}/deploy/services-m5.yaml"
$KUBECTL apply -f "${SCRIPT_DIR}/deploy/snapshot-agent.yaml"

# Orchestrator with correct MODE
ORCH_MODE="snapshot"
if [ "$MODE" = "baseline" ]; then ORCH_MODE="baseline"; fi
sed "s/value: \"snapshot\"/value: \"${ORCH_MODE}\"/" "${SCRIPT_DIR}/orchestrator/pod.yaml" | $KUBECTL apply -f -

# Sampler code ConfigMap (mounts updated sampler.py over the Docker image's copy)
$KUBECTL delete configmap sampler-code --ignore-not-found=true 2>/dev/null || true
$KUBECTL create configmap sampler-code \
    --from-file=sampler.py="${SCRIPT_DIR}/sampler/sampler.py"

echo "  Deploying trainers + samplers..."
TRAINER_CHIPS="${TRAINER_CHIPS:-}"
if [ -n "$TRAINER_CHIPS" ]; then
    echo "  Pinning trainers to TPU_VISIBLE_CHIPS=${TRAINER_CHIPS}"
    sed "s/name: TPU_LIBRARY_PATH/name: TPU_VISIBLE_CHIPS\n          value: \"${TRAINER_CHIPS}\"\n        - name: TPU_LIBRARY_PATH/" \
        "${SCRIPT_DIR}/deploy/trainers-pod.yaml" | $KUBECTL apply -f -
else
    $KUBECTL apply -f "${SCRIPT_DIR}/deploy/trainers-pod.yaml"
fi
$KUBECTL apply -f "${SCRIPT_DIR}/deploy/samplers-pod.yaml"

$KUBECTL wait --for=condition=Ready pod/tpu-orchestrator --timeout=60s
$KUBECTL wait --for=condition=Ready pod/tpu-trainers --timeout=120s 2>/dev/null || true
for i in $(seq 1 24); do
    phase=$($KUBECTL get pod tpu-samplers -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
    if [ "$phase" = "Running" ]; then break; fi
    sleep 5
done

# -- 3. Start TPU duty cycle scrapers inside TPU pods -------------------------
# Standalone scraper pods can't access TPU metrics (no device allocation).
# Instead, run scrapers inside sampler-a and trainer-a containers which have
# TPU access via the device plugin.

echo ""
echo "=== Start TPU Duty Cycle Scrapers ==="

for role_container in "sampler:tpu-samplers:sampler-a" "trainer:tpu-trainers:trainer-a"; do
    role="${role_container%%:*}"
    rest="${role_container#*:}"
    pod="${rest%%:*}"
    container="${rest##*:}"

    $KUBECTL cp "${SCRIPT_DIR}/telemetry/tpu_duty_cycle.py" "${pod}:/tmp/tpu_duty_cycle.py" -c "$container" 2>/dev/null || true
    $KUBECTL exec "$pod" -c "$container" -- bash -c "
        CSV_FILE=/tmp/tpu_duty_cycle_${role}.csv \
        PHASE=${MODE} \
        POLL_INTERVAL=1 \
        nohup python3 /tmp/tpu_duty_cycle.py > /tmp/scraper_${role}.log 2>&1 &
        echo \$!
    " 2>/dev/null || true
    echo "  Scraper started in ${pod}/${container} for ${role} pool"
done

# -- 4. Wait for sampler-a ready ---------------------------------------------

echo ""
echo "=== Waiting for Pods ==="
poll_health "sampler-a vLLM" tpu-samplers sampler-a 8300 vllm_started True 900

# -- 5. Update ConfigMap + launch rl-loops ------------------------------------

echo ""
echo "=== Update RL Loop Code ==="
$KUBECTL delete configmap rl-loop-code --ignore-not-found=true 2>/dev/null || true
$KUBECTL create configmap rl-loop-code \
    --from-file=rl_loop.py="${SCRIPT_DIR}/loop/rl_loop.py" \
    --from-file=reward.py="${SCRIPT_DIR}/loop/reward.py"

echo ""
echo "=== Launch RL Loops ==="
for job in $JOBS; do
    manifest="${SCRIPT_DIR}/loop/rl-loop-${job}.yaml"
    tmp="/tmp/rl-loop-${job}-patched.yaml"
    python3 -c "
import re
text = open('$manifest').read()
for name, val in {'N_RL_STEPS':'${N_RL_STEPS}','PROMPTS_PER_STEP':'${PROMPTS_PER_STEP}','GROUP_SIZE':'${GROUP_SIZE}','MAX_NEW_TOKENS':'${MAX_NEW_TOKENS}','WEIGHT_SYNC_INTERVAL':'${WEIGHT_SYNC_INTERVAL}','GEN_BATCH_SIZE':'${GEN_BATCH_SIZE}'}.items():
    text = re.sub(r'(name: '+name+r'\s+value: )\"[^\"]*\"', r'\g<1>\"'+val+'\"', text)
open('$tmp','w').write(text)
"
    $KUBECTL apply -f "$tmp"
done
echo "  Launched: $JOBS"

# -- 6. Monitor until completion -----------------------------------------------

echo ""
echo "=== Monitoring ==="
a_done=false
b_done=false

for i in $(seq 1 1440); do
    if [ "$a_done" = false ]; then
        if $KUBECTL logs rl-loop-a --tail=5 2>/dev/null | grep -q "RL_JOB_COMPLETED"; then
            echo "  $(date +%H:%M:%S) job-a COMPLETE"
            a_done=true
        fi
    fi
    if [ "$MODE" = "timeslice" ] && [ "$b_done" = false ]; then
        if $KUBECTL logs rl-loop-b --tail=5 2>/dev/null | grep -q "RL_JOB_COMPLETED"; then
            echo "  $(date +%H:%M:%S) job-b COMPLETE"
            b_done=true
        fi
    fi

    if [ "$MODE" = "baseline" ] && [ "$a_done" = true ]; then echo "  Done!"; break; fi
    if [ "$MODE" = "timeslice" ] && [ "$a_done" = true ] && [ "$b_done" = true ]; then echo "  Done!"; break; fi

    if [ $((i % 6)) -eq 0 ]; then
        a_step=$($KUBECTL logs rl-loop-a --tail=5 2>/dev/null | grep "Step.*done" | tail -1 | grep -o "Step [0-9]*" || echo "starting")
        if [ "$MODE" = "timeslice" ]; then
            b_step=$($KUBECTL logs rl-loop-b --tail=5 2>/dev/null | grep "Step.*done" | tail -1 | grep -o "Step [0-9]*" || echo "starting")
            echo "  $(date +%H:%M:%S) job-a: $a_step  job-b: $b_step"
        else
            echo "  $(date +%H:%M:%S) job-a: $a_step"
        fi
    fi
    sleep 5
done

# -- 7. Collect ALL metrics (pods still running — [COMPLETED] printed before exit)

echo ""
echo "=== Collecting Metrics ==="

# Job JSONL (from running pods — kubectl cp works here)
for job in $JOBS; do
    $KUBECTL cp "rl-loop-${job}:/data/rl_logs/metrics_job-${job}.jsonl" "${RUN_DIR}/metrics_job-${job}.jsonl" 2>/dev/null || true
done

# Orchestrator JSONL
$KUBECTL cp "tpu-orchestrator:/data/rl_metrics.jsonl" "${RUN_DIR}/rl_metrics.jsonl" 2>/dev/null || true

# Duty cycle CSVs — scrapers run inside TPU containers
for role_container in "sampler:tpu-samplers:sampler-a" "trainer:tpu-trainers:trainer-a"; do
    role="${role_container%%:*}"
    rest="${role_container#*:}"
    pod="${rest%%:*}"
    container="${rest##*:}"
    # Stop the scraper gracefully
    $KUBECTL exec "$pod" -c "$container" -- bash -c "pkill -f tpu_duty_cycle.py" 2>/dev/null || true
    sleep 1
    $KUBECTL cp "${pod}:/tmp/tpu_duty_cycle_${role}.csv" "${RUN_DIR}/tpu_duty_cycle_${role}.csv" -c "$container" 2>/dev/null || true
done

# Merge duty cycle CSVs into one
if ls "${RUN_DIR}"/tpu_duty_cycle_*.csv 1>/dev/null 2>&1; then
    head -1 "${RUN_DIR}/tpu_duty_cycle_sampler.csv" > "${RUN_DIR}/tpu_duty_cycle.csv" 2>/dev/null || true
    for f in "${RUN_DIR}"/tpu_duty_cycle_*.csv; do
        tail -n +2 "$f" >> "${RUN_DIR}/tpu_duty_cycle.csv" 2>/dev/null || true
    done
    echo "  Duty cycle CSV merged"
fi

# Pod logs
for job in $JOBS; do
    $KUBECTL logs "rl-loop-${job}" > "${RUN_DIR}/rl-loop-${job}.log" 2>&1 || true
done
$KUBECTL logs tpu-orchestrator > "${RUN_DIR}/orchestrator.log" 2>&1 || true
for c in sampler-a sampler-b; do
    $KUBECTL logs tpu-samplers -c "$c" > "${RUN_DIR}/${c}.log" 2>&1 || true
done
for c in trainer-a trainer-b; do
    $KUBECTL logs tpu-trainers -c "$c" > "${RUN_DIR}/${c}.log" 2>&1 || true
done

# Config
cat > "${RUN_DIR}/config.json" << EOF
{
    "run_id": "${RUN_ID}",
    "mode": "${MODE}",
    "n_rl_steps": ${N_RL_STEPS},
    "prompts_per_step": ${PROMPTS_PER_STEP},
    "group_size": ${GROUP_SIZE},
    "max_new_tokens": ${MAX_NEW_TOKENS},
    "weight_sync_interval": ${WEIGHT_SYNC_INTERVAL},
    "gen_batch_size": ${GEN_BATCH_SIZE}
}
EOF

# -- 8. Verify metrics collected -----------------------------------------------

echo ""
echo "=== Metric Verification ==="
missing=0
for f in metrics_job-a.jsonl rl_metrics.jsonl tpu_duty_cycle.csv; do
    path="${RUN_DIR}/${f}"
    if [ -s "$path" ]; then
        lines=$(wc -l < "$path" | tr -d ' ')
        echo "  OK: ${f} (${lines} lines)"
    else
        echo "  MISSING: ${f}"
        missing=$((missing + 1))
    fi
done
if [ "$MODE" = "timeslice" ]; then
    if [ -s "${RUN_DIR}/metrics_job-b.jsonl" ]; then
        echo "  OK: metrics_job-b.jsonl ($(wc -l < "${RUN_DIR}/metrics_job-b.jsonl" | tr -d ' ') lines)"
    else
        echo "  MISSING: metrics_job-b.jsonl"
        missing=$((missing + 1))
    fi
fi
if [ $missing -gt 0 ]; then
    echo "  WARNING: ${missing} metric file(s) missing!"
else
    echo "  All metrics collected."
fi

# -- 9. Clean up pods (after metrics verified) ---------------------------------

# -- 9. Collect Cloud Monitoring TPU metrics -----------------------------------

echo ""
echo "=== Collect Cloud Monitoring Metrics ==="
RUN_END_UTC=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
echo "  Time window: ${RUN_START_UTC} — ${RUN_END_UTC}"
python3 "${SCRIPT_DIR}/telemetry/collect_cloud_metrics.py" \
    --project aishuk-test \
    --start "$RUN_START_UTC" \
    --end "$RUN_END_UTC" \
    --output-dir "$RUN_DIR" 2>&1 || echo "  WARNING: Cloud Monitoring collection failed"

# -- 10. Cleanup ---------------------------------------------------------------

echo ""
echo "=== Cleanup ==="
for job in $JOBS; do
    $KUBECTL delete pod "rl-loop-${job}" --ignore-not-found=true 2>/dev/null || true
done
echo "  rl-loop pods cleaned up"

echo ""
echo "=== Logs saved to ${RUN_DIR}/ ==="
ls -la "${RUN_DIR}/"
echo ""
echo "============================================================"
echo "  Run complete: ${RUN_DIR}"
echo "  To generate dashboard (after both baseline + timeslice):"
echo "    python3 telemetry/dashboard_generator.py \\"
echo "      --baseline-dir runs/${RUN_ID}/baseline \\"
echo "      --timeslice-dir runs/${RUN_ID}/timeslice \\"
echo "      --output runs/${RUN_ID}/dashboard.html"
echo "============================================================"
