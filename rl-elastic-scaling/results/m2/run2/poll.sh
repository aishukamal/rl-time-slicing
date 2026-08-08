#!/usr/bin/env bash
# M2 run2 incremental collector (attempt-1 hardening #6): every 20 min,
# snapshot pod health + controller progress + latest step metrics so a
# successor agent can reconstruct everything from this file + pod stdout.
CTX=gke_aishuk-test_asia-southeast1-b_verl-research-cluster
K=/opt/homebrew/bin/kubectl
POD=elastic-m2-v785d
OUT=/Users/aishuk/workspaces/GPU-CR/elastic-rl-poc/m2-results/run2/incremental.log
END=$(( $(date +%s) + 18000 ))   # cover through training end + margin (~5h)
while [ "$(date +%s)" -lt "$END" ]; do
  {
    echo "===== POLL $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    $K --context=$CTX get pod $POD -n default -o jsonpath='phase={.status.phase} restarts={.status.containerStatuses[0].restartCount} label={.metadata.labels.timeslice\.io/job-id}' 2>&1; echo
    echo "--- last policy lines ---"
    $K --context=$CTX logs $POD -n default --tail=4000 2>/dev/null | grep "\[policy\]" | tail -6
    echo "--- last step/psync lines ---"
    $K --context=$CTX logs $POD -n default --tail=4000 2>/dev/null | grep -E "timing_s/step|timing_s/param_sync|Requesting 64" | tail -4
    echo "--- switch ops so far (decisions/timings) ---"
    $K --context=$CTX exec $POD -n default -- bash -c '
      wc -l /workspace/results/decisions.jsonl 2>/dev/null;
      grep -c "\"operation\"" /workspace/results/switch_timings.jsonl 2>/dev/null || echo "0 switch_timings";
      tail -1 /workspace/results/switch_timings.jsonl 2>/dev/null' 2>/dev/null
    echo
  } >> "$OUT" 2>&1
  # stop early if training section ended (epilogue marker in log)
  if $K --context=$CTX logs $POD -n default --tail=400 2>/dev/null | grep -q "GPUCSV-END"; then
    echo "===== POLL: GPUCSV-END seen, training over $(date -u +%H:%M:%SZ) =====" >> "$OUT"
    break
  fi
  sleep 1200
done
echo "===== POLLER EXIT $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$OUT"
