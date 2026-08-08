#!/usr/bin/env bash
# M3 RUN B incremental collector: every ~20 min snapshot pod health,
# controller progress, flip status, latest steps, switch-op counts.
CTX=gke_aishuk-test_asia-southeast1-b_verl-research-cluster
K=/opt/homebrew/bin/kubectl
POD=$($K --context=$CTX get pods -n default -l app=elastic-m3b -o jsonpath="{.items[0].metadata.name}")
OUT=/Users/aishuk/workspaces/GPU-CR/elastic-rl-poc/m3-results/regime-shift/incremental.log
END=$(( $(date +%s) + 21600 ))   # setup + 16200s training + margin
while [ "$(date +%s)" -lt "$END" ]; do
  {
    echo "===== POLL $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    $K --context=$CTX get pod $POD -n default -o jsonpath='phase={.status.phase} restarts={.status.containerStatuses[0].restartCount} label={.metadata.labels.timeslice\.io/job-id}' 2>&1; echo
    echo "--- last policy lines ---"
    $K --context=$CTX logs $POD -n default --tail=4000 2>/dev/null | grep "\[policy\]" | tail -5
    echo "--- last regime-flip lines ---"
    $K --context=$CTX logs $POD -n default --tail=4000 2>/dev/null | grep "\[regime-flip\]" | tail -3
    echo "--- last step/psync/response-length ---"
    $K --context=$CTX logs $POD -n default --tail=4000 2>/dev/null | grep -oE "step:[0-9]+ - |timing_s/step:[0-9.]+|response_length/mean:[0-9.]+|timing_s/param_sync: [0-9.]+ seconds self.current_param_version: [0-9]+" | tail -8
    echo "--- switch ops / decisions so far ---"
    $K --context=$CTX exec $POD -n default -- bash -c '
      wc -l /workspace/results/decisions.jsonl 2>/dev/null;
      grep -c "\"operation\"" /workspace/results/switch_timings.jsonl 2>/dev/null || echo "0 switch_timings";
      tail -1 /workspace/results/switch_timings.jsonl 2>/dev/null | head -c 400; echo;
      cat /workspace/results/regime_flip.jsonl 2>/dev/null' 2>/dev/null
    echo
  } >> "$OUT" 2>&1
  if $K --context=$CTX logs $POD -n default --tail=400 2>/dev/null | grep -q "GPUCSV-END"; then
    echo "===== POLL: GPUCSV-END seen, training over $(date -u +%H:%M:%SZ) =====" >> "$OUT"
    break
  fi
  phase=$($K --context=$CTX get pod $POD -n default -o jsonpath='{.status.phase}' 2>/dev/null)
  if [ "$phase" != "Running" ]; then
    echo "===== POLL: pod phase=$phase, exiting $(date -u +%H:%M:%SZ) =====" >> "$OUT"
    break
  fi
  sleep 1200
done
echo "===== POLLER EXIT $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$OUT"
