#!/usr/bin/env bash
# M3 RUN C incremental collector (control + armed phases; discovers whichever
# elastic-m3c-* pod is live each cycle).
CTX=gke_aishuk-test_asia-southeast1-b_verl-research-cluster
K=/opt/homebrew/bin/kubectl
OUT=/Users/aishuk/workspaces/GPU-CR/elastic-rl-poc/m3-results/no-harm/incremental.log
END=$(( $(date +%s) + 32400 ))
while [ "$(date +%s)" -lt "$END" ]; do
  POD=$($K --context=$CTX get pods -n default -l app=elastic-m3c -o jsonpath="{.items[?(@.status.phase=='Running')].metadata.name}" 2>/dev/null | awk '{print $1}')
  {
    echo "===== POLL $(date -u +%Y-%m-%dT%H:%M:%SZ) pod=$POD ====="
    if [ -n "$POD" ]; then
      $K --context=$CTX get pod $POD -n default -o jsonpath='phase={.status.phase} label={.metadata.labels.timeslice\.io/job-id}' 2>&1; echo
      echo "--- last policy lines ---"
      $K --context=$CTX logs $POD -n default --tail=4000 2>/dev/null | grep "\[policy\]" | tail -5
      echo "--- last steps: gen-wait vs step decomposition ---"
      $K --context=$CTX logs $POD -n default --tail=4000 2>/dev/null | grep -oE "step:[0-9]+ - |timing_s/step:[0-9.]+|timing_s/gen:[0-9.]+|timing_s/update_actor:[0-9.]+|Requesting 64 samples|sample collected 64/64[^ ]*" | tail -10
      echo "--- decisions/switches ---"
      $K --context=$CTX exec $POD -n default -- bash -c 'wc -l /workspace/results/decisions.jsonl 2>/dev/null; grep -c "\"operation\"" /workspace/results/switch_timings.jsonl 2>/dev/null || echo 0' 2>/dev/null
    else
      echo "(no running elastic-m3c pod)"
    fi
    echo
  } >> "$OUT" 2>&1
  sleep 1200
done
echo "===== POLLER EXIT $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$OUT"
