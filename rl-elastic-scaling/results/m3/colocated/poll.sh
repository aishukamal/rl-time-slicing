#!/usr/bin/env bash
# M3 RUN A incremental collector: every 20 min, snapshot pod health + latest
# step metrics so any successor agent can reconstruct the run from this file
# + pod stdout. (Standing rule: stream results incrementally; never rely on
# end-of-run collection alone.)
CTX=gke_aishuk-test_asia-southeast1-b_verl-research-cluster
K=/opt/homebrew/bin/kubectl
POD=$($K --context=$CTX get pods -n default -l app=colocated-m3a -o jsonpath="{.items[0].metadata.name}")
OUT=/Users/aishuk/workspaces/GPU-CR/elastic-rl-poc/m3-results/colocated/incremental.log
END=$(( $(date +%s) + 16200 ))   # setup (~35min) + 9000s training + margin
while [ "$(date +%s)" -lt "$END" ]; do
  {
    echo "===== POLL $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    $K --context=$CTX get pod $POD -n default -o jsonpath='phase={.status.phase} restarts={.status.containerStatuses[0].restartCount}' 2>&1; echo
    echo "--- last phase/progress lines ---"
    $K --context=$CTX logs $POD -n default --tail=3000 2>/dev/null | grep -E "^=== Phase|step:[0-9]+ - " | tail -6
    echo "--- last step timing lines (full) ---"
    $K --context=$CTX logs $POD -n default --tail=3000 2>/dev/null | grep -E "step:[0-9]+ - " | tail -2
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
