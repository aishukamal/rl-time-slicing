#!/usr/bin/env bash
# M3 RUN B: watch for R2 auto-park, then perform the one-time operator
# relabel (timeslice.io/job-id=elastic-trainer) the controller's agent gate
# needs. Exits after relabel + verification, or on pod failure.
CTX=gke_aishuk-test_asia-southeast1-b_verl-research-cluster
K=/opt/homebrew/bin/kubectl
POD=$($K --context=$CTX get pods -n default -l app=elastic-m3b -o jsonpath="{.items[0].metadata.name}")
OUT=/Users/aishuk/workspaces/GPU-CR/elastic-rl-poc/m3-results/regime-shift/relabel.log
END=$(( $(date +%s) + 7200 ))
echo "watching pod $POD for auto-park" >> "$OUT"
while [ "$(date +%s)" -lt "$END" ]; do
  phase=$($K --context=$CTX get pod $POD -n default -o jsonpath='{.status.phase}' 2>/dev/null)
  if [ "$phase" != "Running" ]; then echo "EVENT: pod phase=$phase" | tee -a "$OUT"; exit 1; fi
  if $K --context=$CTX logs $POD -n default --tail=4000 2>/dev/null | grep -q "R2 parked (auto)"; then
    echo "EVENT: auto-park seen $(date -u +%H:%M:%SZ); relabeling" | tee -a "$OUT"
    $K --context=$CTX label pod $POD -n default timeslice.io/job-id=elastic-trainer --overwrite 2>&1 | tee -a "$OUT"
    $K --context=$CTX get pod $POD -n default -o jsonpath='label={.metadata.labels.timeslice\.io/job-id}' | tee -a "$OUT"; echo | tee -a "$OUT"
    echo "RELABEL DONE" | tee -a "$OUT"
    exit 0
  fi
  sleep 60
done
echo "EVENT: relabel watch expired without auto-park (investigate)" | tee -a "$OUT"
exit 2
