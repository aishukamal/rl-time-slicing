#!/usr/bin/env bash
CTX=gke_aishuk-test_asia-southeast1-b_verl-research-cluster
K=/opt/homebrew/bin/kubectl
OUT=/Users/aishuk/workspaces/GPU-CR/elastic-rl-poc/m3-results/static12/incremental.log
END=$(( $(date +%s) + 18000 ))
HEAD=$($K --context=$CTX get pods -n default -l app=m3d-head -o jsonpath="{.items[0].metadata.name}")
WORKER=$($K --context=$CTX get pods -n default -l app=m3d-worker -o jsonpath="{.items[0].metadata.name}")
while [ "$(date +%s)" -lt "$END" ]; do
  {
    echo "===== POLL $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    $K --context=$CTX get pod $HEAD $WORKER -n default -o jsonpath='{range .items[*]}{.metadata.name}={.status.phase} {end}' 2>&1; echo
    echo "--- head last phase/step ---"
    $K --context=$CTX logs $HEAD -n default --tail=3000 2>/dev/null | grep -oE "=== (Setup|Head)[^=]*===|cluster has [0-9]+ GPUs|step:[0-9]+ - |timing_s/step:[0-9.]+|timing_s/gen:[0-9.]+|RayActorError|NCCL WARN.*" | tail -8
    echo
  } >> "$OUT" 2>&1
  if $K --context=$CTX logs $HEAD -n default --tail=400 2>/dev/null | grep -q "GPUCSV-END"; then
    echo "===== POLL: head GPUCSV-END $(date -u +%H:%M:%SZ) =====" >> "$OUT"; break
  fi
  phase=$($K --context=$CTX get pod $HEAD -n default -o jsonpath='{.status.phase}' 2>/dev/null)
  [ "$phase" != "Running" ] && { echo "===== POLL: head phase=$phase =====" >> "$OUT"; break; }
  sleep 1200
done
echo "===== POLLER EXIT $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$OUT"
