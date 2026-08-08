#!/usr/bin/env bash
# Exits (re-invoking the supervising agent) on: pod not Running, a fatal
# error signature, first completed training step (healthy), or epilogue.
CTX=gke_aishuk-test_asia-southeast1-b_verl-research-cluster
K=/opt/homebrew/bin/kubectl
POD=$($K --context=$CTX get pods -n default -l app=colocated-m3a -o jsonpath="{.items[0].metadata.name}")
END=$(( $(date +%s) + 6000 ))   # covers setup + first long step
while [ "$(date +%s)" -lt "$END" ]; do
  phase=$($K --context=$CTX get pod $POD -n default -o jsonpath='{.status.phase}' 2>/dev/null)
  if [ "$phase" != "Running" ]; then echo "EVENT: pod phase=$phase"; exit 0; fi
  tail=$($K --context=$CTX logs $POD -n default --tail=800 2>/dev/null)
  if echo "$tail" | grep -q "GPUCSV-END"; then echo "EVENT: epilogue complete"; exit 0; fi
  if echo "$tail" | grep -qE "step:[0-9]+ - "; then
    echo "EVENT: first training step logged (healthy)"
    echo "$tail" | grep -E "step:[0-9]+ - " | tail -1
    exit 0
  fi
  if echo "$tail" | grep -qE "Traceback \(most recent call last\)|HYDRA_FULL_ERROR|ConfigAttributeError|RayActorError|CUDA out of memory|torch.OutOfMemoryError|WARNING: training exited rc="; then
    echo "EVENT: error signature in log"
    echo "$tail" | grep -B2 -A15 -E "Traceback|ConfigAttributeError|out of memory|training exited rc=" | tail -60
    exit 0
  fi
  sleep 120
done
echo "EVENT: watch window expired without step or error (investigate)"
