#!/usr/bin/env bash
# Exits (re-invoking the supervising agent) on: pod not Running, controller
# ABORT/exit, or training epilogue complete (GPUCSV-END).
CTX=gke_aishuk-test_asia-southeast1-b_verl-research-cluster
K=/opt/homebrew/bin/kubectl
POD=elastic-m2-v785d
while true; do
  phase=$($K --context=$CTX get pod $POD -n default -o jsonpath='{.status.phase}' 2>/dev/null)
  if [ "$phase" != "Running" ]; then echo "EVENT: pod phase=$phase"; exit 0; fi
  tail=$($K --context=$CTX logs $POD -n default --tail=600 2>/dev/null)
  if echo "$tail" | grep -q "GPUCSV-END"; then echo "EVENT: training epilogue complete (GPUCSV-END)"; exit 0; fi
  if echo "$tail" | grep -qE "\[policy\] ABORT|policy controller exited rc=|GATE-TIMEOUT"; then
    echo "EVENT: controller exit/abort detected"
    echo "$tail" | grep -E "\[policy\]" | tail -20
    exit 0
  fi
  sleep 120
done
