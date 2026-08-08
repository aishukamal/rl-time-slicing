#!/usr/bin/env bash
# M3 RUN B launch sequence (run from ~/workspaces/GPU-CR/elastic-rl-poc).
# Standing rules baked in: explicit --context everywhere; recycle the
# snapshot-agent pod (delete only, never helm) BEFORE the elastic launch;
# pre-flight probe of the exact status-parse path.
set -euo pipefail
CTX=gke_aishuk-test_asia-southeast1-b_verl-research-cluster
K="/opt/homebrew/bin/kubectl --context=$CTX --request-timeout=90s"
cd "$(dirname "$0")/.."   # elastic-rl-poc

echo "=== 1. recycle snapshot-agent pod on trb7 (sticky job-state landmine) ==="
AGENT_POD=$($K get pods -n timeslice-system -l app.kubernetes.io/name=snapshot-agent \
  --field-selector spec.nodeName=gke-verl-research-clus-h100-2gpu-pool-02b8c734-trb7 \
  -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [ -z "$AGENT_POD" ]; then
  AGENT_POD=$($K get pods -n timeslice-system -o name | grep snapshot-agent | head -1 | sed 's|pod/||')
fi
echo "agent pod: $AGENT_POD"
$K delete pod "$AGENT_POD" -n timeslice-system --wait=true
sleep 10
for i in $(seq 1 30); do
  NEW=$($K get pods -n timeslice-system --field-selector spec.nodeName=gke-verl-research-clus-h100-2gpu-pool-02b8c734-trb7 -o name 2>/dev/null | grep snapshot-agent | head -1)
  PHASE=$($K get "$NEW" -n timeslice-system -o jsonpath='{.status.phase}' 2>/dev/null || true)
  [ "$PHASE" = "Running" ] && { echo "fresh agent pod: $NEW Running"; break; }
  sleep 5
done

echo "=== 2. code ConfigMap (m1 sources + controller + regime_flip) ==="
$K create configmap elastic-m3b-code \
  --from-file=m1/fully_async_main_elastic.py --from-file=m1/elastic_trainer.py \
  --from-file=m1/r2_lifecycle.py --from-file=m2/policy_controller.py \
  --from-file=m3/regime_flip.py \
  -n default --dry-run=client -o yaml | $K apply -f -

echo "=== 3. pre-flight probe (exact status-parse path vs live agent) ==="
$K delete pod m3b-preflight-probe -n default --ignore-not-found=true
$K apply -f m3/preflight-probe-m3b.yaml
for i in $(seq 1 60); do
  PH=$($K get pod m3b-preflight-probe -n default -o jsonpath='{.status.phase}' 2>/dev/null || true)
  [ "$PH" = "Succeeded" ] || [ "$PH" = "Failed" ] && break
  sleep 5
done
$K logs m3b-preflight-probe -n default | tail -5
PH=$($K get pod m3b-preflight-probe -n default -o jsonpath='{.status.phase}')
$K delete pod m3b-preflight-probe -n default --ignore-not-found=true
[ "$PH" = "Succeeded" ] || { echo "FATAL: preflight probe $PH"; exit 1; }

echo "=== 4. launch job ==="
$K apply -f m3/k8s-job-m3b-regime-shift.yaml
sleep 15
$K get pods -n default -l app=elastic-m3b -o wide
echo "REMINDER: after auto-park, relabel: kubectl --context=$CTX label pod <pod> timeslice.io/job-id=elastic-trainer --overwrite"
