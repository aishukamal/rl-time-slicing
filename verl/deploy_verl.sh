#!/bin/bash
# deploy_verl.sh — Deploy verl time-slice integration on GKE
#
# Cleans up and redeploys everything: orchestrator, node daemon,
# RayCluster, and uploads all code to GCS. Run this after any change.
#
# Usage:
#   bash deploy_verl.sh                    # full deploy
#   bash deploy_verl.sh --skip-data        # skip GSM8K + model upload (already done)
#   bash deploy_verl.sh --skip-gcs         # skip all GCS uploads (cluster changes only)
#   bash deploy_verl.sh --skip-ray         # skip RayCluster (orch changes only)

set -e

# ── Config ────────────────────────────────────────────────────────────────────
CLUSTER=${CLUSTER:-verl-research-cluster}
ZONE=${ZONE:-asia-southeast1-b}
NAMESPACE=${NAMESPACE:-rl-demo}
NODE_POOL=${NODE_POOL:-h100-2gpu-pool}
GSBUCKET=${GSBUCKET:-verl-timeslice-$(gcloud config get-value project 2>/dev/null)}
KSA_NAME=${KSA_NAME:-verl-ksa}
IMAGE=${IMAGE:-verlai/verl:vllm017.latest}
DAEMON_IMAGE=${DAEMON_IMAGE:-vllm/vllm-openai:latest}
ORCH_PORT=${ORCH_PORT:-9000}
DAEMON_PORT=${DAEMON_PORT:-9001}

# Flags
SKIP_DATA=false
SKIP_GCS=false
SKIP_RAY=false
DEPLOY_MODE="timeslice"
for arg in "$@"; do
  case $arg in
    --skip-data) SKIP_DATA=true ;;
    --skip-gcs)  SKIP_GCS=true  ;;
    --skip-ray)  SKIP_RAY=true  ;;
    --mode=*)    DEPLOY_MODE="${arg#*=}" ;;
  esac
done

# Script dir — all source files expected alongside this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "verl Timeslice Deploy"
echo "  cluster:   $CLUSTER ($ZONE)"
echo "  namespace: $NAMESPACE"
echo "  bucket:    gs://$GSBUCKET"
echo "  image:     $IMAGE"
echo "  skip-data: $SKIP_DATA"
echo "  skip-gcs:  $SKIP_GCS"
echo "  skip-ray:  $SKIP_RAY"
echo "  mode:      $DEPLOY_MODE"
echo "========================================"

# ── Credentials ───────────────────────────────────────────────────────────────
gcloud container clusters get-credentials $CLUSTER --zone $ZONE
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# ── Pick GPU node ─────────────────────────────────────────────────────────────
GPU_NODE=$(kubectl get nodes \
  -l cloud.google.com/gke-nodepool=$NODE_POOL \
  -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
[ -z "$GPU_NODE" ] && echo "ERROR: no node in pool $NODE_POOL" && exit 1
echo "GPU node: $GPU_NODE"

# ── GCS: upload timeslice code ────────────────────────────────────────────────
if [ "$SKIP_GCS" = false ]; then
  echo ""
  echo "=== Uploading timeslice code to GCS ==="
  for f in gpu_client.py config.py orchestrator.py node_daemon.py gpu_duty_cycle.py \
            verl_timeslice_sync_trainer.py main_ppo_timeslice_sync.py; do
    [ ! -f "$SCRIPT_DIR/$f" ] && echo "ERROR: $SCRIPT_DIR/$f not found" && exit 1
  done
  gcloud storage cp \
    $SCRIPT_DIR/gpu_client.py \
    $SCRIPT_DIR/config.py \
    $SCRIPT_DIR/orchestrator.py \
    $SCRIPT_DIR/node_daemon.py \
    $SCRIPT_DIR/gpu_duty_cycle.py \
    $SCRIPT_DIR/verl_timeslice_sync_trainer.py \
    $SCRIPT_DIR/main_ppo_timeslice_sync.py \
    gs://$GSBUCKET/verl/timeslice/
  echo "Code uploaded"

  # Data + model upload (slow — skip with --skip-data after first run)
  if [ "$SKIP_DATA" = false ]; then
    echo ""
    echo "=== Uploading verl source ==="
    [ ! -d "/tmp/verl" ] && git clone https://github.com/volcengine/verl.git /tmp/verl
    gcloud storage rsync -r /tmp/verl gs://$GSBUCKET/verl \
      --delete-unmatched-destination-objects \
      --exclude="timeslice/.*"  # preserve timeslice/ we just uploaded
    
    # Overwrite the clean framework's base.py with our hacked local version
    gcloud storage cp $SCRIPT_DIR/verl/single_controller/ray/base.py gs://$GSBUCKET/verl/verl/single_controller/ray/base.py

    echo ""
    echo "=== Preprocessing GSM8K ==="
    if ! gcloud storage ls gs://$GSBUCKET/data/gsm8k/train.parquet &>/dev/null; then
      cd /tmp/verl && pip install -e . -q
      python3 examples/data_preprocess/gsm8k.py --local_save_dir /tmp/gsm8k
      gcloud storage cp /tmp/gsm8k/*.parquet gs://$GSBUCKET/data/gsm8k/
    else
      echo "GSM8K already uploaded, skipping"
    fi

    echo ""
    echo "=== Checking model ==="
    if ! gcloud storage ls gs://$GSBUCKET/models/Qwen2.5-0.5B-Instruct/ &>/dev/null; then
      [ -z "$HF_TOKEN" ] && echo "ERROR: HF_TOKEN not set" && exit 1
      python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-0.5B-Instruct',
                  local_dir='/tmp/qwen', token='$HF_TOKEN')
"
      gcloud storage cp -r /tmp/qwen gs://$GSBUCKET/models/Qwen2.5-0.5B-Instruct
    else
      echo "Model already uploaded, skipping"
    fi
  fi
fi

# ── ConfigMap ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Updating ConfigMap ==="
kubectl delete configmap rl-orch-code -n $NAMESPACE --ignore-not-found=true
kubectl create configmap rl-orch-code \
  --from-file=$SCRIPT_DIR/gpu_client.py \
  --from-file=$SCRIPT_DIR/config.py \
  --from-file=$SCRIPT_DIR/orchestrator.py \
  --from-file=$SCRIPT_DIR/node_daemon.py \
  --from-file=$SCRIPT_DIR/gpu_duty_cycle.py \
  -n $NAMESPACE

# ── Orchestrator ──────────────────────────────────────────────────────────────
echo ""
echo "=== Cleaning up old logs and metrics ==="
# Do this before deleting pods so new pod logs aren't wiped, and use old orchestrator if it exists
kubectl exec gpu-orchestrator -n $NAMESPACE -- sh -c 'rm -rf /data/rl_logs/* /data/rl_metrics.jsonl 2>/dev/null || true' 2>/dev/null || true

echo "=== Deploying orchestrator ==="
kubectl delete pod gpu-orchestrator -n $NAMESPACE \
  --ignore-not-found=true --force --grace-period=0 2>/dev/null || true
kubectl wait --for=delete pod/gpu-orchestrator \
  -n $NAMESPACE --timeout=30s 2>/dev/null || true

kubectl apply -n $NAMESPACE -f - << YAML
apiVersion: v1
kind: Service
metadata:
  name: gpu-orchestrator
spec:
  selector: {app: gpu-orchestrator}
  ports: [{port: $ORCH_PORT, targetPort: $ORCH_PORT}]
---
apiVersion: v1
kind: Pod
metadata:
  name: gpu-orchestrator
  labels: {app: gpu-orchestrator}
  annotations:
    gke-gcsfuse/volumes: "true"
spec:
  serviceAccountName: $KSA_NAME
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: $GPU_NODE
  containers:
  - name: orchestrator
    image: python:3.11-slim
    command: [sh, -c, "pip install fastapi uvicorn httpx -q && python3 /app/orchestrator.py"]
    ports: [{containerPort: $ORCH_PORT}]
    env:
    - {name: MODE, value: "$DEPLOY_MODE"}
    - {name: ORCH_PORT, value: "$ORCH_PORT"}
    - {name: DAEMON_PORT, value: "$DAEMON_PORT"}
    volumeMounts:
    - {name: app, mountPath: /app}
    - {name: data, mountPath: /data}
  volumes:
  - name: app
    configMap: {name: rl-orch-code}
  - name: data
    persistentVolumeClaim: {claimName: verl-bucket-pvc}
YAML

# ── Node Daemon ───────────────────────────────────────────────────────────────
echo "=== Deploying node daemon ==="
kubectl delete pod gpu-swap-daemon -n $NAMESPACE \
  --ignore-not-found=true --force --grace-period=0 2>/dev/null || true
kubectl wait --for=delete pod/gpu-swap-daemon \
  -n $NAMESPACE --timeout=30s 2>/dev/null || true

kubectl apply -n $NAMESPACE -f - << YAML
apiVersion: v1
kind: Service
metadata:
  name: gpu-swap-daemon
spec:
  selector: {app: gpu-swap-daemon}
  ports: [{port: $DAEMON_PORT, targetPort: $DAEMON_PORT}]
  internalTrafficPolicy: Local
---
apiVersion: v1
kind: Pod
metadata:
  name: gpu-swap-daemon
  labels: {app: gpu-swap-daemon}
  annotations:
    gke-gcsfuse/volumes: "true"
spec:
  serviceAccountName: $KSA_NAME
  restartPolicy: Never
  hostPID: true
  hostNetwork: true
  nodeSelector:
    kubernetes.io/hostname: $GPU_NODE
  initContainers:
  - name: install-cuda-checkpoint
    image: busybox
    command: [sh, -c, "wget -qO /usr/local/bin/cuda-checkpoint https://raw.githubusercontent.com/NVIDIA/cuda-checkpoint/main/bin/x86_64_Linux/cuda-checkpoint && chmod +x /usr/local/bin/cuda-checkpoint"]
    volumeMounts:
    - {name: tools, mountPath: /usr/local/bin}
  containers:
  - name: daemon
    image: $DAEMON_IMAGE
    command: [python3, /app/node_daemon.py]
    securityContext: {privileged: true}
    ports: [{containerPort: $DAEMON_PORT, hostPort: $DAEMON_PORT}]
    env:
    - {name: DAEMON_PORT, value: "$DAEMON_PORT"}
    - {name: LD_LIBRARY_PATH, value: "/usr/local/nvidia/lib64"}
    volumeMounts:
    - {name: app, mountPath: /app}
    - {name: nvidia-bin, mountPath: /usr/local/nvidia/bin, readOnly: true}
    - {name: nvidia-lib, mountPath: /usr/local/nvidia/lib64, readOnly: true}
    - {name: tools, mountPath: /usr/local/bin}
    - {name: data, mountPath: /data}
  volumes:
  - name: app
    configMap: {name: rl-orch-code}
  - name: nvidia-bin
    hostPath: {path: /home/kubernetes/bin/nvidia/bin}
  - name: nvidia-lib
    hostPath: {path: /home/kubernetes/bin/nvidia/lib64}
  - name: tools
    emptyDir: {}
  - name: data
    persistentVolumeClaim: {claimName: verl-bucket-pvc}
YAML

# ── GPU duty cycle scraper ────────────────────────────────────────────────────
echo ""
echo "=== Launching GPU duty cycle scraper ==="
kubectl delete pod gpu-scraper -n $NAMESPACE --ignore-not-found=true --force --grace-period=0 2>/dev/null || true
kubectl apply -n $NAMESPACE -f - << YAML
apiVersion: v1
kind: Pod
metadata:
  name: gpu-scraper
  annotations:
    gke-gcsfuse/volumes: "true"
spec:
  serviceAccountName: $KSA_NAME
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: $GPU_NODE
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  containers:
  - name: scraper
    image: vllm/vllm-openai:latest
    command: ["python3", "/app/gpu_duty_cycle.py"]
    securityContext:
      privileged: true
    env:
    - name: POLL_INTERVAL
      value: "1"
    - name: LOG_DIR
      value: "/data/rl_logs"
    - name: PHASE
      value: "timeslice"
    - name: NVIDIA_SMI
      value: "/usr/local/nvidia/bin/nvidia-smi"
    - name: LD_LIBRARY_PATH
      value: "/usr/local/nvidia/lib64"
    volumeMounts:
    - {name: app, mountPath: /app}
    - {name: nvidia-bin, mountPath: /usr/local/nvidia/bin, readOnly: true}
    - {name: nvidia-lib, mountPath: /usr/local/nvidia/lib64, readOnly: true}
    - {name: data, mountPath: /data}
  volumes:
  - name: app
    configMap: {name: rl-orch-code}
  - name: nvidia-bin
    hostPath: {path: /home/kubernetes/bin/nvidia/bin}
  - name: nvidia-lib
    hostPath: {path: /home/kubernetes/bin/nvidia/lib64}
  - name: data
    persistentVolumeClaim: {claimName: verl-bucket-pvc}
YAML

echo "Waiting for orchestrator, gpu-swap-daemon, and gpu-scraper..."
kubectl wait pod/gpu-orchestrator pod/gpu-swap-daemon pod/gpu-scraper \
  -n $NAMESPACE --for=condition=Ready --timeout=180s

# Verify orchestrator health
ORCH_STATUS=$(kubectl exec -n $NAMESPACE gpu-orchestrator -- \
  curl -s http://localhost:$ORCH_PORT/health 2>/dev/null || echo "unreachable")
echo "Orchestrator health: $ORCH_STATUS"

# ── RayCluster ────────────────────────────────────────────────────────────────
if [ "$SKIP_RAY" = false ]; then
  MAX_RETRIES=${MAX_RETRIES:-10}

  echo ""
  echo "=== Deploying RayCluster (retry until GPU invariant satisfied) ==="
  for attempt in $(seq 1 $MAX_RETRIES); do
    echo ""
    echo "--- Attempt $attempt / $MAX_RETRIES ---"
  kubectl delete raycluster verl-grpo-cluster -n $NAMESPACE \
    --ignore-not-found=true 2>/dev/null || true

  # Wait for worker pods to fully terminate before recreating
  echo "Waiting for old worker pods to terminate..."
  kubectl wait pods \
    -l ray.io/cluster=verl-grpo-cluster \
    -n $NAMESPACE \
    --for=delete --timeout=120s 2>/dev/null || true

  kubectl apply -n $NAMESPACE -f - << YAML
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: verl-grpo-cluster
spec:
  rayVersion: '2.40.0'
  headGroupSpec:
    rayStartParams:
      dashboard-host: '0.0.0.0'
    template:
      metadata:
        annotations:
          gke-gcsfuse/volumes: "true"
      spec:
        serviceAccountName: $KSA_NAME
        nodeSelector:
          cloud.google.com/gke-nodepool: $NODE_POOL
        initContainers:
        - name: install-deps
          image: $IMAGE
          command: ["sh", "-c", "pip install --target /ray-deps torchdata && rm -rf /ray-deps/torch /ray-deps/torch-*"]
          volumeMounts:
          - {name: ray-deps, mountPath: /ray-deps}
        containers:
        - name: ray-head
          image: $IMAGE
          resources:
            limits: {cpu: "12", memory: 50Gi}
          env:
          - {name: HF_HOME, value: /data/models}
          - {name: PYTHONPATH, value: /ray-deps:/data/verl:/data/verl/timeslice}
          - {name: ORCHESTRATOR_HOST, value: gpu-orchestrator.$NAMESPACE.svc.cluster.local}
          - {name: ORCH_PORT, value: "$ORCH_PORT"}
          volumeMounts:
          - {name: data, mountPath: /data}
          - {name: ray-deps, mountPath: /ray-deps}
        volumes:
        - name: data
          persistentVolumeClaim: {claimName: verl-bucket-pvc}
        - name: ray-deps
          emptyDir: {}
  workerGroupSpecs:
  - replicas: 2
    minReplicas: 2
    maxReplicas: 2
    groupName: gpu-workers
    rayStartParams:
      num-cpus: "16"
      num-gpus: "1"
    template:
      metadata:
        annotations:
          gke-gcsfuse/volumes: "true"
      spec:
        serviceAccountName: $KSA_NAME
        nodeSelector:
          kubernetes.io/hostname: $GPU_NODE
        tolerations:
        - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
        initContainers:
        - name: install-deps
          image: $IMAGE
          command: ["sh", "-c", "pip install --target /ray-deps torchdata && rm -rf /ray-deps/torch /ray-deps/torch-*"]
          volumeMounts:
          - {name: ray-deps, mountPath: /ray-deps}
        containers:
        - name: ray-worker
          image: $IMAGE
          env:
          - {name: HF_HOME, value: /data/models}
          - {name: PYTHONPATH, value: /ray-deps:/data/verl:/data/verl/timeslice}
          - {name: ORCHESTRATOR_HOST, value: gpu-orchestrator.$NAMESPACE.svc.cluster.local}
          - {name: ORCH_PORT, value: "$ORCH_PORT"}
          - {name: TIMESLICE_JOB_ID, value: "job-a"}
          - {name: TIMESLICE_POOL, value: "gpu"}
          - {name: LD_LIBRARY_PATH, value: "/usr/local/nvidia/lib64:/usr/local/cuda/lib64"}
          resources:
            limits:
              cpu: "16"
              memory: 170Gi
              nvidia.com/gpu: "1"
          volumeMounts:
          - {name: data, mountPath: /data}
          - {name: ray-deps, mountPath: /ray-deps}
          - {name: nvidia, mountPath: /usr/local/nvidia}
          - {name: shm, mountPath: /dev/shm}
        volumes:
        - name: data
          persistentVolumeClaim: {claimName: verl-bucket-pvc}
        - name: ray-deps
          emptyDir: {}
        - name: nvidia
          hostPath: {path: /home/kubernetes/bin/nvidia}
        - name: shm
          emptyDir: {medium: Memory, sizeLimit: 10Gi}
YAML

      echo "Waiting for Ray workers to exist..."
      for i in $(seq 1 30); do
        COUNT=$(kubectl get pods -n $NAMESPACE \
          -l ray.io/group=gpu-workers \
          --no-headers 2>/dev/null | wc -l)
        [ "$COUNT" -ge 2 ] && break
        sleep 5
      done

      get_uuid() {
          local pod=$1
          for i in $(seq 1 5); do
              local res
              res=$(kubectl exec -n $NAMESPACE "$pod" -- \
                  bash -c 'LD_LIBRARY_PATH=/usr/local/nvidia/lib64 /usr/local/nvidia/bin/nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null' | head -n1 || true)
              [ -n "$res" ] && echo "$res" && return
              sleep 2
          done
          echo "pending"
      }

      # Get pod names
      PODS=($(kubectl get pods -n $NAMESPACE -l ray.io/group=gpu-workers -o jsonpath='{.items[*].metadata.name}'))
      pod_a=${PODS[0]}
      pod_b=${PODS[1]}

      if [ -z "$pod_a" ] || [ -z "$pod_b" ]; then
          echo "  ✗ Error: Could not find two worker pods."
          continue
      fi

      UUID_A=$(get_uuid $pod_a)
      UUID_B=$(get_uuid $pod_b)

      echo "  Worker 1 ($pod_a): $UUID_A"
      echo "  Worker 2 ($pod_b): $UUID_B"

      if [ "$UUID_A" != "pending" ] && [ "$UUID_B" != "pending" ]; then
          if [ "$UUID_A" != "$UUID_B" ]; then
              echo "  ✓ GPU invariant satisfied (workers on different GPUs)"
              break
          else
              echo "  ✗ GPU invariant failed (workers on SAME GPU). Restarting cluster..."
          fi
      else
          echo "  ✗ GPU invariant check failed. UUIDs pending. Restarting cluster..."
      fi

      if [ $attempt -eq $MAX_RETRIES ]; then
          echo "ERROR: GPU invariant not satisfied after $MAX_RETRIES attempts."
          exit 1
      fi
  done

  echo "Waiting for Ray head node to be Ready..."
  kubectl wait pods \
    -l ray.io/node-type=head \
    -n $NAMESPACE \
    --for=condition=Ready --timeout=600s || { echo "ERROR: Ray head node failed to become Ready."; exit 1; }

  echo "Waiting for Ray workers to be Ready..."
  kubectl wait pods \
    -l ray.io/group=gpu-workers \
    -n $NAMESPACE \
    --for=condition=Ready --timeout=600s || { echo "ERROR: Ray worker nodes failed to become Ready."; exit 1; }

  # Verify CUDA visible
  echo ""
  echo "=== Verifying CUDA ==="
  for WORKER in $(kubectl get pods -n $NAMESPACE -l ray.io/group=gpu-workers -o jsonpath='{.items[*].metadata.name}'); do
    echo "Verifying $WORKER..."
    if kubectl exec -n $NAMESPACE $WORKER -- \
      python3 -c "
import torch
print('cuda available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
assert torch.cuda.is_available(), 'CUDA not visible — check volume mounts'
print('CUDA OK')
"; then
      echo "✓ Worker $WORKER CUDA check passed"
    else
      echo "ERROR: CUDA not visible on $WORKER"
      exit 1
    fi
  done

  # Patch worker pods
  echo ""
  echo "=== Patching worker pods ==="
  for WORKER in $(kubectl get pods -n $NAMESPACE -l ray.io/cluster=verl-grpo-cluster \
    -o jsonpath='{.items[*].metadata.name}'); do
    echo "=== Patching $WORKER ==="
    kubectl exec -n $NAMESPACE $WORKER -- bash -c "
      # 1. numpy upgrade for nccl CuPy dependency
      pip install --upgrade numpy --break-system-packages -q 2>&1 | tail -1

      # 2. layered_summon and peft_merge missing on DetachActorWorker
      sed -i 's/layered_summon=self\.layered_summon/layered_summon=getattr(self, \"layered_summon\", False)/g' \
        /data/verl/verl/workers/engine_workers.py
      sed -i 's/if not self\.peft_merge/if not getattr(self, \"peft_merge\", False)/g' \
        /data/verl/verl/workers/engine_workers.py

      # 3. num_cpus too high
      sed -i 's/@ray.remote(num_cpus=10)/@ray.remote(num_cpus=1)/' \
        /data/verl/verl/experimental/fully_async_policy/fully_async_trainer.py
      sed -i 's/@ray.remote(num_cpus=10, max_concurrency=100)/@ray.remote(num_cpus=1, max_concurrency=100)/' \
        /data/verl/verl/experimental/fully_async_policy/fully_async_rollouter.py

      # 4. nccl engine not imported in fully_async files
      grep -q 'nccl_checkpoint_engine' \
        /data/verl/verl/experimental/fully_async_policy/fully_async_trainer.py || \
        sed -i '28 a try:\n    from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine\nexcept ImportError:\n    pass' \
        /data/verl/verl/experimental/fully_async_policy/fully_async_trainer.py

      grep -q 'nccl_checkpoint_engine' \
        /data/verl/verl/experimental/fully_async_policy/fully_async_rollouter.py || \
        sed -i '1 a try:\n    from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine\nexcept ImportError:\n    pass' \
        /data/verl/verl/experimental/fully_async_policy/fully_async_rollouter.py

      # Verify
      echo 'numpy:' \$(python3 -c 'import numpy; print(numpy.__version__)')
      python3 -c 'from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine; print(\"nccl: OK\")'
      echo 'layered_summon fixes:' \$(grep -c 'getattr.*layered_summon' /data/verl/verl/workers/engine_workers.py)
    " 2>/dev/null || echo "skipped"
  done
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Deploy complete"
echo ""
kubectl get pods -n $NAMESPACE -o wide
echo ""
echo "Next steps:"
echo "  1. Port-forward dashboard:"
echo "     kubectl port-forward svc/verl-grpo-cluster-head-svc 8265:8265 -n $NAMESPACE"
echo ""
POD_A=$(kubectl get pods -n $NAMESPACE -l ray.io/group=gpu-workers -o jsonpath='{.items[0].metadata.name}')
POD_B=$(kubectl get pods -n $NAMESPACE -l ray.io/group=gpu-workers -o jsonpath='{.items[1].metadata.name}')
POD_A_IP=$(kubectl get pod $POD_A -n $NAMESPACE -o jsonpath='{.status.podIP}')
POD_B_IP=$(kubectl get pod $POD_B -n $NAMESPACE -o jsonpath='{.status.podIP}')

echo "  2. Submit jobs:"
echo "     See testing_sync.md for command examples."
echo ""
echo "  3. Watch orchestrator:"
echo "     kubectl logs -f gpu-orchestrator -n $NAMESPACE"
echo "========================================"