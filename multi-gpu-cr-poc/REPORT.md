# Multi-GPU Checkpoint/Restore Test Report

## Summary

Full end-to-end checkpoint/restore validated across 3 workloads, 5 GPU topologies, on H100 80GB. All tests pass with post-restore inference/training verification.

**Recipe:** NCCL TCP transport (3 env vars) + LD_PRELOAD shim (ncclCommSuspend/Resume) + framework-specific CUDA graph disable.

## Environment

| Component | Version |
|-----------|---------|
| GPU | NVIDIA H100 80GB HBM3 (2 per node, 2 nodes) |
| Driver | 580.126.20 |
| cuda-checkpoint | 580.126.20 |
| NCCL | 2.30.7 (ncclCommSuspend/Resume) |
| vLLM | 0.23.0 |
| SGLang | 0.5.14 |
| Model | facebook/opt-1.3b |

## Test Results

| # | Test | Topology | Shim | Freeze | Restore | Total |
|---|------|----------|------|--------|---------|-------|
| 1 | vLLM TP=1 | 1 GPU, 1 pod | no | 15.1s | 5.8s | 23.9s |
| 2 | vLLM TP=2 | 2 GPU, 1 pod | yes | 30.9s | 11.2s | 50.1s |
| 3 | SGLang TP=1 | 1 GPU, 1 pod | no | 16.3s | 6.0s | 25.3s |
| 4 | SGLang TP=2 | 2 GPU, 1 pod | yes | 32.7s | 11.9s | 52.6s |
| 5 | FSDP 1-GPU | 1 GPU, 1 pod | no | 2.7s | 1.7s | 7.3s |
| 6 | FSDP DP=2 | 2 GPU, 1 pod | yes | 0.1s | 1.7s | 13.8s |
| 7 | vLLM TP=2 DP=2 (node 0) | 2 GPU, 1 pod per node | yes | 30.9s | 12.0s | 50.9s |
| 8 | vLLM TP=2 DP=2 (node 1) | 2 GPU, 1 pod per node | yes | 31.1s | 12.1s | 51.2s |
| 9 | vLLM DP=2 intra-node | 2×TP=1, 1 pod | no | 30.9s | 10.9s | 41.9s |

**9/9 PASS. All with post-restore inference/training verification.**

## Required Configuration

### NCCL Transport (multi-GPU only)

Forces NCCL to use TCP loopback instead of NVLink P2P / SHM / NVLS:

```bash
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NVLS_ENABLE=0
```

**Why:** NCCL's SHM and P2P transports create cross-process GPU shared memory that cuda-checkpoint cannot restore ([NVIDIA/cuda-checkpoint#27](https://github.com/NVIDIA/cuda-checkpoint/issues/27)). TCP transport uses no cross-process GPU state.

### LD_PRELOAD Shim (multi-GPU only)

```bash
export LD_PRELOAD="/path/to/libcr-shim.so:/path/to/libnccl.so.2"
```

The shim intercepts `ncclCommInitRank` to track communicator handles, then provides `ncclCommSuspend`/`Resume` via real-time signals. Without the shim, cuda-checkpoint hangs at freeze even with TCP transport.

Source: `universal_cr_shim.c` (~120 lines). Build: `gcc -shared -fPIC -o libcr-shim.so universal_cr_shim.c -ldl`

### Framework-Specific Flags

| Framework | Flags | Reason |
|-----------|-------|--------|
| vLLM | `--enforce-eager --disable-custom-all-reduce` + `VLLM_ALLREDUCE_USE_SYMM_MEM=0` | CUDA graphs and custom allreduce create persistent GPU state |
| SGLang | `--disable-cuda-graph` | CUDA graphs |
| FSDP/DDP | None | No CUDA graphs by default |

### Single-GPU Workloads

No shim, no NCCL env vars, no framework flags needed. Just `cuda-checkpoint --toggle --pid <PID>`.

## C/R Orchestration Protocol

### Sequence

```
1. SUSPEND   kill -35 <all_gpu_pids>                      # SIGRTMIN+1 → ncclCommSuspend
2. FREEZE    for pid in pids; cuda-checkpoint --toggle --pid $pid  # sequential
   [GPU memory fully released — other workload can use GPUs]
3. RESTORE   for pid in pids; cuda-checkpoint --toggle --pid $pid  # sequential
4. RESUME    kill -36 <all_gpu_pids>                      # SIGRTMIN+2 → ncclCommResume
```

### Ordering

| Step | Ordering | Why |
|------|----------|-----|
| Suspend | All PIDs at once | Signal delivery is async, order doesn't matter |
| Freeze | Sequential, one PID at a time | Parallel freeze can deadlock |
| Restore | Sequential, one PID at a time | Avoids restore ordering issues |
| Resume | All PIDs at once | Signal delivery is async |

### Cross-Node (DP)

Each node runs its C/R cycle independently. No cross-node coordination required for TP groups. For active DP training (gradient sync in-flight), suspend all nodes simultaneously to avoid NCCL timeouts.

## Performance Impact

| Metric | NVLink P2P | TCP Loopback | Impact |
|--------|-----------|--------------|--------|
| NCCL bandwidth | ~900 GB/s | ~10-20 GB/s | 50-100x slower collectives |
| NCCL latency | ~1-5 μs | ~50-100 μs | 10-50x higher |
| Inference throughput | baseline | ~10-30% loss for 70B+ TP models | Negligible for small models |
| Training throughput | baseline | 2-5x slower for comm-bound workloads | Significant |

NCCL env vars must be set at pod launch (before NCCL init). Cannot be toggled at C/R time.

## Reproduction

### Prerequisites

- GKE cluster with H100 nodes (2+ GPUs per node)
- `kubectl` configured
- Manifests: `test-shell.yaml` (single node), `test-tp2-dp2.yaml` (two nodes)

### Setup (run once per pod)

```bash
# Deploy pod
kubectl apply -f test-shell.yaml
kubectl wait --for=condition=Ready pod/cr-test-shell --timeout=180s

# Install deps inside pod
kubectl exec cr-test-shell -c shell -- bash -c '
  export PATH=$PATH:/usr/local/nvidia/bin:/opt/bin
  pip install --target /tmp/nccl_new "nvidia-nccl-cu12>=2.29.7" -q
  apt-get update -qq && apt-get install -y -qq build-essential
'

# Copy and build shim
kubectl cp universal_cr_shim.c cr-test-shell:/tmp/shim.c -c shell
kubectl exec cr-test-shell -c shell -- bash -c '
  gcc -shared -fPIC -o /opt/bin/libcr-shim.so \
    -I/tmp/nccl_new/nvidia/nccl/include /tmp/shim.c -ldl
'
```

### Test 1: vLLM TP=1 (single GPU, no shim)

```bash
kubectl exec cr-test-shell -c shell -- bash -c '
  export PATH=$PATH:/usr/local/nvidia/bin:/opt/bin
  vllm serve facebook/opt-1.3b --tensor-parallel-size 1 \
    --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.4 \
    --enforce-eager > /tmp/w.log 2>&1 &

  # Wait for health
  for i in $(seq 1 90); do curl -sf http://localhost:8000/health && break; sleep 2; done

  # Pre-check
  curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"facebook/opt-1.3b\",\"prompt\":\"Hello\",\"max_tokens\":5}"

  # C/R
  PID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -1 | tr -d " ")
  cuda-checkpoint --toggle --pid $PID          # freeze
  cuda-checkpoint --get-state --pid $PID       # should be "checkpointed"
  cuda-checkpoint --toggle --pid $PID          # restore
  cuda-checkpoint --get-state --pid $PID       # should be "running"

  # Post-check
  sleep 3
  curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"facebook/opt-1.3b\",\"prompt\":\"Hello\",\"max_tokens\":5}"
'
```

### Test 2: vLLM TP=2 (multi-GPU, shim + TCP)

```bash
# Requires fresh pod (delete and recreate if reusing from test 1)
kubectl exec cr-test-shell -c shell -- bash -c '
  export PATH=$PATH:/usr/local/nvidia/bin:/opt/bin
  export LD_PRELOAD="/opt/bin/libcr-shim.so:/tmp/nccl_new/nvidia/nccl/lib/libnccl.so.2"
  export LD_LIBRARY_PATH=/tmp/nccl_new/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}
  export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NVLS_ENABLE=0
  export VLLM_ALLREDUCE_USE_SYMM_MEM=0

  vllm serve facebook/opt-1.3b --tensor-parallel-size 2 \
    --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.4 \
    --enforce-eager --disable-custom-all-reduce > /tmp/w.log 2>&1 &

  for i in $(seq 1 120); do curl -sf http://localhost:8000/health && break; sleep 2; done

  # Pre-check
  curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"facebook/opt-1.3b\",\"prompt\":\"Hello\",\"max_tokens\":5}"

  PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -un | head -2 | tr -d " ")
  PA=($PIDS)

  # Suspend → Freeze → Restore → Resume
  for p in "${PA[@]}"; do kill -35 $p; done; sleep 3         # suspend NCCL
  for p in "${PA[@]}"; do cuda-checkpoint --toggle --pid $p; done  # freeze
  for p in "${PA[@]}"; do cuda-checkpoint --toggle --pid $p; sleep 1; done  # restore
  for p in "${PA[@]}"; do kill -36 $p; done; sleep 5         # resume NCCL

  # Post-check
  curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"facebook/opt-1.3b\",\"prompt\":\"Hello\",\"max_tokens\":5}"
'
```

### Test 3: SGLang TP=2 (multi-GPU, shim + TCP)

```bash
# Install SGLang first:
kubectl exec cr-test-shell -c shell -- pip install "sglang[all]" -q

kubectl exec cr-test-shell -c shell -- bash -c '
  export PATH=$PATH:/usr/local/nvidia/bin:/opt/bin
  export LD_PRELOAD="/opt/bin/libcr-shim.so:/tmp/nccl_new/nvidia/nccl/lib/libnccl.so.2"
  export LD_LIBRARY_PATH=/tmp/nccl_new/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}
  export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NVLS_ENABLE=0

  python3 -m sglang.launch_server --model-path facebook/opt-1.3b \
    --tp 2 --host 0.0.0.0 --port 8000 --mem-fraction-static 0.4 \
    --disable-cuda-graph > /tmp/w.log 2>&1 &

  for i in $(seq 1 120); do curl -s http://localhost:8000/v1/models | grep -q opt && break; sleep 3; done

  # Same C/R sequence as Test 2 (suspend → freeze → restore → resume → verify)
  PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -un | head -2 | tr -d " ")
  PA=($PIDS)
  for p in "${PA[@]}"; do kill -35 $p; done; sleep 3
  for p in "${PA[@]}"; do cuda-checkpoint --toggle --pid $p; done
  for p in "${PA[@]}"; do cuda-checkpoint --toggle --pid $p; sleep 1; done
  for p in "${PA[@]}"; do kill -36 $p; done; sleep 5

  curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"facebook/opt-1.3b\",\"prompt\":\"Hello\",\"max_tokens\":5}"
'
```

### Test 4: FSDP DP=2 (multi-GPU training)

```bash
kubectl cp test_fsdp_trainer.py cr-test-shell:/tmp/trainer.py -c shell

kubectl exec cr-test-shell -c shell -- bash -c '
  export PATH=$PATH:/usr/local/nvidia/bin:/opt/bin
  export LD_PRELOAD="/opt/bin/libcr-shim.so:/tmp/nccl_new/nvidia/nccl/lib/libnccl.so.2"
  export LD_LIBRARY_PATH=/tmp/nccl_new/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}
  export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NVLS_ENABLE=0

  torchrun --nproc_per_node=2 /tmp/trainer.py > /tmp/w.log 2>&1 &
  sleep 20

  # Pre-check: training producing loss values
  grep "step=" /tmp/w.log | tail -1

  PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -un | head -2 | tr -d " ")
  PA=($PIDS)
  for p in "${PA[@]}"; do kill -35 $p; done; sleep 3
  for p in "${PA[@]}"; do cuda-checkpoint --toggle --pid $p; done
  for p in "${PA[@]}"; do cuda-checkpoint --toggle --pid $p; sleep 1; done
  for p in "${PA[@]}"; do kill -36 $p; done; sleep 8

  # Post-check: training resumed
  grep "step=" /tmp/w.log | tail -1
'
```

### Test 5: vLLM TP=2 DP=2 (2 nodes × 2 GPUs)

```bash
# Deploy 2 pods on 2 nodes
kubectl apply -f test-tp2-dp2.yaml
kubectl wait --for=condition=Ready pod/dp-node0 pod/dp-node1 --timeout=180s

# Setup both pods (same as single-node setup, for each pod)
for POD in dp-node0 dp-node1; do
  kubectl cp universal_cr_shim.c $POD:/tmp/shim.c -c shell
  kubectl exec $POD -c shell -- bash -c '
    pip install --target /tmp/nccl_new "nvidia-nccl-cu12>=2.29.7" -q
    apt-get update -qq && apt-get install -y -qq build-essential
    gcc -shared -fPIC -o /opt/bin/libcr-shim.so \
      -I/tmp/nccl_new/nvidia/nccl/include /tmp/shim.c -ldl
  '
done

# Start vLLM TP=2 on both
for POD in dp-node0 dp-node1; do
  kubectl exec $POD -c shell -- bash -c '
    export PATH=$PATH:/usr/local/nvidia/bin:/opt/bin
    export LD_PRELOAD="/opt/bin/libcr-shim.so:/tmp/nccl_new/nvidia/nccl/lib/libnccl.so.2"
    export LD_LIBRARY_PATH=/tmp/nccl_new/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}
    export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NVLS_ENABLE=0
    export VLLM_ALLREDUCE_USE_SYMM_MEM=0
    vllm serve facebook/opt-1.3b --tensor-parallel-size 2 \
      --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.4 \
      --enforce-eager --disable-custom-all-reduce > /tmp/vllm.log 2>&1 &
  '
done

# Wait for both healthy
for POD in dp-node0 dp-node1; do
  for i in $(seq 1 120); do
    kubectl exec $POD -c shell -- curl -sf http://localhost:8000/health && break; sleep 3
  done
done

# C/R each node independently (self-contained per pod)
for POD in dp-node0 dp-node1; do
  kubectl exec $POD -c shell -- bash -c '
    export PATH=$PATH:/usr/local/nvidia/bin:/opt/bin
    PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -un | head -2 | tr -d " ")
    PA=($PIDS)
    for p in "${PA[@]}"; do kill -35 $p; done; sleep 3
    for p in "${PA[@]}"; do cuda-checkpoint --toggle --pid $p; done
    for p in "${PA[@]}"; do cuda-checkpoint --toggle --pid $p; sleep 1; done
    for p in "${PA[@]}"; do kill -36 $p; done; sleep 5
    curl -s http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"facebook/opt-1.3b\",\"prompt\":\"Hello\",\"max_tokens\":5}"
  '
done
```

## Known Limitations

1. **NCCL transport permanently degraded.** TCP loopback is set at pod launch. Cannot switch to NVLink mid-run. NVIDIA needs to fix `ncclCommSuspend` to fully tear down SHM/P2P state.

2. **CUDA graphs must be disabled.** CUDA graphs create persistent GPU state that cuda-checkpoint cannot restore. Framework-specific flags required (vLLM: `--enforce-eager`, SGLang: `--disable-cuda-graph`).

3. **Requires NCCL ≥ 2.29.7.** For `ncclCommSuspend`/`Resume` API. Older NCCL versions don't have this.

4. **cuda-checkpoint driver bugs.** Multi-GPU restore with SHM/P2P transport is broken ([#27](https://github.com/NVIDIA/cuda-checkpoint/issues/27), [#47](https://github.com/NVIDIA/cuda-checkpoint/issues/47)). TCP transport works around this.
