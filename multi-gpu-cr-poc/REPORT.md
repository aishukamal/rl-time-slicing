# Multi-GPU Checkpoint/Restore Test Report

## Summary

Full end-to-end checkpoint/restore validated across 3 workloads, 5 GPU topologies, on H100 80GB. All tests pass with post-restore inference/training verification.

Two shim generations:

- **Shim v1** (`universal_cr_shim.c`): ncclCommSuspend/Resume across the freeze. Requires NCCL on TCP transport — **no NVLink at steady state** (50-100x slower collectives).
- **Shim v2** (`universal_cr_shim_v2.c`): destroys NCCL comms before freeze, recreates them (fresh uniqueId rendezvous + handle indirection) after restore. **NVLink P2P stays enabled at steady state — zero performance tax.** See the "Shim v2" section below.

**v1 recipe:** NCCL TCP transport (3 env vars) + LD_PRELOAD shim (ncclCommSuspend/Resume) + framework-specific CUDA graph disable.

**v2 recipe:** `NCCL_NVLS_ENABLE=0` + LD_PRELOAD shim v2 + framework-specific CUDA graph disable. NVLink P2P active.

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

NCCL env vars must be set at pod launch (before NCCL init). Cannot be toggled at C/R time — which is exactly why shim v2 (below) exists: it removes the TCP requirement entirely.

## Shim v2: NVLink at Steady State (destroy/recreate)

`universal_cr_shim_v2.c` eliminates v1's performance tax. Instead of suspending comms across the freeze (which requires the checkpoint-safe TCP transport), v2 **destroys** all NCCL communicators before the freeze and **recreates** them after restore:

```
Steady state:  NVLink P2P — full speed (NCCL_NVLS_ENABLE=0 is the only restriction)
C/R window:    quiesce workload
               SIGRTMIN+1  → ncclCommDestroy all comms   (~350-450ms)
                             (NCCL itself tears down ALL P2P/SHM cross-process state)
               cuda-checkpoint freeze                     (sees only process-private state)
               ... GPU free for other workloads ...
               cuda-checkpoint restore
               SIGRTMIN+2  → arms lazy recreate (flag only, async-signal-safe)
               workload resumes → first collective call performs:
                   fresh ncclUniqueId rendezvous (rank 0 generates + publishes;
                   original uniqueId is stale — bootstrap sockets die at freeze)
                   + collective ncclCommInitRank on the app's own thread
               NVLink P2P re-established
```

The framework (PyTorch, vLLM) still holds the original `ncclComm_t` handles: the shim keeps an `app_handle → current_handle` table and translates on every intercepted NCCL call, so the recreate is invisible. While comms are destroyed, query calls (e.g. PyTorch's watchdog polling `ncclCommGetAsyncError`) are answered from cached values.

### v2 test results (2x H100 NV18, opt-1.3b, driver 580.126.20)

| Test | NVLink steady state | Destroy | Freeze | Restore | Recreate | Post-C/R verification |
|------|--------------------|---------| -------|---------|----------|----------------------|
| FSDP DP=2 (3 C/R cycles) | confirmed (traffic counters) | ~360ms | PASS, VRAM=0 | PASS | PASS ×3 | training continues, NVLink traffic confirmed |
| vLLM TP=2 | confirmed (traffic counters) | ~410ms | PASS, VRAM=0 | PASS | PASS | inference correct, NVLink traffic confirmed |

### vLLM specifics: PyNCCL routing

vLLM's PyNCCL loads NCCL via ctypes `dlopen`+`dlsym`, bypassing LD_PRELOAD interposition — its comms would be invisible to the shim (and their P2P state would break the freeze). Fix: point `VLLM_NCCL_SO_PATH` at the shim itself. The shim exports the full PyNCCL surface (version/error/group/mem functions) and forwards to the real NCCL (located via `CR_NCCL_LIB`), so PyNCCL's comms are tracked and translated like everything else. Also set `VLLM_ALLREDUCE_USE_SYMM_MEM=0` (symmetric-memory all-reduce creates cuMem state outside NCCL comms).

```bash
NCCL_NVLS_ENABLE=0 \
VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
LD_PRELOAD="/opt/bin/libcr-shim-v2.so:/path/to/libnccl.so.2" \
CR_NCCL_LIB=/path/to/libnccl.so.2 \
VLLM_NCCL_SO_PATH=/opt/bin/libcr-shim-v2.so \
vllm serve ... --tensor-parallel-size 2 --enforce-eager --disable-custom-all-reduce
```

### v2 design notes (lessons from failed variants)

1. **Recreate must NOT run in the signal handler.** `ncclCommInitRank` spawns threads, allocates, opens sockets — async-signal-unsafe. Running it from the handler wedged rank 0's bootstrap listener ("Connection refused" on rank 1 after 35 retries). The handler only sets a flag; the first intercepted collective after resume performs the re-init on the app's thread. Destroy in the handler is fine in practice (workload quiesced).
2. **Fresh uniqueId is mandatory.** The original uniqueId encodes a bootstrap TCP rendezvous that is dead after restore. Reusing it fails with `ncclUnhandledCudaError`/`ncclRemoteError`.
3. **NVLS must be off at launch — verified driver limitation (`mc_test.c`).** `ncclNvlsSetup` fails post-restore (`Cuda failure 101 'invalid device ordinal'` in `nvlsAllocateMem`), reproduced with the lazy on-app-thread recreate too, so it is not a signal-context artifact. A minimal NCCL-free repro (`mc_test.c`: cuMulticastCreate/AddDevice across a cuda-checkpoint cycle) isolates it: post-restore, `cuMulticastCreate` still succeeds but `cuMulticastAddDevice` fails with error 101 for every device — **including from a freshly created CUDA context**, i.e. the multicast device-binding state is stale process-wide in the driver, and no userspace workaround exists inside the restored process. Only NVIDIA can fix this (cuda-checkpoint/driver). Setting `NCCL_NVLS_ENABLE=0` at recreate time is also not enough: NCCL caches parsed env params in statics, so it must be off at launch. NVLink **P2P** (the 50-100x win over TCP) is unaffected; NVLS in-fabric reduction is a modest additional gain at 2 GPUs.
4. **CUDA graphs still disabled** (`--enforce-eager` / `--disable-cuda-graph`) — unchanged from v1; cuda-checkpoint limitation, independent of transport.

### Remaining v2 limitations

- **NVLS off** (see note 3) — needs NVIDIA driver support for multicast re-creation after restore.
- **Rendezvous is same-host** (file in `/dev/shm`, override with `CR_RENDEZVOUS_DIR`). Multi-node TP needs a shared volume or a TCP rendezvous extension.
- **`ncclCommRegister`-registered buffers** are not re-registered after recreate (not used by stock PyTorch/vLLM paths tested here).

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

## GPU-CR Combo (experimental)

[GPU-CR](https://github.com/gpu-os/GPU-CR) (v0.2.1) provides a faster data plane: hugepage-backed staging buffers dump GPU memory at ~12.4 GB/s before cuda-checkpoint runs, so freeze/restore handles only metadata.

Combined with our ncclCommSuspend shim (4 patches to upstream GPU-CR), inference workloads see ~2.7x faster C/R.

### Complete GPU-CR Combo Test Matrix

| # | Test | Topology | Combo ckpt | Combo rst | Status |
|---|------|----------|-----------|----------|--------|
| 1 | vLLM TP=1 | 1 GPU | 2.4s | 0.6s | **PASS** |
| 2 | vLLM TP=2 | 2 GPU, 1 pod | 5.7s | 1.9s | **PASS** |
| 3 | SGLang TP=1 | 1 GPU | 2.3s | 0.5s | **PASS** |
| 4 | SGLang TP=2 | 2 GPU, 1 pod | 7.7s | 1.9s | **PASS** |
| 5 | FSDP single GPU | 1 GPU | OK | OK | **PASS** |
| 6 | FSDP DP=2 | 2 GPU, 1 pod | 4.8s | 0.9s | **PASS** |

All tests on 2xH100 80GB, opt-1.3b, gpu_util=0.15, driver 580.126.20.

**6/6 PASS. All with post-restore inference/training verification.**

### Speedup vs Pure Shim

| Test | Pure shim ckpt | Pure shim rst | Combo ckpt | Combo rst | Speedup |
|------|---------------|---------------|-----------|----------|---------|
| vLLM TP=2 | 15.8s | 5.4s | **5.7s** | **1.9s** | **2.8x** |
| SGLang TP=2 | 32.7s | 11.9s | **7.7s** | **1.9s** | **4.7x** |

### Patches applied to upstream GPU-CR

1. **RT signals**: `CR_CKPT_SIGNAL`/`CR_RESTORE_SIGNAL` changed from SIGUSR1/2 to SIGRTMAX-4/-5 (vLLM 0.25 API server overrides SIGUSR1)
2. **ncclCommSuspend re-enabled**: upstream removed suspend ("NO LONGER used"); re-implemented in `nccl_hooks.cpp` using `find_real_nccl_sym`
3. **Suspend wired into ckpt handler**: called in `vGPU.cpp` after data dump + P2P disable, before cuda-checkpoint freeze; resume in IPC import handler
4. **Sequential freeze**: `multi_cr_client.cpp` changed from `cuda_ckpt_all_parallel` to `cuda_ckpt_all` (sequential)
5. **SGLang compatibility**: `--disable-custom-all-reduce` required (GPU-CR's cuMemMap hook breaks legacy cudaIpcGetMemHandle)

### FSDP combo: the NCCL allocation problem and fix

GPU-CR hooks ALL `cudaMalloc` calls via `cuGetProcAddress` interception and redirects them through cuMem VMM. This includes NCCL's internal `cudaMalloc` calls (16MB at `0x402000000`, etc.). When `ckpt()` calls `cuMemUnmap` on these NCCL-owned VMM allocations, the driver crashes (SIGSEGV) because NCCL retains internal references even after `ncclCommSuspend`.

vLLM/SGLang TP=2 avoids this because their NCCL creates IPC-exported allocations — GPU-CR's IPC teardown (Phase 1) handles them cleanly. FSDP's NCCL creates local-only allocations (no IPC), so teardown is a no-op.

**Fix (3 changes to GPU-CR):**
1. `vGPU.cpp ckpt()`: When NCCL comms are present, skip `releasePhysicalMemory` entirely. The data dump to hugepages still runs (copies GPU data at ~12 GB/s). `cuda-checkpoint` freeze releases ALL GPU memory at the kernel level — VRAM is fully freed.
2. `vGPU.cpp restore_ptr_and_content()`: When NCCL comms are present, skip `remapPhysicalMemory` and data restore. `cuda-checkpoint` restore already brings back all memory contents.
3. `vGPU.cpp RESTORE_MSG handler`: Add `nccl_resume_all_comms()` after P2P re-enable, so NCCL resume happens even when `-n` flag skips IPC rebuild.

Use `multi_cr_client -n` for FSDP (no IPC to tear down/rebuild).

**Also attempted (did not work):**
- **ncclCommDestroy before ckpt():** Destroy properly frees NCCL allocations via cudaFree hook (GPU-CR VMM cleanup works). But `ncclCommInitRank` after restore fails with `ncclUnhandledCudaError` (rc=6) — NCCL's bootstrap TCP sockets are stale after cuda-checkpoint restore, so the rendezvous can't re-establish.
- **Track NCCL allocs via IPC local alloc system:** Removed the `requestedHandleTypes != 0` filter in `hook_cuMemCreate` so `g_created_allocs` tracks ALL cuMem allocations. `ipc_save_and_teardown_local_allocs` then finds 278MB of local allocs — but it calls the same `cuMemUnmap` that crashes (SIGSEGV).
- **NCCL bypass during cudaMalloc:** Thread-local flag during `ncclCommInitRank` to route NCCL's cudaMalloc to real cudaMalloc. Doesn't work — NCCL allocations happen before `ncclCommInitRank` (during `dist.init_process_group`).

**Root cause:** GPU-CR's cudaMalloc→VMM hook and NCCL both operate on the cuMem VMM layer. After `ncclCommSuspend`, NCCL retains internal VMM state (mappings, handles) that makes `cuMemUnmap` from GPU-CR crash. Fixing this requires either NCCL exposing a "release cuMem handles" API in suspend, or GPU-CR detecting and excluding NCCL's allocations from its tracking at cudaMalloc time (needs reliable caller identification beyond thread-local flags).

**Trade-off:** For FSDP DP=2 with NCCL comms, the GPU-CR data plane (hugepage dump/restore) is bypassed — no speedup over pure shim. For inference workloads (vLLM/SGLang TP=2), full GPU-CR data plane is active and provides 2.8-4.7x speedup.

### Additional requirements (beyond pure shim)

- **Hugepages**: 40GB+ hugetlbfs mount at `/mnt/huge-ckpt`; pod cgroup `hugetlb.2MB.max` and `hugetlb.2MB.rsvd.max` set to `max` at every cgroup level
- **Env vars**: `NCCL_CUMEM_ENABLE=1`, `NCCL_CUMEM_HOST_ENABLE=1` (in addition to TCP transport vars)
- **Build**: `SHM_SIZE_GB=15` for opt-1.3b at 0.15 util (default 25GB per worker is too large)
- **GPU-CR build**: `vGPU-NVIDIA.so` + `multi_cr_client` (cmake, ~2MB)
- **Warm-up inference**: required for SGLang before first checkpoint (NCCL comm creation is lazy)
- **Quiesced workloads**: training must pause between steps before checkpoint

### Why upstream GPU-CR multi-GPU fails on modern vLLM

Upstream claims multi-GPU support for PP=2 on vLLM 0.14.1 + A100. On vLLM 0.25+ with TP=2, it fails at three layers:
1. SIGUSR1 conflict (vLLM API server overrides it)
2. No ncclCommSuspend (removed from upstream, NCCL proxy threads block cuda-checkpoint drain)
3. Parallel freeze (deadlocks with active NCCL)

**Note:** Upstream's PP=2 claim was NOT independently tested by us. Their benchmark configuration (vLLM 0.14.1, PP=2, A100) differs significantly from our test environment.

## C/R Approaches: Options Matrix

Three approaches exist, each with different trade-offs. All validated on 2x H100 NV18.

### Option A: vLLM sleep only (cuMemUnmap-based, no cuda-checkpoint)

vLLM's `/sleep` + `/wake_up` endpoints release and restore model weights + KV cache via cuMemUnmap/cuMemMap. No cuda-checkpoint, no shim signals, no process freeze.

| Feature | Status |
|---------|--------|
| CUDA graphs | **ON** — survive sleep/wake (never destroyed) |
| NVLS | **ON** — survives sleep/wake (NCCL comms never torn down) |
| NVLink P2P | **ON** — full speed |
| GPU memory freed | **96%** — ~3.4 GB residual (NCCL buffers, graphs, CUDA contexts, PyTorch runtime) |
| Multi-cycle | **PASS** — verified 2+ consecutive cycles |
| Framework support | vLLM only (requires `--enable-sleep-mode` + `VLLM_SERVER_DEV_MODE=1`) |
| Snapshot agent backend | `app-endpoint` (already integrated) |

**Limitation:** NCCL state stays on GPU (~1 GB of P2P/NVLS buffers), so the incoming workload gets ~77 GB instead of 80 GB. Sufficient for most workloads.

**Why vLLM sleep + cuda-checkpoint doesn't compose:** after cuMemUnmap, cuda-checkpoint's `--action lock` succeeds but `--action checkpoint` hangs — the half-unmapped VMM state confuses the driver's checkpoint code path.

**Why shim destroy + vLLM sleep doesn't compose:** CUDA graphs reference NCCL comm handles. To destroy comms, graphs must be reset first ([PyTorch #115388](https://github.com/pytorch/pytorch/issues/115388)). But if the shim resets vLLM's graphs externally, vLLM's internal CUDAGraphRunner state becomes inconsistent → `cudaErrorInvalidValue` on the next inference. The graph reset must come from inside vLLM (a vLLM feature gap, not a driver issue).

### Option B: Shim v2 + cuda-checkpoint (100% GPU release, no graphs, TCP or NVLink)

The shim destroys NCCL comms (clearing all cross-process state), cuda-checkpoint freezes the process (releasing ALL GPU memory to zero), then restores and the shim recreates comms with a fresh rendezvous. Full GPU release — another workload gets 100% of VRAM.

| Feature | Shim v2 (NVLink) | Shim v1 (TCP) |
|---------|------------------|---------------|
| CUDA graphs | **OFF** (`--enforce-eager`) — driver refuses to freeze multi-device process with captured graphs, verified via both CLI and in-process API (`cuCheckpointProcessLock` rc=1) | **OFF** |
| NVLS | **OFF** (`NCCL_NVLS_ENABLE=0`) — `cuMulticastAddDevice` returns error 101 post-restore, verified as driver bug with minimal repro (`mc_test.c`) | **OFF** |
| NVLink P2P | **ON** (v2) / OFF (v1) | **OFF** (TCP loopback) |
| GPU memory freed | **100%** | **100%** |
| Framework support | Universal (LD_PRELOAD, any NCCL app) | Universal |

**CUDA graphs finding (verified):**
- Single-GPU + graphs + cuda-checkpoint: **PASS** (graphs survive address-preserving restore)
- Multi-GPU + graphs + cuda-checkpoint: **FAIL** — even with all graph executables destroyed from userspace, the driver's checkpoint path retains internal graph bookkeeping and refuses the freeze. Same result via CLI (`--toggle`) and in-process API (`cuCheckpointProcessLock`). Tested: piecewise mode with no NCCL collectives in graphs, graph exec + template destroy, cudaDeviceSynchronize — nothing unblocks it.
- [PyTorch #115388](https://github.com/pytorch/pytorch/issues/115388): separately, `ncclCommDestroy` hangs when CUDA graphs hold comm references. Known issue with documented workaround (reset graphs before destroy).
- Impact: `--enforce-eager` costs 15-130% decode throughput depending on batch size (2.3x at BS=1 per [Fireworks AI benchmark](https://fireworks.ai/blog/speed-python-pick-two-how-cuda-graphs-enable-fast-python-code-for-deep-learning)).

### Option C: GPU-CR combo (hugepage data plane + cuda-checkpoint)

GPU-CR's `vGPU.so` hooks cudaMalloc → cuMem VMM for all allocations, dumps to hugepages at ~12 GB/s before cuda-checkpoint freeze. With our NCCL shim patches (RT signals, ncclCommSuspend re-enabled, sequential freeze), provides 2.8-4.7x faster C/R for inference.

**CUDA graphs: NOT SUPPORTED.** GPU-CR's cudaMalloc→VMM hook is incompatible with CUDA graph capture on multi-device processes. The vLLM worker dies during graph capture when GPU-CR's vGPU.so is loaded. Confirmed by control: same vLLM config without vGPU.so starts successfully with graphs. GPU-CR can only be used with `--enforce-eager`.

**NVLS: not tested** (GPU-CR requires `NCCL_CUMEM_ENABLE=1` which adds a separate conflict layer; and the cuda-checkpoint multicast restore bug applies regardless).

| GPU-CR test | Graphs | NVLS | Result |
|---|---|---|---|
| Baseline (enforce-eager, TCP) | OFF | OFF | **PASS** — ckpt 5.2s, restore 1.6s |
| Default graphs | ON | OFF | **FAIL** — worker killed during graph capture |

### Summary

| | Option A (vLLM sleep) | Option B (shim + cuda-ckpt) | Option C (GPU-CR + cuda-ckpt) |
|---|---|---|---|
| Graphs | **ON** | OFF (driver limit) | OFF (VMM hook breaks capture) |
| NVLS | **ON** | OFF (driver limit) | OFF |
| NVLink P2P | **ON** | **ON** (v2) / OFF (v1) | OFF (TCP, v1 patches) |
| GPU freed | 96% (3.4 GB stays) | **100%** | **100%** |
| C/R speed | instant sleep/wake | shim ~400ms + freeze ~4s | data dump 1.2s + freeze 4s |
| Frameworks | vLLM only | Universal | Universal |
| Steady-state perf impact | **Zero** | `--enforce-eager` tax (15-130%) | `--enforce-eager` + TCP tax |

## Parallelism Dimension C/R Matrix

Validated on 2x H100 80GB (h100-no-sharing pool, asia cluster). All using Option B (shim v2 + cuda-checkpoint, `--enforce-eager`, `NCCL_NVLS_ENABLE=0`).

### Training

| Test | Parallelism | NCCL pattern | Destroy | Freeze | Restore | Recreate | Result |
|------|------------|-------------|---------|--------|---------|----------|--------|
| T1 | FSDP DP=2 | AllReduce | 58-156ms | ~4.7s | ~750ms | 28ms | **PASS** |
| T2 | FSDP+TP=2 | AllReduce (TP comm) | 272ms | ~4.6s | ~726ms | 29ms | **PASS** |
| T3 | FSDP+CP=2 | AllGather/ReduceScatter (ring attn) | — | — | — | — | **SKIP** (PyTorch `context_parallel()` API bug) |
| T4 | PP=2 | Send/Recv (pipeline) | 188ms | ~4.7s | ~765ms | 28ms | **PASS** |
| T5 | MoE EP=2 | AllReduce (expert dispatch) | 366-410ms | ~4.6s | ~716ms | 30ms | **PASS** |

**4/5 PASS.** VRAM fully freed (4 MiB per GPU) during freeze on all passing tests. Total C/R window ~10-11s including safety waits. T3 (CP=2) skipped due to PyTorch API crash (`RuntimeError: cannot resize variables that require grad` in `context_parallel()`) — not a C/R issue, needs PyTorch nightly.

The shim correctly handles **all tested NCCL communication patterns**: AllReduce (DP, TP), Send/Recv (PP), and AllReduce with expert routing (EP). Handle translation works across all patterns. Fresh-uniqueId rendezvous works for all comm group types.

### NCCL CommCheck race condition (known issue, workaround in place)

After cuda-checkpoint restore, `ncclCommInitRank` on the recreated comms **consistently fails with rc=6 ("corrupted comm object")** unless `NCCL_DEBUG=INFO` is set. Root cause: cuda-checkpoint restores driver-level CUDA state that NCCL's `CommCheck` validation (in `misc/argcheck.cc`) interprets as stale/corrupted. The `NCCL_DEBUG=INFO` logging introduces memory barriers that change timing enough to avoid the race.

This is a fragile workaround — the proper fix is either:
- A `cudaDeviceSynchronize` + explicit barrier in the shim between restore and recreate
- An NCCL-side fix to make `CommCheck` tolerate restored process state
- Both (belt and suspenders)

### Inference (pending — to be tested next)

| Test | Framework | Parallelism | Model | C/R Mechanism | Status |
|------|-----------|-------------|-------|---------------|--------|
| I1 | vLLM | TP=2 | opt-1.3b | Option A (sleep) | **PASS** (validated) |
| I2 | vLLM | PP=2 | opt-1.3b | Option A (sleep) | Pending |
| I3 | vLLM | EP=2 | Qwen3-30B-A3B | Option A (sleep) | Pending |
| I4 | SGLang | TP=2 | opt-1.3b | Option B (shim v2) | **PASS** (validated) |
| I5 | SGLang | EP=2 | Qwen3-30B-A3B | Option B (shim v2) | Pending |

## Known Limitations

1. **cuda-checkpoint cannot freeze multi-GPU processes with captured CUDA graphs.** Driver limitation, not a CLI bug — verified via in-process API. Affects Options B and C. Repro: `graph_cr_api_test.py`.

2. **cuda-checkpoint: `cuMulticastAddDevice` broken post-restore.** NVLS multicast objects cannot be created in a restored process. Driver limitation, process-wide (even fresh CUDA contexts fail). Affects Options B and C. Repro: `mc_test.c`.

3. **vLLM sleep + shim destroy incompatible.** Graph lifecycle deadlock: graphs must be reset before comm destroy (PyTorch #115388), but external graph reset leaves vLLM's CUDAGraphRunner inconsistent. Requires vLLM-internal graph cleanup before comm teardown — a vLLM feature gap.

4. **vLLM sleep + cuda-checkpoint incompatible.** After cuMemUnmap, cuda-checkpoint's checkpoint phase hangs — the driver's checkpoint code path cannot handle half-unmapped VMM state.

5. **NCCL CommCheck race after cuda-checkpoint restore.** `ncclCommInitRank` fails with rc=6 unless `NCCL_DEBUG=INFO` is set. Fragile timing workaround — needs proper fix (cudaDeviceSynchronize barrier or NCCL patch).

6. **Requires NCCL ≥ 2.29.7.** For ncclCommSuspend/Resume (v1) or proper ncclCommDestroy behavior (v2).
