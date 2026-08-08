# C/R for GPU Time-Slicing: Slime Benchmark Guide

Sampler (SGLang) and trainer run on separate GPU pools (disaggregated). Time-slicing reclaims idle GPUs: when the trainer waits for rollouts, its GPUs are checkpointed and handed to the sampler pool (or vice versa).

## Sampler C/R (SGLang)

Use the **app-endpoint** snapshot agent backend. No shim, no cuda-checkpoint, no LD_PRELOAD.

**Launch flags:**
```
--enable-memory-saver
```
Without this flag, the release endpoint is a no-op.

**Snapshot agent config:**
```
backend_config:
  app_endpoint:
    app: APP_SGLANG
    endpoints: ["http://<worker>:30000"]
```

**What happens:** `POST /release_memory_occupation {}` → cuMemUnmap releases weights + KV cache → `POST /resume_memory_occupation {}` reloads.

**Expected VRAM after snapshot:** ~2-3 GB residual per GPU (NCCL + runtime). On 80 GB H100, ~77-78 GB available for the incoming workload.

**Steady-state perf tax:** None. CUDA graphs and NVLS survive the release/resume cycle (comms never torn down).

**Parallelism validated:**

| Parallelism | Tested | Result |
|---|---|---|
| TP=2 | 2x H100 (opt-1.3b) | **PASS** (84% freed, 2 cycles verified) |
| PP=2 | 2x H100 (Qwen2.5-0.5B) | **PASS** (81% freed, 15.3→2.9 GB/GPU) |
| EP=2 (MoE) | 2x H100 (Mixtral-8x7B) | **PASS** (96.6% freed, 70.5→2.4 GB/GPU) |
| Multi-node | Not tested | |

**Drawback:** VRAM not fully freed (~2-3 GB stays). NCCL comms and cross-GPU transport state persist during the C/R window. Incoming workload must fit in ~77 GB.

---

## Trainer C/R (FSDP / torchrun)

Use **shim v2 + cuda-checkpoint**. The shim (`libcr-shim-v2.so`) destroys NCCL comms before freeze and recreates them after restore, making it transparent to the training framework.

**Shim source:** `multi-gpu-cr-poc/universal_cr_shim_v2.c` in this repo ([rl-time-slicing](https://github.com/aishukamal/rl-time-slicing/blob/main/multi-gpu-cr-poc/universal_cr_shim_v2.c)).

**Build:** `gcc -shared -fPIC -o libcr-shim-v2.so universal_cr_shim_v2.c -ldl -lpthread`

**Launch flags:**
```bash
LD_PRELOAD="/path/to/libcr-shim-v2.so:/path/to/libnccl.so.2"
CR_NCCL_LIB=/path/to/libnccl.so.2
NCCL_NVLS_ENABLE=0       # driver bug: multicast broken post-restore
```
**Requires NCCL ≥ 2.30.** Earlier versions (e.g., 2.26) cause symbol errors (`ncclDevCommDestroy` missing). Install: `pip install nvidia-nccl-cu12==2.30.7`.

**Manual C/R sequence (for debugging / standalone use):**
```
1. Quiesce trainer (pause between training steps)
2. kill -35 <all_gpu_pids>                              # shim destroys NCCL comms (~200-400ms)
3. cuda-checkpoint --toggle --pid <PID> (sequential)    # freeze GPU state
4. [GPU fully free — 0 VRAM]
5. cuda-checkpoint --toggle --pid <PID> (sequential)    # restore
6. kill -36 <all_gpu_pids>                              # arms lazy NCCL recreate
7. Resume trainer — first collective triggers comm rebuild (~30ms)
```

### Snapshot agent integration (automated C/R)

The `cuda-multi-gpu` backend automates the manual sequence above. It is **not yet in the mainline** [llm-d-rl-time-slicing](https://github.com/llm-d-incubation/llm-d-rl-time-slicing) repo. All source code is included in this repo at [`multi-gpu-cr-poc/snapshot-agent-backend/`](snapshot-agent-backend/) — ready to PR.

**What the backend does:**
```
Snapshot:
  1. Send SIGRTMIN+1 (35) to all GPU PIDs → shim destroys NCCL comms
  2. Wait 3s for destroy to complete
  3. cuda-checkpoint --action lock + checkpoint per PID (sequential)

Restore:
  1. cuda-checkpoint --action restore + unlock per PID (sequential, 1s delay between)
  2. Send SIGRTMIN+2 (36) to all PIDs → shim arms lazy NCCL recreate
```

**Files to integrate (all in [`snapshot-agent-backend/`](snapshot-agent-backend/)):**

| File | Destination in mainline |
|------|------------------------|
| `snapshot_agent.proto` | `pkg/snapshot-agent/api/v1alpha1/` |
| `snapshot_agent.pb.go` | `pkg/snapshot-agent/api/v1alpha1/` |
| `snapshot_agent_grpc.pb.go` | `pkg/snapshot-agent/api/v1alpha1/` |
| `checkpoint.go` | `pkg/snapshot-agent/backends/` (adds `BackendCudaMultiGPU` constant) |
| `cuda-checkpoint.go` | `pkg/snapshot-agent/backends/` (adds `checkpointSinglePID`/`restoreSinglePID` helpers) |
| `cuda-checkpoint-multi-gpu.go` | `pkg/snapshot-agent/backends/` (**the new backend**, ~130 lines) |
| `server.go` | `pkg/snapshot-agent/server/` (adds `BACKEND_CUDA_MULTI_GPU` routing) |
| `main.go` | `cmd/snapshot-agent/` (registers `NewCudaMultiGPUCheckpoint()`) |
| `*_test.go` | `pkg/snapshot-agent/backends/` (tests) |

See [`snapshot-agent-backend/README.md`](snapshot-agent-backend/README.md) for details.

**Steps to automate trainer C/R for the benchmark:**
1. PR the files above into mainline [llm-d-rl-time-slicing](https://github.com/llm-d-incubation/llm-d-rl-time-slicing)
2. Build `libcr-shim-v2.so` from [`universal_cr_shim_v2.c`](universal_cr_shim_v2.c)
3. Launch trainer pods with:
   ```
   LD_PRELOAD=libcr-shim-v2.so:libnccl.so.2
   CR_NCCL_LIB=/path/to/libnccl.so.2
   NCCL_NVLS_ENABLE=0
   ```
4. Deploy snapshot agent DaemonSet on trainer nodes (`helm install` from `deploy/snapshot-agent/`)
5. Call snapshot agent gRPC `Snapshot()`/`Restore()` with the `cuda-multi-gpu` backend config

**Expected VRAM after snapshot:** 0 (100% freed).

**Total C/R time:** ~10-11s (including safety waits).

**Parallelism validated:**

| Parallelism | NCCL pattern | Tested | Result |
|---|---|---|---|
| FSDP DP=2 | AllReduce | 2x H100 | **PASS** |
| FSDP+TP=2 | AllReduce (TP comm) | 2x H100 | **PASS** |
| PP=2 | Send/Recv (pipeline) | 2x H100 | **PASS** |
| MoE EP=2 | AllReduce (expert dispatch) | 2x H100 | **PASS** |
| FSDP+CP=2 | AllGather/ReduceScatter | Not tested (PyTorch API bug) | |
| Multi-node | Not tested | |

**Drawbacks:**
- **`NCCL_NVLS_ENABLE=0` required.** NVLS multicast objects can't be created post-restore (driver bug). NVLink P2P still works — only in-switch NVLS reduction is lost. Impact on training: negligible at 2 GPUs, ~5-15% on large gradient all-reduces at 8+ GPUs.
- **C/R window ~10s.** Dominated by cuda-checkpoint freeze (~4.5s).

**Note:** `NCCL_DEBUG=INFO` is NOT required. Earlier test reports claiming it was needed were caused by using NCCL 2.26 (missing `ncclDevCommDestroy` symbol). With NCCL ≥ 2.30, recreate works cleanly without debug logging.

---

## Quick reference

| Workload | Mechanism | Agent backend | GPU freed | Steady-state tax |
|---|---|---|---|---|
| SGLang sampler | release/resume | `app-endpoint` APP_SGLANG | 84-97% | **None** |
| Trainer | shim v2 + cuda-ckpt | `cuda-multi-gpu` (needs to be added to mainline) | 100% | `NCCL_NVLS_ENABLE=0` |

## Disaggregated time-slicing flow

```
Sampler pool (SGLang):
  Launch:     sglang serve ... --tp N --enable-memory-saver
  Checkpoint: POST /release_memory_occupation → ~84% GPU freed
  Restore:    POST /resume_memory_occupation → model reloaded

Trainer pool (torchrun):
  Launch:     LD_PRELOAD=libcr-shim-v2.so NCCL_NVLS_ENABLE=0
              torchrun --nproc_per_node=N train.py
  Checkpoint: signal 35 → cuda-checkpoint freeze → 100% GPU freed
  Restore:    cuda-checkpoint restore → signal 36 → training resumes

Flow:
  1. Trainer waiting for rollouts → checkpoint trainer GPUs → expand sampler pool
  2. Sampler generates rollouts on expanded pool
  3. Rollouts ready → checkpoint sampler on reclaimed GPUs → restore trainer
```
