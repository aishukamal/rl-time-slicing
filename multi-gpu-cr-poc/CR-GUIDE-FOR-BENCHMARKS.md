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

**C/R sequence:**
```
1. Quiesce trainer (pause between training steps)
2. kill -35 <all_gpu_pids>                              # shim destroys NCCL comms (~200-400ms)
3. cuda-checkpoint --toggle --pid <PID> (sequential)    # freeze GPU state
4. [GPU fully free — 0 VRAM]
5. cuda-checkpoint --toggle --pid <PID> (sequential)    # restore
6. kill -36 <all_gpu_pids>                              # arms lazy NCCL recreate
7. Resume trainer — first collective triggers comm rebuild (~30ms)
```

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

**Snapshot agent integration:** The shim v2 is **not integrated into the mainline snapshot agent** yet. The `cuda-multi-gpu` backend in the snapshot agent repo uses shim v1 (ncclCommSuspend/Resume, TCP transport). To use shim v2 (destroy/recreate, NVLink), drive C/R manually via signals + cuda-checkpoint as shown above. Integration of shim v2 into the snapshot agent is pending.

---

## Quick reference

| Workload | Mechanism | Agent backend | GPU freed | Steady-state tax |
|---|---|---|---|---|
| SGLang sampler | release/resume | `app-endpoint` APP_SGLANG | 84% | **None** |
| Trainer | shim v2 + cuda-ckpt | manual (shim v2 not in mainline yet) | 100% | `NCCL_NVLS_ENABLE=0` |

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
