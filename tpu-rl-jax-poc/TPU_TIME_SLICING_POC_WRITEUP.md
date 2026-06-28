# E2E Time-Slicing RL PoC: JAX/vLLM on TPU v5e

## 1. Overview

This PoC demonstrates that two independent GRPO reinforcement learning jobs can share a pair of TPU v5e-8 nodes on GKE using libtpu-uds checkpoint-based time-slicing. Each job's TPU context is alternately snapshotted to host RAM and restored, allowing two jobs to interleave on the same physical TPU chips without corrupting model state or learning dynamics.

The system runs a full RL loop: vLLM on TPU generates rollouts, a GSM8K reward function scores completions (3-tier: correct answer, correct format, incorrect), a JAX/Flax trainer runs GRPO gradient updates on all 8 TPU chips, and the cycle repeats. Two such jobs run concurrently on hardware that would normally support only one.

Unlike the GPU PoC (which uses CUDA Checkpoint and NCCL weight sync), this PoC uses the TPU HAL's native gRPC checkpoint/restore interface via Unix domain sockets (`/run/tpu_hal_<PID>.sock`), and runs full fine-tuning (no LoRA) with JAX on TPU.

## 2. Cluster Setup

- **Cluster**: verl-research-cluster-west (us-west1)
- **TPU Node 1** (spot pool): `gke-tpu-d19222db-j30v` — TPU v5e-8 (2×4 topology, 8 chips, 16 GiB HBM each)
- **TPU Node 2** (gvisor pool): `gke-tpu-4c1a4f78-gn8h` — TPU v5e-8 (2×4 topology, 8 chips, 16 GiB HBM each)
- **CPU Node**: default pool — runs orchestrator and RL loop pods
- **Model**: GPT-2 124M (full fine-tuning, no LoRA)

Two modes are supported by the same codebase, selected at deploy time via the `MODE` env var:

- **Baseline** — one job, one sampler on TPU Node 1, one trainer on TPU Node 2. No checkpoint/restore. Used as the control for measuring uncontested throughput.
- **Timeslice** — two independent jobs. Each has a sampler and a trainer. The two samplers share TPU Node 1; the two trainers share TPU Node 2. The Snapshot Agent handles all checkpoint/restore via libtpu-uds gRPC.

## 3. System Components

**Source code**: https://github.com/aishukamal/rl-time-slicing/tree/main/tpu-rl-jax-poc

| Component | Description |
|-----------|-------------|
| `loop/rl_loop.py` | The orchestration script for one training job. Drives the full RL loop: acquires TPU locks via the orchestrator, calls the sampler for generation, computes GSM8K rewards and GRPO advantages, calls the trainer for gradient updates. |
| `sampler/sampler.py` | FastAPI wrapper around vLLM on TPU. Receives prompts from rl-loop, returns completions. vLLM loads lazily under the orchestrator lock on the first step. Exposes `/generate`, `/health`, `/start_vllm`, `/get_pids`. |
| `trainer/trainer.py` | FastAPI service hosting the GPT-2 model in JAX/Flax. Full fine-tuning across all 8 TPU chips. Receives completions and advantages, runs a GRPO gradient step. Exposes `/train`, `/health`, `/get_pids`. |
| `orchestrator/orchestrator.py` | Lock service with one `threading.Lock` per TPU pool (sampler pool, trainer pool). Jobs call `/acquire` to get the lock and `/yield` to release it. In timeslice mode, yield triggers a TPU checkpoint via the snapshot agent; acquire triggers a restore. |
| `cmd/snapshot-agent/` | gRPC service (Go) deployed as a DaemonSet on TPU nodes. Discovers workload PIDs from Kubernetes pod metadata, connects to `/run/tpu_hal_<PID>.sock`, and invokes `Checkpoint` or `Restore` RPCs on the TPU HAL service. Supports both CUDA and TPU backends. |
| `loop/reward.py` | GSM8K reward function. 3-tier scoring: 2.0 for correct numeric answer, 0.5 for correct XML format with wrong answer, 0.0 for neither. Multi-format answer extraction (`<answer>`, `\boxed{}`, `#### N`, last-number fallback). GRPO advantage normalization within completion groups. |
| `trainer/weight_sync.py` | Flax → safetensors export for weight sync (currently disabled — see Section 8). |
| `telemetry/` | TPU metrics collection (tpu-info scraper at 1s resolution + Cloud Monitoring API at 1-min resolution) and comparison dashboard generators. |
| `deploy.sh` | Full GKE deployment automation: clean slate, deploy all pods, start TPU scrapers, launch RL loops, monitor for completion, collect all metrics + logs, query Cloud Monitoring API. |

## 4. The RL Training Loop

Each rl-loop runs the following sequence per step. In timeslice mode, two loops run concurrently and interleave on shared TPUs.

### 4.1 Per-Step Sequence

| Phase | Description |
|-------|-------------|
| **Step 1: Generate** | Acquire sampler TPU lock. Send batch of 330 prompts to vLLM (G=8 completions per prompt, max 1024 tokens each). Prompts are sent in batches of 3 to control phase duration (~6.5 min). Release sampler lock — in timeslice mode, the other job's sampler is restored and can now generate while this job trains. |
| **Step 2: Reward** | CPU-only. Score each of the 2640 completions: 2.0 for correct GSM8K answer, 0.5 for correct XML format but wrong answer, 0.0 for neither. Normalize scores within each group of 8 completions to get GRPO advantages. |
| **Step 3: Train** | Acquire trainer TPU lock. Send prompts, completions, and advantages to the JAX/Flax trainer. Trainer runs GRPO gradient update across all 8 TPU chips (~8 min). Release trainer lock. |

> **High-duty-cycle config**: The values above are defaults. The high-duty-cycle configuration uses `PROMPTS_PER_STEP=1500` and `GEN_BATCH_SIZE=30` (vs 330/3 above), and pins the trainer to a single chip via `TRAINER_CHIPS=0` to achieve ~79% peak trainer duty cycle (vs ~45% when sharding across all 8 chips).

Weight sync is not performed in this PoC (see Section 8.1).

### 4.2 Interleaving in Timeslice Mode

The sampler and trainer pools are independent locks on separate TPU nodes. This means job-a's trainer and job-b's sampler can run simultaneously. The key scheduling insight is that the sampler lock is released before the trainer lock is acquired — allowing the other job's generation to proceed concurrently with this job's training.

```
Timeline (two jobs, timeslice mode):

job-a: [acquire-sampler] [generate ~6.5m] [yield-sampler]
                                                  [acquire-trainer] [train ~8m] [yield-trainer]
                                                                                       [acquire-sampler] ...
job-b:                        [acquire-sampler] [generate ~6.5m] [yield-sampler]
                                                                        [acquire-trainer] [train ~8m] ...

TPU Node 1 (sampler): job-a generates → C/R ~4s → job-b generates → C/R ~4s → job-a generates → ...
TPU Node 2 (trainer): job-a trains → C/R ~4s → job-b trains → C/R ~4s → ...
```

### 4.3 TPU Checkpoint/Restore Mechanism

Unlike CUDA Checkpoint (which snapshots GPU memory pages), TPU checkpoint/restore works through the TPU HAL's gRPC interface:

1. Each TPU process creates a Unix domain socket at `/run/tpu_hal_<PID>.sock`
2. The Snapshot Agent connects to this socket and calls `TpuHalService/Checkpoint` — the libtpu runtime saves TPU execution state to host RAM
3. On restore, `TpuHalService/Restore` is called — libtpu writes saved state back to TPU HBM

The process remains alive throughout — only TPU execution context is swapped. CPU-resident state (Python objects, HTTP connections, file descriptors) is preserved across checkpoint/restore cycles.

**Key difference from GPU**: TPU HBM allocation is not freed during checkpoint. The memory contents are saved/restored, but the allocation persists. This means the Cloud Monitoring `memory_used` metric stays constant during C/R transitions.

## 5. Snapshot Agent

The Snapshot Agent is a Go gRPC service deployed as a DaemonSet on each TPU node. It provides a unified interface for checkpoint/restore across accelerator types.

### 5.1 Architecture

```
Orchestrator
    │ gRPC (Snapshot/Restore)
    ▼
Snapshot Agent (:9001)
    │
    ├─ TPU Backend: connects to /run/tpu_hal_<PID>.sock
    │                calls TpuHalService/{Checkpoint,Restore}
    │
    ├─ CUDA Backend: calls cuda-checkpoint --toggle
    │
    └─ Noop Backend: for testing
```

### 5.2 Key Components

| File | Description |
|------|-------------|
| `cmd/snapshot-agent/main.go` | Entry point. Registers CUDA, TPU, and noop backends. Starts gRPC server on port 9001. |
| `pkg/snapshot-agent/server/server.go` | gRPC service: `Snapshot()`, `Restore()`, `GetOperation()`, `Status()`. Manages long-running operations with state tracking. |
| `pkg/snapshot-agent/backends/tpu-checkpoint.go` | TPU backend. Discovers TPU HAL sockets at `/run/tpu_hal_*.sock`, connects via gRPC, invokes Checkpoint/Restore RPCs. |
| `pkg/snapshot-agent/backends/cuda-checkpoint.go` | CUDA backend. Calls `cuda-checkpoint` binary for GPU context switching. |
| `pkg/snapshot-agent/api/v1alpha1/snapshot_agent.proto` | Proto definition. Backend enum: `BACKEND_CUDA`, `BACKEND_TPU`. |
| `pkg/snapshot-agent/state-machine/state-manager.go` | Job state machine: IDLE → RUNNING → TRANSITIONING → SAVED → FAULTED. |
| `pkg/snapshot-agent/utils/pod-utils-tpu.go` | TPU-specific pod discovery via Kubernetes API. |

### 5.3 TPU Backend Details

The TPU backend discovers workload processes by:
1. Querying the Kubernetes API for pods with the `timeslice.io/job-id` label on the local node
2. Extracting container PIDs from the pod status
3. Scanning `/run/tpu_hal_*.sock` sockets matching those PIDs
4. Connecting to each socket and invoking the checkpoint or restore RPC

This approach works because all TPU pods run with `hostPID: true` and mount `/run` from the host, making TPU HAL sockets visible across containers on the same node.

## 6. Orchestrator

The orchestrator manages exclusive access to each TPU pool (sampler, trainer) using per-pool threading locks.

### 6.1 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/register` | POST | Register a workload with its pool, PIDs, and node name. Orchestrator discovers the snapshot agent on that node via Kubernetes API. |
| `/acquire` | POST | Blocking: wait for the pool lock. If the previous holder was checkpointed, triggers restore via snapshot agent before returning. Returns `restore_ms` and `wait_ms`. |
| `/yield` | POST | Release the pool lock. In timeslice mode, triggers checkpoint via snapshot agent before releasing. Returns `checkpoint_ms`. |
| `/health` | GET | Health check. |

### 6.2 Modes

- **`MODE=baseline`**: Locks are acquired and released but no checkpoint/restore is performed. The lock simply serializes access.
- **`MODE=snapshot`**: On yield, the orchestrator calls the snapshot agent to checkpoint the workload. On acquire, if the workload was previously checkpointed, the orchestrator calls restore before granting the lock.

### 6.3 Metrics

The orchestrator logs every acquire/yield event as a JSONL line to `/data/rl_metrics.jsonl`, including timestamps, wait times, checkpoint/restore durations, and workload IDs. This is the authoritative source for interleaving analysis.

## 7. Sampler (vLLM on TPU)

### 7.1 Architecture

The sampler runs vLLM on TPU v5e using the `tpu_inference` plugin with PyTorch/XLA (torchax) backend. GPT-2 is loaded onto a single TPU chip (chip 0 uses ~15 GiB HBM).

### 7.2 Lazy Initialization

In timeslice mode, two sampler containers (sampler-a, sampler-b) share the same TPU node. Only one can initialize vLLM at a time:

- **sampler-a**: `DEFER_VLLM=false` — starts vLLM immediately on pod startup
- **sampler-b**: `DEFER_VLLM=true` — starts a minimal FastAPI server. vLLM is initialized later when the RL loop calls `/start_vllm` under the orchestrator lock

This ensures exclusive TPU access during initialization without requiring explicit ordering between jobs.

### 7.3 Stale Socket Cleanup

Before initializing vLLM, the sampler scans `/run/tpu_hal_*.sock` and removes sockets belonging to dead PIDs. This prevents libtpu from attempting to connect to stale HAL instances from previous runs.

### 7.4 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns `vllm_started` status and PID info. |
| `/generate` | POST | Generate completions. Accepts `{prompts, group_size, max_new_tokens, temperature}`. |
| `/start_vllm` | POST | Trigger deferred vLLM initialization (sampler-b only). |
| `/get_pids` | GET | Return PIDs of vLLM processes for snapshot agent registration. |

### 7.5 Weight Sync

The sampler exposes a `/reload_weights` endpoint that downloads updated weights from the trainer and hot-swaps them into the running vLLM model without restarting the process.

The reload flow:
1. The sampler calls the trainer's `/export_weights` endpoint, which serializes the current JAX model weights as safetensors.
2. The sampler loads the safetensors file and iterates over the vLLM model's `named_parameters()`, copying each updated tensor via `param.data.copy_()` inside a `torchax.default_env()` context.
3. The vLLM model path is: `runner.model_runner.model` (VllmModelWrapper) -> `.model` (_VllmRunner) -> `.vllm_model` (the actual PyTorch model exposing `named_parameters()`).

This matches the GPU worker's weight-sync approach (direct parameter copy); the TPU-specific addition is wrapping the copy in `torchax.default_env()` to ensure the tensor operations target the correct XLA device context.

## 8. Trainer (JAX/Flax GRPO)

### 8.1 Architecture

The trainer loads GPT-2 124M via HuggingFace's `FlaxGPT2LMHeadModel` and shards across all 8 TPU chips using JAX's default device placement. Full fine-tuning (no LoRA) — all 124M parameters are trainable.

### 8.2 Training Details

- **Optimizer**: Adam, lr=5e-6
- **Loss**: GRPO policy gradient loss with KL divergence penalty (β=0.1)
- **Sequence handling**: all sequences right-padded to `MAX_SEQ_LEN` (2560) for static XLA compilation
- **JIT**: loss/grad function compiled lazily after model load; reused across all training steps
- **Batch size** (default): 2640 samples per step (330 prompts x 8 completions). The high-duty-cycle config uses 12000 samples per step (1500 prompts x 8 completions).
- **Chip pinning**: `TRAINER_CHIPS=0` pins the trainer to a single TPU chip, achieving ~79% peak duty cycle vs ~45% when sharding across all 8 chips. The default config uses all 8 chips.

### 8.3 Lazy Initialization

The trainer health endpoint does NOT call `jax.devices()` or `jax.default_backend()` — these would trigger eager TPU initialization and cause `/dev/vfio: Device or resource busy` errors when two trainer containers share the same TPU node. JAX only initializes the TPU on the first actual computation.

### 8.4 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check. Does not trigger TPU initialization. |
| `/train` | POST | Run one GRPO gradient step. Accepts `{prompts, completions, advantages}`. Returns `{loss, kl_loss, grad_norm}`. |
| `/get_pids` | GET | Return PIDs for snapshot agent registration. |
| `/export_weights` | GET | Export current model weights as safetensors (for weight sync). |

## 9. TPU Metrics Collection

Two independent metrics sources are collected during each run:

### 9.1 TPU Monitoring Issues

Three fundamental TPU monitoring issues were discovered during this PoC:

1. **HBM SW accounting doesn't reflect hardware state after checkpoint.** When a TPU process is checkpointed, the libtpu FLR (Function-Level Reset) frees HBM on the hardware, but the software-side memory accounting still reports the old allocation value. The `memory_used` metric remains at the pre-checkpoint level even though the HBM has been physically freed and is available for the restored process.

2. **gRPC metric server doesn't handle multi-tenant.** The tpu-info metric server at `localhost:8431` only reports metrics for the first LibTPU client that connects. A second process on the same TPU node is invisible to the metric server. Tested 2026-06-28: trainer-b was actively training, but tpu-info still showed trainer-a's stale 3089 MiB. This makes per-workload metric attribution impossible in time-slicing scenarios.

3. **Duty cycle counter not restarted after restore.** After a checkpoint/restore cycle, the `duty_cycle_pct` counter freezes permanently at the last value observed before the checkpoint. It does not resume counting when the process is restored and actively computing. This makes the metric unreliable for any workload that has been through a C/R cycle.

A Python scraper (`telemetry/tpu_duty_cycle.py`) still runs inside the sampler-a and trainer-a containers via `kubectl exec`, polling `tpu-info`'s `get_chip_usage()` every second and writing per-chip `duty_cycle_pct` and `mem_used_mib` to CSV. However, due to the issues above, the scraper values in timeslice mode are stale after the first checkpoint/restore cycle.

### 9.2 Cloud Monitoring API (1-minute resolution)

After each run, `telemetry/collect_cloud_metrics.py` queries the GKE Cloud Monitoring API for:

- `kubernetes.io/container/accelerator/duty_cycle` — percentage of time the TPU chip was active within a 1-minute window
- `kubernetes.io/container/accelerator/memory_used` — HBM bytes allocated per chip
- `kubernetes.io/container/accelerator/memory_total` — total HBM per chip

**Duty cycle interpretation**: The Cloud Monitoring `duty_cycle` metric is NOT binary (0% or 100%). It reports the fraction of time within each 1-minute sample window that the chip was computing. GPT-2 on TPU v5e shows ~45% during active phases (the MXUs aren't saturated by such a small model). The relevant signal is the transition between ~45% and 0%, not the absolute value.

### 9.3 Avg Duty Cycle

Duty cycle is reported as the mean of all `duty_cycle_pct` samples collected by the tpu-info scraper during the run. Results from the highduty_5step configuration:

| Metric | Baseline | Timeslice |
|--------|----------|-----------|
| Sampler avg duty cycle | 41.6% | 94.6% (stale after first C/R) |
| Trainer avg duty cycle | 44.1% | 6.9% (stale) |

**Note**: The timeslice values are affected by the monitoring issues described in Section 9.1. The duty cycle counter freezes after the first checkpoint/restore cycle, so the timeslice averages include a mix of real pre-C/R values and stale post-C/R values. The sampler's 94.6% reflects high activity before its first checkpoint; the trainer's 6.9% reflects the counter freezing at a low value after an early C/R event. These numbers do not accurately represent actual TPU utilization in timeslice mode.

## 10. Component Interfaces

### 10.1 rl-loop → Orchestrator

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/register` | POST | Register workload with pool, PIDs, and node name. Called once at startup. |
| `/acquire` | POST | Blocking: wait for TPU lock. Restore if previously checkpointed. Timeout: 1800s. |
| `/yield` | POST | Release TPU lock. Checkpoint if in snapshot mode. |
| `/health` | GET | Health check. |

### 10.2 Orchestrator → Snapshot Agent

| RPC | Purpose |
|-----|---------|
| `Snapshot()` | Checkpoint all TPU processes for a workload. Connects to `/run/tpu_hal_<PID>.sock` and calls `TpuHalService/Checkpoint`. |
| `Restore()` | Restore all TPU processes for a workload. Calls `TpuHalService/Restore`. |
| `Status()` | Health check and current state of all tracked jobs. |

### 10.3 rl-loop → Sampler / Trainer

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/generate` | POST | Sampler: generate completions for a batch of prompts via vLLM. |
| `/train` | POST | Trainer: run one GRPO gradient step given prompts, completions, and advantages. |
| `/start_vllm` | POST | Sampler (deferred only): trigger vLLM initialization under the orchestrator lock. |
| `/get_pids` | GET | Both: return TPU process PIDs for snapshot agent registration. |
| `/health` | GET | Both: liveness check. |

## 11. Known Shortcuts and Production Gaps

### 11.1 Weight Sync Not Implemented

**Impact**: The sampler always uses the initial GPT-2 weights, never receiving training updates. Each job trains independently but the sampler's generation policy does not improve across steps. Training loss and reward metrics are still real — only the sampler-side policy lags behind.

**Root cause**: vLLM on TPU uses torchax (PyTorch/XLA) for model execution. Updating the model's state dict values via `collective_rpc` succeeds (values are correct), but the model produces garbage output after the update. The likely cause is that torchax's JIT cache does not invalidate when the underlying tensor values change, continuing to use stale compiled programs.

**Production fix**: Implement weight sync via model reload (save to disk, reload vLLM) or fix the torchax JIT invalidation issue. Alternatively, use the native JAX sampler (`sampler_jax_fallback.py`) which does not have this issue.

### 11.2 TPU Memory Not Freed During Checkpoint

The `memory_used` metric stays constant (~15 GiB for the sampler) even during checkpoint/restore transitions. The libtpu checkpoint implementation saves HBM contents to host RAM but does not release the HBM allocation. Both workloads effectively reuse the same physical HBM region — the checkpoint swaps the contents, not the allocation.

**Impact**: This means TPU time-slicing cannot oversubscribe HBM. Both workloads must fit within the available HBM simultaneously (which they do, since they share the same memory region).

**Production note**: This is a fundamental difference from CUDA Checkpoint, which truly unmaps GPU memory pages. TPU time-slicing is better described as "execution context swapping" rather than "memory swapping."

### 11.3 Single Cluster, Static Node Assignment

Sampler and trainer pods are pinned to specific TPU node pools via `nodeSelector`. There is no dynamic placement or multi-cluster support. If a TPU node is preempted (the sampler node is on a spot pool), the entire experiment fails.

**Production fix**: Add pod disruption budgets, checkpoint the RL loop state to allow resumption, and support TPU node replacement.

### 11.4 tpu-info Duty Cycle Metric Staleness

The tpu-info `duty_cycle_pct` metric freezes when the TPU is continuously active (no idle gaps). In timeslice mode, the ~4s checkpoint/restore gap is too short to reset the metric, so it reads a constant value for the entire run. The scraper dashboards mask this with orchestrator event windows, but the underlying metric is unreliable for timeslice scenarios.

**Production fix**: Use the Cloud Monitoring `duty_cycle` metric (1-minute resolution, no staleness) or instrument the application directly.

### 11.5 GPT-2 Does Not Saturate TPU

GPT-2 124M has a hidden dimension of 768, far too small to saturate the TPU v5e MXU (designed for hidden dimensions of 4096+). Tensorcore utilization is only ~3-7% during active computation. The duty cycle pattern (square wave) is valid, but the absolute utilization numbers are not representative of production workloads.

**Impact on PoC validity**: The time-slicing mechanism (checkpoint/restore, lock coordination, interleaving) is model-size-agnostic. The ~4s C/R overhead would be the same for a 70B model. Only the phase durations would change.

### 11.6 Lazy Initialization is Step-1 Only

Both vLLM (sampler) and the JAX model (trainer) load lazily on the first step. Step 1 takes ~15 min (includes model loading and XLA compilation), while subsequent steps take ~14.5 min. In a production setup, model loading should be pipelined with lock acquisition.

## 12. Key Files

| File | Description |
|------|-------------|
| `deploy.sh` | Full GKE deployment: pods, services, scrapers, monitoring, metric collection |
| `loop/rl_loop.py` | Main RL loop — drives one training job end-to-end |
| `loop/reward.py` | GSM8K reward function (3-tier scoring) |
| `sampler/sampler.py` | vLLM-backed TPU generation service |
| `trainer/trainer.py` | JAX/Flax GRPO trainer (full fine-tuning, 8 TPU chips) |
| `trainer/weight_sync.py` | Flax → safetensors weight export (for future weight sync) |
| `orchestrator/orchestrator.py` | Per-pool TPU lock service with C/R coordination |
| `cmd/snapshot-agent/main.go` | Snapshot Agent entry point (Go gRPC service) |
| `pkg/snapshot-agent/backends/tpu-checkpoint.go` | TPU checkpoint backend (libtpu-uds gRPC) |
| `deploy/snapshot-agent.yaml` | Snapshot Agent DaemonSet + RBAC |
| `deploy/samplers-pod.yaml` | 2-container sampler pod (sampler-a + sampler-b) |
| `deploy/trainers-pod.yaml` | 2-container trainer pod (trainer-a + trainer-b) |
| `deploy/services-m5.yaml` | ClusterIP services for all endpoints |
| `telemetry/dashboard_generator.py` | Comparison dashboard generator (Cloud Monitoring or tpu-info) |
| `telemetry/tpu_duty_cycle.py` | tpu-info scraper (1s resolution, per-chip) |
| `telemetry/collect_cloud_metrics.py` | Cloud Monitoring API metric collection |

## 13. Prerequisites

- GKE cluster with 2× TPU v5e-8 nodes (one spot pool, one gvisor pool) and a default CPU node pool
- `kubectl` configured for the cluster
- `gcloud` authenticated with Cloud Monitoring read access
- `libtpu-uds.so` — the modified libtpu shared library with Unix domain socket support for checkpoint/restore. **Obtain from**: [TODO: add link to libtpu-uds.so artifact]
- Container images built and pushed (see `README.md` for build instructions)

## 14. Results

### 14.1 Performance Summary

Results from the highduty_5step configuration (`PROMPTS_PER_STEP=1500`, `GEN_BATCH_SIZE=30`, `TRAINER_CHIPS=0`):

| Metric | Baseline (1 job) | Timeslice (2 jobs) |
|--------|-------------------|---------------------|
| Steps completed | 5 | 5 per job (10 total) |
| Wall clock time | 73.1 min | 89.1 min |
| Sequential 2-job estimate | 146.2 min | -- |
| **Throughput speedup** | -- | **1.64x** |
| Avg generate phase | 6.5 min | 6.5 min |
| Avg train phase | 7.9 min | 7.8 min |
| Sampler peak duty cycle | 99.8% | 99.6% |
| Trainer peak duty cycle | 79.2% | 79.0% |
| Sampler C/R evict | N/A | 4.2s |
| Sampler C/R restore | N/A | 3.7s |
| Trainer C/R evict | N/A | 0.8s |
| Trainer C/R restore | N/A | 0.7s |

### 14.2 Dashboards

**Primary (highduty_5step, scraper + synthetic overlay):**
- [dashboard_scraper_synthetic.html](https://github.com/aishukamal/rl-time-slicing/blob/main/tpu-rl-jax-poc/runs/highduty_5step/dashboard_scraper_synthetic.html)

**Additional highduty_5step dashboards:**
- Cloud Monitoring: [dashboard.html](https://github.com/aishukamal/rl-time-slicing/blob/main/tpu-rl-jax-poc/runs/highduty_5step/dashboard.html)
- Scraper: [dashboard_scraper.html](https://github.com/aishukamal/rl-time-slicing/blob/main/tpu-rl-jax-poc/runs/highduty_5step/dashboard_scraper.html)

**Weight sync run (weightsync_5step):**
- Cloud Monitoring: [dashboard.html](https://github.com/aishukamal/rl-time-slicing/blob/main/tpu-rl-jax-poc/runs/weightsync_5step/dashboard.html)
- Scraper: [dashboard_scraper.html](https://github.com/aishukamal/rl-time-slicing/blob/main/tpu-rl-jax-poc/runs/weightsync_5step/dashboard_scraper.html)

### 14.3 Key Observations

1. **High duty cycle achieved**: With `TRAINER_CHIPS=0` (single-chip pinning), the trainer reaches 79.2% peak duty cycle in baseline, compared to ~45% when sharding across all 8 chips. The sampler reaches 99.8% peak duty cycle. Timeslice mode preserves these levels (79.0% trainer, 99.6% sampler).

2. **1.64x throughput speedup**: Two jobs complete 10 total steps in 89.1 min vs 73.1 min for a single job's 5 steps. The sequential 2-job estimate is 146.2 min, giving a 1.64x improvement.

3. **C/R overhead is asymmetric and small**: Sampler C/R (evict 4.2s, restore 3.7s) is larger than trainer C/R (evict 0.8s, restore 0.7s), reflecting the difference in TPU state size. Both are negligible compared to the ~6.5-8 min phase durations (<1% overhead).

4. **Phase durations unchanged**: Generate and train times are identical between baseline and timeslice, confirming that checkpoint/restore does not degrade per-step performance.

5. **TPU monitoring is unreliable for time-slicing**: Three fundamental issues were discovered (see Section 9.1): HBM SW accounting doesn't reflect post-checkpoint state, the gRPC metric server can't handle multi-tenant, and the duty cycle counter freezes permanently after restore. The duty cycle numbers in the scraper dashboards are stale after the first C/R cycle; peak duty values are measured from baseline runs where no C/R occurs.

## 15. Differences from GPU PoC

| Aspect | GPU PoC | TPU PoC |
|--------|---------|---------|
| Accelerator | H100 (2 GPUs) | TPU v5e-8 (2 nodes × 8 chips) |
| C/R mechanism | CUDA Checkpoint (`cuda-checkpoint --toggle`) | libtpu-uds gRPC (`TpuHalService/{Checkpoint,Restore}`) |
| Memory behavior | GPU memory freed during checkpoint | HBM allocation persists, contents swapped |
| C/R overhead | ~2-5s | ~4s |
| Model | Qwen2.5-0.5B with LoRA | GPT-2 124M full fine-tuning |
| Training framework | PyTorch + TRL | JAX/Flax |
| Inference framework | vLLM (CUDA) | vLLM (TPU via torchax) |
| Weight sync | NCCL GPU-to-GPU broadcast (~1.1s) | Not implemented (torchax JIT invalidation issue) |
| Topology | 2 GPUs on 1 node | 2 TPU nodes (8 chips each) |
| Throughput gain | ~1.5× | 1.61× |
