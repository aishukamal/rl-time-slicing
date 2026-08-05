# Slime Time-Slicing Integration — PoC Report

## Summary

Two independent Slime GRPO jobs (Qwen2.5-0.5B-Instruct, DAPO-Math-17k) successfully time-sliced H100 GPUs using the PhaseCallback-based `timeslice-slime` package. Zero Slime source code changes — integration is purely via `--phase-callback-path timeslice_slime.callback.TimesliceCallback` + environment variables.

## Architecture

- **PhaseCallback protocol**: Added to Slime fork ([aishukamal/slime@feat/phase-callbacks](https://github.com/aishukamal/slime/tree/feat/phase-callbacks)) — 4 commits
- **timeslice-slime package**: Implements `TimesliceCallback` with `RoleLocks` (trainer-first ordering, atexit safety, env-gated no-op)
- **Platform**: Accelerator Orchestrator (gRPC lock service) + Snapshot Agent (cuda-checkpoint DaemonSet) + NVIDIA DRA driver

## Upstream Commits (Slime Fork)

4 commits on [aishukamal/slime@feat/phase-callbacks](https://github.com/aishukamal/slime/tree/feat/phase-callbacks), each self-contained and PR-able to [THUDM/slime](https://github.com/THUDM/slime):

### 1. `d1020001` — Add driver-level phase callback protocol and CLI arg

Adds a `PhaseCallback` base class (`slime/utils/phase_callback.py`) and a `--phase-callback-path` CLI argument. The callback is a simple interface with two methods: `on_phase_begin(phase, role, context)` and `on_phase_end(phase, role, context)`. Users pass a dotted Python path to a `PhaseCallback` subclass, and the training driver instantiates it once. Phases are: `init`, `generate`, `train`, `weight_sync`, `save`, `eval`. Roles indicate which GPU pool is active: `trainer`, `sampler`, or `both`. Follows Slime's existing `--custom-*-path` CLI convention.

### 2. `8f5a9bc5` — Emit phase callbacks in sync training driver

Wraps each GPU phase boundary in `train.py` with `on_phase_begin`/`on_phase_end` calls. For example, `rollout_manager.generate.remote()` is bracketed with `on_phase_begin("generate", "sampler")` / `on_phase_end("generate", "sampler")`, and `actor_model.update_weights()` with `on_phase_begin("weight_sync", "both")` / `on_phase_end("weight_sync", "both")`. All emissions are guarded by `if phase_cb:` — zero overhead when the flag is not set.

### 3. `8f324ae8` — Emit phase callbacks in async training driver

Same pattern for `train_async.py`. In the async driver, generation N+1 overlaps with training N (pipelined). Callbacks fire at the `ray.get` sync point when the driver blocks for generation completion, not at the non-blocking `.remote()` dispatch. Before `update_weights()`, pending generation is drained first (existing Slime behavior), then the `weight_sync` callback fires.

### 4. `f6ab5727` — Fix use-after-free in sleep()

Adds `torch.cuda.empty_cache()` between `destroy_process_groups()` and `torch_memory_saver.pause()` in `MegatronTrainRayActor.sleep()`. Without this, `torch_memory_saver` captures stale block metadata from NCCL communicators that were already freed, causing `cudaErrorIllegalAddress` on the next `update_weights()`. This is a standalone bug fix independent of time-slicing — it affects any Slime workload using `--offload-train` with NCCL weight sync.

## Results

| Metric | Job A | Job B |
|--------|-------|-------|
| Rollout cycles | 2 | 2 |
| Training steps | 2 | 2 |
| update_weights | 3.04s / 0.44s | 0.47s / 0.4s |
| Train TFLOPS | 20.0 | 20.5 |
| Tokens/sec | 6,490 | 6,641 |
| Lock handoff wait | 120s | 198s |
| Total wall time | ~10 min (both jobs, interleaved) |

## Lock Protocol (observed)

```
Job A: ACQUIRE trainer (120s wait) → ACQUIRE sampler → init + generate + train + update_weights → RELEASE → ...
Job B: ACQUIRE trainer (198s wait, context_restored=True) → ACQUIRE sampler → ... → RELEASE
```

Both jobs alternated on the shared trainer GPU via orchestrator lock handoff with cuda-checkpoint snapshot/restore.

## Key Findings

1. **`--offload-train` / `--offload-rollout` must NOT be used** with cuda-checkpoint. Slime's cooperative VRAM offloading (`torch_memory_saver`) conflicts with cuda-checkpoint's GPU state management — causes `cudaErrorIllegalAddress` in `update_weights`. The platform handles GPU state transparently.

2. **KubeRay CRD patch required** for DRA. KubeRay 1.6.2's CRD schema uses `source.resourceClaimName` but the Go struct has flat `resourceClaimName`. Patch: promote `source.resourceClaimName` to top-level in the CRD.

3. **`NCCL_CUMEM_ENABLE=0` + `NCCL_NVLS_ENABLE=0`** recommended (standard for cuda-checkpoint compatibility).

## Cluster

- GKE `verl-research-cluster` (asia-southeast1-b)
- 2× H100 80GB nodes from `h100-2gpu-pool` (on-demand)
- Node 1 (`*-7yup`): trainer pool, Node 2 (`*-hxbl`): sampler pool
- DRA ResourceClaims: `shared-trainers-gpu-claim`, `shared-samplers-gpu-claim`

## User Experience

```bash
pip install timeslice-slime

# Add to Slime launch command:
--phase-callback-path timeslice_slime.callback.TimesliceCallback

# Set environment variables:
TIMESLICE_JOB_ID=my-job
TIMESLICE_ORCH_ADDR=timeslice-acceleratororchestrator.timeslice-system:50051
TIMESLICE_TRAINER_GROUP=trainers
TIMESLICE_SAMPLER_GROUP=samplers
```

Zero changes to Slime training scripts or model code.

## Artifacts

- `package/` — timeslice-slime pip package (callback.py, locks.py, tests)
- `manifests/` — RayJob template, launch script, Dockerfile, Cloud Build config
