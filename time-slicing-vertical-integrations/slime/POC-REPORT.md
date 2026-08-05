# Slime Time-Slicing Integration — PoC Report

## Summary

Two independent Slime GRPO jobs (Qwen2.5-0.5B-Instruct, DAPO-Math-17k) successfully time-sliced H100 GPUs using the PhaseCallback-based `timeslice-slime` package. Zero Slime source code changes — integration is purely via `--phase-callback-path timeslice_slime.callback.TimesliceCallback` + environment variables.

## Architecture

- **PhaseCallback protocol**: Added to Slime fork ([aishukamal/slime@feat/phase-callbacks](https://github.com/aishukamal/slime/tree/feat/phase-callbacks)) — 3 upstream-ready commits (~105 lines total)
- **timeslice-slime package**: Implements `TimesliceCallback` with `RoleLocks` (trainer-first ordering, atexit safety, env-gated no-op)
- **Platform**: Accelerator Orchestrator (gRPC lock service) + Snapshot Agent (cuda-checkpoint DaemonSet) + NVIDIA DRA driver

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
