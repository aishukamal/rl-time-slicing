# Snapshot Agent: cuda-multi-gpu Backend

Reference implementation for the multi-GPU checkpoint/restore backend.
Not yet in the mainline [llm-d-rl-time-slicing](https://github.com/llm-d-incubation/llm-d-rl-time-slicing) repo.

## Files

| File | What it is | Where it goes in mainline |
|------|-----------|--------------------------|
| `snapshot_agent.proto` | Proto with `BACKEND_CUDA_MULTI_GPU = 2` enum | `pkg/snapshot-agent/api/v1alpha1/` |
| `snapshot_agent.pb.go` | Generated proto (Go) | `pkg/snapshot-agent/api/v1alpha1/` |
| `snapshot_agent_grpc.pb.go` | Generated gRPC (Go) | `pkg/snapshot-agent/api/v1alpha1/` |
| `checkpoint.go` | Backend constants (`BackendCudaMultiGPU`) | `pkg/snapshot-agent/backends/` |
| `cuda-checkpoint.go` | Base single-GPU backend (adds `checkpointSinglePID`/`restoreSinglePID` helpers) | `pkg/snapshot-agent/backends/` |
| `cuda-checkpoint-multi-gpu.go` | **The multi-GPU backend** — signal 35/36 + sequential cuda-checkpoint | `pkg/snapshot-agent/backends/` |
| `server.go` | Server routing (adds `BACKEND_CUDA_MULTI_GPU` case) | `pkg/snapshot-agent/server/` |
| `main.go` | Registers `NewCudaMultiGPUCheckpoint()` | `cmd/snapshot-agent/` |
| `cuda_checkpoint_multi_gpu_test.go` | Tests for the multi-GPU backend | `pkg/snapshot-agent/backends/` |
| `cuda_checkpoint_test.go` | Tests for single-GPU helpers | `pkg/snapshot-agent/backends/` |
| `export_test.go` | Test helpers | `pkg/snapshot-agent/backends/` |

## How it works

The `CudaMultiGPUCheckpoint` backend wraps the base `CudaCheckpoint`:

```
Snapshot:
  1. Send SIGRTMIN+1 (35) to all PIDs → shim v2 destroys NCCL comms
  2. Wait 3s for destroy to complete
  3. cuda-checkpoint --action lock + checkpoint per PID (sequential)

Restore:
  1. cuda-checkpoint --action restore + unlock per PID (sequential)
  2. 1s delay between PIDs
  3. Send SIGRTMIN+2 (36) to all PIDs → shim v2 arms lazy NCCL recreate
```

The workload must have shim v2 (`libcr-shim-v2.so`) loaded via `LD_PRELOAD`.
