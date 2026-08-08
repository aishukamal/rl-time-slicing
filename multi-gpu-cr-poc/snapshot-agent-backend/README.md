# Snapshot Agent: cuda-multi-gpu Backend

Reference implementation for the multi-GPU checkpoint/restore backend.
Not yet in the mainline [llm-d-rl-time-slicing](https://github.com/llm-d-incubation/llm-d-rl-time-slicing) repo.

All files are based on mainline (same `BackendConfig` oneof, same `Request` interface) with the multi-GPU additions applied. Ready to PR.

## Files

| File | What it is | Where it goes in mainline |
|------|-----------|--------------------------|
| `snapshot_agent.proto` | Proto — adds `CudaMultiGPUBackendConfig` message + field 5 in `BackendConfig` oneof | `pkg/snapshot-agent/api/v1alpha1/` |
| `checkpoint.go` | Backend constants — adds `BackendCudaMultiGPU` | `pkg/snapshot-agent/backends/` |
| `cuda-checkpoint.go` | Base single-GPU backend (mainline, unmodified) | `pkg/snapshot-agent/backends/` |
| `cuda-checkpoint-multi-gpu.go` | **The multi-GPU backend** — signal 35/36 + sequential cuda-checkpoint | `pkg/snapshot-agent/backends/` (new file) |
| `server.go` | Server routing — adds `GetCudaMultiGpu()` case to `getSnapshotBackendType` | `pkg/snapshot-agent/server/` |
| `main.go` | Registers `NewCudaMultiGPUCheckpoint()` in backend map | `cmd/snapshot-agent/` |
| `cuda_checkpoint_multi_gpu_test.go` | Tests for the multi-GPU backend | `pkg/snapshot-agent/backends/` (new file) |
| `cuda_checkpoint_test.go` | Tests for base backend (mainline) | `pkg/snapshot-agent/backends/` |
| `export_test.go` | Test helpers | `pkg/snapshot-agent/backends/` |

**Note:** Generated pb.go files are not included — run `protoc` on the proto to generate them.

## What changed vs mainline

Only 5 diffs on top of mainline:

1. **Proto:** added `CudaMultiGPUBackendConfig` message + field 5 in `BackendConfig` oneof
2. **checkpoint.go:** added `BackendCudaMultiGPU BackendType = "cuda-multi-gpu"` constant
3. **server.go:** added `config.GetCudaMultiGpu()` routing in `getSnapshotBackendType`
4. **main.go:** added `backends.BackendCudaMultiGPU: backends.NewCudaMultiGPUCheckpoint()` registration
5. **cuda-checkpoint-multi-gpu.go:** new file (~130 lines)

## How it works

The `CudaMultiGPUCheckpoint` backend wraps the base `CudaCheckpoint`:

```
Snapshot:
  1. Send SIGRTMIN+1 (35) to all PIDs → shim v2 destroys NCCL comms
  2. Wait 3s for destroy to complete
  3. cuda-checkpoint --action lock + checkpoint per PID (sequential)

Restore:
  1. cuda-checkpoint --action restore + unlock per PID (sequential)
  2. Send SIGRTMIN+2 (36) to all PIDs → shim v2 arms lazy NCCL recreate
```

The workload must have shim v2 (`libcr-shim-v2.so`) loaded via `LD_PRELOAD`.

## gRPC usage

```protobuf
// Snapshot request with multi-GPU backend:
SnapshotRequest {
  job_id: "trainer-0"
  backend_config {
    cuda_multi_gpu {
      explicit_target {
        pids: [1234, 1235]  // GPU worker PIDs (discovered via NVML or nvidia-smi)
      }
    }
  }
}
```
