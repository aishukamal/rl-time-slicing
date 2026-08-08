package backends

import (
	"context"
	"fmt"
	"log/slog"
	"strconv"
	"syscall"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
)

const (
	// SIGRTMIN on Linux is 34.
	// Signal 35 (SIGRTMIN+1): shim v2 destroys NCCL comms (ncclCommDestroy).
	// Signal 36 (SIGRTMIN+2): shim v2 arms lazy NCCL recreate.
	sigNCCLDestroy  = 35
	sigNCCLRecreate = 36

	ncclDestroyWait = 3 * time.Second
	restorePIDDelay = 1 * time.Second
)

// CudaMultiGPUCheckpoint implements the Backend interface for multi-GPU (TP/DP/PP/EP)
// workloads. It wraps CudaCheckpoint with NCCL destroy/recreate signals and
// sequential per-PID checkpoint/restore.
//
// The workload must have shim v2 (libcr-shim-v2.so) loaded via LD_PRELOAD,
// which registers signal handlers for:
//   - SIGRTMIN+1 (35): ncclCommDestroy all tracked comms
//   - SIGRTMIN+2 (36): arm lazy ncclCommInitRank with fresh uniqueId rendezvous
//
// Required env vars on the workload:
//
//	NCCL_NVLS_ENABLE=0  (driver bug: multicast broken post-restore)
//	CR_NCCL_LIB=/path/to/libnccl.so.2  (shim needs to find real NCCL)
type CudaMultiGPUCheckpoint struct {
	base       *CudaCheckpoint
	sendSignal func(pid int, sig syscall.Signal) error
}

// NewCudaMultiGPUCheckpoint creates a new multi-GPU checkpoint backend.
func NewCudaMultiGPUCheckpoint() *CudaMultiGPUCheckpoint {
	return &CudaMultiGPUCheckpoint{
		base: NewCudaCheckpoint(),
		sendSignal: func(pid int, sig syscall.Signal) error {
			return syscall.Kill(pid, sig)
		},
	}
}

// Snapshot destroys NCCL communicators, then checkpoints all PIDs.
func (c *CudaMultiGPUCheckpoint) Snapshot(ctx context.Context, req Request) error {
	pids := ExtractMultiGPUPIDs(req.Config)
	if len(pids) == 0 {
		return fmt.Errorf("at least one PID is required for multi-GPU snapshot")
	}

	c.base.mu.Lock()
	defer c.base.mu.Unlock()

	slog.InfoContext(ctx, "Multi-GPU snapshot: destroying NCCL comms", "pids", pids)
	t0 := time.Now()
	if err := c.signalAll(pids, sigNCCLDestroy); err != nil {
		return fmt.Errorf("NCCL destroy signal failed: %w", err)
	}
	time.Sleep(ncclDestroyWait)
	slog.InfoContext(ctx, "NCCL comms destroyed", "duration", time.Since(t0))

	slog.InfoContext(ctx, "Multi-GPU snapshot: checkpointing PIDs", "pids", pids)
	t1 := time.Now()
	if err := c.base.checkpointPIDs(ctx, pids); err != nil {
		return fmt.Errorf("checkpoint failed: %w", err)
	}
	slog.InfoContext(ctx, "All PIDs checkpointed", "count", len(pids), "duration", time.Since(t1))
	return nil
}

// Restore restores all PIDs, then arms lazy NCCL recreate.
func (c *CudaMultiGPUCheckpoint) Restore(ctx context.Context, req Request) error {
	pids := ExtractMultiGPUPIDs(req.Config)
	if len(pids) == 0 {
		return fmt.Errorf("at least one PID is required for multi-GPU restore")
	}

	c.base.mu.Lock()
	defer c.base.mu.Unlock()

	slog.InfoContext(ctx, "Multi-GPU restore: restoring PIDs", "pids", pids)
	t0 := time.Now()
	if err := c.base.restorePIDs(ctx, pids); err != nil {
		return fmt.Errorf("restore failed: %w", err)
	}
	slog.InfoContext(ctx, "All PIDs restored", "count", len(pids), "duration", time.Since(t0))

	slog.InfoContext(ctx, "Multi-GPU restore: arming NCCL recreate", "pids", pids)
	if err := c.signalAll(pids, sigNCCLRecreate); err != nil {
		return fmt.Errorf("NCCL recreate signal failed: %w", err)
	}
	slog.InfoContext(ctx, "NCCL recreate armed")
	return nil
}

// HealthCheck delegates to the base CudaCheckpoint health check.
func (c *CudaMultiGPUCheckpoint) HealthCheck(ctx context.Context) error {
	return c.base.HealthCheck(ctx)
}

func (c *CudaMultiGPUCheckpoint) signalAll(pids []string, sig syscall.Signal) error {
	for _, pidStr := range pids {
		pid, err := strconv.Atoi(pidStr)
		if err != nil {
			return fmt.Errorf("invalid pid %q: %w", pidStr, err)
		}
		if err := c.sendSignal(pid, sig); err != nil {
			return fmt.Errorf("failed to send signal %d to pid %d: %w", sig, pid, err)
		}
	}
	return nil
}

// ExtractMultiGPUPIDs extracts PID strings from a BackendConfig.
// Supports CudaMultiGPUBackendConfig (preferred) and falls back to CudaBackendConfig.
func ExtractMultiGPUPIDs(config *pb.BackendConfig) []string {
	if config == nil {
		return nil
	}
	if mg := config.GetCudaMultiGpu(); mg != nil {
		target := mg.GetExplicitTarget()
		if target == nil {
			return nil
		}
		return int32sToStrings(target.GetPids())
	}
	return ExtractPIDStrings(config)
}

func int32sToStrings(pids []int32) []string {
	out := make([]string, 0, len(pids))
	for _, pid := range pids {
		out = append(out, strconv.Itoa(int(pid)))
	}
	return out
}
