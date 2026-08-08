package backends

import (
	"context"
	"fmt"
	"log/slog"
	"strconv"
	"syscall"
	"time"
)

const (
	// SIGRTMIN on Linux is 34. Suspend = SIGRTMIN+1 = 35, Resume = SIGRTMIN+2 = 36.
	sigNCCLSuspend = 35
	sigNCCLResume  = 36

	ncclSuspendWait = 3 * time.Second
	restorePIDWait  = 1 * time.Second
)

// CudaMultiGPUCheckpoint implements the Backend interface for multi-GPU (TP)
// workloads. It wraps CudaCheckpoint with NCCL suspend/resume signals and
// sequential per-PID checkpoint/restore.
//
// The workload must have the CR shim loaded via LD_PRELOAD, which registers
// signal handlers for ncclCommSuspend (SIGRTMIN+1) and ncclCommResume (SIGRTMIN+2).
// The workload must also have NCCL TCP transport forced via:
//
//	NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NVLS_ENABLE=0
type CudaMultiGPUCheckpoint struct {
	base      *CudaCheckpoint
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

// Snapshot suspends NCCL communicators, then sequentially checkpoints each PID.
func (c *CudaMultiGPUCheckpoint) Snapshot(ctx context.Context, pids []string) error {
	if len(pids) == 0 {
		return fmt.Errorf("at least one PID is required")
	}

	c.base.mu.Lock()
	defer c.base.mu.Unlock()

	slog.InfoContext(ctx, "Multi-GPU snapshot: suspending NCCL", "pids", pids)
	t0 := time.Now()
	if err := c.suspendNCCL(pids); err != nil {
		return fmt.Errorf("NCCL suspend failed: %w", err)
	}
	slog.InfoContext(ctx, "NCCL suspended", "duration", time.Since(t0))

	slog.InfoContext(ctx, "Multi-GPU snapshot: checkpointing PIDs sequentially", "pids", pids)
	t1 := time.Now()
	for _, pid := range pids {
		if err := c.base.checkpointSinglePID(ctx, pid); err != nil {
			return fmt.Errorf("checkpoint pid %s failed: %w", pid, err)
		}
	}
	slog.InfoContext(ctx, "All PIDs checkpointed", "count", len(pids), "duration", time.Since(t1))
	return nil
}

// Restore sequentially restores each PID, then resumes NCCL communicators.
func (c *CudaMultiGPUCheckpoint) Restore(ctx context.Context, pids []string) error {
	if len(pids) == 0 {
		return fmt.Errorf("at least one PID is required")
	}

	c.base.mu.Lock()
	defer c.base.mu.Unlock()

	slog.InfoContext(ctx, "Multi-GPU restore: restoring PIDs sequentially", "pids", pids)
	t0 := time.Now()
	for _, pid := range pids {
		if err := c.base.restoreSinglePID(ctx, pid); err != nil {
			return fmt.Errorf("restore pid %s failed: %w", pid, err)
		}
		time.Sleep(restorePIDWait)
	}
	slog.InfoContext(ctx, "All PIDs restored", "count", len(pids), "duration", time.Since(t0))

	slog.InfoContext(ctx, "Multi-GPU restore: resuming NCCL", "pids", pids)
	t1 := time.Now()
	if err := c.resumeNCCL(pids); err != nil {
		return fmt.Errorf("NCCL resume failed: %w", err)
	}
	slog.InfoContext(ctx, "NCCL resumed", "duration", time.Since(t1))
	return nil
}

// HealthCheck delegates to the base CudaCheckpoint health check.
func (c *CudaMultiGPUCheckpoint) HealthCheck(ctx context.Context) error {
	return c.base.HealthCheck(ctx)
}

func (c *CudaMultiGPUCheckpoint) suspendNCCL(pids []string) error {
	for _, pidStr := range pids {
		pid, err := strconv.Atoi(pidStr)
		if err != nil {
			return fmt.Errorf("invalid pid %q: %w", pidStr, err)
		}
		if err := c.sendSignal(pid, sigNCCLSuspend); err != nil {
			return fmt.Errorf("failed to send suspend signal to pid %d: %w", pid, err)
		}
	}
	time.Sleep(ncclSuspendWait)
	return nil
}

func (c *CudaMultiGPUCheckpoint) resumeNCCL(pids []string) error {
	for _, pidStr := range pids {
		pid, err := strconv.Atoi(pidStr)
		if err != nil {
			return fmt.Errorf("invalid pid %q: %w", pidStr, err)
		}
		if err := c.sendSignal(pid, sigNCCLResume); err != nil {
			return fmt.Errorf("failed to send resume signal to pid %d: %w", pid, err)
		}
	}
	return nil
}
