package backends

import "context"

// BackendType represents the type of accelerator backend.
type BackendType string

const (
	// BackendCuda is the CUDA-based checkpointing backend.
	BackendCuda BackendType = "cuda"
	// BackendNoop is a dummy backend for testing.
	BackendNoop BackendType = "noop"
	// BackendGCR is the GCR (GPU Checkpoint/Restore) backend using VMM-based selective C/R.
	BackendGCR BackendType = "gcr"
)

// Backend defines the interface for checkpoint and restore operations.
type Backend interface {
	// Snapshot triggers a snapshot of the accelerator context for a job.
	// Returns storageBytes, deviceBytes, and error.
	Snapshot(ctx context.Context, pids []string) error

	// Restore triggers a restoration of the accelerator context for a job.
	Restore(ctx context.Context, pids []string) error

	// HealthCheck checks if the backend is healthy by initializing the backend
	// and the discovery provider.
	HealthCheck(ctx context.Context) error
}

// MemoryRegion represents a contiguous GPU memory region identified by its device pointer and size.
type MemoryRegion struct {
	Address uint64
	Size    uint64
}

// SelectiveBackend extends Backend with selective checkpoint/restore of specific memory regions.
type SelectiveBackend interface {
	Backend
	SelectiveSnapshot(ctx context.Context, pid string, regions []MemoryRegion) error
	SelectiveRestore(ctx context.Context, pid string, regions []MemoryRegion) error
}
