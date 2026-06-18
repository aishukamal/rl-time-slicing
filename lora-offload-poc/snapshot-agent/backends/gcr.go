package backends

import (
	"context"
	"fmt"
	"log/slog"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// GCRBackend implements the Backend and SelectiveBackend interfaces using
// GCR (GPU Checkpoint/Restore) cr_client for VMM-based checkpoint/restore.
type GCRBackend struct {
	mu          sync.Mutex
	execCommand func(ctx context.Context, name string, args ...string) ([]byte, error)
	lookPath    func(string) (string, error)
}

// NewGCRBackend creates a new GCRBackend instance.
func NewGCRBackend() *GCRBackend {
	return &GCRBackend{
		execCommand: func(ctx context.Context, name string, args ...string) ([]byte, error) {
			return exec.CommandContext(ctx, name, args...).CombinedOutput()
		},
		lookPath: exec.LookPath,
	}
}

func (g *GCRBackend) getCRClientPath() string {
	if path, err := g.lookPath("cr_client"); err == nil {
		return path
	}
	return "/usr/local/bin/cr_client"
}

func (g *GCRBackend) runCommand(ctx context.Context, name string, args ...string) error {
	slog.InfoContext(ctx, "Running command", "name", name, "args", args)
	out, err := g.execCommand(ctx, name, args...)
	if err != nil {
		return fmt.Errorf("command failed: %w, output: %s", err, string(out))
	}
	slog.InfoContext(ctx, "Command completed", "output", string(out))
	return nil
}

// Snapshot performs a full GCR checkpoint (all GPU memory) for each PID.
// Uses buffer-only mode (-b) since GCR manages its own VMM-based memory release.
func (g *GCRBackend) Snapshot(ctx context.Context, pids []string) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	slog.InfoContext(ctx, "GCR Snapshot: full checkpoint", "pids", pids)
	binaryPath := g.getCRClientPath()

	t0 := time.Now()
	for _, pid := range pids {
		if err := g.runCommand(ctx, binaryPath, "-c", "-b", "-p", pid); err != nil {
			return fmt.Errorf("cr_client checkpoint failed for PID %s: %w", pid, err)
		}
	}
	slog.InfoContext(ctx, "GCR Snapshot completed", "duration", time.Since(t0))
	return nil
}

// Restore performs a full GCR restore (all GPU memory) for each PID.
func (g *GCRBackend) Restore(ctx context.Context, pids []string) error {
	if len(pids) == 0 {
		return fmt.Errorf("at least one PID is required")
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	slog.InfoContext(ctx, "GCR Restore: full restore", "pids", pids)
	binaryPath := g.getCRClientPath()

	t0 := time.Now()
	for _, pid := range pids {
		if err := g.runCommand(ctx, binaryPath, "-r", "-b", "-p", pid); err != nil {
			return fmt.Errorf("cr_client restore failed for PID %s: %w", pid, err)
		}
	}
	slog.InfoContext(ctx, "GCR Restore completed", "duration", time.Since(t0))
	return nil
}

// formatRegionsArg formats memory regions as the -s flag value: "ptr1:size1,ptr2:size2,..."
func formatRegionsArg(regions []MemoryRegion) string {
	parts := make([]string, len(regions))
	for i, r := range regions {
		parts[i] = fmt.Sprintf("0x%x:%d", r.Address, r.Size)
	}
	return strings.Join(parts, ",")
}

// SelectiveSnapshot checkpoints only the specified GPU memory regions for a single PID.
func (g *GCRBackend) SelectiveSnapshot(ctx context.Context, pid string, regions []MemoryRegion) error {
	if len(regions) == 0 {
		return fmt.Errorf("at least one memory region is required")
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	slog.InfoContext(ctx, "GCR SelectiveSnapshot", "pid", pid, "numRegions", len(regions))
	binaryPath := g.getCRClientPath()
	regionsArg := formatRegionsArg(regions)

	t0 := time.Now()
	if err := g.runCommand(ctx, binaryPath, "-c", "-s", regionsArg, "-p", pid); err != nil {
		return fmt.Errorf("cr_client selective checkpoint failed for PID %s: %w", pid, err)
	}
	slog.InfoContext(ctx, "GCR SelectiveSnapshot completed", "duration", time.Since(t0))
	return nil
}

// SelectiveRestore restores only the specified GPU memory regions for a single PID.
func (g *GCRBackend) SelectiveRestore(ctx context.Context, pid string, regions []MemoryRegion) error {
	if len(regions) == 0 {
		return fmt.Errorf("at least one memory region is required")
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	slog.InfoContext(ctx, "GCR SelectiveRestore", "pid", pid, "numRegions", len(regions))
	binaryPath := g.getCRClientPath()
	regionsArg := formatRegionsArg(regions)

	t0 := time.Now()
	if err := g.runCommand(ctx, binaryPath, "-r", "-s", regionsArg, "-p", pid); err != nil {
		return fmt.Errorf("cr_client selective restore failed for PID %s: %w", pid, err)
	}
	slog.InfoContext(ctx, "GCR SelectiveRestore completed", "duration", time.Since(t0))
	return nil
}

// HealthCheck verifies the cr_client binary is available.
func (g *GCRBackend) HealthCheck(ctx context.Context) error {
	binaryPath := g.getCRClientPath()
	if _, err := g.lookPath(binaryPath); err != nil {
		return fmt.Errorf("cr_client executable not found: %w", err)
	}
	return nil
}
