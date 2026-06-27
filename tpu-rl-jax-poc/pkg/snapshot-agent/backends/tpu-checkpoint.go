package backends

import (
	"context"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/encoding"
	"k8s.io/klog/v2"
)

func init() {
	encoding.RegisterCodec(rawCodec{})
}

const (
	tpuHalSocketPrefix = "/run/tpu_hal_"
	tpuHalSocketSuffix = ".sock"

	methodCheckpoint = "/tpu.TpuHalService/Checkpoint"
	methodRestore    = "/tpu.TpuHalService/Restore"
)

// rawCodec passes bytes through without protobuf marshaling.
// Implements encoding.Codec for gRPC registration.
type rawCodec struct{}

func (rawCodec) Marshal(v any) ([]byte, error) {
	if b, ok := v.([]byte); ok {
		return b, nil
	}
	if b, ok := v.(*[]byte); ok {
		return *b, nil
	}
	return nil, fmt.Errorf("rawCodec: unsupported type %T", v)
}

func (rawCodec) Unmarshal(data []byte, v any) error {
	if b, ok := v.(*[]byte); ok {
		*b = append([]byte(nil), data...)
		return nil
	}
	return fmt.Errorf("rawCodec: unsupported type %T", v)
}

func (rawCodec) Name() string { return "raw" }

// TpuCheckpoint implements the Backend interface using libtpu-uds.so gRPC RPCs.
// It connects directly to the Unix domain socket created by libtpu at
// /run/tpu_hal_<PID>.sock and calls tpu.TpuHalService/Checkpoint or Restore.
type TpuCheckpoint struct {
	mu sync.Mutex
}

// NewTpuCheckpoint creates a new TpuCheckpoint backend.
func NewTpuCheckpoint() *TpuCheckpoint {
	return &TpuCheckpoint{}
}

func tpuSocketPath(pid string) string {
	return fmt.Sprintf("%s%s%s", tpuHalSocketPrefix, pid, tpuHalSocketSuffix)
}

// discoverAllSockets returns all TPU HAL socket paths on the host.
// Each libtpu-uds process (including subprocesses like VLLM::EngineCore)
// creates its own socket. All must be checkpointed/restored together.
func discoverAllSockets() ([]string, error) {
	return filepath.Glob(tpuHalSocketPrefix + "*" + tpuHalSocketSuffix)
}

func (t *TpuCheckpoint) dialSocketPath(ctx context.Context, sockPath string) (*grpc.ClientConn, error) {
	if _, err := os.Stat(sockPath); err != nil {
		return nil, fmt.Errorf("TPU HAL socket not found: %s", sockPath)
	}

	dialCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	return grpc.DialContext(dialCtx, "passthrough:///unix",
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.CallContentSubtype("raw")),
		grpc.WithBlock(),
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", sockPath)
		}),
	)
}

func (t *TpuCheckpoint) invokeRPC(ctx context.Context, conn *grpc.ClientConn, method string) error {
	rpcCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()

	req := []byte{}
	var resp []byte
	return conn.Invoke(rpcCtx, method, req, &resp)
}

// activeSockets returns TPU HAL socket paths whose owning PID is alive
// and matches one of the given PIDs. This filters out stale sockets from
// previous pods that would corrupt TPU state if checkpointed.
func (t *TpuCheckpoint) activeSockets(pids []string) []string {
	pidSet := make(map[string]bool, len(pids))
	for _, p := range pids {
		pidSet[p] = true
	}

	all, _ := discoverAllSockets()
	var active []string
	for _, sock := range all {
		base := filepath.Base(sock)
		pid := strings.TrimPrefix(base, "tpu_hal_")
		pid = strings.TrimSuffix(pid, ".sock")

		// Skip if PID doesn't match any of the job's PIDs
		if len(pidSet) > 0 && !pidSet[pid] {
			continue
		}

		// Skip if process is dead (stale socket)
		if _, err := os.Stat(fmt.Sprintf("/proc/%s", pid)); err != nil {
			continue
		}
		active = append(active, sock)
	}
	return active
}

// Snapshot checkpoints all TPU processes belonging to the job.
// Uses the PID list to filter sockets, skipping stale sockets from old pods.
func (t *TpuCheckpoint) Snapshot(ctx context.Context, pids []string) error {
	logger := klog.FromContext(ctx)
	t.mu.Lock()
	defer t.mu.Unlock()

	sockets := t.activeSockets(pids)
	if len(sockets) == 0 {
		return fmt.Errorf("no active TPU HAL sockets found for PIDs %v", pids)
	}

	logger.Info("Checkpointing TPU sockets", "sockets", sockets, "pids", pids)
	t0 := time.Now()

	for _, sock := range sockets {
		conn, err := t.dialSocketPath(ctx, sock)
		if err != nil {
			return fmt.Errorf("failed to connect to %s: %w", sock, err)
		}

		if err := t.invokeRPC(ctx, conn, methodCheckpoint); err != nil {
			conn.Close()
			return fmt.Errorf("TPU checkpoint failed on %s: %w", sock, err)
		}
		conn.Close()
		logger.Info("TPU checkpoint complete", "socket", sock)
	}

	logger.Info("TPU checkpoint all sockets complete", "duration", time.Since(t0), "count", len(sockets))
	return nil
}

// Restore restores all TPU processes belonging to the job.
func (t *TpuCheckpoint) Restore(ctx context.Context, pids []string) error {
	logger := klog.FromContext(ctx)
	t.mu.Lock()
	defer t.mu.Unlock()

	sockets := t.activeSockets(pids)
	if len(sockets) == 0 {
		return fmt.Errorf("no active TPU HAL sockets found for PIDs %v", pids)
	}

	logger.Info("Restoring TPU sockets", "sockets", sockets, "pids", pids)
	t0 := time.Now()

	for _, sock := range sockets {
		conn, err := t.dialSocketPath(ctx, sock)
		if err != nil {
			return fmt.Errorf("failed to connect to %s: %w", sock, err)
		}

		if err := t.invokeRPC(ctx, conn, methodRestore); err != nil {
			conn.Close()
			return fmt.Errorf("TPU restore failed on %s: %w", sock, err)
		}
		conn.Close()
		logger.Info("TPU restore complete", "socket", sock)
	}

	logger.Info("TPU restore all sockets complete", "duration", time.Since(t0), "count", len(sockets))
	return nil
}

// HealthCheck verifies TPU HAL sockets are discoverable on the host.
func (t *TpuCheckpoint) HealthCheck(ctx context.Context) error {
	sockets, err := filepath.Glob(tpuHalSocketPrefix + "*" + tpuHalSocketSuffix)
	if err != nil {
		return fmt.Errorf("failed to glob TPU HAL sockets: %w", err)
	}

	if len(sockets) == 0 {
		// No sockets yet is OK — processes may not have started.
		// Just check that /run is accessible.
		if _, err := os.Stat("/run"); err != nil {
			return fmt.Errorf("/run not accessible: %w", err)
		}
		klog.Info("No TPU HAL sockets found yet (no TPU processes running)")
		return nil
	}

	klog.Info("TPU HAL sockets found", "count", len(sockets), "sockets", strings.Join(sockets, ", "))
	return nil
}
