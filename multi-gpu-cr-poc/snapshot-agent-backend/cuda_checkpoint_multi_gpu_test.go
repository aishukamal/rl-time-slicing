package backends_test

import (
	"context"
	"fmt"
	"sync"
	"syscall"
	"testing"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
)

func TestNewCudaMultiGPUCheckpoint(t *testing.T) {
	c := backends.NewCudaMultiGPUCheckpoint()
	if c == nil {
		t.Fatal("NewCudaMultiGPUCheckpoint returned nil")
	}
}

func TestMultiGPUSnapshot(t *testing.T) {
	tests := []struct {
		name        string
		pids        []string
		execErr     error
		signalErr   error
		expectedErr bool
	}{
		{
			name:        "Success",
			pids:        []string{"100", "101"},
			execErr:     nil,
			signalErr:   nil,
			expectedErr: false,
		},
		{
			name:        "SignalFailure",
			pids:        []string{"100"},
			execErr:     nil,
			signalErr:   fmt.Errorf("signal error"),
			expectedErr: true,
		},
		{
			name:        "CheckpointFailure",
			pids:        []string{"100", "101"},
			execErr:     fmt.Errorf("exec error"),
			signalErr:   nil,
			expectedErr: true,
		},
		{
			name:        "NoPIDs",
			pids:        []string{},
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := backends.NewCudaMultiGPUCheckpoint()
			c.SetExecCommand(func(ctx context.Context, name string, args ...string) ([]byte, error) {
				return nil, tt.execErr
			})
			c.SetSendSignal(func(pid int, sig syscall.Signal) error {
				return tt.signalErr
			})

			err := c.Snapshot(context.Background(), tt.pids)
			if (err != nil) != tt.expectedErr {
				t.Errorf("Snapshot() error = %v, expectedErr %v", err, tt.expectedErr)
			}
		})
	}
}

func TestMultiGPURestore(t *testing.T) {
	tests := []struct {
		name        string
		pids        []string
		execErr     error
		signalErr   error
		expectedErr bool
	}{
		{
			name:        "Success",
			pids:        []string{"100", "101"},
			execErr:     nil,
			signalErr:   nil,
			expectedErr: false,
		},
		{
			name:        "RestoreFailure",
			pids:        []string{"100"},
			execErr:     fmt.Errorf("exec error"),
			signalErr:   nil,
			expectedErr: true,
		},
		{
			name:        "ResumeSignalFailure",
			pids:        []string{"100"},
			execErr:     nil,
			signalErr:   fmt.Errorf("signal error"),
			expectedErr: true,
		},
		{
			name:        "NoPIDs",
			pids:        []string{},
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := backends.NewCudaMultiGPUCheckpoint()
			c.SetExecCommand(func(ctx context.Context, name string, args ...string) ([]byte, error) {
				return nil, tt.execErr
			})
			c.SetSendSignal(func(pid int, sig syscall.Signal) error {
				return tt.signalErr
			})

			err := c.Restore(context.Background(), tt.pids)
			if (err != nil) != tt.expectedErr {
				t.Errorf("Restore() error = %v, expectedErr %v", err, tt.expectedErr)
			}
		})
	}
}

func TestMultiGPUSnapshotSignalOrder(t *testing.T) {
	var mu sync.Mutex
	var signals []struct {
		pid int
		sig syscall.Signal
	}
	var commands []string

	c := backends.NewCudaMultiGPUCheckpoint()
	c.SetSendSignal(func(pid int, sig syscall.Signal) error {
		mu.Lock()
		defer mu.Unlock()
		signals = append(signals, struct {
			pid int
			sig syscall.Signal
		}{pid, sig})
		return nil
	})
	c.SetExecCommand(func(ctx context.Context, name string, args ...string) ([]byte, error) {
		mu.Lock()
		defer mu.Unlock()
		if len(args) > 0 {
			commands = append(commands, args[0])
		}
		return nil, nil
	})

	err := c.Snapshot(context.Background(), []string{"100", "101"})
	if err != nil {
		t.Fatalf("Snapshot() unexpected error: %v", err)
	}

	// Verify suspend signals sent to both PIDs
	if len(signals) != 2 {
		t.Fatalf("expected 2 suspend signals, got %d", len(signals))
	}
	for _, s := range signals {
		if s.sig != 35 {
			t.Errorf("expected signal 35 (NCCL suspend), got %d", s.sig)
		}
	}

	// Verify sequential lock+checkpoint for each PID (4 exec calls: lock,ckpt,lock,ckpt)
	if len(commands) != 4 {
		t.Fatalf("expected 4 exec commands, got %d: %v", len(commands), commands)
	}
	if commands[0] != "--action" || commands[2] != "--action" {
		t.Errorf("expected --action commands, got %v", commands)
	}
}

func TestMultiGPUHealthCheck(t *testing.T) {
	c := backends.NewCudaMultiGPUCheckpoint()
	c.SetLookPath(func(path string) (string, error) { return path, nil })
	c.SetNvmlClient(&mockNvmlClient{
		initRet:        0,
		shutdownRet:    0,
		deviceCount:    2,
		deviceCountRet: 0,
	})

	err := c.HealthCheck(context.Background())
	if err != nil {
		t.Errorf("HealthCheck() unexpected error: %v", err)
	}
}
