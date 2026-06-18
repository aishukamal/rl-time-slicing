package backends_test

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
)

func TestNewGCRBackend(t *testing.T) {
	g := backends.NewGCRBackend()
	if g == nil {
		t.Fatal("NewGCRBackend returned nil")
	}
}

func TestGCRSnapshot(t *testing.T) {
	tests := []struct {
		name        string
		pids        []string
		execErr     error
		expectedErr bool
	}{
		{
			name:        "Success",
			pids:        []string{"123", "456"},
			execErr:     nil,
			expectedErr: false,
		},
		{
			name:        "ExecFailure",
			pids:        []string{"123"},
			execErr:     fmt.Errorf("exec error"),
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := backends.NewGCRBackend()
			g.SetExecCommand(func(ctx context.Context, name string, args ...string) ([]byte, error) {
				return nil, tt.execErr
			})

			err := g.Snapshot(context.Background(), tt.pids)
			if (err != nil) != tt.expectedErr {
				t.Errorf("Snapshot() error = %v, expectedErr %v", err, tt.expectedErr)
			}
		})
	}
}

func TestGCRRestore(t *testing.T) {
	tests := []struct {
		name        string
		pids        []string
		execErr     error
		expectedErr bool
	}{
		{
			name:        "Success",
			pids:        []string{"123"},
			execErr:     nil,
			expectedErr: false,
		},
		{
			name:        "NoPIDs",
			pids:        []string{},
			execErr:     nil,
			expectedErr: true,
		},
		{
			name:        "ExecFailure",
			pids:        []string{"123"},
			execErr:     fmt.Errorf("exec error"),
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := backends.NewGCRBackend()
			g.SetExecCommand(func(ctx context.Context, name string, args ...string) ([]byte, error) {
				return nil, tt.execErr
			})

			err := g.Restore(context.Background(), tt.pids)
			if (err != nil) != tt.expectedErr {
				t.Errorf("Restore() error = %v, expectedErr %v", err, tt.expectedErr)
			}
		})
	}
}

func TestGCRSelectiveSnapshot(t *testing.T) {
	tests := []struct {
		name        string
		pid         string
		regions     []backends.MemoryRegion
		execErr     error
		expectedErr bool
		checkArgs   func(t *testing.T, name string, args []string)
	}{
		{
			name: "Success",
			pid:  "123",
			regions: []backends.MemoryRegion{
				{Address: 0x7f0001000000, Size: 4194304},
				{Address: 0x7f0002000000, Size: 2097152},
			},
			execErr:     nil,
			expectedErr: false,
			checkArgs: func(t *testing.T, name string, args []string) {
				argStr := strings.Join(args, " ")
				if !strings.Contains(argStr, "-s") {
					t.Error("expected -s flag in command args")
				}
				if !strings.Contains(argStr, "-c") {
					t.Error("expected -c flag in command args")
				}
				if !strings.Contains(argStr, "0x7f0001000000:4194304") {
					t.Error("expected first region in args")
				}
				if !strings.Contains(argStr, "0x7f0002000000:2097152") {
					t.Error("expected second region in args")
				}
			},
		},
		{
			name:        "NoRegions",
			pid:         "123",
			regions:     []backends.MemoryRegion{},
			execErr:     nil,
			expectedErr: true,
		},
		{
			name: "ExecFailure",
			pid:  "123",
			regions: []backends.MemoryRegion{
				{Address: 0x1000, Size: 1024},
			},
			execErr:     fmt.Errorf("exec error"),
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := backends.NewGCRBackend()
			g.SetExecCommand(func(ctx context.Context, name string, args ...string) ([]byte, error) {
				if tt.checkArgs != nil {
					tt.checkArgs(t, name, args)
				}
				return nil, tt.execErr
			})

			err := g.SelectiveSnapshot(context.Background(), tt.pid, tt.regions)
			if (err != nil) != tt.expectedErr {
				t.Errorf("SelectiveSnapshot() error = %v, expectedErr %v", err, tt.expectedErr)
			}
		})
	}
}

func TestGCRSelectiveRestore(t *testing.T) {
	tests := []struct {
		name        string
		pid         string
		regions     []backends.MemoryRegion
		execErr     error
		expectedErr bool
		checkArgs   func(t *testing.T, name string, args []string)
	}{
		{
			name: "Success",
			pid:  "123",
			regions: []backends.MemoryRegion{
				{Address: 0x7f0001000000, Size: 4194304},
			},
			execErr:     nil,
			expectedErr: false,
			checkArgs: func(t *testing.T, name string, args []string) {
				argStr := strings.Join(args, " ")
				if !strings.Contains(argStr, "-s") {
					t.Error("expected -s flag in command args")
				}
				if !strings.Contains(argStr, "-r") {
					t.Error("expected -r flag in command args")
				}
			},
		},
		{
			name:        "NoRegions",
			pid:         "123",
			regions:     []backends.MemoryRegion{},
			execErr:     nil,
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := backends.NewGCRBackend()
			g.SetExecCommand(func(ctx context.Context, name string, args ...string) ([]byte, error) {
				if tt.checkArgs != nil {
					tt.checkArgs(t, name, args)
				}
				return nil, tt.execErr
			})

			err := g.SelectiveRestore(context.Background(), tt.pid, tt.regions)
			if (err != nil) != tt.expectedErr {
				t.Errorf("SelectiveRestore() error = %v, expectedErr %v", err, tt.expectedErr)
			}
		})
	}
}

func TestGCRHealthCheck(t *testing.T) {
	tests := []struct {
		name        string
		lookPathErr error
		expectedErr bool
	}{
		{
			name:        "Success",
			lookPathErr: nil,
			expectedErr: false,
		},
		{
			name:        "BinaryNotFound",
			lookPathErr: fmt.Errorf("not found"),
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := backends.NewGCRBackend()
			g.SetLookPath(func(path string) (string, error) {
				if tt.lookPathErr != nil {
					return "", tt.lookPathErr
				}
				return path, nil
			})

			err := g.HealthCheck(context.Background())
			if (err != nil) != tt.expectedErr {
				t.Errorf("HealthCheck() error = %v, expectedErr %v", err, tt.expectedErr)
			}
		})
	}
}

func TestGCRImplementsSelectiveBackend(t *testing.T) {
	g := backends.NewGCRBackend()
	var _ backends.Backend = g
	var _ backends.SelectiveBackend = g
}
