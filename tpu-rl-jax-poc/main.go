package main

import (
	"context"
	"fmt"
	"net"
	"os"
	"strconv"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const version = "0.1.0"

type rawCodec struct{}

func (rawCodec) Marshal(v interface{}) ([]byte, error) {
	if b, ok := v.([]byte); ok {
		return b, nil
	}
	if b, ok := v.(*[]byte); ok {
		return *b, nil
	}
	return nil, fmt.Errorf("rawCodec: unsupported type %T", v)
}

func (rawCodec) Unmarshal(data []byte, v interface{}) error {
	if b, ok := v.(*[]byte); ok {
		*b = append([]byte(nil), data...)
		return nil
	}
	return fmt.Errorf("rawCodec: unsupported type %T", v)
}

func (rawCodec) Name() string { return "raw" }

func socketPath(pid int) string {
	return fmt.Sprintf("/run/tpu_hal_%d.sock", pid)
}

func dial(pid int) (*grpc.ClientConn, error) {
	sock := socketPath(pid)
	if _, err := os.Stat(sock); err != nil {
		return nil, fmt.Errorf("socket not found: %s\nis PID %d a TPU process with libtpu-uds loaded?", sock, pid)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	return grpc.DialContext(ctx, "passthrough:///unix",
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.ForceCodecCallOption{ForceCodec: rawCodec{}}),
		grpc.WithBlock(),
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", sock)
		}),
	)
}

func invoke(conn *grpc.ClientConn, method string, timeoutSec int) error {
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeoutSec)*time.Second)
	defer cancel()

	req := []byte{}
	var resp []byte
	return conn.Invoke(ctx, method, req, &resp)
}

func doCheckpoint(pid int, timeoutSec int) error {
	conn, err := dial(pid)
	if err != nil {
		return err
	}
	defer conn.Close()

	t0 := time.Now()
	if err := invoke(conn, "/tpu.TpuHalService/Checkpoint", timeoutSec); err != nil {
		return fmt.Errorf("Checkpoint RPC failed: %w", err)
	}
	fmt.Printf("checkpoint complete (%.3fs)\n", time.Since(t0).Seconds())
	return nil
}

func doRestore(pid int, timeoutSec int) error {
	conn, err := dial(pid)
	if err != nil {
		return err
	}
	defer conn.Close()

	t0 := time.Now()
	if err := invoke(conn, "/tpu.TpuHalService/Restore", timeoutSec); err != nil {
		return fmt.Errorf("Restore RPC failed: %w", err)
	}
	fmt.Printf("restore complete (%.3fs)\n", time.Since(t0).Seconds())
	return nil
}

func usage() {
	fmt.Fprintf(os.Stderr, `TPU checkpoint and restore utility.
Version %s.

Operations:
  --action checkpoint | restore --pid <pid> [--timeout <seconds>]
        Performs the specified action on <pid>.

  --get-state --pid <pid>
        Prints the current checkpoint state (unimplemented).

Options:
  --pid|-p <pid>           Target process PID
  --timeout|-t <seconds>   RPC timeout in seconds (default: 300)
  --help|-h                Print this help message

Socket:
  Connects to /run/tpu_hal_<PID>.sock (created by libtpu-uds.so)
`, version)
}

func fatal(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "tpu-checkpoint: "+format+"\n", args...)
	os.Exit(1)
}

func main() {
	args := os.Args[1:]
	if len(args) == 0 {
		usage()
		os.Exit(1)
	}

	var (
		pid        = -1
		action     string
		getState   bool
		timeoutSec = 300
	)

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--help", "-h":
			usage()
			os.Exit(0)
		case "--pid", "-p":
			i++
			if i >= len(args) {
				fatal("--pid requires a value")
			}
			var err error
			pid, err = strconv.Atoi(args[i])
			if err != nil || pid <= 0 {
				fatal("invalid PID: %s", args[i])
			}
		case "--action":
			i++
			if i >= len(args) {
				fatal("--action requires a value")
			}
			action = strings.ToLower(args[i])
		case "--get-state":
			getState = true
		case "--timeout", "-t":
			i++
			if i >= len(args) {
				fatal("--timeout requires a value")
			}
			var err error
			timeoutSec, err = strconv.Atoi(args[i])
			if err != nil || timeoutSec <= 0 {
				fatal("invalid timeout: %s", args[i])
			}
		default:
			fatal("unknown option: %s\nrun 'tpu-checkpoint --help' for usage", args[i])
		}
	}

	if pid <= 0 {
		fatal("--pid is required")
	}

	var err error
	switch {
	case getState:
		fatal("--get-state is not implemented (libtpu GetState always returns READY regardless of state)")
	case action == "checkpoint":
		err = doCheckpoint(pid, timeoutSec)
	case action == "restore":
		err = doRestore(pid, timeoutSec)
	case action != "":
		fatal("unknown action: %s (supported: checkpoint, restore)", action)
	default:
		fatal("no operation specified (use --action checkpoint|restore)\nrun 'tpu-checkpoint --help' for usage")
	}

	if err != nil {
		fatal("%v", err)
	}
}
