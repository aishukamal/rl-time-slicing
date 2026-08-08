package backends

import (
	"context"
	"syscall"
)

func (c *CudaCheckpoint) SetExecCommand(f func(ctx context.Context, name string, args ...string) ([]byte, error)) {
	c.execCommand = f
}

func (c *CudaCheckpoint) SetNvmlClient(n nvmlClient) {
	c.nvml = n
}

func (c *CudaCheckpoint) SetLookPath(f func(string) (string, error)) {
	c.lookPath = f
}

func (c *CudaMultiGPUCheckpoint) SetExecCommand(f func(ctx context.Context, name string, args ...string) ([]byte, error)) {
	c.base.execCommand = f
}

func (c *CudaMultiGPUCheckpoint) SetSendSignal(f func(pid int, sig syscall.Signal) error) {
	c.sendSignal = f
}

func (c *CudaMultiGPUCheckpoint) SetNvmlClient(n nvmlClient) {
	c.base.nvml = n
}

func (c *CudaMultiGPUCheckpoint) SetLookPath(f func(string) (string, error)) {
	c.base.lookPath = f
}
