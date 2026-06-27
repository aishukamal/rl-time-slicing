"""
Minimal FSDP training loop for C/R testing.
Runs continuous training steps on a small model, printing loss each step.
Uses NCCL backend for multi-GPU gradient sync.

Usage:
  # Single GPU:
  python test_fsdp_trainer.py

  # Multi-GPU (2 GPUs, same node):
  torchrun --nproc_per_node=2 test_fsdp_trainer.py
"""
import os
import sys
import time
import signal
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")

    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
    ).to(device)

    if world_size > 1:
        model = FSDP(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    step = 0
    print(f"[rank {local_rank}] Training started (world_size={world_size})", flush=True)

    while True:
        x = torch.randn(64, 1024, device=device)
        target = torch.randn(64, 1024, device=device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        step += 1
        if step % 10 == 0:
            print(f"[rank {local_rank}] step={step} loss={loss.item():.4f}", flush=True)

        time.sleep(0.5)

if __name__ == "__main__":
    main()
