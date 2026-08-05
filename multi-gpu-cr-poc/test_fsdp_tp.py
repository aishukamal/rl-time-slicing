"""
FSDP + Tensor Parallelism (TP=2) training for C/R testing.
Uses PyTorch DeviceMesh with TP dimension.

Usage: torchrun --nproc_per_node=2 test_fsdp_tp.py
"""
import os, time, torch, torch.nn as nn, torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel

def main():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("tp",))

    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
    ).to(device)

    parallelize_module(model, mesh, {
        "0": ColwiseParallel(),   # 1024 -> 2048/tp_size per rank
        "2": RowwiseParallel(),   # 2048/tp_size per rank -> 2048 (all-reduce)
    })
    # Layer 4 remains replicated (not parallelized)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    step = 0
    pause_file = os.environ.get("PAUSE_FILE", "/tmp/fsdp_pause")
    print(f"[rank {rank}] FSDP+TP training started (world_size={world_size})", flush=True)

    while True:
        while os.path.exists(pause_file):
            if not getattr(main, "_p", False):
                print(f"[rank {rank}] paused at step={step}", flush=True)
                main._p = True
            time.sleep(0.2)
        main._p = False

        x = torch.randn(64, 1024, device=device)
        target = torch.randn(64, 1024, device=device)
        optimizer.zero_grad()
        output = model(x)
        if isinstance(output, DTensor):
            output = output.full_tensor()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        step += 1
        if step % 10 == 0:
            print(f"[rank {rank}] step={step} loss={loss.item():.4f}", flush=True)
        time.sleep(0.5)

if __name__ == "__main__":
    main()
