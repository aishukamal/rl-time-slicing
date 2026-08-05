"""
Pipeline Parallelism (PP=2) training for C/R testing.
Splits model layers across 2 GPUs with manual pipeline scheduling.

Usage: torchrun --nproc_per_node=2 test_fsdp_pp.py
"""
import os, time, torch, torch.nn as nn, torch.distributed as dist

class Stage(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

def main():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    assert world_size == 2, "PP=2 requires exactly 2 GPUs"
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        model = Stage(1024, 2048, 1024).to(device)
    else:
        model = Stage(1024, 2048, 1024).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    step = 0
    pause_file = os.environ.get("PAUSE_FILE", "/tmp/fsdp_pause")
    print(f"[rank {rank}] PP training started (stage={rank})", flush=True)

    while True:
        while os.path.exists(pause_file):
            if not getattr(main, "_p", False):
                print(f"[rank {rank}] paused at step={step}", flush=True)
                main._p = True
            time.sleep(0.2)
        main._p = False

        if rank == 0:
            x = torch.randn(64, 1024, device=device)
            out = model(x)
            dist.send(out, dst=1)
            grad = torch.empty_like(out)
            dist.recv(grad, src=1)
            optimizer.zero_grad()
            out.backward(grad)
            optimizer.step()
        else:
            hidden = torch.empty(64, 1024, device=device)
            dist.recv(hidden, src=0)
            hidden.requires_grad_(True)
            target = torch.randn(64, 1024, device=device)
            out = model(hidden)
            optimizer.zero_grad()
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            dist.send(hidden.grad, dst=0)

        torch.cuda.synchronize()
        step += 1
        if step % 10 == 0 and rank == 1:
            print(f"[rank {rank}] step={step} loss={loss.item():.4f}", flush=True)
        time.sleep(0.5)

if __name__ == "__main__":
    main()
