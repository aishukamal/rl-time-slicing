"""
Simple MoE with Expert Parallelism (EP=2) training for C/R testing.
4 experts split across 2 GPUs, top-2 routing, all-to-all dispatch.

Usage: torchrun --nproc_per_node=2 test_moe_ep.py
"""
import os, time, torch, torch.nn as nn, torch.distributed as dist

class Expert(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
    def forward(self, x):
        return self.net(x)

class MoELayer(nn.Module):
    def __init__(self, dim, hidden, n_experts_local, n_experts_total, rank, world_size):
        super().__init__()
        self.gate = nn.Linear(dim, n_experts_total, bias=False)
        self.experts = nn.ModuleList([Expert(dim, hidden) for _ in range(n_experts_local)])
        self.n_experts_total = n_experts_total
        self.n_experts_local = n_experts_local
        self.rank = rank
        self.world_size = world_size
        self.top_k = 2

    def forward(self, x):
        bs, seq, dim = x.shape
        flat = x.view(-1, dim)
        logits = self.gate(flat)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1)

        output = torch.zeros_like(flat)
        for k in range(self.top_k):
            expert_idx = indices[:, k]
            w = weights[:, k].unsqueeze(-1)
            for local_e in range(self.n_experts_local):
                global_e = self.rank * self.n_experts_local + local_e
                mask = (expert_idx == global_e)
                if mask.any():
                    tokens = flat[mask]
                    expert_out = self.experts[local_e](tokens)
                    output[mask] += w[mask] * expert_out

        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output.view(bs, seq, dim)

def main():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    dim = 512
    hidden = 1024
    n_experts_total = 4
    n_experts_local = n_experts_total // world_size

    model = nn.Sequential(
        nn.Linear(dim, dim),
    ).to(device)

    moe = MoELayer(dim, hidden, n_experts_local, n_experts_total, rank, world_size).to(device)
    out_proj = nn.Linear(dim, dim).to(device)

    params = list(model.parameters()) + list(moe.parameters()) + list(out_proj.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4)
    criterion = nn.MSELoss()

    step = 0
    pause_file = os.environ.get("PAUSE_FILE", "/tmp/fsdp_pause")
    print(f"[rank {rank}] MoE+EP training started (experts_local={n_experts_local}, total={n_experts_total})", flush=True)

    while True:
        while os.path.exists(pause_file):
            if not getattr(main, "_p", False):
                print(f"[rank {rank}] paused at step={step}", flush=True)
                main._p = True
            time.sleep(0.2)
        main._p = False

        x = torch.randn(8, 32, dim, device=device)
        target = torch.randn(8, 32, dim, device=device)

        h = model(x)
        h = moe(h)
        out = out_proj(h)

        optimizer.zero_grad()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        step += 1
        if step % 10 == 0:
            print(f"[rank {rank}] step={step} loss={loss.item():.4f}", flush=True)
        time.sleep(0.5)

if __name__ == "__main__":
    main()
