"""
FSDP + Context Parallelism (CP=2) training for C/R testing.
Uses PyTorch ring attention via context_parallel().

Usage: torchrun --nproc_per_node=2 test_fsdp_cp.py
"""
import os, time, torch, torch.nn as nn, torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

def main():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
    cp_group = mesh.get_group("cp")

    try:
        from torch.distributed.tensor.experimental._attention import context_parallel
    except ImportError:
        from torch.distributed.tensor.experimental import context_parallel

    d_model = 256
    n_heads = 8
    head_dim = d_model // n_heads
    seq_len = 512  # total sequence, split across CP ranks

    q_proj = nn.Linear(d_model, d_model, bias=False).to(device)
    k_proj = nn.Linear(d_model, d_model, bias=False).to(device)
    v_proj = nn.Linear(d_model, d_model, bias=False).to(device)
    out_proj = nn.Linear(d_model, d_model, bias=False).to(device)
    ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model)).to(device)

    params = list(q_proj.parameters()) + list(k_proj.parameters()) + list(v_proj.parameters()) + list(out_proj.parameters()) + list(ffn.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4)
    criterion = nn.MSELoss()

    local_seq = seq_len // world_size
    step = 0
    pause_file = os.environ.get("PAUSE_FILE", "/tmp/fsdp_pause")
    print(f"[rank {rank}] FSDP+CP training started (cp_size={world_size}, local_seq={local_seq})", flush=True)

    while True:
        while os.path.exists(pause_file):
            if not getattr(main, "_p", False):
                print(f"[rank {rank}] paused at step={step}", flush=True)
                main._p = True
            time.sleep(0.2)
        main._p = False

        x = torch.randn(4, local_seq, d_model, device=device)
        target = torch.randn(4, local_seq, d_model, device=device)

        q = q_proj(x).view(4, local_seq, n_heads, head_dim).transpose(1, 2)
        k = k_proj(x).view(4, local_seq, n_heads, head_dim).transpose(1, 2)
        v = v_proj(x).view(4, local_seq, n_heads, head_dim).transpose(1, 2)

        optimizer.zero_grad()
        with context_parallel(mesh, buffers=(q, k, v), buffer_seq_dims=(2, 2, 2)):
            attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        attn_out = attn_out.transpose(1, 2).contiguous().view(4, local_seq, d_model)
        out = out_proj(attn_out)
        out = ffn(out)

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
