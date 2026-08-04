"""
Test: can the in-process CUDA checkpoint API freeze/restore a multi-GPU
process with captured CUDA graphs?

cuda-checkpoint CLI fails this case ("initialization error").
The CUDA Driver API (cuCheckpointProcess*) is a different code path —
called from INSIDE the process, no ptrace. This test checks whether
the API path has the same limitation.

Usage: torchrun --nproc_per_node=2 graph_cr_api_test.py
"""
import os, sys, time, ctypes, ctypes.util
import torch
import torch.distributed as dist

def get_cuda_driver():
    """Load libcuda.so and resolve checkpoint functions."""
    cuda = ctypes.CDLL("libcuda.so.1")

    # cuCheckpointProcessLock / Checkpoint / Restore / Unlock
    # These are the in-process equivalents of cuda-checkpoint --action
    fns = {}
    for name in ["cuCheckpointProcessLock", "cuCheckpointProcessCheckpoint",
                  "cuCheckpointProcessRestore", "cuCheckpointProcessUnlock"]:
        fn = getattr(cuda, name, None)
        if fn is None:
            print(f"[rank {rank}] {name} not found in driver — too old?")
            return None
        fn.restype = ctypes.c_int  # CUresult
        fns[name] = fn

    return fns


def checkpoint_restore_cycle(fns, rank):
    """Call the checkpoint API directly, in-process."""
    lock = fns["cuCheckpointProcessLock"]
    ckpt = fns["cuCheckpointProcessCheckpoint"]
    rst  = fns["cuCheckpointProcessRestore"]
    unlock = fns["cuCheckpointProcessUnlock"]

    print(f"[rank {rank}] Locking...", flush=True)
    rc = lock()
    if rc != 0:
        print(f"[rank {rank}] Lock FAILED rc={rc}", flush=True)
        return False

    print(f"[rank {rank}] Checkpointing...", flush=True)
    rc = ckpt()
    if rc != 0:
        print(f"[rank {rank}] Checkpoint FAILED rc={rc}", flush=True)
        unlock()
        return False

    print(f"[rank {rank}] Checkpoint done — GPU memory released. Restoring...", flush=True)
    time.sleep(1)

    rc = rst()
    if rc != 0:
        print(f"[rank {rank}] Restore FAILED rc={rc}", flush=True)
        unlock()
        return False

    print(f"[rank {rank}] Unlocking...", flush=True)
    rc = unlock()
    if rc != 0:
        print(f"[rank {rank}] Unlock FAILED rc={rc}", flush=True)
        return False

    print(f"[rank {rank}] C/R cycle complete!", flush=True)
    return True


def main():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Simple model
    model = torch.nn.Linear(1024, 1024).to(device).half()

    # Warm up
    x = torch.randn(4, 1024, device=device, dtype=torch.float16)
    for _ in range(3):
        y = model(x)
        if world_size > 1:
            dist.all_reduce(y)
        y.sum().backward()

    torch.cuda.synchronize()

    # Capture a CUDA graph (compute only — no collectives inside)
    print(f"[rank {rank}] Capturing CUDA graph (compute-only segment)...", flush=True)
    g = torch.cuda.CUDAGraph()
    static_x = torch.randn(4, 1024, device=device, dtype=torch.float16)
    static_y = torch.empty(4, 1024, device=device, dtype=torch.float16)

    with torch.cuda.graph(g):
        static_y = model(static_x)

    # Verify graph replay works before C/R
    g.replay()
    torch.cuda.synchronize()
    print(f"[rank {rank}] Graph captured and replayed OK (pre-C/R).", flush=True)

    # Barrier so all ranks are ready
    dist.barrier()

    # Load checkpoint API
    fns = get_cuda_driver()
    if fns is None:
        print(f"[rank {rank}] Checkpoint API not available, exiting.")
        dist.destroy_process_group()
        return

    # --- The test: in-process checkpoint/restore with graphs alive ---
    print(f"[rank {rank}] === IN-PROCESS C/R WITH GRAPHS ===", flush=True)
    dist.barrier()
    torch.cuda.synchronize()

    ok = checkpoint_restore_cycle(fns, rank)

    if ok:
        # Try replaying the graph after restore
        torch.cuda.synchronize()
        try:
            g.replay()
            torch.cuda.synchronize()
            result = static_y.sum().item()
            print(f"[rank {rank}] POST-RESTORE graph replay OK (sum={result:.2f})", flush=True)
        except Exception as e:
            print(f"[rank {rank}] POST-RESTORE graph replay FAILED: {e}", flush=True)

        # Try an all-reduce after restore (comms should still be alive — no destroy)
        try:
            t = torch.ones(4, device=device)
            dist.all_reduce(t)
            print(f"[rank {rank}] POST-RESTORE all_reduce OK (sum={t[0].item()})", flush=True)
        except Exception as e:
            print(f"[rank {rank}] POST-RESTORE all_reduce FAILED: {e}", flush=True)
    else:
        print(f"[rank {rank}] C/R FAILED — skipping post-restore checks.", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {rank}] Done.", flush=True)


if __name__ == "__main__":
    main()
