# Corrected PyTorch selective C/R validation.
#
# Fixes vs validate_selective_cr.py:
#   1. Run with PYTORCH_NO_CUDA_MEMORY_CACHING=1 so every tensor allocation is
#      a direct (hooked) cudaMalloc — data_ptr() is then a real allocation base.
#   2. The reference copy lives on the HOST (not in a GPU segment that could be
#      evicted alongside the target tensor).
#   3. The evicted tensor is NOT touched between snapshot and restore — that is
#      the eviction contract. Eviction is verified via cudaMemGetInfo instead.
#
# Protocol via files in $TEST_DIR (same as selective_test.cpp):
#   writes  ready         "<pid> <ptr> <size>"
#   waits   ckpt_done
#   writes  midcheck      free-memory delta info
#   waits   restore_done
#   writes  result        PASS/FAIL

import os
import sys
import time

assert os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1", \
    "run with PYTORCH_NO_CUDA_MEMORY_CACHING=1"

import torch

TEST_DIR = os.environ.get("TEST_DIR", "/tmp/seltest-torch")
os.makedirs(TEST_DIR, exist_ok=True)


def write_file(name, content):
    with open(os.path.join(TEST_DIR, name), "w") as f:
        f.write(content)


def wait_file(name):
    path = os.path.join(TEST_DIR, name)
    print(f"[torch-test] waiting for {path}", flush=True)
    while not os.path.exists(path):
        time.sleep(0.2)
    print(f"[torch-test] {path} appeared", flush=True)


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)

    device = torch.device("cuda")

    # 16MB tensor: big enough to see in memory stats, exact multiple of 2MB.
    n = 4 * 1024 * 1024
    tensor = torch.arange(n, dtype=torch.float32, device=device)
    torch.cuda.synchronize()

    addr = tensor.data_ptr()
    size_bytes = tensor.element_size() * tensor.nelement()
    reference = tensor.cpu().clone()  # host copy — NOT in GPU memory

    free0, _ = torch.cuda.mem_get_info()
    print(f"[torch-test] tensor at {hex(addr)} size={size_bytes} pid={os.getpid()} "
          f"free={free0 >> 20}MB", flush=True)
    write_file("ready", f"{os.getpid()} {hex(addr)} {size_bytes}\n")

    # --- driver runs: cr_client -i, then cr_client -c -s <addr>:<size> ---
    wait_file("ckpt_done")

    # Do NOT touch `tensor` here — its physical pages are released.
    # Verify eviction happened via free-memory delta.
    free1, _ = torch.cuda.mem_get_info()
    delta_mb = (free1 - free0) >> 20
    print(f"[torch-test] during evict: free={free1 >> 20}MB (delta {delta_mb:+d}MB)", flush=True)

    # Allocate/compute on OTHER memory while the tensor is evicted — the rest
    # of the CUDA context must remain fully functional.
    other = torch.ones(1024 * 1024, device=device)
    other_sum = other.sum().item()
    ok_other = abs(other_sum - 1024 * 1024) < 1
    del other
    write_file("midcheck", f"free_delta_mb={delta_mb} other_compute_ok={ok_other}\n")

    # --- driver runs: cr_client -r -s <addr>:<size> ---
    wait_file("restore_done")

    # Pointer stability: same tensor object, same data_ptr, data intact.
    assert tensor.data_ptr() == addr, "data_ptr changed after restore!"
    restored = tensor.cpu()
    if torch.equal(restored, reference):
        # Verify the tensor is fully usable: autograd-style compute on it.
        s = (tensor * 2).sum().item()
        expect = float(n) * (n - 1)  # 2 * sum(arange(n)) = n*(n-1)
        if abs(s - expect) / expect < 1e-6:
            print("[torch-test] PASS", flush=True)
            write_file("result", "PASS\n")
            return
        write_file("result", f"FAIL compute-after-restore: got {s} expect {expect}\n")
        sys.exit(1)

    bad = (restored != reference).nonzero()
    write_file("result",
               f"FAIL data mismatch: {bad.numel()} bad elems, first at {bad[0].item() if bad.numel() else '?'}\n")
    sys.exit(1)


if __name__ == "__main__":
    main()
