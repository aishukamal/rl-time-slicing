# Next Steps — Applying the Selective C/R Fixes

The selective C/R corruption is root-caused and fixed; both a pure-CUDA test
and a PyTorch test pass on a 2xH100 GKE node with the changes below. Full
analysis in [FINDINGS.md](FINDINGS.md).

**What's in THIS repo** (see the "apply selective C/R fixes" commit diff on
`lora-offload-poc/`): the two GCR code fixes applied to the parked baseline
(`vGPU.cpp`, `nv.cpp`), the same fixes as a `git apply`-able patch in
[patches/gcr-selective-fixes.patch](patches/gcr-selective-fixes.patch), and
the corrected test suite in [tests/](tests/).

**What is NOT in this repo** — changes you need to make in YOUR repos:

## 1. `Edwinhr716/GPU-CR` branch `memory-allocation`

### Apply the patch (2 files, 3 fixes)
```bash
cd GPU-CR   # on memory-allocation branch
git apply gcr-selective-fixes.patch
```
This gives you:
- `src/GPUs/NVIDIA/nv.cpp` — clear the sticky CUDA error after the tolerated
  `cudaHostRegister` failure in `registerHostMemory()`. **This is the direct
  cause of your `CUDA error: invalid argument`** — the signal handler runs on
  the app's own thread, and PyTorch's next kernel-launch check picks up the
  stale error. It triggers on `cr_client -i` alone.
- `src/vGPU.cpp` — clear the sticky error at both signal-handler exits
  (portable `GCR_CLEAR_LAST_ERROR` macro, works for the AMD build too).
- `src/vGPU.cpp` — `ckpt_selective()` now uses the FULL allocation size from
  `allocated_memory[ptr]` instead of the caller-passed size. Release/remap
  operate on whole allocations, so saving less than the allocation left VA
  unmapped after restore (with PyTorch's caching allocator your 4MB tensor
  lives in a 20MB segment → 16MB got destroyed, including the `original_data`
  clone your test compares against).

### Revert your cr_client full-restore reordering
In `coordinator/cr_client.cpp` you moved `RESTORE_MSG` + `kill()` BEFORE the
`cuda-checkpoint --toggle` call in the full (non-selective) restore path. The
data-restore handler makes CUDA calls, which can't run while CUDA state is
still frozen — this breaks full C/R. Restore the original order (toggle
first, then `RESTORE_MSG`). The selective path is unaffected either way.

### Remove the kernel-launch debug hooks (recommended)
The `cudaLaunchKernel` / `cuLaunchKernel` / `_ptsz` hooks in
`src/ipc_hooks.cpp` fprintf on EVERY kernel launch — that's a large overhead
in real workloads, and they were only debugging aids. The real cause is found;
safe to delete. (Your `pushContext`/`popContext` and `cudaMalloc(0)` additions
are harmless — keep or drop as you like.)

## 2. Your `llm-d-rl-time-slicing` fork — `testing-artifacts/validate_selective_cr.py`

Replace it with [tests/torch_selective_test.py](tests/torch_selective_test.py)
(passed on H100), or make these three changes to yours:

1. **Never touch the evicted tensor between snapshot and restore.**
   `tensor.fill_(0.0)` writes to memory whose physical pages are released —
   illegal by the eviction contract (OpenRL guarantees idle adapters aren't
   touched). Verify eviction via `torch.cuda.mem_get_info()` delta instead.
2. **Keep the reference copy on the host** — `tensor.cpu().clone()`, not a GPU
   clone. Your GPU clone landed in the same 20MB caching-allocator segment as
   the target tensor.
3. **Run with `PYTORCH_NO_CUDA_MEMORY_CACHING=1`** so every tensor is a direct
   (hooked) cudaMalloc — then `data_ptr()` really is an allocation base and
   the size matches. Without it the region lookup silently misses or hits a
   segment (see FINDINGS.md Bug 2). For the real LoRA path, adapter tensors
   should be allocated via dedicated VMM allocation outside the caching
   allocator, which avoids this class of issue entirely.

## 3. Runtime environment reminders (deployment side)

- `GPU_VENDOR=NVIDIA` must be set in the WORKLOAD container (vGPU.so's GPU
  factory exits without it — your deployment.yaml already has it; standalone
  runs need it too).
- The full driver sequence that passed on H100 is in
  [tests/run_tests.sh](tests/run_tests.sh) (uses `EXPORT_FILE_PATH`, no
  hugepages needed for functional testing).
