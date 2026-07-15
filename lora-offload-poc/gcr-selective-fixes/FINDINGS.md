# Selective C/R Corruption — Root Cause & Fixes

Debugging results for the "corruption after restore" seen with the
`Edwinhr716/GPU-CR@memory-allocation` branch driven by
`testing-artifacts/validate_selective_cr.py`. Verified on a 2xH100
(a3-highgpu-2g) GKE node, CUDA 12.4, torch 2.5.1.

**Result: with the two patches in `patches/` applied, selective checkpoint /
restore passes both a pure-CUDA test and a PyTorch test on real hardware —
data intact, `data_ptr()` stable, VRAM measurably freed during eviction, and
the process fully usable afterwards.**

---

## Bug 1 — sticky CUDA error poisons the host app (the reported "corruption")

This is the failure the validate script actually hits, and it happens on
**`cr_client -i` alone** — no eviction required.

During `init_CR()` (which runs inside the GCR signal handler **on the host
app's own thread**), GCR calls `cudaHostRegister` on the 2GB staging buffer.
On file-backed/hugepage-backed buffers this fails with `invalid argument`,
which GCR deliberately tolerates ("continuing without pinned memory")… but the
failure leaves CUDA's **sticky per-thread last-error** set on PyTorch's main
thread. GCR never clears it.

The next time torch launches ANY kernel, its `C10_CUDA_KERNEL_LAUNCH_CHECK`
calls `cudaGetLastError()`, sees the stale `invalid argument`, and throws:

```
RuntimeError: CUDA error: invalid argument
```

The snapshot/restore RPCs themselves succeed — the app then falls over on its
next kernel. Hence "C/R works but something goes wrong after restore."

**Fix** (`patches/gcr-selective-fixes.patch`):
- `nv::registerHostMemory`: call `cudaGetLastError()` to clear the sticky
  error after the tolerated `cudaHostRegister` failure.
- `cr_signal_handler`: clear the last-error before sending `FINISH_MSG` (both
  the init path and the ckpt/restore path), so no C/R-internal error ever
  leaks into the host app's error-checking.

Why the C++ smoke test never caught it: it checks explicit return codes and
never consults `cudaGetLastError()` — only PyTorch-style error checking trips.

## Bug 2 — save/release/remap size mismatch (latent VA corruption)

`validate_selective_cr.py` passes `tensor.data_ptr()` and the **tensor's**
byte size (4MB). But PyTorch's caching allocator services a 4MB tensor from a
**20MB cudaMalloc segment**, so:

- `ckpt_selective` saved only the caller's 4MB,
- `releasePhysicalMemory()` unmapped the **entire 20MB** allocation (it can
  only release whole cuMemMap ranges — it looks up the true size in
  `allocated_memory`),
- restore remapped only the first 4MB → **16MB of the segment left permanently
  unmapped**, destroying everything else living in it (including the
  `original_data` clone the test compares against, which the allocator placed
  in the same segment).

**Fix** (same patch): `ckpt_selective` now normalizes the region size to the
full allocation size from `allocated_memory` (with a NOTE log when the caller
size differs), so save, release, and remap all cover the same range.

## Bug 3 — the validate script's design conflicts with eviction semantics

Two problems in `validate_selective_cr.py` itself:

1. **It writes to evicted memory**: `tensor.fill_(0.0)` runs between snapshot
   and restore, when the tensor's physical pages are released. That's an
   illegal access by design. The eviction contract is "don't touch until
   restore" — exactly what OpenRL guarantees for idle adapters.
2. **Caching-allocator fragility**: `data_ptr()` is only a valid region key if
   it's exactly a `cudaMalloc` base. Run tests with
   `PYTORCH_NO_CUDA_MEMORY_CACHING=1` (1:1 tensor↔cudaMalloc mapping), or, for
   the real LoRA path, allocate adapter tensors via dedicated VMM allocation
   outside the caching allocator.

`tests/torch_selective_test.py` is the corrected validation: reference copy on
the host, no writes to the evicted tensor, eviction verified via
`cudaMemGetInfo`, plus a compute check on *other* tensors during eviction and
on the restored tensor afterwards.

## Test results (2xH100 GKE node)

| Test | Result |
|------|--------|
| `tests/selective_test.cpp` — 3 buffers, evict 1, verify others during eviction, restore, verify all + rewrite | **PASS** (VRAM 551→543MB during eviction) |
| `tests/torch_selective_test.py` — 16MB torch tensor, evict, compute on other tensors during eviction, restore, verify data + `data_ptr()` + compute | **PASS** (free +16MB during eviction, data_ptr unchanged) |

Before the sticky-error fix, the torch test reproduced Edwin's exact failure
(`CUDA error: invalid argument` on the first kernel launch after the handler
ran) — confirmed it triggers with `cr_client -i` alone.

## Review notes on the memory-allocation branch

- The `pushContext`/`popContext` + PyTorch-context capture additions are
  harmless but were not needed to make selective C/R work in these tests.
- The `cudaLaunchKernel`/`cuLaunchKernel` LD_PRELOAD debug hooks log on every
  kernel launch — significant overhead; recommend removing now that the real
  cause is found.
- The reordering of the **full** (non-selective) restore path in
  `cr_client.cpp` (data-restore signal before `cuda-checkpoint --toggle`) will
  break full C/R — the data-restore handler needs live CUDA. Recommend
  reverting that hunk; it isn't needed for the selective path.

## How to apply

```bash
git clone -b memory-allocation https://github.com/Edwinhr716/GPU-CR.git
cd GPU-CR
git apply /path/to/patches/gcr-selective-fixes.patch
```
