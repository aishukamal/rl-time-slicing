# Selective C/R Changes — Per-File Breakdown

All changes are relative to the upstream `main` branches of:
- `gpu-os/GPU-CR` (GCR C++ core)
- `llm-d-incubation/llm-d-rl-time-slicing` (Snapshot Agent)

To see the raw diff: `git diff 16730ca..ebc1445`

---

## Part 1: GCR C++ Changes

### `gcr/src/common.h`

**What changed:** Added three new types and extended `signal_controls`.

| Addition | Purpose |
|----------|---------|
| `selective_cr_region` struct | Holds a single `{void* ptr, uint64_t size}` pair identifying a GPU memory region |
| `selective_cr_request` struct | Array of up to `MAX_SELECTIVE_REGIONS` (4096) regions plus a `num_regions` count |
| `signal_controls.selective_req` field | Carries the selective request inline in the existing PID-keyed shared memory control page (the struct is mmap'd to a HUGE_PAGE_SIZE / 2MB region, so there's ample space) |

**Why:** The full C/R path iterates the global `allocated_memory` map. Selective mode needs the caller to specify which subset of pointers to checkpoint/restore. The request travels through the same shared memory channel (`signal_controls`) that already carries the message type (`signal` field).

---

### `gcr/src/comm/comm.h`

**What changed:** Added two new message type constants.

```c
#define SELECTIVE_CKPT_MSG    20
#define SELECTIVE_RESTORE_MSG 21
```

**Why:** The signal handler in `vGPU.cpp` dispatches on the message type. New messages are needed so the handler knows to read the `selective_req` from shared memory and call the selective functions instead of the full `ckpt()` / `restore_ptr_and_content()`.

---

### `gcr/src/vGPU.cpp`

**What changed:** Three modifications.

#### 1. New function: `ckpt_selective(const selective_cr_request* req)` (~120 lines)

Mirrors the existing `ckpt()` but with these differences:

| `ckpt()` (full) | `ckpt_selective()` (selective) |
|-----------------|-------------------------------|
| Iterates `allocated_memory` (all tracked GPU allocations) | Iterates `req->regions[]` (caller-specified subset) |
| No validation needed (map is authoritative) | Validates each pointer exists in `allocated_memory` before proceeding; skips unknown pointers with a warning |
| After data copy, releases ALL physical pages | After data copy, releases ONLY the checkpointed regions' physical pages |

The D→H copy pipeline is identical: async memcpy to staging buffers, double-buffered flush to `tmp_buf` (hugepage shared memory), same `shared_mem_fs` metadata format.

`releasePhysicalMemory()` (cuMemUnmap + cuMemRelease) is called per-region — this frees the physical VRAM pages while keeping the virtual address reserved. This is the pointer-stability guarantee.

#### 2. New function: `restore_ptr_and_content_selective()` (~80 lines)

Mirrors `restore_ptr_and_content()`. Reads the `shared_mem_fs` metadata from `tmp_buf` to know which pointers/sizes/offsets were stored, then:
1. Calls `remapPhysicalMemory(ptr, size)` for each region (cuMemCreate + cuMemMap at the same VA)
2. Copies data back H→D using the same staging buffer pipeline

No structural differences from the full restore — the metadata already records exactly what was stored, so the restore path is the same regardless of whether the checkpoint was full or selective.

#### 3. Updated `cr_signal_handler()` — new dispatch branches

Added two `else if` cases before the existing `CKPT_MSG` / `RESTORE_MSG` handlers:

- **`SELECTIVE_CKPT_MSG`**: Casts `comm` to `ShareMemComm*` to read `control->selective_req`, calls `gpu->syncAllKernels()`, then `ckpt_selective(&req)`. No P2P disable, no IPC teardown — the process stays live.
- **`SELECTIVE_RESTORE_MSG`**: Calls `restore_ptr_and_content_selective()`. No P2P re-enable, no IPC rebuild.

Both send `FINISH_MSG` when done, same as the full C/R path.

---

### `gcr/coordinator/cr_client.cpp`

**What changed:** Four modifications.

#### 1. Added `#include <vector>` and `parse_selective_regions()` function

Parses a comma-separated string of `ptr:size` pairs (e.g., `0x7f0001000000:4194304,0x7f0002000000:2097152`) into a `selective_cr_request` struct. Supports hex and decimal for both pointer and size values via `strtoull(..., 0)`.

#### 2. Added `-s` option to `getopt`

New command-line flag: `-s <ptr:size,ptr:size,...>`

#### 3. Changed `Comm*` to `ShareMemComm*`

The `comm` variable was previously typed as `Comm*` (base class). Changed to `ShareMemComm*` so we can access `comm->control->selective_req` to write the selective request into shared memory.

#### 4. Added selective checkpoint/restore branches

Two new `else if` branches execute before the existing `ckpt` and `restore` branches:

- **`ckpt && selective_spec`**: Parses regions, writes `selective_cr_request` to `comm->control->selective_req`, sends `SELECTIVE_CKPT_MSG` + signal, waits for `FINISH_MSG`. No `cuda-checkpoint` call.
- **`restore && selective_spec`**: Same flow with `SELECTIVE_RESTORE_MSG`. No `cuda-checkpoint` call.

Usage:
```
cr_client -c -s 0x7f0001000000:4194304,0x7f0002000000:2097152 -p 12345
cr_client -r -s 0x7f0001000000:4194304,0x7f0002000000:2097152 -p 12345
```

---

## Part 2: Snapshot Agent Changes

### `snapshot-agent/backends/checkpoint.go`

**What changed:** Added three new exported types.

| Addition | Purpose |
|----------|---------|
| `BackendGCR BackendType = "gcr"` | New backend type constant for the GCR backend |
| `MemoryRegion` struct | `{Address uint64, Size uint64}` — Go representation of a GPU memory region |
| `SelectiveBackend` interface | Extends `Backend` with `SelectiveSnapshot(ctx, pid, regions)` and `SelectiveRestore(ctx, pid, regions)` methods |

**Why:** The existing `Backend` interface only takes PIDs. Selective mode needs memory regions too. Rather than breaking the existing interface, `SelectiveBackend` is a superset — backends that support selective mode implement both interfaces, and the server uses a type assertion to check.

---

### `snapshot-agent/backends/gcr.go` (NEW FILE)

Implements `Backend` and `SelectiveBackend` using `cr_client` subprocess calls.

| Method | cr_client invocation |
|--------|---------------------|
| `Snapshot(ctx, pids)` | `cr_client -c -b -p <pid>` per PID (full, buffer-only) |
| `Restore(ctx, pids)` | `cr_client -r -b -p <pid>` per PID |
| `SelectiveSnapshot(ctx, pid, regions)` | `cr_client -c -s 0xaddr:size,... -p <pid>` |
| `SelectiveRestore(ctx, pid, regions)` | `cr_client -r -s 0xaddr:size,... -p <pid>` |
| `HealthCheck(ctx)` | Checks `cr_client` is on PATH |

Follows the same patterns as `CudaCheckpoint`: injectable `execCommand` and `lookPath` for testing, `sync.Mutex` for serialization.

---

### `snapshot-agent/backends/gcr_test.go` (NEW FILE)

Unit tests for all `GCRBackend` methods with mocked `execCommand`. Tests cover:
- Success/failure for `Snapshot`, `Restore`, `SelectiveSnapshot`, `SelectiveRestore`, `HealthCheck`
- Empty PID/region validation
- Correct `-s` flag formatting (verifies region args contain the expected `0x...` addresses)
- Interface compliance: `var _ backends.SelectiveBackend = g`

---

### `snapshot-agent/backends/export_test.go`

**What changed:** Added `SetExecCommand` and `SetLookPath` test helpers for `GCRBackend`, same pattern as the existing `CudaCheckpoint` helpers.

---

### `snapshot-agent/api/v1alpha1/snapshot_agent.proto`

**What changed:** Three additions.

| Addition | Purpose |
|----------|---------|
| `BACKEND_GCR = 2` in `Backend` enum | Allows clients to request the GCR backend |
| `MemoryRegion` message | `{uint64 address, uint64 size}` — proto representation of a GPU memory region |
| `repeated MemoryRegion regions = 4` on `SnapshotRequest` and `RestoreRequest` | Optional field; when populated, triggers selective mode |

**Note:** The generated `.pb.go` files need regeneration (`make proto`) after this change.

---

### `snapshot-agent/server/server.go`

**What changed:** Three modifications.

#### 1. `getBackendType()` — added GCR case

```go
case pb.Backend_BACKEND_GCR:
    return backends.BackendGCR
```

#### 2. New helper: `protoRegionsToBackend()`

Converts `[]*pb.MemoryRegion` to `[]backends.MemoryRegion`.

#### 3. `Snapshot()` and `Restore()` handlers — selective dispatch

Both handlers now check `req.GetRegions()`. If regions are provided:
1. Type-assert the resolved backend to `backends.SelectiveBackend`
2. If the assertion fails, return an error ("backend X does not support selective snapshot")
3. If it succeeds, call `SelectiveSnapshot` / `SelectiveRestore` per PID instead of the bulk `Snapshot` / `Restore`

If no regions are provided, the existing full C/R path is used unchanged.
