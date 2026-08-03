# NVIDIA Meeting Brief — Multi-GPU C/R for GPU Time-Slicing

Prepared 2026-08-03. Context: NVIDIA has offered to support the time-slicing
project. This doc lists what to ask for, with the upstream issues/roadmap
items each ask attaches to.

## 30-second context to open with

We time-slice GPUs between workloads using externally-triggered
checkpoint/restore built on `cuda-checkpoint`. Today we have:

- Universal multi-GPU C/R for vLLM, SGLang, and FSDP (TP=2, DP=2, TP=2+DP=2,
  1-2 nodes) with **zero application changes** (LD_PRELOAD shim, external
  signals from our snapshot agent).
- **NVLink P2P at full speed during steady state** (shim v2: destroy NCCL
  comms pre-freeze, recreate with fresh rendezvous post-restore, handle
  indirection keeps it invisible to PyTorch/vLLM).
- A validation matrix we can re-run against any driver/NCCL drop in hours.

Remaining gaps are all on the NVIDIA side of the ioctl/library boundary.
Everything below is an ask attached to evidence.

---

## Ask 1 — Fix multicast (NVLS) in restored processes  [bug, minimal repro in hand]

**Our finding (driver 580.126.20, 2x H100 NV18):** after any
`cuda-checkpoint` restore, `cuMulticastCreate` succeeds but
`cuMulticastAddDevice` fails with `101 (invalid device ordinal)` for every
device. Verified:

- process-wide: also fails from a **freshly created CUDA context**;
- **not** state poisoning: fails even when the process never touched
  multicast before the checkpoint (`SKIP_PRE` variant);
- this is what breaks `ncclNvlsSetup` (`nvlsAllocateMem`, same error 101)
  when NCCL comms are recreated after restore — hence NVLS must be disabled
  for the entire lifetime of any checkpointable workload today.

**Artifact:** `mc_test.c` (~80 lines, pure CUDA driver API, no NCCL;
5-minute repro).

**Related upstream (different mechanisms, same blast radius):**
- [NCCL #2117](https://github.com/NVIDIA/nccl/issues/2117) — NVLS teardown
  silently swallows `cuMulticastUnbind` failures; stale bindings poison the
  checkpoint image. Distinct from our finding (ours reproduces with zero
  prior multicast activity) but same "NVLS × cuda-checkpoint" family.
- [NCCL #2077](https://github.com/NVIDIA/nccl/issues/2077) — NVLS multicast
  slot exhaustion is a fatal error instead of graceful fallback (2.29.7
  regression vs 2.28.9).

**Asks:**
1. Is post-restore `cuMulticastAddDevice` failure a bug or by-design? If
   bug: target driver release. (We'll file the repro; ask for routing to the
   right team.)
2. Until the driver fix: make `ncclNvlsSetup` failure **non-fatal** in comm
   init (graceful fallback to non-NVLS transport, as 2.28.9 did per #2077).
   Today it hard-fails the entire `ncclCommInitRank`, which forces
   `NCCL_NVLS_ENABLE=0` for the whole workload lifetime — with fallback,
   workloads keep NVLS until their first C/R and only the recreated comms
   drop to P2P.

---

## Ask 2 — Native multi-process / shared-state restore in the driver  [roadmap clarity]

**The gap:** `cuda-checkpoint` is per-process and cannot restore
cross-process GPU state: IPC imports, peer (P2P) mappings, SHM-backed
transport buffers. This is why *any* multi-GPU C/R today (ours and everyone
else's) must tear that state down before freeze. The driver already shows
this is being worked on:

- Driver **610** adds `cuIpcGetMemHandle`-based (legacy) CUDA IPC support +
  `--launch-job` job files ([README](https://github.com/NVIDIA/cuda-checkpoint)).
- README still lists `cuMemExportToShareableHandle()` IPC as unsupported —
  but that is exactly what modern NCCL (`NCCL_CUMEM_ENABLE`) and symmetric
  memory use.
- README: "For now, cuda-checkpoint must be invoked on the processes in a
  job sequentially" — "for now" implies coordinated job C/R is planned.

**Open upstream bugs in this family:**
- [cuda-checkpoint #27](https://github.com/NVIDIA/cuda-checkpoint/issues/27)
  — vLLM TP=2 restore fails, open since Apr 2025, no NVIDIA response.
- [#45](https://github.com/NVIDIA/cuda-checkpoint/issues/45) — toggle hangs
  checkpointing one process of a torch.distributed NCCL job.
- [#47](https://github.com/NVIDIA/cuda-checkpoint/issues/47) — restore
  "invalid argument". [#5](https://github.com/NVIDIA/cuda-checkpoint/issues/5)
  — NCCL restore segfault.
- [#53](https://github.com/NVIDIA/cuda-checkpoint/issues/53) — killing a
  restored process hard-hangs the host (595.71.05) — stability concern for
  our platform (we kill restored workloads routinely).
- [#55](https://github.com/NVIDIA/cuda-checkpoint/issues/55) /
  [#56](https://github.com/NVIDIA/cuda-checkpoint/issues/56) — `--launch-job`
  stability; whether IPC support strictly requires 610.

**Clarity to request:**
1. Scope + timeline of the 610 IPC support: legacy `cudaIpcGetMemHandle`
   only? When does **cuMem/fabric-handle IPC** (modern NCCL P2P, symmetric
   memory) land?
2. Is **coordinated multi-process job checkpoint** (atomic freeze of a
   process group, P2P/IPC mappings restored) on the driver roadmap, and
   what's the intended UX — job files? How should a Kubernetes operator
   (our snapshot agent DaemonSet) drive it?
3. Can we get early-access driver builds to validate against our matrix?
   (We are on 580.126.20 on GKE; also ask which GKE/driver channel picks up
   595/610.)

**What it unlocks for us:** if the driver restores shared mappings, the
destroy/recreate machinery in our shim shrinks to just quiesce+trigger —
and non-NCCL sharing (custom all-reduce, NVSHMEM) becomes checkpointable too.

---

## Ask 3 — NCCL "CUDA checkpoint" roadmap item  [exists! get it prioritized + shape it]

The [NCCL Q2 2026 roadmap (#2090)](https://github.com/NVIDIA/nccl/issues/2090)
lists, under **"Under consideration"**:

> "CUDA checkpoint: Enable whole-process CUDA checkpoint/restore so NCCL
> communicators, registered memory, and CUDA Graphs [with NCCL collectives
> remain usable after restore]" — goal: cut model cold-start time.

This is *exactly* our use case, and if it ships as described (comm handles
remain valid across restore) it deletes our entire shim for NCCL apps: no
destroy/recreate, no fresh-uniqueId rendezvous, no handle indirection, no
multi-node rendezvous problem.

Related shipped pieces: `ncclCommSuspend()/ncclCommResume()` (2.29.7,
"dynamic memory offload"; multi-GPU-per-process support slated for 2.30).
Our measurement: suspend does NOT tear down P2P/SHM/NVLS transport state, so
suspend-based C/R only works on TCP transport (50-100x collective slowdown)
— that's the gap between what shipped and what the roadmap item promises.

**Clarity to request:**
1. Status/priority of the "CUDA checkpoint" item — it's "under
   consideration" with no version target. Our production use case + ready
   validation matrix should be an argument to commit it. Timeline?
2. Design: suspend-in-place (comm survives) vs destroy/recreate under the
   hood? Which transports (P2P? SHM? NVLS — blocked on Ask 1?)? One process
   per GPU (torchrun/vLLM style) covered?
3. Interim: would they accept a `ncclCommSuspend` flag for **full transport
   teardown** (reduce comm to process-private state; resume re-bootstraps)?
   That alone removes our TCP restriction for suspend-based flows.
4. Offer: we'll be a design partner — we have the only(?) end-to-end
   external-trigger multi-GPU C/R validation matrix (vLLM/SGLang/FSDP ×
   TP/DP × NVLink/TCP) and a working destroy/recreate reference
   implementation to compare against.

**Ask 2 vs Ask 3 (why both):** Ask 2 is the *driver* restoring shared state
— generic, covers non-NCCL sharing, heavyweight. Ask 3 is *NCCL* removing
and rebuilding its own state around the window — NCCL-only but much cheaper
to ship, and already on their roadmap. Either one eliminates our biggest
maintenance burden; both together cover everything.

---

## Ask 4 — CUDA graphs across C/R  [clarify, then retest]

Upstream evidence says plain captured graphs **survive** cuda-checkpoint
(address-preserving restore keeps embedded device pointers valid — per the
[vLLM C/R RFC #34303](https://github.com/vllm-project/vllm/issues/34303) and
[Modal's production GPU snapshots](https://modal.com/blog/gpu-mem-snapshots),
this is one of the biggest cold-start wins). The real multi-GPU catch:
**graphs that embed NCCL collectives** reference comm handles — after any
comm rebuild (our destroy/recreate, or vLLM's own reinit) those are stale
and replay is undefined. The NCCL roadmap item (Ask 3) explicitly targets
keeping such graphs usable.

**Actions/asks:**
1. (Us, before the meeting if possible:) retest single-GPU C/R with CUDA
   graphs ON — our blanket `--enforce-eager` requirement predates this
   evidence and may be over-conservative for TP=1.
2. (NVIDIA:) confirm graphs-with-collectives is in scope of the NCCL
   checkpoint item; if comm rebuild remains the model, is there a sanctioned
   invalidate-and-recapture hook frameworks should implement?

---

## Ecosystem context worth mentioning

- [vLLM RFC #34303](https://github.com/vllm-project/vllm/issues/34303):
  vLLM upstream is building cuda-checkpoint cold-start support, also using
  destroy + reinit (~1-3s) for NCCL. Our shim v2 architecture matches where
  the ecosystem is converging — NVIDIA fixing Asks 1-3 accelerates everyone.
- [CRIUgpu paper](https://arxiv.org/html/2502.16631v1): "cuda-checkpoint
  does not support NCCL, functionality expected in a future CUDA driver
  release" — corroborates that NCCL-aware C/R is planned driver-side.

## Priority summary

| # | Ask | Type | Upstream anchor | Unlocks |
|---|-----|------|-----------------|---------|
| 1 | Multicast restore fix + NVLS graceful fallback | Bug w/ repro (`mc_test.c`) | NCCL #2117, #2077 | NVLS through C/R (~10-30% on 8-GPU all-reduce) |
| 2 | Multi-process / cuMem-IPC restore, job UX, timeline | Roadmap clarity | cc #27/#45/#47/#5/#53; driver 610 IPC | Simpler shim; non-NCCL workloads; K8s UX |
| 3 | NCCL "CUDA checkpoint" item: commit + design partner | Roadmap commitment | NCCL #2090 | Deletes our shim for NCCL apps |
| 4 | Graphs w/ collectives scope; recapture contract | Clarification | vLLM #34303 | Removes `--enforce-eager` tax |

Relationship asks regardless of the above: named engineering contacts for
cuda-checkpoint + NCCL teams; early-access driver builds; we contribute our
regression matrix as their multi-GPU C/R test bed.
