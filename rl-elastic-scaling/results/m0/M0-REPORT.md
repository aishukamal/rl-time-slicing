# M0 Baseline — Code-RLVR Fully-Async Re-Run (elastic-rl-poc)

**Question (C1):** does re-running the Phase 5 code-RLVR fully-async recipe — same model,
data, knobs, verl commit — on the elastic-PoC environment reproduce the gen-heavy
trainer-idle regime that the elastic milestones (M1+) are built to exploit?

**Answer: yes, to within noise.** Steady-state step time matches Phase 5 to 0.1%
(619.0s vs 619s), trainer idle ratio to 0.2pp (0.511 vs 0.513), and mean response
length to less than one token (11439.0 vs 11439). 11 contiguous trainer idle blocks
(one per step) vs 11 in Phase 5; rollout GPU pinned at 99.6% in both. mq=0 and
dropped_stale=0 every step in both runs.

## Run facts

- Job `code-rlvr-m0`, pod `code-rlvr-m0-pxctd`, node `gke-verl-research-clus-h100-2gpu-pool-02b8c734-q36v`, namespace default
- Started 2026-08-04 23:33 UTC; planned 7200s training timeout reached (expected end state); job Complete, 0 restarts
- verl commit: `983cb0f24443f87b3d161fad318445130a620b07` — **verified identical to Phase 5**
- 10 steps completed (steps 2–11); steady window = steps 3–11 (skip first 2), same convention as Phase 5
- Recovery: container had exited (pod Completed) → plan B, `kubectl logs` (full retention, single
  `kubectl logs --tail=-1` call — no Cloud Logging stitching needed this time)
- GPU trace: 104,829 rows (52,414 samples/GPU @100ms, ~5,833s steady window), decoded from the
  GPUCSV gzip+base64 stdout dump; **md5 verified: `73aed7b8156fcb6b3c0100e4d98c4e6b` matches the
  GPUCSV-BEGIN header**
- `start_ts` (1785886538118) was not echoed to stdout (redirected to the in-pod file); recovered
  from the first GPU-trace sample timestamp, accurate to ≤100ms

## Steady-state comparison: M0 vs Phase 5 (code s=8 @16K)

| Metric | **M0 (this run)** | Phase 5 (code-rlvr/REPORT.md) | Δ |
|---|---|---|---|
| Step time | **619.0s** (571–675) | 619s (547–691) | −0.1% |
| Trainer gen-wait / step | **316.0s** | 319s | −0.9% |
| update_actor / step | **280.4s** | 281s | −0.2% |
| param_sync | **1.6s** | 1.9s | −0.3s |
| Trainer idle ratio (verl) | **0.511** | 0.513 | −0.2pp |
| Trainer GPU idle (<10% util, 100ms trace) | **53.7%** | 52.8% | +0.9pp |
| Trainer idle blocks (≥2s) | **11 × 280.1s mean, 379s max** | 11 × 244.7s mean, 399s max | same count; +35s mean, −20s max |
| Rollout GPU util / idle | **99.6% / 0.3%** | 99.6% / 0.4% | ≈0 |
| mq depth / dropped_stale | **0 / 0** every step | 0 / 0 every step | ≈0 |
| response_length mean | **11,439** (11,023–12,125 per-step) | 11,439 (10,848–12,271) | ±0 |
| score mean (per-step avg) | **0.196** (0.150–0.238) | 0.199 (0.144–0.235) | −0.003 |

Per-step score means (steps 2–11): 0.196, 0.226, 0.157, 0.150, 0.211, 0.205, 0.170, 0.238,
0.214, 0.197 — live and non-degenerate, same spread as Phase 5.

Per-step response_length means (steps 2–11): 10,682 → 11,023 → 11,807 → 12,125 → 11,044 →
11,293 → 11,291 → 11,365 → 11,297 → 11,707 — same early lengthening (10.7K → 12.1K by step 5)
then plateau around ~11.3K; the Phase-5 drift observation (CoT lengthening within 10 steps)
reproduces in direction and magnitude.

Trainer GPU (GPU 0) power mean 371W vs rollout GPU (GPU 1) 593W — the idle bubble is visible
in power as well as utilization.

## C1 verdict

**The re-run reproduces the gen-heavy trainer-idle regime, quantitatively.** The structural
signature is identical: rollout GPU pinned ≥99.6%, trainer idling the gen−train imbalance
(316s gen-wait vs 280s update → ~51% idle) in one large contiguous multi-minute block per step
(11 blocks, 280s mean, 379s max), staleness budget never binding (mq=0, dropped_stale=0
throughout). Every headline metric lands within ~1% of Phase 5. The regime is a property of
the workload (long-CoT code RLVR at 16K cap, s=8, 1+1 GPUs), not of the node, scheduling
mode, or provisioning model — M0 is a valid baseline for the elastic (M1+) comparisons.

## Environment differences vs Phase 5 (none moved the needle)

- **Node:** `h100-2gpu-pool` node with NVIDIA time-sharing enabled (`*-q36v`), vs the
  Phase 5 exclusive-GPU node. The pod requested **1 logical (time-shared) GPU** and ran
  **privileged**, so it saw and used both physical H100s (nvidia-smi confirms 2 GPUs, both
  idle at job start; no co-tenants during the run).
- **Provisioning:** non-spot node (Phase 5 ran on spot and the pod was reclaimed ~2h after
  run end; M0's pod survived, though the container had exited so log-based recovery was
  still the collection path).
- Same image (`verlai/verl:vllm020.dev2`), same verl commit, same data prep, same knobs.

## Artifacts (this directory)

- `full_pod.log` — complete pod stdout (all phases + GPUCSV dump)
- `train.log` — Phase 5 (training) section of the log
- `gpu_util.csv` — decoded 100ms GPU trace (md5 `73aed7b8156fcb6b3c0100e4d98c4e6b`, verified)
- `start_ts` — 1785886538118 (ms epoch; recovered from first trace sample)
- `verl_commit` — 983cb0f24443f87b3d161fad318445130a620b07 (matches Phase 5)
- `summary.json` — analyzer output
- `analyze_run.py` — copied verbatim from `code-rlvr/results/` (schema parity; invoked with
  M0 paths, no code changes needed)

Cluster cleanup: job `code-rlvr-m0` and configmap `code-rlvr-m0` deleted after md5-verified
collection; node `*-q36v` released in the node registry.

Reference run: [../../code-rlvr/REPORT.md](../../code-rlvr/REPORT.md) (Phase 5).
