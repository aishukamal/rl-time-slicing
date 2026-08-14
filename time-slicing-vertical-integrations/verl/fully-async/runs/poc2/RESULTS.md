# POC2 Time-Slicing Rerun -- Final Results

GPU time-slicing PoC: two async-RL training jobs (code-RLVR) sharing a single
trainer GPU via cuda-checkpoint C/R, compared against a solo baseline.

**Topology**

| Config | GPUs | Assignment |
|--------|------|------------|
| Timeslice | 3 | GPU0 = rollout-a, GPU1 = rollout-b, GPU2 = shared trainer |
| Baseline | 2 | GPU0 = rollout, GPU1 = trainer |

**Hardware**: H100 80 GB (gke-verl-research-cl, mega-8gpu-spot nodes)
**Framework**: veRL fully-async GRPO, DeepSeek-R1-Distill-Qwen-1.5B
**Data**: timeslice from `poc2_ts_rerun/` (real agent logs), baseline from `poc2_final/baseline_v2/`

---

## KPI Table

### Step Count

| Metric | Timeslice | Baseline |
|--------|-----------|----------|
| Job-a steps | 13 | -- |
| Job-b steps | 9 | -- |
| **Total steps** | **22** | **10** |

### Step Time (steady-state, first 2 warmup steps excluded)

| Metric | Baseline | TS Job-a | TS Job-b |
|--------|----------|----------|----------|
| Mean step time (s) | 611.1 | 470.0 | 604.3 |
| Median step time (s) | 627.2 | 554.0 | 570.2 |
| N (steady steps) | 8 | 11 | 7 |

### Train Time (update_actor only, steady-state)

| Metric | Baseline | TS Job-a | TS Job-b |
|--------|----------|----------|----------|
| Mean (s) | 281.6 | 188.7 | 286.3 |
| Median (s) | 281.4 | 188.6 | 280.6 |

Job-a trains ~33% faster than baseline because it receives samples while job-b
holds the trainer lock, reducing queue starvation.  Job-b trains at near-baseline
speed.

---

## REAL Swap Durations (from 51 cuda-checkpoint operations)

Enriched from snapshot-agent.log: 24 snapshot ops, 23 restore ops matched to
rl_metrics events.

| Operation | Median | Mean | Min | Max | N |
|-----------|--------|------|-----|-----|---|
| **Restore** (cuda-checkpoint toggle) | **9,168 ms** | 8,618 ms | 4,760 ms | 11,252 ms | 23 |
| **Snapshot** (cuda-checkpoint action) | **21,175 ms** | 20,511 ms | 12,378 ms | 24,200 ms | 27 |
| **Total swap** (median restore+snapshot) | **30,343 ms** | -- | -- | -- | -- |

- Early swaps (steps 0-1) are faster: restore 4.8-6.9 s, snapshot 12.4-13.9 s
- Steady-state swaps stabilize: restore ~9 s, snapshot ~21 s
- Snapshot is ~2.3x slower than restore (asymmetric: full GPU memory dump vs page-fault restore)

---

## Trainer Duty Cycle

| Metric | Value |
|--------|-------|
| Baseline solo trainer GPU | 41.6% |
| **Timeslice shared trainer GPU (steady-state)** | **86.6%** |
| **Improvement** | **+45.0 pp (2.08x)** |

The shared trainer GPU achieves 86.6% utilization during steady-state --
more than double the solo baseline -- by interleaving training from two jobs.

---

## Lock Hold Time (trainer GPU occupancy per acquire/yield cycle)

| Job | Mean hold (s) | Median hold (s) | N |
|-----|---------------|------------------|---|
| Job-a | 200.2 | 198.4 | 14 |
| Job-b | 303.6 | 300.7 | 10 |

Job-b holds the trainer lock ~50% longer than job-a because its batches are
larger / training takes more wall time per step.

---

## Heterogeneous Cadence Analysis

The two jobs exhibit different step cadences because they generate rollout
samples at different rates:

- **Job-a mean step**: 470.0 s (steady-state)
- **Job-b mean step**: 604.3 s (steady-state)
- **Cadence ratio** (b/a): 1.29x
- **Step ratio**: job-a completes 13 steps while job-b completes 9 (1.4x)

Three of job-a's steps (4, 10, 14) have gen_ms < 10 s -- these are steps where
rollout samples were already queued from the previous iteration, so the step
comprises almost entirely training with no rollout wait.  These "fast steps"
bring step time down to ~190 s (vs ~550 s for normal steps).

---

## Sample-Queue-Delay Analysis

The `wait_ms` at `samples_ready` captures how long a job waited for the trainer
lock.  This breaks down into **lock contention** (waiting for the other job to
release) plus **restore time** (cuda-checkpoint restore after acquiring).

### Per-step breakdown

| Job | Step | wait_ms | restore_ms | contention_ms |
|-----|------|---------|------------|---------------|
| job-a | 0 | 23,000 | 6,944 | 16,056 |
| job-b | 0 | 33,001 | 6,261 | 26,740 |
| job-a | 1 | 232,000 | 9,162 | 222,838 |
| job-a | 2 | 1,000 | 0 | 1,000 |
| job-b | 1 | 197,000 | 11,252 | 185,748 |
| job-a | 3 | 246,001 | 7,997 | 238,004 |
| job-b | 2 | 82,000 | 9,333 | 72,667 |
| job-a | 4 | 371,000 | 9,225 | 361,775 |
| job-b | 3 | 62,000 | 9,289 | 52,711 |
| job-a | 5 | 386,000 | 9,365 | 376,635 |
| job-b | 4 | 72,000 | 8,922 | 63,078 |
| job-a | 6 | 365,000 | 9,168 | 355,832 |
| job-b | 5 | 42,000 | 9,298 | 32,702 |
| job-a | 7 | 366,000 | 8,108 | 357,892 |
| job-a | 8 | 1,000 | 0 | 1,000 |
| job-b | 6 | 209,000 | 9,410 | 199,590 |
| job-a | 9 | 364,000 | 7,943 | 356,057 |
| job-b | 7 | 153,000 | 9,951 | 143,049 |
| job-a | 10 | 364,000 | 8,206 | 355,794 |
| job-b | 8 | 128,000 | 10,469 | 117,531 |
| job-a | 11 | 354,000 | 9,808 | 344,192 |
| job-a | 12 | 1,000 | 0 | 1,000 |
| job-b | 9 | 238,000 | 9,612 | 228,388 |
| job-a | 13 | 372,000 | 8,369 | 363,631 |

### Key observations

- **Job-a contention is high** (~350 s steady state) because job-b holds the
  lock for ~300 s per training step.
- **Job-b contention is lower** (~60-230 s) because job-a holds for ~200 s and
  job-b's rollout phase overlaps more of job-a's lock hold.
- Steps with wait_ms=1,000 and restore_ms=0 (job-a steps 2, 8, 12) are "fast
  re-acquires" where samples were already ready and the lock was free --
  no checkpoint swap needed.
- **Restore time is only ~3% of total wait** for high-contention steps,
  confirming that lock contention (not swap overhead) dominates step time.

### total_wait_time (veRL-internal sample queue delay)

The `fully_async/total_wait_time` metric from veRL's training loop tracks how
long accumulated ready samples waited before the trainer consumed them:

| Job | Step | total_wait_time (s) | trainer_idle (%) |
|-----|------|---------------------|------------------|
| job-a | 2 | 455.5 | 68.7 |
| job-a | 3 | 122.5 | 64.3 |
| job-a | 4 | 0.7 | 1.2 |
| job-a | 5 | 119.9 | 67.7 |
| job-a | 6 | 22.0 | 62.3 |
| job-a | 7-14 | 0.65 avg | 1-66% |
| job-b | 2 | 613.3 | 69.0 |
| job-b | 3-10 | 118-257 | 42-61% |

Job-a's total_wait_time drops to <1 s after step 6, indicating the sample
queue stays full and the trainer processes them immediately upon acquiring the
lock.  Job-b's higher total_wait_time (118-257 s) reflects its longer rollout
phase generating samples while waiting for the lock.

---

## Artifacts

| File | Description |
|------|-------------|
| `dashboard.html` | Full matplotlib-rendered dashboard with duty cycle, step time, reward curves |
| `timeslice-replay.html` | Interactive lock-alternation replay with per-GPU utilization |
| `verl_demo_data.json` | Structured data for downstream consumers |
| `RESULTS.md` | This file |

---

## Summary

| KPI | Value |
|-----|-------|
| Total training steps | 22 (13+9) vs 10 baseline |
| Trainer GPU duty (steady) | **86.6%** vs 41.6% baseline (**2.08x**) |
| Median restore | **9.2 s** |
| Median snapshot | **21.2 s** |
| Median total swap | **30.3 s** |
| Job-a median step time | 554 s (vs 627 s baseline) |
| Job-b median step time | 570 s (vs 627 s baseline) |
| Lock cycles | 28 acquire/yield pairs |
| cuda-checkpoint ops | 51 (24 snapshot + 23 restore + 4 init) |
