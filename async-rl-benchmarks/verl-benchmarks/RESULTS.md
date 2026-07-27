# Async RL Time-Slicing Feasibility — Results

## Objective

Determine whether GPU time-slicing (checkpoint/restore) applies to **async RL**
workloads, not just synchronous RL. Do async RL trainer GPUs exhibit a "square wave"
utilization pattern (train → idle → train → idle) that time-slicing can exploit —
and does any of veRL's async modes eliminate it?

## Setup

- **Cluster**: verl-research-cluster-west, `h100-mega-8gpu-spot-a` pool
- **Nodes**: a3-megagpu-8g (8× H100-Mega-80GB each); 1 or 2 nodes per recipe
- **Model**: Qwen2.5-Math-7B — the recipes' exact model
- **Dataset**: DAPO-Math-17k — the recipes' exact dataset
- **Framework**: veRL fully-async policy (`fully_async_policy` recipe family), Ray multi-node
- **Image**: `verlai/verl:vllm020.dev2` + verl from git main + cupy-cuda12x
- **GPU monitoring**: nvidia-smi sampled at 100ms on every node

### What nvidia-smi measures

`nvidia-smi utilization.gpu` reports **% of time during the sample period that at
least one GPU kernel was executing**. Each 100ms sample IS a duty cycle for that
window. 0% = no kernels ran; 50% = kernels ran half the window. We plot each sample
directly.

### How trainer vs sampler GPUs are identified

Roles are classified from the memory signature: vLLM sampler GPUs hold a flat
pre-allocated pool (0.80 × 81GB ≈ 65.5 GiB, spread < 1.5GB for the whole run);
FSDP trainer GPUs fluctuate and grow past it (72-75GB). Validated on all three
topologies, including mixed nodes hosting both roles.

## Recipes Tested

Three standard veRL async recipes, run with recipe-default values:

| Recipe | Trainer GPUs | Sampler GPUs | Total | Nodes | Source |
|--------|-------------|----------------|-------|-------|--------|
| `dapo_7b_4_4` | 4 | 4 | 8 | 1 | `dapo_7b_math_fsdp2_4_4.sh` |
| `dapo_7b_8_8` | 8 | 8 | 16 | 2 | `dapo_7b_math_fsdp2_8_8.sh` |
| `dapo_7b_4_12` | 12 | 4 | 16 | 2 | `dapo_7b_math_fsdp2_4_12.sh` |

Common config (recipe defaults): `max_prompt_length=2048, max_response_length=8192,
n_resp_per_prompt=16, ppo_mini_batch_size=32, require_batches=4, actor_offload=False,
ref_offload=True, checkpoint_engine=nccl`.

Deviations from the published recipes (and why):

- `hostPID + SYS_PTRACE + seccomp Unconfined` pod security — veRL's NCCL checkpoint
  engine needs the `pidfd_getfd` syscall, blocked by GKE's default seccomp profile
- `fsdp_size=4` instead of 2 — recipes target H20 (96GB); H100 is 80GB
- `total_rollout_steps=15360` — bounds run length only (~30 sync cycles)
- 4_12 only: `ppo_mini_batch_size=24` — veRL requires trajectory count divisible
  by trainer GPU count (32×4×16=2048 is not divisible by 12; 24×4×16=1536 is)

### Why these 3 of the 12 available async recipes

| Recipe | Trainer GPUs | Sampler GPUs | Total | Model | Data | Tested? | Blocker |
|--------|-------------|--------------|-------|-------|------|---------|---------|
| `dapo_7b_math_fsdp2_4_4` | 4 | 4 | 8 | Qwen2.5-Math-7B | DAPO-Math-17k | **Yes** | — |
| `dapo_7b_math_fsdp2_8_8` | 8 | 8 | 16 | Qwen2.5-Math-7B | DAPO-Math-17k | **Yes** | — |
| `dapo_7b_math_fsdp2_4_12` | 12 | 4 | 16 | Qwen2.5-Math-7B | DAPO-Math-17k | **Yes** | — |
| `dapo_7b_async_retool` | 4 | 4 | 8 | Qwen2.5-7B | Tool-calling | No | Needs multi-turn tool env |
| `geo3k_qwen25vl_7b_megatron_4_4` | 4 | 4 | 8 | Qwen2.5-VL-7B | Geo3K | No | Different model + dataset |
| `grpo_qwen35_35b_megatron_async` | 8 | 8 | 16 | Qwen3.5-35B-A3B | Math | No | 35B needs megatron + more memory |
| `grpo_30b_a3b_megatron_8_8_trtllm` | 8 | 8 | 16 | Qwen3-30B-A3B | Math | No | Needs TensorRT-LLM backend |
| `dapo_7b_math_fsdp2_16_16` | 16 | 16 | 32 | Qwen2.5-7B | Math (28K resp) | No | Needs 32 GPUs |
| `dapo_30b_a3b_base_math_fsdp` | 16 | 16 | 32 | Qwen3-30B-A3B | Math | No | Needs 32 GPUs |
| `dapo_7b_math_fsdp2_32_32` | 32 | 32 | 64 | Qwen2.5-7B | Math | No | Needs 64 GPUs |
| `dapo_7b_math_fsdp2_64_64` | 64 | 64 | 128 | Qwen2.5-7B | Math | No | Needs 128 GPUs |
| `grpo_30b_a3b_megatron_96_32` | 32 | 96 | 128 | Qwen3-30B-A3B | Math | No | Needs 128 GPUs |

We tested every recipe that fits the 16-GPU cluster and shares the same model/data
(Qwen2.5 7B + math). The untested 8-16 GPU recipes need different models, datasets,
or inference backends; the 32+ GPU recipes exceed cluster capacity.

## The 4 Async Modes

Each recipe runs under all 4 async modes (same hardware; only async params change):

| Mode | staleness_threshold | trigger_parameter_sync_step | partial_rollout | Description |
|------|---------------------|------------------------------|-----------------|-------------|
| 1 | 0 | 1 | False | On-policy pipeline — weights sync after every fetch (most synchronous) |
| 2 | 0 | 4 | False | Stream off-policy — sync every 4 fetches |
| 3 | 0.1 | 4 | False | Async + stale samples — sampler may run 10% ahead |
| 4 | 0.1 | 4 | True | Async + partial rollout — **the recipe default** (most async) |

## Results

### Mode 4 — recipe default (COMPLETE)

All three recipes at scale, ~30+ weight syncs each. Plots show trainer and sampler
separately with an **identical, shared time axis** (same t0), plus an overlay.
All statistics and plot axes are trimmed to the **training window** (first to last
GPU activity), so monitor startup and the post-run idle tail don't dilute the numbers.

| Run | GPUs (T:S) | Syncs | Train window | Trainer active | Sampler active | Trainer idle gaps ≥5s | Sampler idle gaps ≥5s | Weight sync |
|-----|-----------|-------|--------------|----------------|----------------|----------------------|----------------------|-------------|
| dapo44 | 4:4 | 29 | 147 min | 79% (96% util) | 54% (94% util) | 117 gaps, top 109s/106s/68s, total 1628s | 35 gaps, top 169s/150s/147s, total 3832s | 2.9s avg (2.5-7.4s) |
| dapo88 | 8:8 | 29 | 87 min | **66%** (92% util) | 52% (87% util) | 121 gaps, top 112s/64s/48s, total 1407s | 34 gaps, top 93s/84s/83s, total 2095s | 5.8s avg (5.3-9.5s) |
| dapo412 | 12:4 | 39 | 127 min | 89% (76% util) | 82% (71% util) | 21 gaps, top 96s/71s/49s, total 497s | 32 gaps, top 106s/82s/80s, total 1240s | 6.0s avg (5.5-8.1s) |

Weight sync = veRL's own `timing_s/param_sync` metric (NCCL checkpoint engine):
~2.9s single-node (4:4), ~6s when trainer and sampler span two nodes.

#### dapo_7b_4_4

![dapo44 separate](plots/dapo44_mode4_separate.png)

![dapo44 overlay](plots/dapo44_mode4_overlay.png)

![dapo44 gaps](plots/dapo44_mode4_gaps.png)

#### dapo_7b_8_8

![dapo88 separate](plots/dapo88_mode4_separate.png)

![dapo88 overlay](plots/dapo88_mode4_overlay.png)

![dapo88 gaps](plots/dapo88_mode4_gaps.png)

#### dapo_7b_4_12

![dapo412 separate](plots/dapo412_mode4_separate.png)

![dapo412 overlay](plots/dapo412_mode4_overlay.png)

![dapo412 gaps](plots/dapo412_mode4_gaps.png)

#### Reading the gap figures

Each run's gap figure has two panels. The **histogram** (left) shows the full
distribution of idle-gap durations — the headline "top" gaps in the table are the
tail, not the norm. The **recurrence scatter** (right) plots every gap at the time
it occurred: gaps appear continuously across the entire run, once or more per sync
cycle, not as one-off outliers.

#### Mode 4 findings

1. **The GPU split is the dominant lever.** With the async mode held constant,
   trainer activity ranges from 66% (8:8) to 89% (12:4). The trainer:sampler
   throughput ratio sets who idles and by how much.
2. **Trainer and sampler gaps have different shapes.** Trainers accumulate many
   short-to-medium gaps — ~120 gaps per run in 4:4/8:8, the bulk 5-20s, recurring
   at a steady cadence every training step. Samplers have fewer but much longer
   gaps — ~35 per run, mostly 45-170s, one per sync cycle, and they *grow* over the
   run as responses lengthen. Both kinds vastly exceed the 2-3s C/R cost.
3. **8:8 has the most trainer slack**: trainer idle 34% of the window, with 121
   gaps ≥5s totaling 23 of the 87 minutes (27% of the run in sliceable windows).
4. **4:4 has the most sampler slack**: samplers idle 46% of the window — 35 gaps
   totaling 64 of the 147 minutes (43%), waiting for the 4 trainer GPUs.
5. **12:4 is the busiest config** (trainer 89%, sampler 82%) yet still yields
   21 trainer gaps up to 96s and 32 sampler gaps up to 106s.
6. **All runs show the periodic square wave for 29-39 consecutive sync cycles**
   at 76-96% utilization when active — steady state at production-recipe scale.
7. **Weight sync itself is cheap and bounded**: 2.9-6.0s per sync, comparable to
   the 2-3s C/R cost — syncing weights is not the thing that idles the GPUs; the
   pipeline structure is.

### Mode 1 — on-policy pipeline (COMPLETE)

Weights sync after **every** fetch (`trigger_parameter_sync_step=1`, staleness=0,
no partial rollout) — the most synchronous async mode. Same recipes, same rollout
budget as Mode 4; mode 1 logs one training step per sync (120-160 steps per run).

| Run | GPUs (T:S) | Syncs | Train window | Trainer active | Sampler active | Trainer idle gaps ≥5s | Sampler idle gaps ≥5s | Weight sync |
|-----|-----------|-------|--------------|----------------|----------------|----------------------|----------------------|-------------|
| dapo44 | 4:4 | 120 | 238 min | 49% (94% util) | 47% (84% util) | 121 gaps, top 111s/111s/110s, total 6938s | 124 gaps, top 93s/91s/91s, total 7620s | 2.7s avg (2.5-7.1s) |
| dapo88 | 8:8 | 120 | 162 min | **35%** (82% util) | 47% (71% util) | 122 gaps, top 113s/76s/73s, total 5372s | 124 gaps, top 81s/51s/49s, total 4307s | 5.7s avg (5.2-9.6s) |
| dapo412 | 12:4 | 160 | 245 min | 60% (56% util) | 76% (61% util) | 183 gaps, top 96s/80s/78s, total 5134s | 135 gaps, top 81s/43s/42s, total 3489s | 5.5s avg (4.9-8.1s) |

#### dapo_7b_4_4

![dapo44 mode1 separate](plots/dapo44_mode1_separate.png)

![dapo44 mode1 overlay](plots/dapo44_mode1_overlay.png)

![dapo44 mode1 gaps](plots/dapo44_mode1_gaps.png)

#### dapo_7b_8_8

![dapo88 mode1 separate](plots/dapo88_mode1_separate.png)

![dapo88 mode1 overlay](plots/dapo88_mode1_overlay.png)

![dapo88 mode1 gaps](plots/dapo88_mode1_gaps.png)

#### dapo_7b_4_12

![dapo412 mode1 separate](plots/dapo412_mode1_separate.png)

![dapo412 mode1 overlay](plots/dapo412_mode1_overlay.png)

![dapo412 mode1 gaps](plots/dapo412_mode1_gaps.png)

#### Mode 1 findings

1. **On-policy strictness roughly doubles the idle time on both sides.** Trainer
   activity drops from Mode 4's 79%/66%/89% to **49%/35%/60%** across 4:4/8:8/12:4;
   trainer gap totals grow 3-10× (e.g. 4:4: 1628s → 6938s of ≥5s gaps).
2. **The strict alternation shows on the samplers too**: every recipe now has
   ~124-135 sampler gaps (vs ~35 in Mode 4) because samplers must wait for the
   sync after every single fetch — in 4:4, samplers idle 53% of the run.
3. **Wall-clock cost of on-policy**: the same rollout budget takes 1.6-1.9×
   longer than Mode 4 (238 vs 147 min for 4:4; 162 vs 87 for 8:8; 245 vs 127
   for 12:4). The GPUs saved by time-slicing would come on top of an already
   slower training mode.
4. **This is the most time-sliceable mode**: 5100-6900s of ≥5s trainer gaps per
   run — 35-50% of the entire run is sliceable trainer idle, in recurring 20-113s
   windows, on top of an equally gap-rich sampler side.
5. **Sync count ×4, sync cost unchanged** (2.7-5.7s per sync) — but 120-160 syncs
   accumulate: dapo88 spent ~11 min of its 162-min run just syncing weights.

### Mode 2 — stream off-policy (COMPLETE)

Weights sync every 4 fetches (`trigger_parameter_sync_step=4`) with **zero staleness
allowance** and no partial rollout — Mode 4's cadence without its slack.

| Run | GPUs (T:S) | Syncs | Train window | Trainer active | Sampler active | Trainer idle gaps ≥5s | Sampler idle gaps ≥5s | Weight sync |
|-----|-----------|-------|--------------|----------------|----------------|----------------------|----------------------|-------------|
| dapo44 | 4:4 | 30 | 149 min | 75% (96% util) | 54% (95% util) | 116 gaps, top 106s/101s/93s, total 2022s | 34 gaps, top 165s/154s/150s, total 3873s | 2.8s avg (2.5-7.2s) |
| dapo88 | 8:8 | 30 | 89 min | **63%** (92% util) | 52% (86% util) | 121 gaps, top 112s/65s/59s, total 1583s | 34 gaps, top 93s/87s/86s, total 2164s | 5.8s avg (5.4-9.4s) |
| dapo412 | 12:4 | 40 | 146 min | 85% (70% util) | 81% (70% util) | 29 gaps, top 95s/68s/48s, total 721s | 33 gaps, top 94s/83s/80s, total 1532s | 5.6s avg (5.2-7.8s) |

#### dapo_7b_4_4

![dapo44 mode2 separate](plots/dapo44_mode2_separate.png)

![dapo44 mode2 overlay](plots/dapo44_mode2_overlay.png)

![dapo44 mode2 gaps](plots/dapo44_mode2_gaps.png)

#### dapo_7b_8_8

![dapo88 mode2 separate](plots/dapo88_mode2_separate.png)

![dapo88 mode2 overlay](plots/dapo88_mode2_overlay.png)

![dapo88 mode2 gaps](plots/dapo88_mode2_gaps.png)

#### dapo_7b_4_12

![dapo412 mode2 separate](plots/dapo412_mode2_separate.png)

![dapo412 mode2 overlay](plots/dapo412_mode2_overlay.png)

![dapo412 mode2 gaps](plots/dapo412_mode2_gaps.png)

#### Mode 2 findings

1. **Mode 2 is nearly indistinguishable from Mode 4** — trainer active 75%/63%/85%
   vs Mode 4's 79%/66%/89%, with matching wall-clock, gap counts, and gap shapes.
   Removing the staleness allowance and partial rollout costs only a few points of
   trainer activity when the sync cadence stays at every 4 fetches.
2. **Sync cadence, not staleness slack, is the async lever that matters.** The big
   behavioral cliff is between Mode 1 (sync every fetch → trainers 35-60% active)
   and Modes 2/4 (sync every 4 → trainers 63-89% active).
3. Gap structure mirrors Mode 4: trainers accumulate ~120 short-to-medium gaps in
   the 4-GPU-trainer configs; samplers keep ~34 long 80-165s gaps, one per cycle.

### Mode 3 — async + stale samples (COMPLETE)

Sync every 4 fetches with `staleness_threshold=0.1` (sampler may run 10% ahead)
but **no partial rollout** — Mode 4 minus its last ingredient.

| Run | GPUs (T:S) | Syncs | Train window | Trainer active | Sampler active | Trainer idle gaps ≥5s | Sampler idle gaps ≥5s | Weight sync |
|-----|-----------|-------|--------------|----------------|----------------|----------------------|----------------------|-------------|
| dapo44 | 4:4 | 30 | 141 min | 79% (96% util) | 56% (95% util) | 116 gaps, top 107s/104s/67s, total 1620s | 34 gaps, top 166s/139s/138s, total 3497s | 2.8s avg (2.5-7.2s) |
| dapo88 | 8:8 | 30 | 86 min | **66%** (92% util) | 54% (87% util) | 121 gaps, top 116s/66s/46s, total 1404s | 33 gaps, top 94s/84s/81s, total 2018s | 5.8s avg (5.3-9.5s) |
| dapo412 | 12:4 | 40 | 127 min | 88% (76% util) | 82% (72% util) | 21 gaps, top 93s/78s/49s, total 517s | 31 gaps, top 95s/92s/82s, total 1204s | 5.6s avg (5.1-8.2s) |

#### dapo_7b_4_4

![dapo44 mode3 separate](plots/dapo44_mode3_separate.png)

![dapo44 mode3 overlay](plots/dapo44_mode3_overlay.png)

![dapo44 mode3 gaps](plots/dapo44_mode3_gaps.png)

#### dapo_7b_8_8

![dapo88 mode3 separate](plots/dapo88_mode3_separate.png)

![dapo88 mode3 overlay](plots/dapo88_mode3_overlay.png)

![dapo88 mode3 gaps](plots/dapo88_mode3_gaps.png)

#### dapo_7b_4_12

![dapo412 mode3 separate](plots/dapo412_mode3_separate.png)

![dapo412 mode3 overlay](plots/dapo412_mode3_overlay.png)

![dapo412 mode3 gaps](plots/dapo412_mode3_gaps.png)

#### Mode 3 findings

1. **Mode 3 is statistically identical to Mode 4** on every recipe (79/66/88% vs
   79/66/89% trainer active, matching gap counts and totals). Partial rollout —
   the only difference between them — has no visible effect on GPU duty cycles
   at this scale.
2. Together with Mode 2's result, the async knobs rank: **sync cadence ≫ staleness
   allowance > partial rollout**. Only the first changes the utilization pattern.

### Cross-mode comparison — all 12 runs

![Cross-mode summary](plots/cross_mode_summary.png)

**Trainer active (% of training window):**

| Recipe | Mode 1 (sync=1) | Mode 2 (sync=4) | Mode 3 (+stale) | Mode 4 (+partial, default) |
|--------|-----------------|-----------------|-----------------|----------------------------|
| 4:4 | 49% | 75% | 79% | 79% |
| 8:8 | **35%** | 63% | 66% | 66% |
| 12:4 | 60% | 85% | 88% | 89% |

**Sampler active:** 47-76% (Mode 1) vs 52-82% (Modes 2-4) — same ordering.

**Trainer idle locked in ≥5s gaps (minutes per run):**

| Recipe | Mode 1 | Mode 2 | Mode 3 | Mode 4 |
|--------|--------|--------|--------|--------|
| 4:4 | 116 | 34 | 27 | 27 |
| 8:8 | 90 | 26 | 23 | 23 |
| 12:4 | 86 | 12 | 9 | 8 |

**Wall-clock for the same rollout budget (minutes):**

| Recipe | Mode 1 | Mode 2 | Mode 3 | Mode 4 |
|--------|--------|--------|--------|--------|
| 4:4 | 238 | 149 | 141 | 147 |
| 8:8 | 162 | 89 | 86 | 87 |
| 12:4 | 245 | 146 | 127 | 127 |

Cross-mode takeaways:

1. **Two regimes, not four.** The 4 async modes collapse into: sync-every-fetch
   (Mode 1) and sync-every-4 (Modes 2/3/4, near-identical within ±4%). Staleness
   allowance and partial rollout barely move GPU duty cycles; sync cadence
   reshapes them completely.
2. **The square wave never disappears.** Even the most asynchronous configuration
   (Mode 4) on the busiest split (12:4) retains 21 trainer gaps up to 96s and 31
   sampler gaps up to 95s per run. No mode/recipe combination eliminates the
   recurring idle windows.
3. **Time-slicing headroom by configuration**: worst case ~8 min of ≥5s trainer
   gaps per ~2h run (12:4, Mode 4); best case ~116 min (4:4, Mode 1). Balanced 8:8
   at default settings sits at 23 min per 87-min run — 27% of the GPUs' time.
4. **Weight sync is invariant** (2.7-2.9s single-node, 5.5-6.0s two-node) across
   all modes and recipes — the C/R cost (2-3s) is the same order as an operation
   the pipeline already performs 30-160 times per run.
5. **On-policy strictness is doubly expensive**: Mode 1 costs 1.6-1.9× wall-clock
   AND leaves 2-4× more idle — making it simultaneously the worst-performing and
   the most time-sliceable configuration. Teams that need on-policy training have
   the strongest economic case for GPU sharing.

## Conclusion

**Time-slicing IS applicable to async RL** — confirmed across all 12 runs
(3 standard veRL recipes × all 4 async modes) at exact-recipe 7B scale, with
30-160 weight syncs per run:

1. **The square wave exists in every one of the 12 configurations** — trainer GPUs
   alternate between ~0% and 76-96% utilization with recurring idle gaps; no async
   mode or GPU split eliminates it.
2. **Async mode does not rescue utilization.** The four modes collapse into two
   regimes: sync-every-fetch (Mode 1: trainers 35-60% active, 86-116 min of ≥5s
   gaps per run) and sync-every-4 (Modes 2/3/4, within ±4% of each other:
   trainers 63-89% active, 8-34 min of gaps). Staleness allowance and partial
   rollout — the knobs that make async "async" — barely move GPU duty cycles.
3. **GPU allocation ratio determines the opportunity size** — balanced 8:8 leaves
   the most trainer slack at default settings (27% of the run in ≥5s gaps); 4:4
   leaves the most sampler slack (43%); even trainer-heavy 12:4 keeps dozens of
   50-100s gaps per run.
4. **C/R overhead (2-3s) is negligible**: gaps run 5-170s and recur every cycle
   (histograms + recurrence scatters per run above), and the cost matches the
   2.7-6.0s weight sync the pipeline already absorbs 30-160 times per run.
5. **The pattern is steady-state, not startup** — every run holds the periodic
   square wave across its full 30-160 sync cycles.
6. **On-policy (Mode 1) training has the strongest case for time-slicing**: it is
   1.6-1.9× slower for the same rollout budget AND idles 2-4× more — its trainers
   and samplers both sit idle about half the run in 20-113s windows.

## Experiment Artifacts

| File | Purpose |
|------|---------|
| `worklog.md` | Full incremental history: 1-step survey, pidfd blocker, 0.5B + 7B steady-state runs |
| `launch_phase1.sh` | Launcher for the 3 recipes at Mode 4 defaults (full veRL arg list) |
| `launch_phase2.sh` | Parameterized launcher for modes 1-3 (`MODE_NAME/STALENESS/SYNC_STEP/PARTIAL`) |
| `plot_phase1.py` | Aligned separate + overlay plots, automatic trainer/sampler classification |
| `analyze_gaps.py` | Idle-gap histogram + recurrence plots, gap/sync-duration table stats |
| `md_to_html.py` | Renders this report to self-contained RESULTS.html |
| `phase1_results/dapo{44,88,412}_<mode>/` | Raw nvidia-smi CSVs + train logs per run |
| `plots/` | All generated PNGs |
| `PHASE2_RUNBOOK.md` | Operational runbook (launch, monitor, extract, cleanup) |
