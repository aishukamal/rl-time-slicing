# PoC 1 — Time-Slice Two Sweep Trainers on One GPU (Copy-Paste Handoff)

You are building a proof-of-concept that time-slices the **trainer processes of two RL training
jobs onto one shared H100**, while each job's generation GPU runs continuously. Everything below
is self-contained; measured baselines exist for every number you need to hit.

## 1 · The story

A researcher sweeps an RL recipe (different seeds / learning rates of the same run — standard
practice). Each sweep member is a veRL **fully-async disaggregated** job: 1 rollout GPU generating
continuously + 1 trainer GPU that is **~45% busy** — it trains ~185s, then idles ~200s (3.3 min
mean, 4.8 min max) waiting for the next batch, every ~407s cycle, perfectly periodically. Two
members today cost 4 GPUs. With trainer time-slicing they cost 3: two dedicated rollout GPUs +
**one shared trainer GPU running both trainers at ~91% packing** (2 × 185s = 370s of work per
407s cycle) with no step-time impact. Generation is never touched — matching production reality
where the generation fleet runs hot and continuous.

## 2 · The workload (validated, run it as-is)

- Repo: https://github.com/aishukamal/rl-time-slicing — directory `async-rl-benchmarks/`
  (read its README first; full study in `rl-timeslicing-benchmark-report.html`)
- Job spec: `async-rl-benchmarks/async-longcot/k8s-job-async-longcot-16k.yaml`
  (also on-disk at `/Users/aishuk/workspaces/GPU-CR/async-longcot/k8s-job-async-longcot-16k.yaml`)
- What it is: DeepSeek-R1-Distill-Qwen-1.5B + DAPO-Math-17k, long-CoT GRPO, veRL
  `fully_async_policy` (verl main @ 983cb0f2, image `verlai/verl:app-verl0.5-vllm0.10.0-mcore0.13.0-te2.2`
  as referenced in the YAML), STALENESS_THRESHOLD=8, max_response_length=16384,
  1 trainer GPU + 1 rollout GPU, TP=1. Everything self-contained in the ConfigMap
  (data prep from HF, GPU monitor at 100ms, gzip+base64 trace dump to stdout at completion).
- For sweep member B: change ONLY the job/configmap name (e.g. `-b` suffix), plus a sweep
  variable — `actor_rollout_ref.actor.optim.lr` (e.g. 1e-6 → 2e-6) or `data.seed` — and the
  experiment_name label. Do not change anything else; the phase structure must stay identical.

## 3 · Measured solo baseline (your comparison target)

From `async-rl-benchmarks/async-longcot/results/run_s8_16k_summary.json` (+ REPORT.md):

| Metric | Solo value |
|---|---|
| Step cadence | 407s mean (range 322-513s — straggler jitter is real, plan for it) |
| Trainer busy (update_actor) | 185s mean (158-220s) |
| Trainer idle block | 201s mean, 286s max, exactly 1 per step |
| Trainer GPU idle share | 59.6% |
| Rollout GPU util | 99.5% (never pauses; staleness budget never binds) |
| Weight sync (NCCL param_sync) | ~1.7s |
| response_length | 8.2K tokens mean, max pinned 16384 |

## 4 · Topology (one a3-megagpu-8g node, 8×H100)

| GPU | Assignment |
|---|---|
| 0 | Job A rollout (vLLM) — dedicated, never touched |
| 1 | Job B rollout (vLLM) — dedicated, never touched |
| 2 | Job A trainer + Job B trainer — **time-sliced** |
| 3-7 | free |

## 5 · Swap mechanics

The trainer's natural boundaries (visible in verl logs and metrics):
- **Yield point**: end of `update_actor` + param_sync completion (~1.7s NCCL push to its
  rollouter). After this the trainer does nothing but wait to collect 64 samples (~200s).
- **Resume point**: its rollouter has ~64 fresh samples ready (observable via the fully_async
  metrics: `mq_queue_size` / collect progress in the trainer log; or simply when the peer trainer
  yields — with two jobs, strict alternation works).
- **Shock absorber**: at staleness=8, while a trainer is swapped out its rollouter KEEPS
  generating into the queue budget (up to 8×64 samples ahead). Swap latency does not stall
  generation — it shows up only as queue depth. This is why s=8 (a customer-realistic setting)
  is the friendliest regime for trainer time-slicing.
- Swap implementation: your existing rl-time-slicing stack (cuda-checkpoint C/R via
  snapshot-agent / orchestrator, or the gpu_client acquire/yield pattern from
  `rl-time-slicing/verl`). Trainer process holds FSDP 1.5B + Adam state — expect a heavier
  checkpoint than the PoC's sub-second sampler swaps; budget 1-5s. Even 5s is ~2.5% of a 201s
  idle block.
- Straggler jitter: cycles vary 322-513s. Do NOT assume lockstep; block on the actual yield/ready
  signals, not on timers.

## 6 · Success metrics

1. Shared trainer GPU (GPU 2) duty cycle ≥ 90% (solo baseline: 45%).
2. Each job's mean step time within 10% of the 407s solo baseline (expect ~2-5%; 2×185=370 < 407
   leaves 37s mean slack, but jitter will occasionally serialize — that's the interesting data).
3. Both rollout GPUs ≥ 95% util throughout (proves generation was never impacted).
4. verl staleness counters stay within budget (no dropped_stale, mq below 8×64=512).
5. Capture the same artifacts as the baseline runs: 100ms GPU trace for all 3 GPUs + training
   logs for both jobs, so before/after charts are directly comparable.

## 7 · Environment & rules

- Cluster: verl-research-cluster-west (GKE us-west1-c control plane), kubectl at
  /opt/homebrew/bin/kubectl, context already set — NEVER change kubectl context.
- Node pool: **h100-mega-8gpu-spot-b ONLY** (autoscaling 0-2, spot). Capacity note: node
  provisioning has taken 4-25 min lately ("GCE out of resources" retries) — be patient, the
  autoscaler keeps retrying. Jobs on the pool tolerate preemption (traces dumped to Cloud
  Logging via the GPUCSV mechanism at completion).
- NEVER touch: h100-2gpu-spot nodes, TPU pools, default/pathways pools, mega-a pool, or any
  pods you didn't create (dis-a7-*, tsdisagg-util, snapshot-agent-*, node-debugger-*,
  tpu-orchestrator).
- Fresh clones for any repo you modify; no local builds (GCP Cloud Build, project aishuk-test,
  if an image is ever needed); no pushes without explicit go-ahead.
- Monitoring discipline (hard-won): no single command >240s; sleep in ≤240s chunks with echoes
  between; kubectl logs --tail, never -f; rescue-copy results off the pod every ~10 min.

## 8 · Reference artifacts

| What | Where |
|---|---|
| Full benchmark study (all 4 phases, charts) | `async-rl-benchmarks/rl-timeslicing-benchmark-report.html` in the repo |
| 16K baseline job spec | `async-rl-benchmarks/async-longcot/k8s-job-async-longcot-16k.yaml` |
| 16K baseline trace + stats | `async-rl-benchmarks/async-longcot/results/run_s8_16k_*` |
| Staleness-sweep report (why s=8 behaves this way) | `async-rl-benchmarks/async-longcot/REPORT.md` |
| Trace analyzer (reuse for your runs) | `async-rl-benchmarks/async-longcot/results/analyze_run.py` |
| Time-slicing PoC stack (C/R, orchestrator) | repo root: `verl/`, `time-slicing-*` directories |
