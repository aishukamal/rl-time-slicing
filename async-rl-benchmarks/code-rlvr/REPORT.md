# Code-RLVR x verl fully-async — Gen-Heavy Regime on Verifiable Code Rewards (Phase 5)

**Question:** does the trainer-idle regime measured on long-CoT *math* RLVR reproduce on the
other canonical verifiable-reward workload — competitive-programming *code* RL with live
test-case-execution rewards — so a math trainer and a code trainer can be time-sliced together?

**Answer: yes, quantitatively.** Same model, same async knobs, same staleness (s=8), 16K
response cap: the code-RLVR trainer GPU idles **51.3%** (verl metric; 52.8% by 100ms GPU trace)
in **11 contiguous blocks averaging 4.1 min (max 6.7 min)** — one block per step — while the
rollout GPU is pinned at 99.6%. The message queue never buffered a sample (`mq=0` every step,
`dropped_stale=0`): generation is structurally slower than training, exactly the math regime.

## Setup

- **DeepSeek-R1-Distill-Qwen-1.5B** (the base DeepCoder-1.5B trains from) on the
  **PRIME-RL/Eurus-2-RL-Data code split** (512 prompts, codecontests/apps/codeforces/taco),
  single-turn long CoT + final ```python block
- **Reward: verl in-tree `prime_code`** — LOCAL in-pod test execution (multiprocessing +
  signal.alarm timeouts + reliability_guard; suites capped at 8 tests at prep), routed by
  `data_source` with zero glue. `continuous=True` → score = fraction of tests passed.
- max_response_length=16384, max_prompt_length=2048, temp 1.0, rollout.n=8, mini_batch 64
  (8 prompts/step); veRL fully-async disagg, 1 trainer + 1 rollout H100, TP=1, sync_step=1,
  gen_batch_size=1, gpu_mem_util=0.8, **staleness_threshold=8** — no throttling
- Documented deviations: `ppo_max_token_len_per_gpu=32768` (dynamic-bsz throughput packing for
  ~18.4K-token seqs, same rationale as 24576 in the math runs); Eurus system prompt dropped
  (R1-Distill guidance; the ```python instruction lives in the user message)
- verl commit `983cb0f2` — identical to the math sweep. **Zero crashes, one deploy.**

## Steady-state numbers (steps 3-11 of 10 completed steps, 7200s window)

| Metric | **code s=8 @16K** (this run) | math s=8 @16K | math s=8 @8K |
|---|---|---|---|
| Step time | **619s** (547-691) | 407s | 223s |
| Trainer gen-wait / step | **319s** | 210s | 83s |
| update_actor / step | **281s** | 185s | 124s |
| param_sync | 1.9s | ~2s | 1.4s |
| Trainer idle ratio (verl) | **0.513** | 0.513 | 0.37-0.46 |
| Trainer GPU idle (<10% util, 100ms trace) | **52.8%** | 59.6% | 45.5% |
| Trainer idle blocks | **11 × 244.7s mean, 399s max** | 11 × 200.9s, 286s max | 21 × ~96s, 126s max |
| Rollout GPU util / idle | **99.6% / 0.4%** | 99.5% / 0.5% | 99.2% / 0.6% |
| mq depth / dropped_stale | **0 / 0** every step | 0 / 0 | 0 / 0 |
| response_length mean | **11,439** (10,848-12,271; clip 9-18%) | 8,178 | ~6,000 |
| score mean (per step) | **0.199** (0.144-0.235) | −0.588 (±1 scale) | −0.56 |

Per-step score means: 0.206, 0.196, 0.144, 0.177, 0.235, 0.190, 0.209, 0.222, 0.223, 0.198 —
**rewards are live and non-degenerate**: every step has full passes (score/max = 1.0), partials
(fraction-of-8-tests), and failures; the pre-GPU smoke test verified pass=1.0 / partial=0.5 /
no-code=0.0 / infinite-loop bounded (25s) / call-based fn_name=1.0 through the real
`default_compute_score("taco", ...)` path.

## Reward-execution cost: off the trainer path, by construction

Trainer-side `timing_s/reward` ≈ **20 microseconds/step** — in fully-async mode the
RewardLoopManager lives on the rollouter and rule-based scores stream per-sample in 8 Ray CPU
workers as each rollout finishes (`agent_reward_loop`). Test execution (up to ~90s worst-case
per pathological sample, typically ≪5s) is absorbed into the rollouter's per-sample pipeline
(`processing_time/avg` 102-129s, dominated by 11K-token decodes) and never blocks the trainer
separately from generation itself. For time-slicing purposes code-RLVR's reward is free — it
consumes rollouter-side CPU, not GPU.

## Verdict for time-slicing

1. **Code-RLVR reproduces the gen-heavy trainer-idle regime** — 51% trainer idle in
   one ~5.3-min contiguous block per ~10.3-min step, even more pronounced than math@8K
   because code CoT is longer (11.4K vs 6K mean tokens).
2. The regime law from the math sweep holds unchanged: rollout side pinned, trainer idles
   the gen−train imbalance (319s gen-wait vs 281s update → ~50% idle), staleness budget
   never binds (mq=0 throughout).
3. Blocks are large, periodic and predictable — a math-RLVR trainer and a code-RLVR trainer
   (each 1+1 GPUs) expose complementary multi-minute trainer bubbles; the pair is the
   concrete time-slicing demo target this project set out to justify.
4. Response lengths grew 10.5K → 12.3K within just 10 steps (CoT lengthening as training
   progresses) — reinforcing that static pool resizing chases a drifting ratio while
   time-slicing harvests the bubble wherever it is.

## Run notes (honesty items)

- The data-prep token-length filter silently no-op'd (transformers 4.57 `apply_chat_template`
  returns a dict → `len()==2`); measured post-hoc: all 576 sampled prompts ≤1027 tokens, so
  the 7000-char prefilter sufficed and no prompt was truncated. Fix the `len()` before reuse.
- pyext (prime_code dependency) does not install on the image's Python 3.12; a 12-line
  `RuntimeModule.from_string` exec-shim was installed by the job script (smoke-tested).
- The pod was reclaimed ~2h after run end (planned sleep); `train.log` steps 10-11 metric
  lines and the full GPU trace were recovered from Cloud Logging (GPUCSV dump, md5-verified
  `f83da9ec`).

## Artifacts

`results/`: `run_s8_16k_{gpu_util.csv,train.log,train_full.log,summary.json,start_ts,verl_commit,endlog.txt}`,
`analyze_run.py` (verbatim from the math sweep for schema parity). Job spec:
`k8s-job-code-rlvr.yaml` + `data_prep.py`; scoping in `NOTES.md`; verl reference clone `verl/`.
ConfigMap `code-rlvr` left in-cluster; Job deleted.

Prior phases: [../async-longcot/REPORT.md](../async-longcot/REPORT.md) (Phase 4, math gen-heavy),
[../async-multiturn/REPORT.md](../async-multiturn/REPORT.md) (Phase 3, train-heavy),
[../benchmark-deepresearch/REPORT.md](../benchmark-deepresearch/REPORT.md) (Phase 2),
[../benchmark-longtail/REPORT.md](../benchmark-longtail/REPORT.md) (Phase 1).
