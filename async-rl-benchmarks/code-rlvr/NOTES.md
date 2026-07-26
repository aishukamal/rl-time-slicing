# Code-RLVR x verl fully-async: Scoping Notes

Phase 5 of the time-slicing RL benchmark: reproduce the gen-heavy trainer-idle regime
(async-longcot, math s=8) on the OTHER canonical verifiable-reward workload —
competitive-programming code RL — so a math-RLVR trainer and a code-RLVR trainer can
later be time-sliced against each other.

Scoped against: verl-project/verl main @ `983cb0f2` (same commit the longcot runs used),
fresh clone in `/Users/aishuk/workspaces/GPU-CR/code-rlvr/verl/` (read-only reference;
the job clones main at runtime like all prior phases).

## Decision 1 — Reward path: in-tree `prime_code`, local in-pod execution

`verl/utils/reward_score/__init__.py:74-88`: `data_source in ["codecontests", "apps",
"codeforces", "taco"]` routes to code-execution reward. Two branches:

- if `sandbox_fusion_url` is set → external sandbox service (REJECTED: no external service)
- else → **`prime_code.compute_score(solution_str, ground_truth, continuous=True)`** —
  plain local Python execution. CHOSEN.

How prime_code executes (`prime_code/__init__.py`, `utils.py`, `testing_util.py`):

- Extracts code as `completion.split("```python")[-1].split("```")[0]` — the dataset
  prompt MUST instruct "code in a ```python block" (ours does, see Decision 2).
- `ground_truth` = JSON string `{"inputs": [stdin_str...], "outputs": [expected...]}`
  (+ optional `"fn_name"` for call-based problems; both types handled by `run_test`).
- Sandboxing = `multiprocessing.Process` per check + `signal.alarm` per-test timeout +
  `reliability_guard()` (disables subprocess/os.system/ctypes etc. in the child) +
  hard `p.join(timeout+1); p.kill()`. No Docker, no network. Exactly the
  subprocess+timeout level appropriate for a benchmark.
- Timeouts: first a whole-suite pass with timeout=5 (join 6s); on any failure, a
  per-test loop with timeout=10 (join 11s), **capped at 10 test cases**. Worst case
  (infinite-loop generation) ≈ 6 + 10×11 ≈ 116s per sample. We additionally truncate
  test suites to 8 cases at data-prep time → worst case ≈ 94s, typical ≪ 5s.
- `continuous=True` → score = fraction of tests passed in [0,1] (1.0 fast-path if the
  whole suite passes) → non-degenerate GRPO advantages.

**Where it runs (trainer-latency question):** in fully-async mode reward is NOT on the
trainer path. `fully_async_rollouter.py:744-839` creates a `RewardLoopManager` on the
rollouter and, for rule-based rewards (`enable_agent_reward_loop = not use_rm`), hands
`reward_loop_worker_handles` straight to the agent-loop manager → scores are computed
**streaming, per sample, as each rollout finishes**, in `reward.num_workers=8` (default)
Ray CPU actors (`reward_loop.py`, `reward_manager/naive.py` `run_single` via
`run_in_executor`). So test execution overlaps generation; it adds trainer latency only
if a sample's score is still pending when the trainer wants the batch. We report the
gen-wait breakdown either way (it is indistinguishable from gen time in the trainer's
`gen` timing key; noted honestly in REPORT.md).

**Dependency gotcha:** `prime_code/testing_util.py` does `from pyext import RuntimeModule`;
pyext is in `PRIME_REQUIRES`, NOT in the `[gpu]` extra the job installs. pyext 0.7 (2015)
also fails to install on Python ≥3.11 (`inspect.formatargspec` removed). Job handles it:
`pip install pyext || true`, then if `import pyext` still fails, drop a 12-line shim
`pyext.py` (module-from-string via `exec`, semantically identical to
`RuntimeModule.from_string`) into site-packages. A fail-fast reward smoke test (pass /
fail / infinite-loop-timeout cases through `default_compute_score("taco", ...)`) runs
BEFORE any GPU phase.

## Decision 2 — Dataset: PRIME-RL/Eurus-2-RL-Data, code split

Candidates considered:

| Candidate | Verdict |
|---|---|
| **PRIME-RL/Eurus-2-RL-Data** (code rows) | **CHOSEN** — already exact verl RLHFDataset format: `data_source ∈ {codecontests, apps, codeforces, taco}` (routes to prime_code with zero glue), `reward_model.ground_truth` already the prime_code JSON `{"inputs","outputs"}`, user prompt already ends with "Write Python code… ```python block". This is the dataset verl's own PRIME recipe ran with this exact scorer. 25,276 code train rows, MIT. |
| PrimeIntellect/verifiable-coding-problems | Needs transform (`verification_info.test_cases[{type,input,output}]` → inputs/outputs JSON; multi-language fan-out to filter). Viable fallback. |
| agentica-org/DeepCoder-Preview-Dataset | verl-*adjacent* (rllm fork format): `tests` field in LCB/taco-specific shapes, needs per-split conversion + custom prompt build. More work, same outcome. |

Prep (in-job `data_prep.py`, mirrors the math one): download `train.parquet` (~1.7GB),
filter `ability=="code"`, validate ground_truth JSON (≥1 test, parallel inputs/outputs),
**truncate to first 8 test cases** (bounds reward latency; preserves `fn_name`), drop
rows whose truncated ground truth >200KB, dedup on user text, filter to prompts ≤1900
tokens under the R1-Distill chat template (max_prompt_length=2048, no truncation damage),
sample 512 train / 64 val disjoint.

Documented deviation: we DROP the Eurus system message (the `[ASSESS]/[ADVANCE]/...`
action-protocol prompt designed for Eurus-SFT models). DeepSeek-R1-Distill guidance is
no system prompt, and DeepCoder trained this same base model with bare problem prompts.
The reward-relevant format instruction (```python block) lives in the user message and
is kept verbatim.

## Decision 3 — Everything else mirrors k8s-job-async-longcot.yaml

- Model: DeepSeek-R1-Distill-Qwen-1.5B (= DeepCoder-1.5B's base; long CoT for code too).
- Same image `verlai/verl:vllm020.dev2` + runtime verl-main clone; fully_async_policy;
  1 trainer + 1 rollout H100; sync_step=1; gen_batch_size=1; staleness_threshold=8;
  rollout.n=8; ppo_mini_batch_size=64 (8 prompts/step); temp 1.0; no throttling.
- max_response_length=16384 (vs 8192 math) + max_prompt_length=2048 → 18432 total seq;
  dynamic bsz budget `ppo_max_token_len_per_gpu=32768` (fits one max-len seq, packs ~3
  mean-length; same throughput-packing rationale as the documented 24576 in longcot).
  Fallback on trainer OOM: 20480.
- RUN_SECONDS=7200 (~10-17 steps at an expected 400-700s/step), then sleep 7200.
- 100ms GPU monitor, GPUCSV gzip+base64 stdout dump, same summary tooling.
- Job + ConfigMap both named `code-rlvr`, pool h100-mega-8gpu-spot-b, nvidia.com/gpu: 2
  (room alongside the async-longcot-16k job on the same 8-GPU node).

## Expected failure modes

1. pyext broken on image Python → shim path (handled in-script, smoke-tested pre-GPU).
2. All-zero rewards if the model never closes a ```python block within 16K tokens —
   watch first-step score distribution; R1-Distill emits fenced code reliably, and
   `continuous=True` gives partial credit, so expect a mix of 0 / partial / 1.
3. Reward stragglers (infinite-loop generations) delaying batch readiness by up to
   ~90s — visible as inflated gen-wait; reported, not hidden.
4. Trainer OOM at 32768-token packing → redeploy with 20480.
5. verl main drift vs image deps — same risk as prior phases (all of which passed).
