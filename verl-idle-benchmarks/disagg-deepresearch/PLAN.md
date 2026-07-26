# Disaggregated Deep-Research Benchmark — Integration Assessment

Scoping result for running the CMU multi-turn agentic RL workload (verl-agent-deepresearch,
arXiv:2510.06534, Qwen2.5-3B + MHQA + local BM25 Wikipedia search) in a **disaggregated**
layout — sampler GPU(s) doing multi-turn rollouts while the trainer GPU sits idle — instead
of today's colocated hybrid-engine mode.

Fresh clones used for this analysis (do not edit shared checkouts elsewhere):

- `/Users/aishuk/workspaces/GPU-CR/disagg-deepresearch/rl-time-slicing/` (PoC, subdir `verl/`)
- `/Users/aishuk/workspaces/GPU-CR/disagg-deepresearch/verl-agent-deepresearch/` (CMU)
- `/Users/aishuk/workspaces/GPU-CR/disagg-deepresearch/verl-upstream/` (volcengine/verl main, for API verification)

---

## 1. How the PoC's custom sync-disagg loop works

Location: `rl-time-slicing/verl/` — runs against **verl main** (deploy_verl.sh clones
volcengine/verl **unpinned** at deploy time, image `verlai/verl:vllm017.latest`), NOT a
pinned release.

**Entry point** — `main_ppo_timeslice_sync.py`
- Hydra config: `verl/experimental/one_step_off_policy/config/one_step_off_ppo_trainer.yaml`.
- `SyncTimesliceTaskRunner.run()` builds tokenizer/datasets/StatefulDataLoader itself, then:
  - `create_role_worker_mapping(config)` + `create_resource_pool_manager(...)` from
    `verl.experimental.separation.utils` → **two Ray resource pools**: a `DetachActorWorker`
    (FSDP actor) pool sized by `trainer.n_gpus_per_node`/`trainer.nnodes`, and a rollout pool
    sized by `+rollout.n_gpus_per_node`/`+rollout.nnodes` (config forced to 1×1 in the runner).
  - Instantiates `OneStepOffRayTrainer` (verl `experimental/one_step_off_policy/ray_trainer.py`,
    subclass of the `separation` trainer) and calls `init_workers()`; rollout side comes up as
    **server-mode vLLM** (`LLMServerManager` + `AgentLoopManager`; `rollout.mode == "async"` is
    asserted upstream — there is no sync SPMD `generate_sequences` in this path anymore).
  - Monkey-patches `DetachActorWorker.__init__`/`AgentLoopWorker.__init__` to register PIDs
    with the timeslice orchestrator (`gpu_client.py` → HTTP to `orchestrator.py`;
    `node_daemon.py` resolves PIDs on the host). For a pure disagg benchmark these calls can
    be stubbed (no orchestrator needed).
- **Known snag:** it imports `verl_timeslice_sync_modular_trainer` but the file in the repo is
  `verl_timeslice_sync_trainer.py` (deploy_verl.sh uploads it under the original name) — the
  import must be fixed when reusing this code.
- deploy_verl.sh also overwrites `verl/single_controller/ray/base.py` with a "hacked local
  version" that is **not present in the repo clone**, and live-patches worker pods
  (`layered_summon`/`peft_merge` getattr guards in `engine_workers.py`, NCCL checkpoint-engine
  imports in fully_async files). Budget for re-deriving these patches against current verl main.

**The sync loop** — `verl_timeslice_sync_trainer.py` (`SyncTimesliceTrainer.fit()`, ~300 lines):

```
per step:
  trainer_gpu.acquire(); sampler_gpu.acquire()
  1. weight sync   : one_step_trainer.checkpoint_manager.update_weights(step)   # NCCL
                     + async_rollout_manager.clear_kv_cache()
     trainer_gpu.yield()
  2. generate      : gen_batch = one_step_trainer._get_gen_batch(batch)
                     .repeat(rollout.n) → await async_rollout_manager.generate_sequences(...)
     sampler_gpu.yield()
  3. rewards/adv   : extract_reward(...) + core_algos.compute_grpo_outcome_advantage(index=uid)  # CPU
  4. train         : trainer_gpu.acquire(); _compute_old_log_prob → _update_actor
     (trainer keeps GPU rolling into next step's weight sync)
```

- **Weight sync** = `CheckpointEngineManager.update_weights()` (verl
  `experimental/separation/ray_trainer.py:126/:304`, NCCL checkpoint engine) from the detached
  FSDP actor to the vLLM replicas. Workload-agnostic — unchanged for multi-turn.
- **Currently drives:** Qwen2.5-0.5B-Instruct, GSM8K, `rollout.n=16`, `hybrid_engine=False`,
  1 trainer GPU + 1 sampler GPU, GRPO (see `rl-time-slicing/verl/README.md:48-82`).

Key property for us: the sampler phase is a single awaited call on
`async_rollout_manager.generate_sequences(batch)`; **anything** that only issues generate
calls can be substituted there, and the trainer GPU idles for its whole duration.

## 2. CMU's multi-turn loop and what it needs from the trainer

CMU repo pins **verl 0.3.1.dev** (vllm 0.8.5, transformers 4.51.1) and hard-asserts colocated:
`verl/trainer/ppo/ray_trainer.py:500` — `assert self.hybrid_engine, "Currently, only support
hybrid engine"`. Their fit loop is stock RayPPOTrainer with generation swapped for the agent
loop (`ray_trainer.py:1133`): `gen_batch_output = self.traj_collector.multi_turn_loop(gen_batch,
self.actor_rollout_wg, envs=self.envs, ...)`, then `batch = gen_batch_output` wholesale.

**Rollout loop** — `agent_system/multi_turn_rollout/rollout_loop.py`
- `TrajectoryCollector.vanilla_multi_turn_loop()` (lines 534-703) — the mode we run
  (no filter_groups / critique / rule_reward):
  1. `envs.reset()` → per-env question text (questions come from the **env's own JSON dataset**,
     see below — the trainer's parquet is literally `dummy_data/text/train.parquet`).
  2. If obs count ≠ gen_batch size: `gen_batch.repeat(env.rollout.n)` (GRPO grouping is done
     here; `actor_rollout_ref.rollout.n` must stay **1**).
  3. Per turn (up to `env.max_steps=6`): `preprocess_batch()` rebuilds the per-turn prompt from
     obs text via `tokenizer.apply_chat_template` (single user message) →
     pops `input_ids/attention_mask/position_ids` + `raw_prompt_ids` (+ `raw_prompt`,
     `tools_kwargs` if present) into `batch_input` → **`actor_rollout_wg.generate_sequences(batch_input)`**
     → decode responses → `envs.step(responses)` (parse action → search HTTP call → new obs,
     reward, done) → append one training row per env per turn (with `uid` = question id for
     GRPO groups, `traj_uid`, `active_masks`, `rewards`).
     Note: it generates for the **whole batch every turn**, including finished trajectories;
     rows for done envs are dropped later via `active_masks`. Long-tail comes from turn count
     and growing per-turn context.
  4. `gather_rollout_data()` flattens all **active** (env, turn) rows into one training batch —
     batch size = Σ episode lengths, i.e. **variable per step** — and broadcasts
     `episode_rewards`/`episode_lengths` onto every row.
- `preprocess_single_sample` (line 49) needs from the dataloader batch only the non-tensor
  fields **`raw_prompt`** (⇒ `data.return_raw_chat=True`) and **`data_source`**; everything
  else it constructs from env obs.

**Environment** — `agent_system/environments/`
- `make_envs(config)` (`env_manager.py:592`, deepresearch branch at :676) builds
  `DeepResearchMultiProcessEnv` = `env_num × group_n` Ray CPU actors
  (`env_package/deepresearch/envs.py`, `@ray.remote(num_cpus=0.125)`), env_num =
  `data.train_batch_size`, group_n = `env.rollout.n`. Each worker wraps
  `deepresearch/env.py`: parse `<search>`/`<answer>` actions, call the Serper-format search
  endpoint (`retrieval.py`, patched to `http://localhost:8877/search` for the local pyserini
  BM25 server — see `GPU-CR/benchmark-deepresearch/k8s-job.yaml` ConfigMap), max_steps cap.
- **Reward** is computed inside the env at the final turn: `evaluation_reward_fn` (qa mode)
  → `evaluate_afm_answer` → **gpt-4o-mini LLM judge** (`reward/evaluation/afm_eval.py:69`,
  5 retries). With the dummy `OPENAI_API_KEY` used in the colocated run this fails → score 0
  for all rows. Fine for a GPU-trace benchmark, but flag it: rewards are all-zero unless a
  real key (or a rule-based F1 patch) is provided.
- **Trainer-side reward path**: `agent_system/reward_manager/episode.py`
  (`EpisodeRewardManager`) just copies `non_tensor_batch['episode_rewards']` to the last
  response token → `token_level_scores`; GRPO advantage over `uid` groups. Plus
  `adjust_batch` (`multi_turn_rollout/utils.py:86`) truncates/pads the variable-size batch to
  a divisor of `ppo_mini_batch_size` × dp size, and `apply_invalid_action_penalty`
  (CMU `ray_trainer.py:200`).

**Summary of the contract:** the multi-turn loop needs (1) a batch with `raw_prompt` +
`data_source`, (2) an object exposing `generate_sequences(DataProto) -> DataProto`
(prompts/responses/input_ids/attention_mask/position_ids), (3) Ray CPU env actors + the search
HTTP server, (4) tokenizer + a handful of config keys (`env.max_steps`, `env.rollout.n`,
`data.max_prompt_length/truncation/return_raw_chat`). It does **not** need the hybrid engine,
the ref policy, or anything GPU-side beyond generate.

## 3. Candidate approaches, ranked by effort

### (a) Port CMU's multi-turn loop into the PoC's sampler phase — RECOMMENDED
Replace `SyncTimesliceTrainer._generate()`'s single generate call with
`TrajectoryCollector.vanilla_multi_turn_loop()`, feeding it an adapter around
`async_rollout_manager.generate_sequences`.

Why it fits well:
- The per-turn `batch_input` **already contains `raw_prompt`** (CMU pops it in), which is
  exactly what verl main's `AgentLoopManager` needs — `SingleTurnAgentLoop.run()` does
  `messages = list(kwargs["raw_prompt"])` and re-applies the same chat template CMU applied
  to build `input_ids` (`verl-upstream/verl/experimental/agent_loop/single_turn_agent_loop.py:41`).
  `AgentLoopWorker.generate_sequences` forwards all non-tensor keys as kwargs and defaults
  `agent_name` to the single-turn loop (`agent_loop.py:610-612,645`). So the adapter is
  ~10 lines: await the manager, return the DataProto.
- All verl symbols CMU's vendored files import still exist on verl main:
  `tokenize_and_postprocess_data` (`torch_functional.py:560`), `get_response_mask` (:342),
  `compute_position_id_with_mask` (`utils/model.py:240`), `collate_fn`
  (`utils/dataset/rl_dataset.py:41`), `qwen2_vl.get_rope_index`, `DataProto.repeat/union/pop`.
  Only `DataProto.truncate` is gone — used solely in the eval/val path; guard it.
- Weight sync, pools, timeslice acquire/yield points: **unchanged**.
- Envs are Ray CPU actors — they join the PoC's existing Ray cluster for free.

What breaks / must change: the trainer-side reward path (swap `extract_reward` for
`EpisodeRewardManager`), variable batch size (trivial with a 1-GPU trainer, port
`adjust_batch` anyway), and CMU's critique/GiGPO imports (strip them — they pull the whole
langchain/google dependency tree, which is what made the colocated image painful).

**Size:** ~2-3k lines vendored (mostly copied files), ~300-500 lines new/modified PoC code.
**Risk: LOW-MEDIUM.** Biggest unknowns: unpinned verl main drift (pin a commit!), agent-loop
output shape details (`prompt_length` padding per turn), and the deploy-time verl patches
that live only in the GCS bucket. Estimate **1-2 days to first disagg step, +1 day tuning**.

### (b) Run CMU's trainer with hybrid engine disabled — NOT VIABLE CHEAPLY
- On CMU's pinned verl 0.3.1: `hybrid_engine=False` is hard-asserted away, and 0.3.1 has
  **no** separate-rollout worker group, no checkpoint engine, no server-mode vLLM. You'd be
  building disagg + NCCL weight sync from scratch inside a dead verl version.
- Alternative framing — forward-port CMU's modified `RayPPOTrainer` onto verl main's
  separation/one_step_off trainer: their `ray_trainer.py` diff touches init, `_validate`, and
  ~200 lines of `fit()`; upstream `RayPPOTrainer` has drifted enormously since 0.3.1, so this
  is a conflict-heavy merge that converges on the same end state as (a) but drags along the
  whole CMU fit loop, checkpointing, validation, and logging surface.
**Size:** 1-2 weeks. **Risk: HIGH.** Rank 3 — do not pursue.

### (c) Two-process split: standalone vLLM server + external agent loop + PoC trainer
Run vLLM (OpenAI-compatible server) on the sampler GPU; a standalone client re-implements the
CMU env loop over HTTP; trajectories are shipped (files/queue) to the PoC trainer.
- Pros: total version decoupling (could even keep CMU's exact vllm 0.8.5 sampler-side); the
  agent loop becomes plain Python, easy to instrument.
- Cons: you must reconstruct training tensors (exact token ids, per-turn masks, padding) from
  server responses — the classic source of silent train/rollout skew; weight sync must be
  rebuilt (per-step checkpoint reload ≈7GB for 3B, ~10-30s/step, or a custom
  `collective_rpc` NCCL path — i.e., re-deriving what `CheckpointEngineManager` already does);
  plus a new trajectory ingestion path into the trainer's DataProto format.
**Size:** ~1 week+, several new moving parts. **Risk: MEDIUM-HIGH.** Rank 2 — fallback if the
agent-loop adapter in (a) hits a wall.

### (a2) Variant, noted for completeness: native verl AgentLoop
Write a registered `deepresearch` AgentLoop (verl main's own multi-turn mechanism, one
training row per trajectory with observation-masked `response_mask`). Cleanest verl-native
integration and zero per-turn manager round trips, but it **changes the training semantics**
vs the CMU paper (per-turn rows + episode reward broadcast vs single masked row), so the
benchmark would no longer be "the CMU workload". Keep in back pocket; not recommended for
fidelity.

## 4. Recommended plan (approach a) — file-level changes

Target: 1 trainer GPU + 1 sampler GPU on one 8×H100 node (2+2 TP=2 as fallback).
Work in a new dir, e.g. `disagg-deepresearch/timeslice-deepresearch/`, seeded from
`rl-time-slicing/verl/` — never edit the shared checkouts.

### Vendored from CMU (`agent_system/` subset, imports fixed for verl main)
| File | Action |
|---|---|
| `agent_system/multi_turn_rollout/rollout_loop.py` | Keep `TrajectoryCollector.{preprocess_single_sample, preprocess_batch, gather_rollout_data, vanilla_multi_turn_loop, multi_turn_loop}`; delete critique/GiGPO/rule-reward methods and the `from agent_system.critique...` / `behavior_reward` imports (drops langchain + google deps entirely) |
| `agent_system/multi_turn_rollout/utils.py` | Keep `adjust_batch`, `to_list_of_dict`, `torch_to_numpy`; `filter_group_data` optional |
| `agent_system/environments/env_manager.py` | Keep `DeepResearchEnvironmentManager` + deepresearch branch of `make_envs` only |
| `agent_system/environments/base.py` | As-is |
| `agent_system/environments/env_package/deepresearch/**` | As-is (`envs.py`, `projection.py`, `deepresearch/{env,retrieval,utils}.py`, `reward/`, `data/`, prompts); apply the existing k8s-job.yaml runtime patches at build time instead (retrieval URL → `$SEARCH_URL`, optional google imports, hardcoded paths) |
| `agent_system/reward_manager/episode.py` | As-is |
| Import fixes | `DataProto.truncate` (only in the non-train branch) → guard; verify `verl.utils.dataset.rl_dataset.collate_fn` signature at port time |

### PoC files modified
1. **`verl_timeslice_sync_trainer.py`** (module docstring name mismatch: fix
   `main_ppo_timeslice_sync.py`'s import `verl_timeslice_sync_modular_trainer` →
   `verl_timeslice_sync_trainer`):
   - `__init__`: accept `traj_collector`, `envs` (+ optional `val_envs`).
   - New ~10-line adapter:
     ```python
     class AgentLoopRolloutAdapter:            # passed as actor_rollout_wg
         def __init__(self, mgr, loop): ...
         def generate_sequences(self, batch):  # CMU calls this per turn
             batch.meta_info.setdefault("global_steps", ...)
             return run_coroutine(self.mgr.generate_sequences(batch))
     ```
   - `_generate()`: drop the `repeat(rollout.n)` + single generate; instead
     `gen_batch = one_step_trainer._get_gen_batch(batch)` (must retain `raw_prompt`,
     `data_source`) → `traj_collector.multi_turn_loop(gen_batch, adapter, envs)` →
     the returned per-turn batch **replaces** the dataloader batch (CMU semantics) →
     `adjust_batch(...)` → `compute_response_mask`.
   - `_compute_rewards_and_advantages()`: replace `extract_reward` with
     `EpisodeRewardManager(tokenizer)(batch)` → `token_level_rewards`; keep
     `compute_grpo_outcome_advantage(index=uid)` (uid = CMU question uid). Optionally port
     `apply_invalid_action_penalty`.
   - Timeslice boundaries unchanged: sampler lease covers the whole multi-turn loop
     (generation + search I/O = the natural sampler phase); trainer lease covers
     old_log_prob + update + weight sync.
2. **`main_ppo_timeslice_sync.py`**:
   - After tokenizer setup: `envs, val_envs = make_envs(config)`;
     `traj_collector = TrajectoryCollector(config, tokenizer, processor)`; pass both into
     `SyncTimesliceTrainer`.
   - Config additions (hydra `+env.*` or an OmegaConf merge onto the one_step_off config):
     `env.env_name=deepresearch, env.dataset=deepresearch_mhqa, env.max_steps=6,
     env.rollout.n=4, env.seed, env.use_explicit_thinking=True, env.is_evaluation=False,
     env.use_critique=False, env.use_rule_reward=False, env.rule_reward_coef=0,
     env.use_dense_reward=False, env.rule_number=5, algorithm.filter_groups.enable=False`.
   - Make the GpuClient orchestrator calls optional (env flag) so the benchmark can run
     without the timeslice stack for the baseline square-wave trace.
3. **verl pinning**: pin the volcengine/verl commit in deploy (replace the unpinned clone);
   re-derive the pod-patch sed fixes from `deploy_verl.sh:520-556` against that commit.

### Run configuration (1+1 on 8×H100)
- Data: regenerate CMU's dummy parquet via
  `python3 -m examples.data_preprocess.deep_research_data_prepare --train_data_size 128 --val_data_size 64`
  (questions actually come from the env JSON; parquet only feeds batch sizing +
  `raw_prompt`/`data_source`). `data.return_raw_chat=True`, `data.train_batch_size=8`,
  `data.max_prompt_length=10000`, `data.max_response_length=1024`.
- Trainer pool (`trainer.n_gpus_per_node=1`): Qwen2.5-3B FSDP ≈ 6GB params + 6GB grads +
  ~36GB Adam fp32 ≈ 48GB + activations w/ grad ckpt → fits 80GB. Fallback: fsdp
  param/optimizer offload, then 2-GPU trainer.
- Sampler pool (`+rollout.n_gpus_per_node=1`, TP=1): 3B weights ~7GB,
  `gpu_memory_utilization=0.8`, `max_model_len≈12288`, `max_num_batched_tokens=12288`,
  `actor_rollout_ref.rollout.n=1` (grouping via `env.rollout.n=4` → 32 concurrent envs, same
  as colocated run).
- Search server: run the pyserini BM25 server as a **sidecar container reusing the existing
  `deepresearch-benchmark:latest` image** (it has pyserini + the 12GB index bootstrap);
  main container uses the PoC's verl-main image. `SEARCH_URL=http://localhost:8877/search`
  must reach the Ray env actors (pass via ray runtime_env env_vars).
- Reward: keep dummy OPENAI_API_KEY for the GPU-trace benchmark (all-zero rewards, judged
  calls fail fast); if learning curves are wanted, patch `evaluate_afm_answer` to a local
  F1/EM check (~30 lines) instead of gpt-4o-mini.
- Reuse `gpu_monitor.sh` (100ms CSV) from `GPU-CR/benchmark-deepresearch/k8s-job.yaml`.
- Expected trace: sampler busy 75-146s+/step (6 manager round trips/step, tail visible as
  active-count decay), trainer busy ~100-200s (1 GPU vs the 2-GPU colocated ~100s), NCCL
  weight sync a few seconds → real multi-minute trainer idle per step, the reclaimable gap
  the time-slicing benchmark needs.

### Main risks
1. **Unpinned verl main** — PoC deploy tracks HEAD; APIs verified today can drift. Pin first.
2. **Agent-loop padding semantics** — per-turn outputs padded to `rollout.prompt_length`;
   with 10k prompts × 6 turns the per-row tensors are large; verify `union`/`collate` shapes
   on the first smoke run (can lower prompt_length for early turns if needed).
3. **Missing repo artifacts** — the hacked `single_controller/ray/base.py` exists only in the
   GCS bucket; retrieve or re-derive before deploying.
4. **Per-turn generation for finished trajectories** — CMU generates the full batch each turn;
   dropped rows still cost sampler time. This matches the colocated baseline (keep for
   apples-to-apples), with an optional "active-only" optimization later.

### Suggested sequence
1. Pin verl commit; stand up PoC sync-disagg as-is (GSM8K, Qwen2.5-0.5B) on 2 GPUs of the
   8×H100 node — validates pools + weight sync with zero new code.
2. Vendor agent_system subset + adapter + trainer edits; smoke-test with `env.max_steps=2`,
   batch 2, against the search sidecar.
3. Full config (max_steps=6, batch 8×4); capture 30+ min gpu_util.csv; compare against the
   colocated trace from the benchmark-deepresearch run.
