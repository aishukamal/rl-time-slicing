# Async + Multi-Turn Tool Rollout: Scoping Notes

Phase 3 of the time-slicing RL benchmark: does verl's OWN fully-async disaggregated
mode, run essentially out-of-the-box on-policy, show natural trainer idle when the
workload is multi-turn tool-calling (search agent)?

Scoped against: volcengine/verl main @ `578b9f2b` (2026-07-23), fresh clone in
`/Users/aishuk/workspaces/GPU-CR/async-multiturn/verl/`.

## Verdict: composes out of the box (with one piece of user-supplied glue)

The fully-async rollouter IS the agent-loop stack. `FullyAsyncAgentLoopManager`
(verl/experimental/fully_async_policy/fully_async_rollouter.py:305) subclasses
`AgentLoopManager` from verl/experimental/agent_loop/agent_loop.py — the same code
path that runs `ToolAgentLoop` for multi-turn tool calling. There is no separate,
single-turn-only generation path in async mode.

Direct upstream proof: `verl/experimental/fully_async_policy/shell/dapo_7b_async_retool.sh`
is an official fully-async + multi-turn-tool (retool/sandbox-fusion) run script using
`actor_rollout_ref.rollout.multi_turn.enable=True ... tool_config_path=... format=hermes`
under `verl.experimental.fully_async_policy.fully_async_main`.

**The one gap:** verl main REMOVED the in-tree `verl.tools.search_tool.SearchTool`
(see docs/sglang_multiturn/search_tool_example.rst: "must now be provided by users").
`verl/tools/` only ships base_tool / function_tool / schemas / tool_registry. So the
only glue we need is a ~60-line `BaseTool` subclass (`WikiSearchTool`) — which is
actually convenient: we control the HTTP request format, so it talks the Serper-style
schema of our existing local pyserini Wikipedia server (benchmark-deepresearch) directly.
No server-side adaptation needed.

Also gone from main: `recipe/` is empty (recipes moved out of tree) and
`examples/sglang_multiturn/` no longer exists. Only `examples/data_preprocess/preprocess_search_r1_dataset.py`
and the reward fn `verl/utils/reward_score/search_r1_like_qa_em.py` remain in-tree.

## How the pieces wire together

### 1. Agent selection (ToolAgentLoop vs single-turn)

- Registry: `@register("tool_agent") class ToolAgentLoop` (agent_loop/tool_agent_loop.py:99),
  `@register("single_turn_agent")` in single_turn_agent_loop.py.
- Per-sample routing via the dataset's `agent_name` non-tensor field; if absent, falls
  back to `actor_rollout_ref.rollout.agent.default_agent_loop` (default `single_turn_agent`,
  agent_loop.py:610-612, verl/workers/config/rollout.py:74).
- We set BOTH: an `agent_name = "tool_agent"` column in the parquet and
  `actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent`.

### 2. Tool loading

- `AgentLoopWorker` loads tools once from
  `actor_rollout_ref.rollout.multi_turn.tool_config_path` via
  `load_all_tools()` → `initialize_tools_from_config()` (agent_loop.py:530-533,
  verl/tools/tool_registry.py).
- `class_name` is resolved with `importlib.util.find_spec(module)` — the module must be
  importable inside Ray workers. Simplest robust approach: after the runtime
  `git clone verl && pip install -e .`, copy our `wiki_search_tool.py` into
  `verl-src/verl/tools/` so `class_name: verl.tools.wiki_search_tool.WikiSearchTool`
  resolves everywhere without PYTHONPATH games.
- Tool interface (verl/tools/base_tool.py): async `execute(instance_id, parameters, **kwargs)
  -> (ToolResponse(text=...), tool_reward: float, metrics: dict)`. `create`/`release`
  defaults are fine for a stateless search tool.

### 3. Tool-call format

- `multi_turn.format=hermes` → `ToolParser.get_tool_parser("hermes", tokenizer)`;
  Qwen2.5-Instruct emits hermes-style `<tool_call>{"name": "search", "arguments": {...}}</tool_call>`
  natively via its chat template (tool schemas injected by `apply_chat_template(tools=...)`).
- Tool schema (ours, mirrors the removed SearchTool): function `search` with
  `query_list: array[string]`.
- IMPORTANT non-default: `multi_turn.max_tool_response_length` defaults to **256 chars**
  (rollout.yaml:205) — useless for retrieved passages. We raise it to 4096.
- `multi_turn.max_user_turns` / `max_assistant_turns` default to null (no limit,
  ~max_len//3); we cap at 8 to bound trajectories.

### 4. Dataset format (search-R1 style)

From `examples/data_preprocess/preprocess_search_r1_dataset.py` + RLHFDataset
(verl/utils/dataset/rl_dataset.py:387-409):

- `prompt`: chat messages list (system + user with question). Passed through as
  `raw_prompt` — requires `data.return_raw_chat=True`. ToolAgentLoop consumes
  `kwargs["raw_prompt"]` directly.
- `data_source`: `searchR1_hotpotqa` → default reward routing
  (verl/utils/reward_score/__init__.py:94-104) hits `search_r1_like_qa_em.compute_score`,
  which extracts `<answer>...</answer>` and does EM against `ground_truth["target"]`.
  So NO custom reward function needed; default `reward.reward_manager.name=naive` works.
- `reward_model`: `{"style": "rule", "ground_truth": {"target": [golden answers]}}`
  (the `target` key is mandatory — compute_score indexes `ground_truth['target']`).
- `extra_info`: `{"need_tools_kwargs": True, "tools_kwargs": {"search": {"create_kwargs": {...}}}, ...}`
  — RLHFDataset lifts `extra_info.tools_kwargs` into the non-tensor batch; keyed by tool
  name (`search`).
- Source data: HF `PeterJinGo/nq_hotpotqa_train` (the search-R1 dataset; has `question`,
  `golden_answers`, `data_source` columns). We filter to hotpotqa rows and take 512
  train / 64 val. (CMU MHQA JSONs were the alternative; HF parquet is simpler and is
  what the in-tree preprocess script targets.)

### 5. Async trainer config surface (fully_async_ppo_trainer.yaml defaults)

- `async_training.staleness_threshold: 0.1` (we set 0 → on-policy),
  `trigger_parameter_sync_step: 4` (we set 1 → sync every trainer step),
  `require_batches: 1` (default, keep), `partial_rollout: True` (default, keep).
- `data.gen_batch_size=1` and `data.train_batch_size=0` required in async mode (Phase 1 lesson).
- Rollouter finishing `rollout.total_rollout_steps` sends a None sentinel that kills the
  trainer → set 65536 (never reached) + `trainer.total_epochs=10` so the 512-prompt
  dataset recycles; bound the run with `timeout 6000` in the job script instead, then
  dump the GPU trace.
- Reward in async mode runs through the agent/reward loop (`fully_async_main.py` calls
  `migrate_legacy_reward_impl`); default naive manager + default_compute_score is the
  out-of-the-box path.

## Deliberate choices vs defaults (job spec)

Deliberate (the experiment): dataset (hotpotqa multi-hop), multi-turn search tool at
http://localhost:8877 (pyserini Wikipedia sidecar, Serper format — reused verbatim from
benchmark-deepresearch), 1 trainer GPU + 1 rollout GPU, on-policy knobs
(staleness_threshold=0, trigger_parameter_sync_step=1), max_tool_response_length=4096,
turn caps 8.

Defaults kept (no starvation engineering): gpu_memory_utilization=0.8,
require_batches=1, partial_rollout=True, naive reward manager, TP=1,
rollout.n=8, ppo_mini_batch_size=64 (= 8 prompts/step, moderate).

Model: **Qwen2.5-3B-Instruct** (needs real tool-calling ability; 0.5B is the fallback if
3B FSDP OOMs on the 1 trainer GPU — unlikely on H100-80GB: ~36GB weights+grads+Adam for
3B bf16/fp32-master + activations with grad checkpointing + dynamic bsz).

## Expected failure modes

1. Qwen2.5-3B not emitting hermes tool calls reliably at temperature 1.0 → few/zero
   search turns → workload degenerates to single-turn (watch `num_turns` in logs and
   tool-call metrics).
2. 3B trainer OOM on 1 GPU during log_prob/update at 6K total len → fallback to
   Qwen2.5-0.5B-Instruct (weaker tool-calling; note in report).
3. verl main API drift vs the `verlai/verl:vllm020.dev2` image deps (Phase 1 cloned
   main at runtime too and worked, but main moves fast — e.g. `checkpoint_engine`
   config, TransferQueue import).
4. Pyserini index download (~12GB) slow on first start → job waits up to 40 min for the
   sidecar before training (same as Phase 2 behavior).
5. `agent_name` column dtype: must be plain string column in parquet (object array) —
   otherwise falls back to default_agent_loop (which we also set to tool_agent, so benign).
6. Reward always 0 if the model never wraps answers in `<answer>` tags → training still
   runs (GRPO with all-zero advantage is degenerate but the GPU-trace experiment is
   about idle patterns, not learning quality).

## Files

- `k8s-job-async-multiturn.yaml` — the job (ConfigMap: gpu_monitor.sh,
  local_search_server.py, wiki_search_tool.py, search_tool_config.yaml, data_prep.py,
  run_async_multiturn.sh; Job: search sidecar + benchmark container, 2x H100 on
  h100-mega-8gpu-spot-b).
- `wiki_search_tool.py`, `data_prep.py` — standalone copies (py_compile-checked), same
  content as embedded in the ConfigMap.
- `verl/` — scoping clone (read-only reference, not used by the job; the job clones
  verl main at runtime like Phase 1).
