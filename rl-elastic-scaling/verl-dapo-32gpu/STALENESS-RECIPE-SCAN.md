# STALENESS-RECIPE-SCAN: stock recipes with staleness > 1 for the 32-GPU elastic showcase

Paper-only survey (2026-09-02). Successor scoping to GENBOUND-PLAN.md / GB-STATE.md.
Decisive new criterion: the recipe must ship, as-stock, a staleness allowance
greater than strictly-fresh (NeMo-RL `max_trajectory_age_steps > 1`; verl
`async_training.staleness_threshold > 0`), because the genbound campaign proved the
harvest-geometry lesson: with age-1 + full dispatch the tails are welded to the batch
and spares have nothing legal to generate (GB-STATE "HARVEST GEOMETRY FINDING",
S2d window 1: dispatched=0). With staleness headroom, spare samplers can legally
work ahead during trainer-idle windows, which is exactly the verl-PoC benefit
mechanism (PROPOSAL.md: +23% samples/GPU-hour at staleness_threshold=8).

Surfaces examined (all quotes are from files as they exist on disk today):
- NeMo-RL main: `/Users/aishuk/workspaces/nemo-rl-main` @ eaf02ef6 (2026-08-25)
- NeMo-RL v0.7.0 surface: `/Users/aishuk/workspaces/GPU-CR/elastic-rl-poc/mlperf-replica/elastic/ref-v0.7.0/`
- verl PoC pin: `/Users/aishuk/workspaces/GPU-CR/code-rlvr/verl` @ 983cb0f2 (2026-07-25), clean tree
- verl upstream main: `/Users/aishuk/workspaces/verl-recipes-scan` @ 896a9bba (2026-09-02, fresh shallow clone)

---

## A. Survey table

| # | Recipe (path relative to repo) | Framework / pin | Staleness field, AS SHIPPED (quoted) | Workload | Gen-bound? | 32-GPU (4x8 H200) fit |
|---|---|---|---|---|---|---|
| 1 | `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-24n8g-async-8off.yaml` | NeMo-RL main AND v0.7.0 (byte-identical, verified by diff) | `max_trajectory_age_steps: 8` (line 5); also `in_flight_weight_updates: true` (line 6) | GRPO math (OpenMathInstruct-2 via `grpo_math_1B.yaml` parent), Qwen3-30B-A3B, seq len 4096, 64x32=2048 rollouts/step | WEAK: 4096-token cap, same parent as the disqualified CISPO recipe; est. ~1.2-1.5:1 at a 16:16 split | NO stock: documented 24n8g = 192 GPUs (16 trainer nodes : 8 gen nodes); parent `grpo-qwen3-30ba3b-4n8g.yaml` is 32 GPUs but is sync/colocated |
| 2 | `examples/configs/recipes/llm/grpo-nanov3-30BA3B-4n4g-megatron_async_colocated.yaml` | NeMo-RL main ONLY (absent from v0.7.0) | `max_trajectory_age_steps: 4  # weight versions a rollout may span` (line 11) | GRPO math, Nemotron-3-Nano-30B-A3B, seq len 2048, 2 prompts x 8 gens (functional-test scale) | NO: 2048 tokens, tiny batch | 16 GPUs (4n4g), COLOCATED (megatron-backend generation, no disaggregated gen fleet to harvest) |
| 3 | `verl/experimental/fully_async_policy/shell/dapo_30b_a3b_base_math_fsdp.sh` | verl @ 983cb0f2 AND upstream main (only drift: `enable_sleep_mode` False -> True) | `staleness_threshold=0.6 # 0 0.3 1` (line 75); `trigger_parameter_sync_step=$((train_bsz / ( train_prompt_mini_bsz * require_batches))) # 8 16 32` = 16 (line 81); `partial_rollout=True` (line 82); `require_batches=1` (line 76) | DAPO math (dapo-math-17k, AIME-2024 val), Qwen3-30B-A3B-Base, 2K prompt + 20K response, n=16, mini_bsz 32 | YES: 30B-A3B + 20K CoT; same model class where GB S1 measured 2.27:1 on these exact H200s | EXACT: `n_nodes_rollout=2, n_nodes_train=2` x 8 GPUs = 16 gen : 16 train = 32 GPUs, as shipped |
| 4 | `verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_16_16.sh` | verl @ 983cb0f2 AND upstream main (byte-identical) | `staleness_threshold=0.1` (line 74); `trigger_parameter_sync_step=4` (75); `require_batches=4` (76); `partial_rollout=True` (77) | DAPO math (dapo-math-17k), Qwen2.5-Math-7B, 28K response, n=16 | YES: 28K CoT vs 7B trainer; README ships full convergence tables at 16:16/32:32/64:64 with wandb links | EXACT: `NNODES_ROLLOUT=2, NNODES_TRAIN=2` x 8 = 32 GPUs. Published-experiment variant used staleness 0.5 (README "Experiments" section, 16:16 row, 2.66x at 400 steps) |
| 5 | `verl/experimental/fully_async_policy/shell/grpo_30b_a3b_base_math_megatron_8_8_mis_trtllm.sh` | verl both pins | `staleness_threshold=0.5` (line 110); `trigger_parameter_sync_step=4` (111); `partial_rollout=True` (113) | GRPO math, Qwen3-30B-A3B-Base, 8K response, Megatron trainer, TRT-LLM rollout, rollout-IS | MODERATE (8K cap) | EXACT: 2R:2T nodes = 32 GPUs. TRT-LLM engine is outside our validated C/R surface |
| 6 | `verl/experimental/fully_async_policy/shell/run_qwen35_35b_a3b_math_dynamic_megatron.sh` | verl both pins (pin has a dead `${True}` grad_offload line, removed upstream) | `staleness_threshold=0.5` (line 50); `trigger_parameter_sync_step=4` (51); plus `async_training.use_dynamic_resource_scheduling=True` (line 120) | GRPO math, Qwen3.5-35B-A3B, 64K response, Megatron | YES, strongly (64K) | EXACT: `NNODES_ROLLOUT=2, NNODES_TRAIN=2` = 32 GPUs. BUT it runs verl's own native elastic (hybrid replicas on trainer GPUs), see C.3 |
| 7 | `verl/experimental/fully_async_policy/shell/dapo_7b_async_retool.sh` | verl both pins (identical) | `staleness_threshold=0.5` (line 67); `trigger_parameter_sync_step=4` (68); `partial_rollout=True` (70) | ReTool: DAPO-Math-17k with sandbox-fusion code-interpreter tool calls, Qwen2.5-7B, 16K response | MODERATE (tool latency adds gen wall) | Parameterized: `NNODES=${NNODES:-1}`, 4 rollout + 4 train GPUs per node; NNODES=4 gives 16:16=32 via env var only. Needs an external sandbox-fusion service |
| 8 | `verl/experimental/fully_async_policy/shell/grpo_30b_a3b_base_math_megatron_96_32(.mis).sh` | verl both pins | `staleness_threshold=0.5` (line 108) | GRPO math, 30B-A3B-Base, 8K response | MODERATE | NO: named/documented at 96:32 = 128 GPUs (shipped nnodes defaults are 2:2, but batch shape is sized for 128) |
| 9 | verl `fully_async_ppo_trainer.yaml` (config default, both pins, byte-identical) | verl | `staleness_threshold: 0.1` (line 12), `trigger_parameter_sync_step: 4` (16), `partial_rollout: True` (23) | n/a (base config) | n/a | n/a; this is the shipped DEFAULT: staleness and partial rollout on by default for every fully-async run |

Everything else scanned and excluded:
- All other NeMo-RL async recipes are `-async-1off` with `max_trajectory_age_steps: 1`
  (grpo-qwen3-235b-32n8g, nemotron3-super-120B 32n8g, deepseek-v3-64n8g, qwen3-32b-8n8g,
  llama3.1-8b 2n8g, qwen3-30ba3b-4n8g/4n4g, all quoted `max_trajectory_age_steps: 1` at line 5).
- All NeMo-RL gym/coding recipes (nemotron-3-super stage2_swe1/swe2, nanov3, ultra swe_teacher,
  workplace assistant, thinking-swe1-16n8g async-gym) ship `max_trajectory_age_steps: 1`.
- verl NPU shells: `dapo_30b_a3b_math_fsdp_npu.sh` has `staleness_threshold=0.75` (line 52), the
  single largest shipped value, but it is an Ascend-NPU recipe (not our hardware).

## B. NeMo-RL: definitive findings

1. **Stock staleness>1 exists but is thin.** Exactly two recipes in the entire tree ship
   `max_trajectory_age_steps > 1`: the 192-GPU perf recipe (#1, age 8, also in v0.7.0) and the
   16-GPU colocated functional recipe (#2, age 4, main only). The docs actively steer the other
   way: `docs/guides/async-grpo.md` line 204: "Start with `max_trajectory_age_steps: 1` and
   increase if needed for higher throughput".
2. **No stock gen-bound + staleness>1 + 32-GPU recipe.** Recipe #1 has the staleness but is a
   4096-token perf benchmark at 192 GPUs (the same weakly-gen-bound parent shape that
   disqualified the CISPO recipe in GENBOUND-PLAN section (a)3); recipe #2 is colocated and tiny.
   The gen-bound 32K recipe we are currently running (grpo-math-qwen3-30ba3b-megatron-tp4-32k)
   ships NO async block at all; the genbound campaign added age-1 async as delta D1.
   Combining the 32K recipe with age 8 would be config surgery on the decisive knob, exactly
   what "stock" forbids.
3. **No coding RLVR outside sandboxed gym.** Every code/SWE workload in NeMo-RL is a nemo-gym
   sandbox recipe (swe1/swe2/dynamo-swe, all age 1). There is no non-sandbox code-RLVR recipe,
   and no coding recipe at any staleness > 1. Definitive.
4. NeMo-RL still lacks partial rollout (abort-with-token-retention); GB-STATE built it
   out-of-tree (gen7 v3 segmented dispatch). Any NeMo-RL candidate inherits that machinery cost.

## C. verl: findings

1. **Staleness > fresh is the shipped default** for the fully-async recipe family
   (`verl/experimental/fully_async_policy`, present and structurally identical at our PoC pin
   983cb0f2 and upstream main; config file byte-identical, shells differ by one-line nits).
   Default config: `staleness_threshold: 0.1`, `trigger_parameter_sync_step: 4`,
   `partial_rollout: True`. Shipped shells go up to 0.6 (H200-relevant) and 0.75 (NPU).
2. **Semantics** (README, both pins): rollouter may produce up to
   `(1+staleness_threshold) * trigger_parameter_sync_step * require_batches * ppo_mini_batch_size`
   samples between parameter syncs. The work-ahead budget is therefore
   `staleness_threshold x sync-block`. For candidate #3 that is 0.6 x 512 prompts x 16 responses
   = 4915 legal ahead-of-need responses per sync block (~10 trainer updates worth). Partial
   rollout interrupts in-flight requests at sync and resumes them on new weights with tokens
   retained, so drains are lossless and tails are never welded to a batch. This is natively the
   dispatch geometry the genbound campaign had to hand-build (admission control + segmented
   continuation) and still could not make stock.
3. **verl now ships its own elastic** (both pins): `async_training.use_dynamic_resource_scheduling`
   parks sleeping hybrid rollout replicas on trainer GPUs and activates them dynamically
   (recipe #6 runs it at 32 GPUs stock). Strategic note for the showcase: the mechanism verl
   ships is sleep/wake colocation of rollout replicas; ours is C/R time-slicing of whole
   processes with a predictive controller. Recipe #6 is the natural third arm (stock-elastic vs
   our-elastic) if we want a comparative headline; it is not required for the primary claim.
4. **Provenance**: the fully-async README publishes full convergence + wallclock tables
   (Qwen2.5-Math-7B DAPO, 32/64/128 GPUs, 2.35x-2.67x at 400 steps, acc preserved, wandb links:
   hou-zg-meituan/fully-async-policy-*), plus a staleness ablation (0 / 0.1 / 0.3 / 0.5) showing
   larger staleness = faster with comparable final acc. This is the most production-credible
   async provenance in either framework.
5. **Coding preference, honest**: verl ships NO pure code-RLVR fully-async shell. Closest stock
   is ReTool (#7, code-interpreter tool calls, needs a sandbox-fusion service, excluded by the
   same no-sandbox design rule as SWE2). Our PoC's code-RLVR (Eurus-2-RL-Data code split,
   in-tree prime_code scorer, code-rlvr/NOTES.md decisions 1-2) reuses only shipped verl
   machinery but the dataset pairing is ours, not a shipped recipe. Offered below as an
   optional secondary arm, clearly labeled a delta.

---

## D. Top-2 deep dives

### D1. RECOMMENDED: verl `dapo_30b_a3b_base_math_fsdp.sh` @ 983cb0f2 (candidate #3)

Stock shape: Qwen3-30B-A3B-Base (3.3B active MoE), DAPO on dapo-math-17k, AIME-2024 val,
2K prompt + 20K response, n=16, temp 1.0, GRPO adv, token-mean loss, mini_bsz 32 prompts,
sync block 512 prompts (16 updates), FSDP trainer sp4, vLLM gen TP4 with CUDA graphs
(`enforce_eager=False`), gpu_memory_utilization 0.50, streaming gen_batch_size 1,
`staleness_threshold=0.6`, `trigger_parameter_sync_step=16`, `require_batches=1`,
`partial_rollout=True`. Resource split as shipped: 2 rollout nodes + 2 trainer nodes,
8 GPUs each = 16:16 on exactly our 4x8 H200.

Gen:train arithmetic at 32 GPUs (per trainer-update equivalent = 512 responses):
- Tokens: 512 x ~5k avg (base-model DAPO CoT at 20K cap, growing over training; band 3-7k)
  = ~2.6M gen tokens/update.
- Gen wall on 16 H200 = 4 x TP4 vLLM: GB S1 measured 4.1k tok/s aggregate on the same
  GPUs/model-class at 32K with enforce_eager; CUDA graphs + 22K ctx gives ~1.5-2x,
  so ~6-8k tok/s => ~325-430s per update-equivalent.
- Train wall on 16 H200 FSDP: fwd+bwd 6 x 3.3e9 x 2.6e6 = ~5.1e16 FLOPs; no ref logprob
  (KL fully off), no old_logprob recompute (`bypass_mode: True` default,
  `use_rollout_log_probs: True`); at 5-8% MoE MFU => ~40-65s compute, ~90-150s with
  optimizer/data/overhead.
- **Predicted gen:train ~2.4:1 (band 1.8:1 to 4.5:1), structurally gen-bound**, consistent
  with the 2.27:1 we already measured for this model class on this hardware (GB S1) and with
  meituan shipping this exact split with the largest staleness value in the H200 shell set.

Staleness semantics for the harvest: work-ahead budget 0.6 x 8192 = 4915 responses per sync
block. Trainer-idle windows occur whenever the MessageQueue underfills (chronic when
gen-bound); spares that join the AgentLoop load balancer generate real, legal, consumable
samples the entire window, and at sync time partial rollout retains their in-flight tokens.
No admission-control redesign, no segmentation patch, no welded tails: the two structural
blockers of the genbound campaign do not exist in this stack.

Topology sketch (elastic arm):
- Fleet: 16 gen GPUs = 4 x TP4 vLLM AgentLoop servers (stock).
- Trainer: 16 GPUs, one FSDP world (stock).
- Spares: 2-4 x TP4 vLLM engines parked on trainer GPUs (TP4 = producer sharding, wcache
  1:1 shard invariant carries over from GB design decision 1); controller = PoC controller
  (buffer depth + fill-rate ETA gates) pointed at the message-queue RPC, switch mechanism =
  trainer rank offload/suspend + spare wake, drain = deregister engine from load balancer,
  partial-rollout client preserves in-flight work (PoC-proven lossless).

Machinery reuse and drift risk:
- **Version drift: effectively zero.** The PoC stack targets verl@983cb0f2; this shell,
  its config, the message-queue, load-balancer membership, and partial-rollout client all
  exist at that pin; pin-to-main drift on the recipe is one cosmetic line
  (`enable_sleep_mode=False` -> `True`).
- Reused as-is: launcher rewire, spare-sampler subclass, controller policy core, decision
  logging, drain/no-harm protocol (26-cycle + regime-shift + no-harm evidence already in
  PROPOSAL.md at 2-3 GPUs).
- Genuinely new: (1) multi-rank trainer suspend, 16-GPU FSDP world (PoC was 1 rank;
  GB proved 12-15s/rank yield offloads on 16-rank Megatron on these nodes, and the multi-GPU
  C/R stack covers the C/R variant, but FSDP-world suspend at 16 ranks is the S2-class risk
  item); (2) spare wcache for a 61 GB model fed from the vLLM fleet (GB pipelined-wcache
  design ports directly). Known shipped nit: the shell references
  `--config-name='fully_async_dapo_trainer.yaml'` which does not exist at either pin; the
  one-line fix is `fully_async_ppo_trainer.yaml` (all DAPO-ness is passed as CLI overrides);
  registered delta, both arms identical.
- Model staging: Qwen3-30B-A3B-**Base** (~61 GB) is a sibling of the already-mirrored
  Qwen3-30B-A3B; one-time GCS mirror. dapo-math-17k + aime-2024 parquets are ~30 MB.

>15% claim feasibility, honest:
- Baseline trainer idle at 2.4:1 gen:train is chronic ~55-65% of wallclock (PoC measured 51%
  at 1:1 scale and converted it to +23%/GPU-hr with ~30s switches). Here, lending 16 trainer
  GPUs as 4 TP4 spares doubles serving capacity during windows; even at the conservative
  1.8:1 edge of the band the fleet-seconds added exceed 25%, and the staleness budget
  (~10 update-equivalents) is far larger than any window can produce, so nothing generated
  is wasted.
- Predicted: **+18% to +30% samples/GPU-hour, -15% to -25% time-per-sync-block**, clearing
  >15% if (gate S1) measured gen:train >= 1.5:1 and (gate S2) switch-out-to-serving <= 60s
  median. Miss risk concentrates in the multi-rank suspend wall; GB measured 12-15s offloads
  for the same tensor volume, giving margin.
- Structural tailwind: DAPO response lengths grow over training, the band drifts upward.

### D2. Runner-up: NeMo-RL `grpo-qwen3-30ba3b-24n8g-async-8off.yaml` (candidate #1)

The only at-scale stock staleness>1 recipe in NeMo-RL (age 8, in-flight weight updates true,
disaggregated 16T:8G nodes, vLLM TP2 async engines) and it is present byte-identical in the
v0.7.0 run image lineage. It fails the other two criteria:
- Gen-bound: parent `grpo-qwen3-30ba3b-4n8g.yaml` sets `max_total_sequence_length: 4096`
  and 64x32=2048 rollouts, GBS 512. Tokens/step ~5.5M but at 4K cap the gen wall on 16 GPUs
  (8 x TP2) is ~450-600s vs train+refit ~400-500s on 16 GPUs: **~1.2-1.5:1**, the CISPO
  problem again; the harvest at best funds a ~10% claim.
- 32-GPU fit: documented at 192 GPUs; an honest 32-GPU map (16T:16G, its 2:1 node ratio is
  not expressible at 4 nodes anyway) is a registered delta of the same class as GB's D2/D3,
  acceptable but weaker than a recipe whose shipped nnodes are exactly ours.
- Machinery: NeMo-RL stack is current and battle-tested by GB, but age-8 changes nothing
  about the burst-dispatch geometry by itself; we would still carry the out-of-tree
  admission/segmentation layer (gen7 v3) to convert windows, and its replay-buffer eviction
  math (`buffer_size = num_prompts_per_step x max_trajectory_age_steps x 2`) merely permits
  older samples rather than providing streaming credit semantics.
Verdict: legal but weakly harvestable; keep only if a NeMo-RL-branded showcase becomes a
hard requirement, with the claim renegotiated toward ~10%.

---

## E. Recommendation

**Run the elastic showcase on verl `verl/experimental/fully_async_policy/shell/dapo_30b_a3b_base_math_fsdp.sh` at our existing PoC pin 983cb0f2**, stock 16 gen : 16 train on the 4x8 H200 pool, with the shipped
`staleness_threshold=0.6`, `trigger_parameter_sync_step=16`, `partial_rollout=True`.
Rationale, in order: (1) it is the only recipe in either framework that is simultaneously
strongly gen-bound (30B-A3B + 20K CoT), staleness-legal as shipped (0.6 of a 512-prompt sync
block = ~4.9k responses of work-ahead), and documented at exactly 32 GPUs; (2) it runs at the
pin our entire PoC stack already targets, so version drift is one cosmetic line; (3) its
recipe family carries the strongest published async provenance available (meituan convergence
+ staleness-ablation tables with wandb links); (4) partial rollout at the framework level
removes both structural blockers the genbound campaign spent S2 fighting. Coding preference:
not satisfiable stock (section B.3, C.5); optional secondary arm swaps in the PoC's
Eurus-2 code split + prime_code scorer (machinery-stock, dataset delta, registered), which
also reconnects the showcase narrative to the original code-RLVR PoC.

NeMo-RL finding, stated definitively: NeMo-RL ships NO stock recipe that is gen-bound,
staleness>1, and 32-GPU-mappable; its only at-scale staleness>1 recipe (age 8) is a
4096-token perf benchmark at 192 GPUs, and its only other one (age 4) is a 16-GPU colocated
functional config absent from v0.7.0.

## F. Run-plan sketch (32 GPUs, spot H200 pool, node-registry discipline)

| Stage | What | Gate | Wall | GPU-hr |
|---|---|---|---|---|
| S0 staging | Mirror Qwen3-30B-A3B-Base (61 GB) to GCS + 4 nodes; stage dapo-math-17k/aime-2024 parquets; image = PoC verl image at 983cb0f2 + PoC elastic code; fix config-name nit (both arms) | pulls + sha verified | ~2h | ~0 |
| S1 smoke | Stock baseline, 2 sync blocks (~32 updates); measure trainer idle_ratio, gen tok/s, gen:train, response-length histogram (trainer/idle_ratio and rollouter/idle_ratio are shipped metrics) | gen:train >= 1.5:1, trainer idle >= 30% | ~3h | 96 |
| S2 shakedown | Spares boot (2-4 x TP4 on trainer GPUs) + controller; 10 consecutive clean windows: 16-rank suspend, spare wake+serve (load-balancer join), spare samples consumed (fully_async/count/stale_samples_processed > 0), partial-rollout drain, resume | 10/10 clean, switch-out-to-serving <= 60s, zero lost samples | ~4h | 128 |
| S3 baseline scored | Stock arm, N sync blocks (N pre-registered from S1, target ~6h wall), val at start/end via shipped test_freq | completes <= 8h | ~5-6h | 160-192 |
| S4 elastic scored | Identical N, seed, endpoints; controller auto-live after dry-run | completes <= 8h | ~4-5h | 128-160 |
| slack | one S2 retry or scored incident | | ~2h | 64 |

Totals: ~2.5-3 days wall, **~580-640 GPU-hours** (larger than GB's 390-450 because S1/S2
carry the new multi-rank-suspend risk and sync blocks are long). Headline metrics: samples
per GPU-hour, time per sync block at identical endpoints, trainer idle_ratio conversion,
reward/acc overlay as the no-harm check. Optional third arm (stretch, +~150 GPU-hr):
verl-native `use_dynamic_resource_scheduling=True` on the same workload for a
stock-elastic vs C/R-elastic comparison.
