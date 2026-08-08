# M3 — Experiment Matrix: Elastic vs Static/Colocated Arms + Dynamism Evidence (elastic-rl-poc)

*Runs executed 2026-08-07/08 on `verl-research-cluster` (asia-southeast1-b), nodes trb7/q36v/w0x7.
All arms: verl `983cb0f24443f87b3d161fad318445130a620b07`, DeepSeek-R1-Distill-Qwen-1.5B, Eurus-2
code split (512 prompts, seed 17), GRPO, n=8, 16K response cap, 64 groups (=512 trajectories) per
optimizer step, lr 1e-6 — except where an arm's own design says otherwise (Run C inverts the
balance; Run B flips the cap mid-run by design).*

## The two headline claims

**Comparison currencies (all cross-arm numbers):** wall-clock per optimizer step of 64
message-queue samples = 64 prompt GROUPS = 512 trajectories; and samples per GPU-hour counting
ALL GPUs the arm uses. Phase decompositions appear per-arm for diagnostics only — sync and async
architectures account phases differently and their phase timers must not be compared directly.

### Angle 1 — same GPUs, faster (banked: M0 vs M2)

| | static 1:1 disagg (M0) | **elastic (M2)** | Δ |
|---|---|---|---|
| GPUs | 2 | 2 | = |
| Wall-clock / 64-group step | 619.0 s | **502.7 s** | **−18.8%** |
| Groups / GPU-hour | 186.1 | **229.2** | **+23.2%** |
| Trajectories / GPU-hour | 1,489 | 1,833 | +23.2% |
| Response tokens / GPU-hour | 17.0 M | 20.9 M | +23% |

Same hardware, same recipe, same NCCL worlds — the elastic controller time-multiplexes the
trainer's gen-wait blocks into a warm second rollout replica and returns **~23% more samples per
GPU-hour**.

### Angle 2 — fewer GPUs, near-identical per-GPU efficiency (M2 vs Run D)

| | **elastic (M2)** | static 1:2 disagg (Run D) |
|---|---|---|
| GPUs | **2** | 3 |
| Wall-clock / 64-group step | 502.7 s | **321.0 s** |
| Groups / GPU-hour | 229.2 | 239.2 |
| Trajectories / GPU-hour | 1,833 | 1,914 |
| Regime | rollout-bound, harvested | trainer-bound; idle moved to samplers (2-15% + trainer 17.5%) |

The 1:2 sampler:trainer ratio is (approximately) this workload's optimum, and it is faster in
wall-clock — but it needs a third GPU. Per GPU-hour the elastic 2-GPU system delivers
95.8% of the static-1:2 arm's efficiency, without provisioning the extra sampler, and
with the drift-tracking / no-harm properties Runs B and C demonstrate (a static 1:2 split is
frozen at launch; when the workload's balance moves, its idle moves — see Run D's own idle
profile below).

## Reference ladder (all 2-GPU code-rlvr arms, 64-group steps)

| Arm | Step time | Source |
|---|---|---|
| M0 static 1:1 disagg | 619.0 s | m0-results |
| Elastic scaffolding idle (0 switches) | 615.6 s | m2-results/attempt1 |
| M1 manual switching | 553.1 s | m1-results/run4 |
| **RUN A2 colocated sleep/wake (stock verl, batch-parity)** | **522.7 s ± 29.9** | m3-results/colocated-b64 |
| M2 autonomous controller | 502.7 s ± 25.2 | m2-results/run2 |
| M2 best-case reconstruction | 501.8 s | m2-results |
| RUN B pre-flip live window (M2-config reproduction) | 506.8 s | m3-results/regime-shift |

## RUN A — colocated sleep/wake arm (C3b, the architecture-alternative)

**Question:** does stock verl colocated hybrid (both GPUs shared by FSDP trainer + vLLM engines,
sleep/wake between phases — no elastic machinery, no shim, stock `verlai/verl:vllm020.dev2`)
beat the elastic disaggregated system on the same 2 GPUs?

**Answer: no — elastic wins by ~4% (502.7 vs 522.7 s/step); call it a win-to-tie given σ≈30s.**

### Config parity audit (required before any cross-arm number)

- **Batch-size semantics trap (first launch invalidated):** in the fully-async path a
  message-queue "sample" is one prompt's whole GRPO group, so `ppo_mini_batch_size=64` =
  **64 groups = 512 trajectories** per optimizer step (`required_samples = ppo_mini_batch_size ×
  require_batches`, fully_async_trainer.py:159). In the sync path `ppo_mini_batch_size` is in
  PROMPT units ×n (ray_trainer.py:1327-28). The first colocated launch (train_batch_size=8) did
  512/8 = **1/8 the work per step**; its data is kept as a diagnostic only
  (m3-results/colocated/, steady 125.4 s/step, and per the units directive it is excluded from
  all cross-arm tables). **RUN A2** (m3-results/colocated-b64/) reran with
  `train_batch_size=64, ppo_mini_batch_size=64` → 512-trajectory minibatch, 1 optimizer step per
  training step — true parity.
- **update_actor discrepancy reconciled:** async 281 s (1 trainer GPU) vs colocated 145.8 s
  (2 GPUs) is exactly the GPU count: per-token update time 0.0460 vs 0.0236 ms/token, MFU/actor
  0.31 in BOTH arms. No hidden config difference; the async trainer is not less efficient per
  token, it just has half the training silicon and 8× the tokens the first launch had.
- **Verified identical:** model, dataset+seed (512 prompts), rollout.n=8, temperature 1.0 /
  top_p 1.0 / top_k −1, 16384 cap / 2048 prompt cap, 64 groups per optimizer step, lr 1e-6,
  grad_clip 1.0, token-mean loss, dynamic bsz @ 32768 tok/GPU, gradient checkpointing,
  remove-padding. Response-length mean 11,506 (A2) vs 11,439 (M0) / ≈11.4K (M2); reward mean
  0.196 (A2 steps 3-17) vs 0.196 (M0) — same workload, quantitatively.
- **Documented deviations (forced or stock-inherent):** `gpu_memory_utilization` 0.6 vs 0.8
  (FSDP state stays resident during colocated generation; 0.8 does not fit),
  `checkpoint_engine=naive` (stock colocated default; async arms use the nccl backend),
  and the sync trainer **recomputes old_log_prob on the actor (41.2 s/step)** while the
  fully-async path trains on rollout-computed logprobs — an algorithmic difference of the
  architectures themselves (measured logprob agreement in A2: rollout_probs_diff_mean ≈ 0.005).

### RUN A2 result (steady = steps 3-17, n=15)

- Step time **522.7 s ± 29.9** (449.8-576.3); groups/GPU-hour **220.4** vs elastic **229.2**
- Diagnostic decomposition (per-arm only): gen 332.3 s, old_log_prob 41.2 s, update_actor
  145.8 s, **update_weights (wake+weight-sync) 1.9 s/step ≈ 0.4%** — sleep cost is inside the
  gen timer; vLLM server-actor sleep logs don't surface in the driver log, so 1.9 s is the
  measurable sleep/wake overhead floor, and the gen/step totals bound the rest
- GPU util: both GPUs 84% mean, 14.7% idle (<10%) — colocation's residual idle is the
  phase-transition tail, not a per-GPU imbalance
- Rewards healthy: per-step score mean band 0.11-0.27, mean 0.196

### Semantics footnote (applies to every colocated-vs-async comparison)

Colocated-sync is strictly **on-policy**: every batch is generated by the exact current weights.
The fully-async arms run at `staleness_threshold=8` — but in every 2-GPU async run (M0, M2, B)
the staleness budget was **never touched**: mq depth 0 at every collect, dropped_stale 0,
dropped_samples 0; trajectories trained at their generation version. The elastic result does not
buy its throughput with data freshness on this workload. The one arm whose queue DOES buffer is
Run D (static 1:2): its wall-clock win engages the staleness budget (see the Run D freshness
note) — that asymmetry is in the summary table.

## RUN B — forced regime shift 16K→8K mid-run (C5: it's *dynamic*)

**Setup:** byte-identical to the M2 PASS config (same image `verl-cr-shim:m1`, same controller,
s=8, RUN_SECONDS=16200, node trb7) plus an in-pod driver (`m3/regime_flip.py`) that, after step
12 completed, flipped the generation length cap on the LIVE vLLM server actors (R1 + parked R2)
from 16384 → 8192 — restart-free. Mechanism (verified against the pin): per-request `max_tokens`
defaults to the server's `config.response_length` (vllm_async_server.py:574-590), and
`RolloutConfig._mutable_fields` explicitly whitelists `response_length`; the flip is a
`__ray_call__` setattr on each server actor. Tensor shapes/padding stay 16384 — batches keep one
layout across the boundary; in-flight 16K requests finish at their old cap.

**Flip executed 05:16:07Z after step 12, readback-verified** (`regime_flip.jsonl`:
r1_server_0 16384→8192, r2_server 16384→8192).

### The workload moved, and the controller followed

| Window | Steps | Step time | Trainer gen-wait | update_actor | resp. length |
|---|---|---|---|---|---|
| Pre-flip live | 6-12 (n=7) | **506.8 s** | 207.5 s | 276.9 s | 11,241 |
| Transition (in-flight 16K drain) | 13-14 | 438.1 s | 143.3 s | 280.1 s | 10,979 |
| Post-flip | 15-46 (n=31) | **271.2 s** | 71.7 s | 176.0 s | 7,720 |

The pre-flip window independently **reproduces M2** (506.8 vs 502.7 s) — the M2 result is not a
one-off.

- **Adaptation ≤ 2 steps:** steps 13-14 are the drain; from step 15 the new regime is fully
  established and the controller's behavior has already changed with it.
- **Switch rate:** pre-flip 1.0 pair/step (every gen-wait block harvested, M2 behavior);
  post-flip **0.65/step with block-skipping** — the gate stayed closed for blocks 18, 20, 21,
  23, 26, 28, 30, 38, 39, 41, 44… as ETAs fell against the `1.5×round_trip` (~98-130 s)
  threshold.
- **The boundary evidence in `decisions.jsonl` no-action reasons:** `eta_below_threshold`
  **0 pre-flip → 310 post-flip** (the ETA gate rejecting now-too-short blocks is a post-flip
  phenomenon by construction), `not_blocked` 1,171 → 3,005 (blocks are a smaller fraction of
  each step), `eta_above_trigger` (R2-active, switch-back gate) 181 → 6 (R2 windows collapsed
  from ~109 s toward the failsafe). Every one of 5,000+ no-action ticks carries its reason.
- **No harm through the shift:** **29/29 switch pairs completed and param_sync-verified**
  (9 pre-flip, 20 post-flip), weight versions 0→45 strictly sequential, 0 RayActorError,
  0 failed cycles, 0 late wakes (all recorded wake-vs-batch-ready gaps ≤ −2 s; post-flip backs
  fire on the hard-collect failsafe at 60/64, i.e. before batch-ready by construction),
  mq drops 0 throughout. Switch latencies unchanged across the boundary (post-flip mean
  28.0 s in / 37.0 s out).
- **Post-flip switching remained net-positive:** steps with a switch pair averaged **261.8 s**
  vs **288.2 s** for gate-closed steps — the controller kept harvesting only where the ~65 s
  round-trip still fit profitably inside the (now ~60-130 s) blocks.
- Honest nuance for the design note: with `c=1.5` the gate sits close to break-even in the 8K
  regime — the controller trims frequency rather than halting outright, and the per-cycle margin
  is small. If the flip had been deeper (e.g. 4K), `eta_below_threshold` would dominate and
  switching would stop entirely — that end-state is what Run C demonstrates.
- Footnote: post-flip reward mean drops 0.20 → 0.11 — expected, the 8K cap truncates long-CoT
  solutions (this is a property of the forced shift, not of the elastic machinery; response
  clip behavior is identical in a static 8K run).

## RUN C — train-heavy no-harm control (C5: degrade to exactly nothing)

**Setup:** the M2 elastic config with the balance inverted on the same code-rlvr recipe:
`max_response_length=2048`, `rollout.n=4` (still 64 groups/step = 256 trajectories), s=8, node
w0x7 (freshly timeslice-labeled; its own snapshot-agent). Two phases, RUN_SECONDS=5400 each —
both completed the FULL 80-step (10-epoch) workload inside the window:

| Phase | Steps | Step time | Gen-wait | update_actor | Switches |
|---|---|---|---|---|---|
| Control (scaffolding, R2 parked, **no controller**) | 2-80 (n=77) | **42.24 s ± 4.00** | 11.87 s | 21.78 s | 0 (none possible) |
| **Armed (controller LIVE)** | 2-80 (n=77) | **42.58 s ± 4.10** | 11.90 s | 21.82 s | **0** |

- **Zero switches, zero would-switches.** The dry-run window produced no would-ins (nothing to
  align on), so `--auto-live` engaged via its documented failsafe
  (`via=auto_failsafe_no_would_in` after dry_run_steps+2) — live gate ≡ shadow gate, so this
  changes nothing about the property under test. The controller then ran live for the entire
  remaining run: **1,656 decision records, every no-action tick with its reason:**
  `not_blocked` 1,181, `eta_below_threshold` 454, `fill_ema_warmup` 15, `signal_error_mq` 2
  (transient), actions: none.
- **Step-time parity: +0.34 s (+0.8%), well inside noise (σ ≈ 4 s).** The armed system is
  indistinguishable from the controller-absent control.
- Regime verification (as specified): gen-wait blocks 11.9 s mean ≪ the ~98-128 s ETA gate and
  ≪ the ~65 s switch round-trip — blocks exist but are strictly unprofitable, which is precisely
  the no-harm scenario ("small/absent"); the ETA gate never cleared once in ~48 minutes of live
  polling.

**C5 no-harm verdict: PASS.** Elastic machinery fully armed on a trainer-bound workload does
exactly nothing, on evidence, at zero measurable cost.

## RUN D — static 1:2 disaggregated arm (the true static competitor)

**Setup:** stock verl at the pin, no elastic machinery/shim/controller. 1 FSDP trainer GPU +
2 always-on vLLM rollout engines (`rollout.nnodes=2, n_gpus_per_node=1`, TP=1, both engines full
members of the weight-sync group), spanning two nodes via a 2-pod Ray cluster (head q36v:
trainer + engine 1; worker w0x7: engine 2; NCCL weight sync over the pod network,
`NCCL_SOCKET_IFNAME=eth0`). Same recipe, s=8, RUN_SECONDS=9000. **3 GPUs counted.**

**Result (steady = steps 3-26, n=24): 321.0 s ± 24.8 per 64-group step** (280-368);
planned 9000 s timeout, 25 steps, rc=124 path confirmed in the epilogue.

- Regime flipped to **trainer-bound**, as predicted: trainer gen-wait collapsed to **25.0 s/step**
  (vs 316 s at 1:1) while update_actor is the floor at 281.5 s. Response length 11,450 and reward
  mean 0.193 — same workload band as every other 16K arm.
- **Where the 1:2 split leaves the idle:** trainer GPU 17.5% idle (<10% util, 100 ms trace);
  engine-1 GPU 2.0% idle; engine-2 GPU 14.7% idle — the samplers now outrun the trainer and
  pause on the staleness budget, so the residual idle migrated to the rollout side (the Phase-3
  sweep's s≥1 signature, reproduced here at 1:2). A second-tier elastic system could harvest
  *those* blocks; a static split cannot.
- **Freshness note:** unlike every 2-GPU async arm (mq=0 throughout), the 1:2 arm's queue
  actually buffers (1,532 collect events with nonzero mq depth) — samples are trained up to a
  few versions stale within the s=8 budget; dropped_stale = 0. The wall-clock win engages the
  staleness budget; the elastic-2GPU result never did.
- Cross-node mechanics were uneventful: 2-pod Ray cluster formed on first attempt, both engines
  joined the weight-sync NCCL group over the pod network (param_sync ~4-5 s/step at 1.5B),
  zero RayActorError.

## Cross-arm summary table (normalized units only)

| Arm | GPUs | s / 64-group step | Groups/GPU-hr | Traj/GPU-hr | Resp-tokens/GPU-hr | Freshness |
|---|---|---|---|---|---|---|
| M0 static 1:1 disagg | 2 | 619.0 | 186.1 | 1,489 | 17.0 M | s=8 (unused) |
| RUN A2 colocated sleep/wake | 2 | 522.7 | 220.4 | 1,763 | 20.3 M | strictly on-policy |
| **M2 elastic** | **2** | **502.7** | **229.2** | **1,833** | **20.9 M** | s=8 (unused) |
| RUN D static 1:2 disagg | 3 | 321.0 | 239.2 | 1,914 | 21.9 M | s=8 (budget engaged: queue buffers, 0 drops) |

Reward-vs-wallclock sanity: all 16K arms sit in the same per-step score band (means ≈0.19-0.20:
M0 0.196, A2 0.196, M2 ≈0.19, B pre-flip 0.20) — no arm buys throughput with degenerate rewards.

## C-claim verdicts after M3

| Claim | Verdict | Evidence |
|---|---|---|
| C3 it pays (vs best static AND colocated) | **Win vs 1:1 (+23% /GPU-hr); win-to-tie vs colocated (+4% wall-clock, +4% /GPU-hr); vs static 1:2 (3 GPUs): 95.8% of its per-GPU-hour efficiency with one-third fewer GPUs; static 1:2 wins wall-clock 1.57x** | Angle 1 + Run A2 + Run D |
| C5 it's dynamic (shift + no-harm) | **PASS** | Run B (adapts ≤2 steps, 29/29 clean cycles across the boundary, eta_below_threshold 0→310) + Run C (0 switches, +0.8% ≈ noise) |
| C2 switch is cheap | Re-confirmed | 28.0/37.0 s post-flip, unchanged across regimes |
| C4 it's correct | Re-confirmed | param_sync versions strictly sequential in every elastic run; mq/dropped_stale 0; rewards in band |

## Operational notes / artifacts

- Every kubectl operation used the explicit `--context=gke_aishuk-test_asia-southeast1-b_verl-research-cluster`;
  the snapshot-agent pod was recycled (pod delete, never helm) before BOTH elastic launches
  (B on trb7; C-armed on w0x7 — C-control's auto-park leaves sticky `elastic-r2=SAVED` state,
  recycle before the armed phase was mandatory and verified); pre-flight probes of the exact
  status-parse path passed before both elastic launches.
- Incremental collection streamed to `m3-results/<run>/incremental.log` every ~20 min for every
  arm; all epilogues carried gzip+base64 insurance dumps (BEGIN/END+md5); all collections were
  md5-verified where dumped.
- First RUN A launch lost 45 min to a dead-but-ESTABLISHED HF CDN connection (download stalled at
  256000000 bytes, no read-timeout fire); all subsequent manifests carry a bounded
  resume-retry download loop (`timeout 300` × 20 + `HF_HUB_DOWNLOAD_TIMEOUT=30`).
- Artifacts: `m3-results/colocated/` (batch-8 diagnostic + parity analysis),
  `m3-results/colocated-b64/` (RUN A2), `m3-results/regime-shift/` (RUN B: decisions.jsonl
  10,000+ records, regime_flip.{jsonl,log}, switch_timings.jsonl, policy_controller.log, GPU
  traces), `m3-results/no-harm/{control,armed}/` (RUN C), `m3-results/static12/` (RUN D).
- Nodes: trb7 (A, B), q36v (A2, D-head), w0x7 (C, D-worker); q36v/w0x7 released and trb7 set
  RESERVED in the node registry at completion; protected nodes/pods untouched.
