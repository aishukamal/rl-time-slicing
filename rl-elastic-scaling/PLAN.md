# PoC Plan: Elastic Trainer↔Rollout GPU Reassignment via Transparent C/R

*Status: ACTIVE (go-ahead 2026-08-04). Target: the "dynamic trainer↔rollout GPU switch" open at verl 26Q3 [#6985](https://github.com/verl-project/verl/issues/6985), Miles ("elastic rollout-vs-training scheduling"), NeMo-RL ("auto-scaling", planned).*
*Session rule: no GitHub artifact modifications (PRs/issues/comments) without explicit user go-ahead.*

## 1. Thesis

In disaggregated async RL the GPU split between rollout and training is frozen at launch, but the workload's balance drifts (code-rlvr response lengths grew 10.5K→12.3K tokens in 10 steps). Measured on our own Phase 5 run: trainer GPU idle **51.3%** in 11 contiguous blocks averaging **4.1 min** (max 6.7) while the rollout GPU is pinned at 99.6%.

**Claim:** you don't need elastic world resize. Transparent GPU C/R (cuda-checkpoint + ncclCommSuspend) time-multiplexes shared GPUs under *unchanged* NCCL worlds: a warm rollout replica (R2) lives checkpointed in host RAM on the trainer's GPUs; during each trainer gen-wait block the scheduler suspends the trainer, activates R2 (generation runs on all GPUs), and reverses before the next step. Zero framework code changes; the one visible change — R2 appearing/disappearing — happens at HTTP router level, not in any collective.

**The asymmetry, stated honestly:** readers scale horizontally, the writer doesn't. Rollout replicas are stateless consumers behind a router; the trainer is a fixed-membership gradient all-reduce over sharded optimizer state, and C/R preserves membership — it cannot mint ranks. So reassignment is rollout-favoring by construction. Consequences we embrace: (a) provision the trainer for its worst case and harvest the slack; (b) offload the trainer's separable read-only work (old-logprob recompute, ref logprobs, eval) to time-sliced replicas; (c) train-bound regimes → controller does nothing (no-harm), idle rollout GPUs need a co-tenant (cross-job tier, out of PoC scope). No framework has shipped elastic trainer resize either — this scope is defensible, and the measured waste is on the solvable side.

## 2. Static options, dynamic decisions (the policy)

What's static is only the **option**: R2 pre-created, warmed, suspended (costs host RAM only). Exercising it is a per-block decision from live signals — verl emits the regime decomposition every step (`gen_wait` vs `update_actor`, mq depth, `dropped_stale`).

- **Switch in** when trainer blocks AND `ETA = (samples_needed − mq_depth)/fill_rate > c × switch_round_trip` (fill rate as EMA).
- **Switch back predictively**: begin R2 drain + trainer restore when ETA ≈ restore latency, so the trainer wakes as the batch completes.
- **Guards**: min-dwell (anti-thrash), staleness headroom check, R2 drain cost accounting.
- **No-harm property**: if the run is train-bound the gate never clears and the system degrades to exactly the static baseline. If gen time shrinks mid-run, blocks shrink and switching naturally stops — no discrete "regime change" detection needed.
- **Headline metric alongside speedup: regret vs hindsight-oracle schedule**, per phase of the run.

## 3. R2 weight path

R2's checkpoint image weights are garbage by design; only process structure is preserved. R2 is **never** in the trainer→R1 NCCL `param_sync` group (a suspended rank would stall the collective). Instead: trainer stages weights at each sync point (host pinned buffer/shared storage; ~3.5GB for the 1.5B model); on each activation R2 pulls once via the rollout server's update-weights path before router registration. **Invariant:** trainer is suspended while R2 is active → no new version can appear during R2's window → R2 loads exactly version k (same as R1); R2 is never staler than R1. v2: P2P pull from R1's GPU copy (checkpoint-engine pattern). KV hygiene: drain in-flight before suspend; flush prefix cache on post-restore weight load (ServiceNow correctness trap).

## 4. Workload & platform

- **Workload:** bring-up on math@8K (223s steps, fast iteration); **headline on code-rlvr @16K** — strongest measured case, reproducible (verl `983cb0f2`, zero crashes), reward exec is rollouter-CPU-side (~20µs trainer path) so sandbox latency doesn't confound. Harness: `aishukamal/rl-time-slicing/async-rl-benchmarks` (clone: `~/workspaces/rl-time-slicing-elastic-poc`), job spec `GPU-CR/code-rlvr/k8s-job-code-rlvr.yaml`.
- **Cluster:** west (`gke_aishuk-test_us-west1-c_verl-research-cluster-west`), node pool **`h100-2gpu-spot`** (currently 1 node up = 2×H100; snapshot agent + DRA driver already deployed in `timeslice-system`). Spot preemption risk doubles as the durability stretch demo. M3 multi-split sweeps need the pool's second node.
- **Mechanism components:** `llm-d-incubation/llm-d-rl-time-slicing` mainline (clone: `~/workspaces/llm-d-rl-time-slicing-elastic-poc`): snapshot-agent (cuda-checkpoint + app_channel backends), TimeSlice Orchestrator (M2 controller home), Python client `timeslice.snapshot_agent` with `VLLMAdapter` (rollout drain/flush hooks) and `CallbackAdapter` (trainer side). Builds via GCP Cloud Build (aishuk-test) only.
- **Layout:** trainer GPU 0, R1 GPU 1, R2 time-sliced on GPU 0. Model 1.5B TP=1 → small C/R images, fast weight pulls.

## 5. verl integration unknowns (investigation running)

1. **Router membership** at `983cb0f2` fully-async: engine set fixed at init vs dynamic registration. Preferred workaround if fixed: register R2 at init, health-check as drained while suspended (GLM-5 heartbeat pattern). Keep shims in our plugin, out-of-tree.
2. **Update-weights path** the R2 pull can use (disk/HTTP), and staging format.
3. **In-flight drain/abort API** for switch-back; behavior of mq samples if an engine dies mid-generation.
4. Metrics access at ~1s granularity for the controller; staleness accounting when an engine skips sync rounds.

## 6. What the case must prove

| # | Claim | Evidence | Bar |
|---|---|---|---|
| C1 | Waste is real | Phase 5 data (done) + colocated sleep/wake arm + multi-split sweep | 51% idle documented; "no static split dominates" figure |
| C2 | Switch is cheap | suspend/restore/weight-pull/drain breakdown | round-trip ≪ 245s mean blocks (target <30s) |
| C3 | It pays | time-to-N-steps, tokens/GPU-hour vs best-static AND colocated | ≥1.2×; win/tie vs colocated with differentiators intact |
| C4 | It's correct | reward parity, staleness accounting, KV hygiene checks | curves within seed noise; mq/dropped_stale clean |
| C5 | It's *dynamic* | regime-shift matrix (below) + regret vs oracle | adapts within ~2 steps; no-harm in train-bound control |

**Key scientific risk (measure first in M1):** the win assumes rollout is throughput-bound (99.6% util suggests yes) so R2 ≈ doubles fill rate; long-tail sequential decode is a floor extra engines can't remove. Measure R2's marginal fill rate immediately. Fallback narrative if it disappoints: paired-trainer time-slicing (math+code, Phase 5 verdict #3), which doesn't depend on this assumption.

## 7. Milestones

- **M0 (shrunk — Phase 5 is the baseline):** re-run code-rlvr baseline for stability; add colocated sleep/wake arm; optional multi-split sweep.
- **M1 — manual switch:** scripted trainer↔R2 swap both directions, N cycles, no divergence. Produces C2 breakdown + R2 marginal fill rate (go/no-go on the R2 story).
- **M2 — controller:** policy loop in TimeSlice Orchestrator consuming verl metrics via workload channel; ETA gate, predictive switch-back, min-dwell.
- **M3 — benchmark + report:** static splits vs elastic × cadence; **regime-shift matrix:** (a) natural drift (long code-rlvr run, dwell tracking CoT growth), (b) forced shift (8K↔16K cap flip mid-run, adapt ≤2 steps), (c) no-harm control (train-heavy multiturn workload, parity with static). Report with C1–C5 verdicts.
- **Stretch:** spot-preemption durability demo; separable-work offload (old-logprob replica); 7B/TP=2; slime port for generality.

## 8. Risks

Short phases → break-even gate + measured phase distribution (4.1 min mean says fine). Stale R2 weights → §3 invariant. Host RAM → 1.5B images are small; sequential switching. Restore fragility → validated patch set, pinned image. verl fully-async experimental → pinned `983cb0f2`, known-good. Colocated "good enough" → honest C3 arm + capability differentiators (trainer-side suspend, cross-node, preemption durability). Throughput-bound assumption → §6 go/no-go + paired-trainer fallback.

## 9. External case (HOLD until explicit go-ahead)

M3 report + trace visuals + design note mapped to verl #6985 / #3624. No PRs, issues, comments, or any GitHub artifact modification this session without explicit instruction.
