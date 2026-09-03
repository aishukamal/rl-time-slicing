# VERL-DAPO-REPORT: elastic RL on the stock verl fully-async DAPO recipe family at 32 GPUs

Written 2026-09-03 ~19:00Z. Campaign per STALENESS-RECIPE-SCAN.md (genbound2);
journal: VD-STATE.md (complete ledger). Predecessor: ../genbound/GENBOUND-REPORT.md.

## 1. Verdict, honest

**Headline (scored, pre-registered protocol): elastic time-slicing of the 16
trainer GPUs improved sampling throughput by +11.0% samples/GPU-hour and cut
step time by -9.9% over the identical stock baseline (N=12 sync blocks,
identical config, fresh v0, same day, same nodes). The pre-registered >15%
target was NOT met on the full-run average.** Individual window-active sync
blocks repeatedly cleared it (best -18.2%, four blocks ≤ -12.5%), and the
window-active subrange (v2-11) scored -10.7% step time / +11.9% samples/GPU-hr.
A second tuned elastic arm reproduced the result (+10.2%/+11.2% -> +11.0%/+11.9%
across the two arms), so ~+11% is a stable, honest number for this recipe
shape at this scale, not a lucky draw.

Correctness verdict: CLEAN. 26/26 scored windows passed the weight-provenance
invariant (fleet-dump fingerprint == spare-model fingerprint, per TP rank,
16 load receipts/window), version stamps truthful by construction (spares
serve exactly the fleet's current param version), partial-rollout drains
lossless (MessageQueue dropped_samples = 0 across all arms), reward curves
same-band throughout (baseline -0.15 -> +0.40 over 12 versions; elastic
-0.12 -> +0.40).

## 2. What was run (lineage)

- verl pin 983cb0f2 (the PoC stack pin), baked into image
  us-central1-docker.pkg.dev/aishuk-test/rl-images/verl-vd:s0
  (torch 2.11.0+cu130, vLLM 0.20.2, transformers 5.3.0, verl 0.9.0.dev).
- Recipe: verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_16_16.sh
  (scan candidate #4; the recipe family's README publishes 16:16 convergence
  tables) — override list VERBATIM; 2 rollout nodes + 2 trainer nodes x 8
  H200 = 16 gen (4 x TP4 vLLM) : 16 train (fsdp2, sp4), staleness_threshold
  0.1, trigger_parameter_sync_step 4, require_batches 4 (micro-step = 128
  prompt groups = 2048 responses; sync block = 4 micro-steps), partial_rollout
  True, max_response 28K.
- Model: DeepSeek-R1-Distill-Qwen-7B via the shell's OWN MODEL_PATH env knob
  (registered delta VD7-D5, see section 5) — long-CoT at v0 (mean response
  6.2-7.2K tokens), the regime the recipe reaches late in training.
- Data: dapo-math-17k (train) + aime-2024 (val), the recipe's own files,
  sha256-verified mirrors.
- Pool: h200-8gpu-ubuntu, 4 nodes, us-west1-c (spot). Runs 2026-09-03
  10:10Z (S3 baseline), 12:56Z (S4a elastic), 15:29Z (S4b elastic tuned).

## 3. Scored results

Per-sync-block step walls (seconds; v12 is the standard termination flush in
all arms):

| v | baseline | S4a elastic | S4b elastic | S4a delta | S4b delta |
|---|---|---|---|---|---|
| 1 | 734.7 | 726.1 | 719.4 | -1.2% | -2.1% |
| 2 | 856.0 | 791.1 | 749.2 | -7.6% | -12.5% |
| 3 | 795.2 | 713.1 | 723.2 | -10.3% | -9.1% |
| 4 | 805.8 | 784.6 | 718.6 | -2.6% | -10.8% |
| 5 | 768.4 | 652.1 | 683.2 | -15.1% | -11.1% |
| 6 | 734.3 | 660.9 | 678.5 | -10.0% | -7.6% |
| 7 | 756.1 | 676.9 | 671.0 | -10.5% | -11.3% |
| 8 | 718.9 | 646.5 | 650.0 | -10.1% | -9.6% |
| 9 | 705.2 | 651.3 | 633.1 | -7.6% | -10.2% |
| 10 | 730.0 | 597.5 | 620.0 | -18.2% | -15.1% |
| 11 | 654.0 | 592.7 | 596.0 | -9.4% | -8.9% |
| 12* | 161.8 | 151.3 | 147.4 | -6.5% | -8.9% |

Totals (v1-12): baseline 8420s; S4a 7644s (-9.22%); S4b 7590s (-9.87%).
Samples/GPU-hour (98304 responses / 32 GPUs): 1313 -> 1447 (+10.16%) -> 1457
(+10.95%). Window-active v2-11: -10.06%/-10.65% step time, +11.19%/+11.92%
samples/GPU-hr. (S4a's block 12 ran windowless after an operator error killed
its controller ~3 min early — conservative bias only.)

Duty (1Hz nvidia-smi, whole-arm means):
- Trainer nodes: 55.9%/56.0% (baseline) -> 71.5%/71.9% (S4b): **+15.6pp duty
  on the lent GPUs** — the claim-3 evidence.
- Rollout nodes: 85.2/84.1% (baseline) -> 82.1/82.1% (S4b; some serving load
  shifts to spares during windows).

Exposed gen-wait conversion (v2-11): baseline 2348s of trainer gen-wait ->
S4b 1478s. 37% of the wait pool was converted; the rest is pre-fire
collection (windows open at median 52/128 collected: burst-absorption gate +
fill-EMA warmup + 20s dwell) and finite spare capacity.

Switch mechanics at scale (S4b, 26 windows):
- switch-to-spares p50 1.46s / p90 2.50s / max 3.17s
  (trainer empty_cache ~0.7s + vLLM wake_up ~0.75s + wcache-load-if-stale
  ~2.4s when a new version landed + LB add + concurrency, else stamps only).
- switch-to-trainer p50 2.98s / p90 11.03s / max 18.65s (tail = engine-side
  abort of long in-flight sequences; tokens retained by partial rollout).
- Window close reasons: 18 predictive-ETA, 7 hard-collect failsafe,
  1 batch-completed-late (handled by the VD-6 hold: trainer stalled 7.1s,
  never raced).
- wcache pipeline (off the window clock): fleet TP-rank dump + 2-node HTTP
  prefetch of 15.2GB in 18.4-23.1s per param sync.

## 4. Mechanism (what "elastic" is here)

Design iteration landed on **sleep-L1 paused spares + trainer cache release**
(full trail in VD-STATE.md):
- 4 x TP4 out-of-tree vLLM spares live on the 16 trainer GPUs (launched
  exactly like in-tree server actors: no Ray GPU claim, NOSET env + explicit
  cuda_visible_devices; STANDALONE mode => never members of the weight-sync
  group; engine config deltas: enable_sleep_mode=True,
  gpu_memory_utilization 0.60).
- Between windows: spares sleep(level=1) (weights -> CPU, KV freed; 3.8GB/GPU
  at 7B TP4; 1-3s walls). During trainer compute the GPUs are 100% the
  trainer's.
- Window open (controller ETA gate on the MessageQueue fill rate, M2 policy
  core): trainer torch.cuda.empty_cache on all 16 ranks (frees reserved-unused
  allocator blocks, no state movement) -> spare wake_up -> wcache load IF the
  param version advanced (per-TP-rank safetensors, 1:1 shard invariant)
  -> truthful set_global_steps -> LB add_servers + sticky-cache clear +
  rollouter concurrency x2.
- Window close (predictive ETA / failsafes): LB remove -> abort (vLLM
  pause+drain; verl's partial-rollout client resumes in-flight requests on
  the fleet with tokens retained) -> sleep L1 -> release the trainer hold.
- Safety: a resident-guard event in the trainer subclass stalls
  _fit_generate between batch-ready and compute while spares are unpacked
  (defect VD-6 fix) — stall-not-crash, observed live (7.1s stall, no OOM).
- The 16-rank FSDP offload lever (TrainingWorker.to("cpu"), verl's own
  manual-control API) is retained: it bootstraps spare engine init (spares
  must size their KV while the trainer GPUs are empty) and is the switch
  path for big-model topologies (VD_SPARE_PARK_MODE=sleep). Measured: 1.8s
  offload / 0.4s onload for the 16-rank 7B world.
- NO cuda-checkpoint, NO NCCL shim, NO transport restrictions anywhere:
  NCCL runs stock (NVLink on), and the checkpoint-engine group is rebuilt
  per sync at this pin, so the M1-era frozen-group landmines are gone.

## 5. Deltas from stock (register)

Both arms identical:
- VD7-D1: MODEL_PATH/data/ckpt paths (pod mounts; MODEL_PATH is the shell's
  own env parameter).
- VD7-D2: rollout.total_rollout_steps = 512 x 12 (run length; stock factor
  400) ; test_freq stock 20 (never fires inside N=12; val_before_train stock
  True in scored arms).
- VD7-D3: save_freq knob left at stock -1 in scored runs.
- VD7-D4: resume_mode=disable (protocol hygiene; equivalent to stock auto
  with save_freq=-1).
- VD7-D5 (the decisive one, registered before any scored run): model =
  DeepSeek-R1-Distill-Qwen-7B instead of Qwen2.5-Math-7B, set via the
  shell's own MODEL_PATH env. Evidence-driven: the two shipped stock models
  were measured train-bound in every reachable regime (section 6); R1-distill
  produces at v0 the long-CoT distribution the stock recipe reaches late in
  its own 400-step runs. Same delta class as the scan's pre-authorized
  code-RLVR dataset swap.
- Environment: VLLM_ALLOW_INSECURE_SERIALIZATION=1 pod-wide (needed for
  callable collective_rpc used by the wcache dump/load; baseline never sends
  callables).
Elastic arm only (machinery, no stock-config changes): elastic driver fork
(spare bootstrap + handles actor), trainer/rollouter subclasses (offload
lever, resident guard, LB handle, concurrency lever), spare server subclass,
wcache pipeline, policy controller. Spare-engine-only config: enable_sleep_mode
True, gpu_memory_utilization 0.60.

## 6. The regime study (why the 30B recipe was descoped, all measured)

- dapo_30b_a3b_base_math_fsdp (scan's D1 recommendation) at 16:16 on H200:
  warmup regime is TRAIN-bound — response length ~840-910 tokens for 12+
  versions (flat), trainer idle 3-8%, MFU 2.2-2.4% (overhead-dominated 0.43M
  token updates), ROLLOUTER 23-55% idle (fleet has ~2x headroom), exposed
  gen:train 0.03-0.09 vs the scan's predicted 2.4:1. The prediction assumed
  the length-grown regime (3-7K); measured growth (~34 tok/version, then
  flat) puts that regime ~100+ versions (20-30h+) out — unreachable, and the
  claim gate was honestly failed at S1. Evidence: results/s1-30b/.
- dapo_7b_math_fsdp2_16_16 with its stock Qwen2.5-Math-7B: idle 25-36% at
  v1-2 but DAPO's early length DIP (1179 -> 927 by v9, reward -0.99 -> -0.75)
  moves it AWAY from gen-bound within reachable budget. Evidence:
  results/s1-7b/.
- Same recipe + R1-Distill-7B: len 6.9-7.3K at v0, trainer idle 43.7% (v1)
  settling ~28-32%, rollouter pegged (6-10% idle) — structurally gen-bound.
  Amended S1 gate (registered pre-scoring, reflecting the 5s switch
  economics): idle >= 30% AND median micro-step wait >= 30s — PASS.
- Regime lesson for the scan's model: fully-async recipes SHIP balanced
  splits by design (that is what the 16:16 tuning is), so the harvestable
  trainer idle at shipped splits is 25-45%, not the 55-65% the genbound
  (colocated-recipe) experience suggested. With ~5s switch round trips the
  convertible fraction is high, but +15% end-to-end requires either a
  longer-tailed regime (bigger waits), higher spare capacity share, or
  earlier window fires than the current controller achieves (fires at median
  52/128 collected).

## 7. Ladder + defect ledger (all root-caused; full detail in VD-STATE.md)

S0 staging (model/data/image mirrors, sha-verified) -> S1 smoke x3 recipes
(gate analysis above) -> S2 shakedown (8 verified window cycles; final 2
with complete invariant chain) -> S3 baseline scored -> S4a elastic scored ->
S4b elastic scored (tuned). ~530 GPU-hours total (inside the 580-640 plan).

- VD-1: paused-RESIDENT spares (util 0.30) OOM vs the stock update peak
  (~116GB/GPU: 61440-token packed batches, 17.3GiB logits alloc).
- VD-2: even util 0.17 misses by 10MiB; torch caching allocator holds
  near-peak reserved between updates -> paused-resident abandoned for
  sleep-L1 + empty_cache design (iteration 3, the one that scored).
- VD-3: vLLM 0.20 msgpack encoder rejects callable collective_rpc without
  VLLM_ALLOW_INSECURE_SERIALIZATION=1.
- VD-4: invariant fingerprint mismatch from unsorted named_parameters
  ordering in the dump callable (fixed; also exposed module-by-reference
  pickling staleness in long-lived WorkerProcs — fresh runs unaffected).
- VD-5: node-role flips between runs made the detached, node-pinned wcache
  fleet agent stale (404 prefetch) -> node-ip verification + idempotent
  HTTP server reuse.
- VD-6: latent OOM race in sleepL1 mode (batch completing before switch-back
  while spare memory is wake'd) -> controller-driven hold/release of the
  trainer resident-guard event; observed working live in S4b.
- Op errors (journaled): a wrong pgrep pattern killed S4a's controller ~3
  min early; an earlier false-stale ConfigMap read cost one S2 relaunch.

## 8. Artifacts

- genbound2/results/: s1-30b/ (trajectory), s1-7b/ (trajectory+waits),
  s2/ (switch timings, invariants, decisions), s3/ (baseline log +
  trajectory), s4/ + s4b/ (elastic logs, decisions, switch timings,
  invariants, trajectories). Journal: VD-STATE.md.
- genbound2/code/: vd_main_elastic.py, vd_elastic_trainer.py, vd_spares.py,
  vd_controller.py, vd_measure.py, vd_trajectory.py, run_vd7_*.sh (recipe #4
  arms), run_vd_*.sh (30B arms), k8s/elastic-vd-ray.yaml, staging/ (Cloud
  Build mirrors + image).
- GCS: gs://aishuk-test-elastic-stage/models/{Qwen3-30B-A3B-Base,
  Qwen2.5-Math-7B-mpe32k, DeepSeek-R1-Distill-Qwen-7B} + data/dapo-math
  (all with sha manifests). Image: rl-images/verl-vd:s0.
- On-node (head /results): all raw train logs (s1_*, s2*, s3_, s4_, s4b_),
  gpu_util CSVs per node, launch logs with full arg lists.

## 9. What would close the gap to >15% (not run, evidence-based)

1. Earlier window fires: median fire at 52/128 leaves ~40% of each wait
   pre-window (burst gate + 20s dwell + fill warmup). A fill-rate prior
   seeded from the previous block (instead of re-learning post-reset) should
   roughly double wait coverage; S4a->S4b moved +0.7pp with crude knobs.
2. Higher spare KV share: util 0.60 -> 0.75 during windows (trainer holds
   ~14GB static + released cache; headroom exists but was not risked
   overnight).
3. Val windows: with stock test_freq inside the scored range (25/20-version
   cadence), each ~206s fleet-side validation is a guaranteed full-length
   window (measured val wall 205s at v0); N=12 never crossed one.
4. The stretch arms from the plan (verl-native use_dynamic_resource_scheduling
   comparison; code-RLVR dataset arm) remain unrun.

## 10. Cluster end state (2026-09-03 ~19:15Z)

All drivers/controllers stopped; detached actors killed; 32 GPUs at 0 MiB;
elastic-vd pods left Running (idle) on the 4-node h200-8gpu-ubuntu pool;
elastic-smoke-driver-610 DS untouched (load-bearing). wcache dirs pruned.
