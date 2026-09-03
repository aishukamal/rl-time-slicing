# VD-STATE: verl-DAPO elastic showcase campaign journal
Campaign start: 2026-09-02 (successor to genbound; recipe per STALENESS-RECIPE-SCAN.md D1/E)

GOAL: elastic RL >15% step-time + samples/GPU-hr at 32-GPU scale on STOCK
verl dapo_30b_a3b_base_math_fsdp.sh @ 983cb0f2 (Qwen3-30B-A3B-Base, dapo-math-17k,
staleness 0.6, partial_rollout=True, 16 gen : 16 train on 4x8 H200).

Ladder: S0 staging -> S1 smoke (GATE gen:train >= 1.5:1) -> S2 shakedown (6 clean
elastic windows) -> S3 baseline scored -> S4 elastic scored -> S5 report.

k8s prefix: elastic-vd. Context: gke_aishuk-test_us-west1-c_verl-research-cluster-west
(always explicit). Pool: h200-8gpu-ubuntu, 4 nodes. driver-610 DS: DO NOT TOUCH.

## Registered deltas (both arms unless noted)
- VD-D1: --config-name fully_async_dapo_trainer.yaml does not exist at pin ->
  fully_async_ppo_trainer.yaml (shipped nit; all DAPO-ness is CLI overrides).
(more as they occur)

## Ledger
- 2026-09-02: campaign start. Read scan, GENBOUND-REPORT, m1/m2 assets, recipe shell.
  Nodes: 4 Ready on h200-8gpu-ubuntu; only elastic-smoke-driver-610 DS pods present.
  gsutil auth broken; gcloud storage (ADC) works — using gcloud storage everywhere.
- S0 staging: model mirror (Qwen3-30B-A3B-Base) SUCCESS (build 23a19f3d), dataset
  parquets SUCCESS (274de21e) -> gs://aishuk-test-elastic-stage/{models/Qwen3-30B-A3B-Base,data/dapo-math}.
  Image build attempt 1 FAILED (RUN heredoc unsupported by classic parser); fixed,
  resubmitted as fc7ceb11 (verl-vd:s0 = verl-cr-shim:m1 + verl baked at 983cb0f2).
- SEAM FOUND (yield): TrainingWorker.to(device, model, optimizer, grad) is a
  @register(ONE_TO_ALL) method at the pin (engine_workers.py:159) ->
  FSDPEngine.to() -> offload_fsdp_model_to_cpu/offload_fsdp_optimizer
  (transformer_impl.py:786-812). 16-rank yield = actor_wg.to("cpu"); onload =
  actor_wg.to("device"). No __ray_call__ lambdas needed -> genbound
  device-pinning lesson moot on this path (registered dispatch runs in worker
  device context). Trainer role for stock fully-async (no hybrid) = Role.Actor.
- Trainer gen-wait signals at pin identical to M2 parser expectations:
  'Requesting N samples from queue' / 'sample collected i/N. mq_len: m' /
  'timing_s/param_sync ... current_param_version' (fully_async_trainer.py:384,415,759).
  required_samples = ppo_mini_batch_size(32) x require_batches(1) = 32/micro-step,
  sync every trigger_parameter_sync_step=16 micro-steps.
- OPEN DESIGN FORK: spare weight path = (a) fleet-side wcache dump + cross-node
  prefetch (genbound-proven, more moving parts) vs (b) spares as extra members
  of the main nccl CheckpointEngineManager group, parked via vLLM sleep mode,
  wake_up(tags=weights-only) at each sync. Reading checkpoint_engine + 
  vllm_async_server at pin to decide.
- DESIGN FORK RESOLVED -> option (a) WCACHE. Evidence trail:
  * Pin's CheckpointEngineManager.update_weights REBUILDS the NCCL group every
    sync from self.replicas (base.py:498-538, temp RayWorkerGroup per round;
    add_replicas/remove_replicas exist "for elastic scale up/down"). So per-round
    membership tolerance DOES exist structurally (SWE2-G2 conflict is absent at
    this pin in that specific sense).
  * BUT option (b) still loses on evidence: (1) release_kv_cache/resume_kv_cache
    on vLLMHttpServer are EMPTY TODOs (vllm_async_server.py:835-845) -> a synced
    spare must hold weights+KV GPU-resident through sync (KV cannot be released),
    i.e. full VRAM co-residency with the trainer at sync time — the exact G2
    conflict, confirmed in code; (2) sleep/wake are no-ops in STANDALONE mode
    (:790-820) so parked-member tricks need a server subclass INSIDE the scored
    sync path; (3) out-of-tree CheckpointEngineWorker launch is unproven.
  * Option (a) seams all verified in-tree: vLLMHttpServer.collective_rpc accepts
    str|Callable (:228-240, forwards to AsyncLLM.collective_rpc) -> fleet-side
    per-TP-rank dump callable + spare-side load callable, zero fleet code changes;
    spares stay workers=[] STANDALONE (M1-proven launch), invisible to sync,
    CANNOT corrupt the scored sync path. Truthful stamping: set_global_steps(v of
    wcache dump); invariant jsonl per window (wcache v == fleet current_param_version
    == per-rank load receipts).
- Spare park mode: vLLM sleep L2 via server SUBCLASS for spares only
  (ElasticSpareServer adds spare_sleep/spare_wake/spare_load_wcache driving
  self.engine directly, since STANDALONE server methods no-op). Spare engine
  config = stock rollout config + registered spare-only overrides
  (+enable_sleep_mode=True, gpu_memory_utilization tuned ~0.70).
- Init ordering (chicken-egg: spares must init on TRAINER nodes before trainer
  memory exists, but trainer placement decides which nodes those are):
  keep UPSTREAM order (trainer first), then actor_wg.to("cpu") (early exercise of
  the yield lever), discover trainer nodes/GPUs, launch+warm spares, sleep L2,
  actor_wg.to("device"), then rollouter/MQ/sync/fit as stock.
- SAFETY GUARD (new, load-bearing): FSDP compute while offloaded = crash (not
  stall — offloaded FSDP params on CPU break the first dispatched kernel/allgather).
  ElasticFullyAsyncTrainer overrides _fit_generate to await an
  "elastic_resident" event after batch collection, before ANY compute dispatch.
  Restores the stall-not-crash property M1 had with cuda-checkpoint freeze.
- Wcache transport: fleet pod dumps per-TP-rank safetensors to /dev/shm/wcache
  (collective_rpc callable, off-window right after each param_sync), fleet pod
  serves it via python http.server; trainer pods prefetch with curl during
  trainer-busy. 61GB/sync-block/node at node-to-node ~1.7-3GB/s = 20-40s, off-clock.
- S0 COMPLETE (staging):
  * Model: gs://.../models/Qwen3-30B-A3B-Base mirrored + sha manifest (build
    23a19f3d). config.json max_position_embeddings is NATIVELY 32768 on the
    Base variant — the shell's "modify after download" note needs no action.
  * Data: dapo-math-17k.parquet + aime-2024.parquet + sha256 (build 274de21e).
  * Image: verl-vd:s0 (build fc7ceb11 SUCCESS) = verl-cr-shim:m1 base + verl
    baked at 983cb0f2 + datasets/TransferQueue/cupy.
- Code written (genbound2/code/): vd_elastic_trainer.py (offload lever +
  resident guard + self-heal), vd_spares.py (ElasticSpareServer sleep-L2,
  wcache dump/load callables with receipts, VDNodeAgent http/prefetch,
  launch_spare TP4 out-of-tree), vd_main_elastic.py (driver fork, spare
  bootstrap between trainer init and rollouter init), vd_controller.py
  (M2 policy port: ETA gates + wcache-fresh gate + invariant jsonl),
  run_vd_common.sh (stock shell override list VERBATIM + deltas),
  run_vd_{baseline,elastic}.sh. k8s/elastic-vd-ray.yaml (4 pods, hostNetwork,
  GCS staging initContainers, per-pod 1Hz GPU duty monitor).
- Registered deltas so far: VD-D1 config-name nit; VD-D2 logger console (no
  wandb creds); VD-D3 run length N_VERSIONS + test_freq protocol knobs;
  VD-D4 path mapping; VD-D5 resume_mode=disable (protocol hygiene; save_freq
  is stock -1 so auto-resume would find nothing anyway).
  Elastic-arm machinery (not config): driver fork + spares (+enable_sleep_mode
  and gpu_memory_utilization on SPARE engines only) + controller.
- WATCH ITEM (image): pip resolver flagged vllm 0.20.2 requires
  transformers!=5.3.* but 5.3.0 is installed (verl pin constraint pulled it).
  M1 ran the same combo (runtime install) on a dense 1.5B without issue;
  Qwen3MoE + vLLM 0.20.2 under transformers 5.3.0 gets its first real test at
  S1 engine init. Fallback if it breaks: rebuild image pinning transformers
  to a 4.56+/5.2.x version satisfying both.
- Cluster deployed: elastic-vd-{head,w1,w2,w3} on {hf1v,bc29,jqtb,79fs},
  hostNetwork, head=10.138.0.46:6379. Staging initContainers pulling model.
- S1 PROTOCOL (pre-registered): stock baseline arm (run_vd_baseline.sh),
  N_VERSIONS=2 (2 sync blocks = 32 micro-steps), val_before_train=True
  (stock). Measure from train.log console metrics per micro-step:
  gen wall = timing_s/gen (dominated by MQ wait when gen-bound),
  train wall = step - gen (update_actor + logprob etc.), plus
  fully_async/total_wait_time and param_sync walls. GATE: exposed gen:train
  >= 1.5:1 steady-state (exclude warmup micro-steps 1-2 and val).
  Also record: rollouter tok/s, response-length stats, trainer idle share,
  per-GPU duty from gpu_util_*.csv. STOP+reassess if gate fails.
- 03:40Z S1 LAUNCHED (baseline arm, N_VERSIONS=2 -> rollout.total_rollout_steps
  =1024, stock everything else; full arg list archived in /results/s1_launch.log
  on head). Trainer FSDP pool landed on jqtb(10.138.0.56) + bc29(10.138.0.7);
  rollout fleet therefore hf1v + 79fs. Model load ~1-2 min/rank in flight.
  Launch hygiene note: kubectl exec + nohup holds the stream; future launches
  add </dev/null and disown.
- 04:25Z S1 (2 blocks, pre-registered) RESULT: warmup regime is TRAIN-BOUND.
  Block1: len 837, gen 39.9s vs update 513s, trainer idle 7.2%, MFU 2.2%,
  psync 27-31s. Block2: len 906, gen 20s, update 548s, idle 3.4%, MFU 2.4%.
  Mechanics all healthy (MQ 819 cap = shipped staleness math; rollouter
  throttled at cap, 23% idle; partial_ratio 0; fleet ~10.8K tok/s decode).
  GATE VERDICT AT WARMUP: FAIL (ratio ~0.06) - but the S1 protocol's
  "steady state" for this recipe is the length-grown regime: DAPO on a BASE
  model starts at ~840 tok and grows CoT over versions (scan band 3-7K).
  Train wall is overhead-dominated (MFU 2.3% at 0.43M tok/update), so ratio
  improves superlinearly as length grows (gen scales ~linear, train sublinear).
- DECISION (pre-registered adaptation, not knob surgery): S1b = same stock
  baseline, N_VERSIONS=25, save_freq=10 (VD-D6: checkpoint cadence for A/B
  warm-start; applied identically to both scored arms via resume). S1b serves
  as (a) ratio-vs-version trajectory probe, (b) trainer of the warm-start
  checkpoint for S3/S4 (both arms resume from the SAME checkpoint at a
  version where the gate holds -> scored window = the recipe's gen-bound
  regime; standard A/B practice; dilution symmetric).
  FALLBACK if trajectory saturates below 1.5: recipe #4
  (dapo_7b_math_fsdp2_16_16, published 16:16 tables) per scan D2 runner-up
  logic within verl.
- 04:27Z S1b launched (baseline, N=25, save_freq=10, no val) as trajectory
  probe + warm-start producer. Block1: len 877, ratio 0.089, idle 8.2%,
  reward mean -0.887 (early Base, expected). Fleet decode ~12.8K tok/s.
- TRAJECTORY DECISION TREE (pre-registered now):
  * If gen:train trend clearly rises toward >=1.5 within reachable versions:
    S3/S4 = both arms resumed from the SAME S1b checkpoint at the crossing
    version; scored window = gen-bound regime.
  * If ratio saturates ~1.0-1.2: recipe is roughly balanced at 16:16 on
    H200 (meituan's 0.6-staleness gen-bound experience was H800-class);
    fallbacks in order: (b1) extend baseline training overnight (length
    growth accelerates post-warmup in DAPO; stays maximally stock);
    (b2) switch to scan runner-up recipe #4 (7B/28K published 16:16) or
    #6 (35B-A3B/64K, enables stock-elastic vs our-elastic comparison).
  * S2 shakedown (mechanics only, unscored): after S1b, elastic arm resumed
    from v10/v20 ckpt; if no natural gen-waits at that regime, manufacture
    real gen-bound conditions via our own rollouter concurrency lever
    (elastic_set_max_concurrent_samples low) - real stack, real windows,
    no stock-config surgery; scored arms untouched by this.
- Checkpoint portability note for S3 prep: FSDP per-rank shards land on
  node-local hostPath of whichever trainer node hosts the rank -> MIRROR
  ckpt dirs across both trainer nodes before any resumed arm (genbound
  defect #1 class).
- 05:10Z S1b TERMINATED at v3 by decision-tree branch (b): length FLAT
  (877/911/874), ratio 0.032-0.089, rollouter idle 55% (fleet has ~2x
  headroom). The 30B-A3B recipe's gen-bound regime is len>~5K, reachable only
  after ~100+ versions (~20-30h) - outside budget. 30B S1/S1b evidence
  archived (results/s1-30b/trajectory.tsv + s1b_train_30b.log on head node).
  HONEST FINDING for the report: at 16:16 on H200, dapo_30b_a3b warmup regime
  is train-bound at MFU ~2.3% (overhead-dominated small updates); meituan's
  gen-bound experience is presumably H800-class and/or later-regime.
- PIVOT (pre-registered fallback): recipe #4 dapo_7b_math_fsdp2_16_16.sh
  @ pin - Qwen2.5-Math-7B, 28K response, gen TP4 (spare design ports 1:1),
  fsdp2 (offload utils branch internally), staleness 0.1, sync block =
  4 micro-steps x 128 samples, published 16:16 convergence tables in the
  recipe family README. Model mirrored WITH the recipe's own config.json
  edit (max_position_embeddings 4096->32768, quoted instruction in shell)
  as Qwen2.5-Math-7B-mpe32k (build d390bf34).
  Controller adaptation: --hard-collect 112 (of 128).
- 05:35Z S1-7B (recipe #4, 3 blocks): len ~1180, MFU 33-35% (compute-real,
  unlike 30B's 2.3%), trainer idle 36%/27% (v1/v2), ratio 0.56/0.37,
  ro_idle ~21% with queue pinned at the 0.1-staleness cap (51 prompts).
  GATE at this regime: below 1.5, BUT (a) idle 27-36% is already a real
  harvest pool (2x capacity during waits converts up to ~half of idle),
  (b) len growth (published Math-7B DAPO curves: 3-10K over training) grows
  exposed gen superproportionally (long tails). Window convertibility needs
  per-micro-step waits > ~70s; at len ~1200 waits ~40-60s -> controller
  correctly would not fire (no-harm); at len >=2.5-3K windows open.
- 05:36Z S1b-7B launched: N=30, save_freq=10 (ckpts v10/v20/v30, keep 2),
  no val. Purpose: trajectory to the convertible regime + warm-start ckpt.
  Plan: S2 shakedown = elastic arm resumed from best ckpt (concurrency
  throttle as fallback window generator); S3/S4 = both arms resumed from the
  SAME ckpt, N pre-registered from trajectory; S5 report.
- 06:05Z S1b-7B v1-v9: len DECLINES 1179->927 (DAPO early dip) while reward
  climbs -0.99->-0.75; idle steady ~25%; waits = ONE ~30-42s pre-sync window
  per block (p50 micro-step wait 4s, p90 24s) - staleness-cap drain shape.
  Long-CoT regime unreachable in budget on Math-7B. KILLED at v9; evidence
  archived (results/s1-7b/).
- PIVOT P2 (registered delta VD7-D5): MODEL_PATH = DeepSeek-R1-Distill-Qwen-7B
  via the stock shell's OWN env knob (everything else stock shell #4).
  Rationale: long-CoT at v0 = the regime the stock recipe reaches late in
  training (S1b trajectories as evidence); delta class pre-authorized by the
  scan (code-RLVR dataset-swap precedent). Simplification: scored arms start
  fresh at v0 (no ckpt resume/mirroring needed). Mirror build 739467b2.
- 06:40Z S1-R1D block 1: len 6928, step 714s, exposed gen 302s, update 412s,
  trainer idle 42.3%, ROLLOUTER PEGGED (6.6% idle), ratio 0.73, MFU 37%,
  reward -0.128, waits per micro-step 31/50/148/47s, fleet ~85K tok/s at
  ~2048-way concurrency. GEN-BOUND CONFIRMED in the R1D arm.
- DESIGN AMENDMENT (registered NOW, before any scored run - no post-hoc
  gate shopping): at 7B the C/R-offload window economics are dominated by
  switch cost (~25-35s round trip vs 30-150s windows -> est gain only
  8-12%). H200 memory admits a strictly better mechanism with the SAME
  elastic control plane: spares stay MEMORY-RESIDENT (weights+KV, util 0.30
  ~= 42GB/GPU) and PAUSED (zero SM usage) during trainer compute; window
  open = wcache-load-if-stale + resume_generation + LB add + concurrency
  bump (~2-5s); window close = LB remove + abort (partial-rollout lossless
  drain) + pause (~2-5s). Trainer is never offloaded in the window path
  (resident guard stays as belt-and-braces; FSDP offload lever remains in
  the bootstrap and available as VD_SPARE_PARK_MODE=sleep for big-model
  topologies). Differentiation vs verl-native dynamic scheduling: mid-block
  window granularity, predictive ETA gate, out-of-tree spares (no stock
  config change), pause-resident (no sleep/wake wall), truthful wcache
  version stamping.
- AMENDED S1 GATE (registered): trainer idle share >= 30% AND median
  micro-step wait >= 30s (old ratio>=1.5 gate encoded 86s-round-trip
  economics that no longer apply). R1D arm: 42.3% idle, waits 31-148s ->
  PASS both.
- Sizing note for S3/S4: block wall ~714s at N_VERSIONS blocks; scored runs
  N=15 (~3h/arm) pre-registered, test_freq stock 20 (one val window falls
  inside the scored range at v20? with N=15 no val; set val at start+end via
  protocol instead - final decision at S2 exit, before S3 starts).
- 07:15Z S2 attempt 1: BOOTSTRAP CLEAN END-TO-END (16-rank offload 1.8s,
  4x TP4 spares launched on trainer GPUs, warm probes ok, paused-resident
  park, onload 0.4s, bootstrap 419s total) -> first fit_step OOMed in
  update_actor: trainer peak ~100GB/GPU (82 in use + 17.4GiB logits alloc =
  61440 tok x 152K vocab bf16) vs spare-resident 44GB at util 0.30.
  Defect VD-1: spare util 0.30 too high for the 61K-token packed update peak.
  Fix: VD_SPARE_GPU_UTIL=0.17 (~24GB: 3.8 weights + ~19 KV, ~390 concurrent
  seqs at len 7K across 4 spares). Also validates the ray-restart hygiene:
  stale detached handles actor must be killed between attempts (fresh ray
  restart done at 06:52).
- 07:30Z S2b (util 0.17): update_actor OOM again by 10MiB (!) - trainer peak
  ~116GB/GPU (logits alloc 17.3GiB on 61440-token packed batches dominates).
  Defect VD-2: paused-RESIDENT spares cannot coexist with stock update peaks
  at ANY useful KV size; torch caching allocator also keeps ~peak reserved
  between updates, so high-util spares can't wake against a merely-idle
  trainer either.
- DESIGN ITERATION 3 (sleepL1 mode, now default):
  * spares: enable_sleep_mode=True, sleep(level=1) between windows (weights
    3.8GB/GPU to CPU, KV freed; ~1-3s walls at 7B), gpu_memory_utilization
    0.60 for BIG window KV.
  * window open: trainer elastic_release_cache (torch.cuda.empty_cache on
    all 16 ranks - frees reserved-unused blocks, NO state movement, ~1s,
    only ever fired while blocked in gen-wait) -> spare wake_up (weights
    H2D + KV alloc) -> wcache-load-if-stale -> resume -> LB add.
  * window close: LB remove -> abort (partial rollout) -> drain -> sleep L1.
    Trainer proceeds; first post-window update re-allocs (seconds).
  * rt estimate ~8-14s round trip - still 5x better than the C/R-offload
    economics; FSDP offload lever remains for the bootstrap + big-model mode.
- 07:45Z S2c RUNNING (sleepL1 design): bootstrap clean (L1 parks 3.3s/spare,
  util 0.60), training started, controller up (auto-live, dry 4 blocks,
  max-cycles 6, hard-collect 112/128, dwell 20s, rt seeds 8/8). Fleet wcache
  source replica0 @ 10.138.15.209 + HTTP :18080.
- 07:55Z Defect VD-3: vLLM 0.20 msgpack encoder rejects callable
  collective_rpc without VLLM_ALLOW_INSECURE_SERIALIZATION=1 -> wcache dump
  callable failed on fleet server. Fix: pod-wide env (registered environment
  delta, BOTH arms identical; baseline arm never sends callables so it is
  env-neutral there). Pods recreated (S2c torn down; bootstrap+controller
  wiring already validated end-to-end in S2c: L1 park 3.3s, steady gate,
  wcache pipeline reached the dump call).
- 08:56-09:06Z S2d LIVE WINDOWS WORKING at scale:
  * switch-to-spares 2.3-3.9s (release_cache 0.7 + wake 0.75 + wcache load
    2.4 + LB/resume/concurrency <0.1); switch-to-trainer 1.3s. Round trip
    ~5s (vs 86s M1-era C/R economics: 17x better).
  * wcache pipeline: dump+2-node prefetch 18.4s off-window, 15.2GB/node at
    ~1.5GB/s; version gate enforced (wcache_not_fresh blocks windows until
    refresh lands).
  * cycles verified: 2 (old ctl) + 2+ (new ctl); windows 10-100s, mostly
    closed by hard_collect failsafe (fill-rate EMA underestimates the
    spare-boosted fill -> predictive gate conservative; acceptable).
  * Defect VD-4 (evidence-only): invariant fingerprint mismatch persists
    within THIS run because fleet/spare WorkerProcs cached vd_spares from
    first use (module-by-reference pickling); fix (sorted fp) is on disk and
    takes effect in any FRESH run. Plan: short fresh S2e run to show
    invariant_ok=True before scoring. Consider closure-pickling later.
- S3/S4 PRE-REGISTRATION (locked now):
  * Both arms FRESH from v0, identical stock shell #4 config + MODEL_PATH=
    R1-Distill-7B + VD_VAL_BEFORE_TRAIN=True + stock test_freq=20 +
    N_VERSIONS=12 (12 sync blocks = 48 micro-steps; est ~3.5h/arm at
    ~700-900s/block + val).
  * Headline metrics: wallclock to N=12 syncs (time-per-sync-block),
    samples/GPU-hour (48x2048 responses / 32 GPUs / wall), trainer idle
    ratio conversion, reward curve same-band overlay, duty from
    gpu_util csv. Baseline arm runs the STOCK module (no handles/spares).
  * Elastic arm: controller --auto-live --dry-run-blocks 1 --hard-collect
    112 --min-dwell 20 --wake-margin 10 (no max-cycles, no throttle).
- 09:52-10:05Z S2e (fresh run, fixed code): invariant_ok=TRUE windows 1-2
  (v1, v2), cycles 2/2, wcache refresh 18.6-23.1s. Defect VD-5 (stale
  node-pinned detached fleet agent after role flip) found+fixed (node-ip
  verification + idempotent http reuse). S2 GATE: PASS - 8 total verified
  window cycles, final 2 with full invariant chain.
- 10:07Z S3 BASELINE SCORED LAUNCHED: stock module, N_VERSIONS=12, val at
  start (stock val_before_train) - fresh v0, R1-Distill-7B via shell env.
  Cleanup verified: all detached actors killed, 32 GPUs 0 MiB.
- 14:00-15:00Z S4a scored (elastic, N=12) in flight. Per-block vs baseline:
  v1 -1.2% (dry), v2 -7.6%, v3 -10.3%, v4 -2.6%, v5 -15.1%, v6 -10.0%,
  v7 -10.5%, v8 -10.1%. 21+ verified cycles, 0 aborts, invariants clean.
  Cumulative v1-8: -8.4% step time.
- ANALYSIS (cap on conversion): windows fire at ~50/128 (fill-EMA warmup
  after regime resets delays the gate ~30-60s) and close at hard_collect
  112/128 -> coverage ~48% of the wait phase.
- Defect VD-6 (LATENT MEMORY RACE, sleepL1): the resident-guard event is
  never cleared in sleepL1 mode, so a batch completing before switch-back
  finishes lets the trainer enter update_actor while spare weights+KV
  (~85GB at util 0.60) are still wake'd -> OOM race, survived so far only
  because hard_collect 112 leaves 10-30s of headroom > drain walls.
  FIX (for S4b): trainer elastic_hold()/elastic_release() drive the SAME
  resident-guard event from the controller around every window (stall-not-
  crash restored); then hard_collect can extend to 124 and fill_warmup
  post-reset drops to 5 -> higher window coverage.
- PLAN: S4a completes and stands as scored evidence (report both);
  S4b = same arms/protocol, controller tuning iteration (registered):
  hold/release guard + --hard-collect 124 + fill warmup 5.
- 15:30Z S4a COMPLETE + SCORED (vs S3 baseline, N=12, identical protocol):
  step time -9.22% (v1-12), -10.06% (v2-11 window-active);
  samples/GPU-hr +10.16% / +11.19%; rewards same-band (-0.15->0.40 vs
  -0.13->0.40); 26 verified cycles, 0 aborts, invariants clean, 0 OOM.
  Op note: my pgrep pattern killed the S4a controller ~3 min early (block 12
  ran windowless; conservative bias only).
- 15:29Z S4b LAUNCHED (VD-6 hold/release + hard_collect 124 + post-reset
  fill warmup 5). 15:58Z v2: 749.2s vs baseline 856.0 = -12.5% (S4a v2 was
  -7.6%). VD-6 guard observed working live: batch-ready mid-window -> trainer
  stalled 7.1s until spares parked -> proceeded (stall-not-crash).
- 19:15Z S4b COMPLETE + SCORED. FINAL CAMPAIGN NUMBERS (see
  VERL-DAPO-REPORT.md): baseline 8420s vs S4b elastic 7590s over 12 sync
  blocks = -9.87% step time, +10.95% samples/GPU-hr (v2-11: -10.65%/+11.92%);
  26/26 window invariants clean, 0 dropped samples, rewards same-band,
  trainer-node duty 56% -> 72%. Switch walls: in p50 1.46s, back p50 2.98s.
  >15% VERDICT: NOT MET on full-run average (peaks -15.1/-18.2% per block);
  ~+11% is stable across two elastic arms. Gap analysis + close-path in
  report section 9.
- CAMPAIGN CLOSED. Cluster end state: all drivers/controllers dead, detached
  actors killed, 32 GPUs 0 MiB, wcache pruned, elastic-vd pods left Running
  idle, driver-610 DS untouched, pool at 4 nodes. ~530 GPU-hours consumed.
