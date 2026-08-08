# M2 RUN 2 — Elastic Policy Controller Autonomous Run (attempt 2, enum-parse bug fixed)

**Question (M2):** does the closed-loop policy controller (`m2/policy_controller.py`)
autonomously harvest the trainer's gen-wait idle blocks — dry-run alignment, then ≥10 live
steps of correct switch-in/switch-back decisions?

**Answer: YES — decisively.** After the attempt-1 fix (`state_name()` str fallback) plus the
post-mortem hardening, the controller ran the entire 4.5h window fully autonomously:
2 aligned dry-run steps → auto-live → **26 complete switch cycles over 24 fully-live steps,
zero aborts, zero failed operations, zero operator interventions after the one-time relabel.**
Fully-live mean step time **502.7s vs the 501.8s best-case reconstruction — regret
+0.9s/step (0.2%)**, i.e. the controller captured essentially all of the theoretically
harvestable idle time. −18.8% vs the M0 baseline (619.0s), beating M1's manual driving
(553.1s) by 9%.

## Run facts

- Job `elastic-m2`, pod `elastic-m2-v785d`, node `*-trb7`, image `verl-cr-shim:m1`,
  s=8, RUN_SECONDS=16200 (4.5h) + keep-alive
- Pod start 2026-08-06 23:03:34Z; training stopped by the planned 16200s timeout at
  03:34:40Z (rc=124, expected end state); controller exited cleanly on the stop flag at
  03:34:48Z; 0 restarts
- verl commit `983cb0f24443f87b3d161fad318445130a620b07` (same as M0/M1/attempt-1)
- 28 steps logged (2–29), 30 param_syncs (versions 0–29), steady window = steps 3–29
  (27 steps, same skip-first-logged-step convention)
- snapshot-agent `timeslice-snapshot-agent-npxgp` (image ts-demo7), restarted fresh at
  23:02:59Z before launch (clears sticky job state — same precondition as attempt 1)

## Hardening applied before launch (all attempt-1 post-mortem items)

1. **Pre-flight probe** (`m2/preflight-probe.yaml`): pod on trb7 imported the controller
   module from the deployed ConfigMap and exercised the EXACT status-parse path
   (`agent_job_states()` → `state_name()`) against the live agent, on both str and int
   inputs. PASSED before launch (twice: before and after the agent restart). This probe
   would have caught the attempt-1 bug in ~2 minutes instead of 4 hours.
2. **`--auto-live` with alignment check**: goes live only after ≥2 dry-run steps AND ≥1
   `would_switch_to_rollout` fired inside an observed gen-wait block (blocked=True is the
   gate's first conjunct, so any would-in proves alignment). Fired `via=auto` with 2/2
   aligned would-pairs. No operator exec needed mid-run.
3. **Controller stdout in the pod log**: launched by the job itself, piped through the run
   script's stdout chain (note: `/proc/1/fd/1` is the HOST init under `hostPID: true`, so
   fd inheritance is the correct mechanism). All `[policy]` lines survive in `full_pod.log`.
4. **Insurance dumps + progress heartbeat**: epilogue dumps CTRLLOG/DECISIONS/SWTIMINGS/
   GPUCSV to stdout with BEGIN/END + md5 (all present, all md5s match the collected files);
   controller emitted a one-line PROGRESS summary every 10 min to pod stdout.
5. **Startup-gate timeouts**: agent-RUNNING and handles-actor gates hard-cap at 20 min with
   loud exit(7); steady-state gate warns at 20 min and hard-caps at 45 min. Deviation from
   the blanket 20-min spec, and it was load-bearing: steady state legitimately took 27.3 min
   (3 param syncs from cold start — the 20-min WARNING fired at 2/3 syncs, exactly as
   designed; a 20-min hard cap would have falsely aborted this healthy run, as it would have
   attempt 1's, which took 25.3 min).
6. **Incremental collection**: `run2/incremental.log` snapshots (pod health, controller
   progress, latest metrics, switch counts) taken during the run from the operator side.

## Timeline

| Time (UTC) | Event |
|---|---|
| 23:02:59 | snapshot-agent restarted (fresh state); pre-flight probe PASS |
| 23:03:34 | Job/pod start; phases 1–4 fast (warm caches) |
| 23:06:41 | Controller launched by job (train start + 120s); `wait_steady_state` |
| 23:07:31–23:08:00 | Auto-park of R2: 29.08s, clean |
| ~23:08–23:15 | Operator relabel → `timeslice.io/job-id=elastic-trainer`; agent shows trainer RUNNING |
| 23:26:41 | Controller 20-min WARNING (2/3 param syncs) — heartbeat working as designed |
| 23:33:56 | Steady state (3rd param_sync); **agent-RUNNING gate cleared in 32ms** (attempt-1 failure point) |
| 23:34:35–23:49:41 | Dry-run: 2 would-pairs, both switch-ins inside gen-wait blocks 3 and 4 |
| 23:56:20 | **LIVE via auto** (2 dry steps, would_in=2, would_back=2, aligned=True) |
| 23:56:20–03:30:05 | 26 autonomous switch cycles, one per gen-wait block, blocks 5–30 |
| 03:34:40 | Planned RUN_SECONDS timeout → SIGINT; run-script epilogue + insurance dumps |
| 03:34:48 | Controller sees stop flag, exits rc=0 |

## (a) Dry-run phase: PASS

2 dry steps observed, 2 shadow would-pairs, both would-switch-ins fired while
`blocked=True` inside gen-wait blocks (ETA 151.9s / 424.0s ≫ 128.4s threshold), both
shadow switch-backs predictive at ETA ~50s. Decisions aligned with observed gen-wait
blocks → auto-live triggered on the first tick after the 2nd dry param_sync.

## (b) Live phase: 26/26 cycles clean, one pair per gen-wait block

Trigger accuracy: **28/28 switch-in fires (2 dry + 26 live) had blocked=True at fire**;
zero `skip_switch_in` (block never closed mid-decision); zero spurious switches (the
one-pair-per-block guard held: 197 live `already_switched_this_block` no-action ticks).
Live switch-in ETA at fire: mean 269s (range 148.5–675.5s), always above the c·roundtrip
threshold (~100–128s, tightening as the rt EMAs converged 37.8→28.3s / 47.8→37.4s).

Per-cycle C2 table (26 cycles):

| cyc | block | in @coll | in ETA(s) | in dur(s) | back reason | back @coll | back dur(s) | R2 window(s) | wake−ready(s) | psync ver |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 31/64 | 675.5 | 29.3 | predictive_eta | 56 | 37.1 | 122.3 | −2.0 | 5 ✓ |
| 2 | 6 | 39/64 | 213.2 | 27.3 | predictive_eta | 56 | 37.6 | 110.3 | −40.1 | 6 ✓ |
| 3 | 7 | 37/64 | 225.8 | 27.3 | predictive_eta | 57 | 38.1 | 132.3 | −18.0 | 7 ✓ |
| 4 | 8 | 41/64 | 228.4 | 27.1 | predictive_eta | 56 | 37.3 | 94.2 | −48.1 | 8 ✓ |
| 5 | 9 | 38/64 | 307.1 | 27.3 | predictive_eta | 58 | 37.6 | 136.3 | −2.0 | 9 ✓ |
| 6 | 10 | 41/64 | 244.9 | 27.3 | predictive_eta | 58 | 37.4 | 120.3 | −2.0 | 10 ✓ |
| 7 | 11 | 43/64 | 205.8 | 28.3 | predictive_eta | 58 | 37.7 | 112.3 | −40.1 | 11 ✓ |
| 8 | 12 | 23/64 | 324.1 | 28.3 | predictive_eta | 56 | 37.1 | 66.2 | −76.2 | 12 ✓ |
| 9 | 13 | 41/64 | 265.2 | 28.3 | predictive_eta | 57 | 37.5 | 92.2 | −72.2 | 13 ✓ |
| 10 | 14 | 40/64 | 192.1 | 27.3 | predictive_eta | 59 | 37.5 | 132.3 | −2.0 | 14 ✓ |
| 11 | 15 | 25/64 | 356.3 | 28.3 | predictive_eta | 57 | 37.5 | 130.3 | −16.0 | 15 ✓ |
| 12 | 16 | 38/64 | 296.0 | 27.3 | predictive_eta | 58 | 37.5 | 100.2 | −32.1 | 16 ✓ |
| 13 | 17 | 43/64 | 212.7 | 28.3 | predictive_eta | 57 | 37.2 | 60.1 | −80.2 | 17 ✓ |
| 14 | 18 | 40/64 | 208.7 | 27.3 | predictive_eta | 56 | 37.4 | 102.2 | −62.1 | 18 ✓ |
| 15 | 19 | 42/64 | 150.9 | 27.3 | predictive_eta | 58 | 37.5 | 114.3 | −32.1 | 19 ✓ |
| 16 | 20 | 45/64 | 199.1 | 27.1 | predictive_eta | 58 | 37.5 | 96.2 | −28.1 | 20 ✓ |
| 17 | 21 | 34/64 | 232.1 | 28.3 | predictive_eta | 59 | 37.5 | 134.3 | −22.1 | 21 ✓ |
| 18 | 22 | 40/64 | 267.9 | 27.3 | predictive_eta | 59 | 37.6 | 132.3 | −18.0 | 22 ✓ |
| 19 | 23 | 42/64 | 306.9 | 28.3 | predictive_eta | 57 | 37.2 | 62.2 | −82.2 | 23 ✓ |
| 20 | 24 | 9/64 | 231.7 | 30.3 | hard_collect_failsafe | 61 | 37.6 | 86.2 | −2.0 | 24 ✓ |
| 21 | 25 | 32/64 | 370.6 | 28.3 | hard_collect_failsafe | 60 | 37.2 | 162.4 | −2.0 | 25 ✓ |
| 22 | 26 | 47/64 | 150.3 | 28.3 | predictive_eta | 58 | 37.2 | 62.1 | −86.2 | 26 ✓ |
| 23 | 27 | 42/64 | 148.5 | 28.3 | predictive_eta | 57 | 37.3 | 98.2 | −2.0 | 27 ✓ |
| 24 | 28 | 37/64 | 303.3 | 28.3 | predictive_eta | 54 | 37.1 | 70.2 | −104.2 | 28 ✓ |
| 25 | 29 | 39/64 | 176.8 | 28.3 | predictive_eta | 59 | 37.8 | 150.4 | −2.0 | 29 ✓ |
| 26 | 30 | 8/64 | 503.8 | 28.3 | predictive_eta | 58 | 37.3 | 162.4 | −2.0 | n/a* |

\* cycle 26's post-resume param_sync (v30) was preempted by the planned RUN_SECONDS
timeout mid-step-30; both switch ops were clean and the trainer woke normally — not a
failure, run simply ended.

**Switch-back gap distribution** (wake − batch-ready, reconstructed from 2s-cadence
decision ticks; negative = trainer awake before the batch completed): n=26, mean −33.7s,
median −25.1s, range −104.2 … −2.0s. **Success criterion "wake ≤30s after batch-ready":
26/26 PASS** — the trainer was never late; the predictive trigger is in fact ~30s
conservative on average (safe-side bias; tunable via wake_margin/rt_out EMA if we want to
squeeze the last few seconds).

**Guards / no-action accounting** (6,301 no-action decisions, every one with a reason):
`not_blocked` 4,003, `min_dwell` 812, `eta_above_trigger` 830, `fill_ema_warmup` 379,
`already_switched_this_block` 275, `signal_error_mq` 2 (Ray actors dying at planned
shutdown, 03:34:44–46 — correctly resulted in no action). Staleness guard and
mq-dropped guard never activated (no stale drops, no queue drops all run).
Zero op hangs, zero switch failures, zero aborts, zero RayActorError.

C2 switch latencies (26 cycles, all sub-40s and metronome-stable):

| op | mean (s) | min–max | dominant phases (mean) |
|---|---|---|---|
| switch-to-rollout (rt_in) | **27.93** | 27.05–30.27 | cuda_snapshot_trainer 17.06, cuda_restore_r2 10.59 |
| switch-to-trainer (rt_out) | **37.43** | 37.07–38.06 | cuda_snapshot_r2 29.03, cuda_restore_trainer 6.01 |

Round trip 65.4s vs run4's 85.6s (−24%): NCCL suspend confirm ~0.19s both ways;
faster snapshots than run4 likely because the controller consistently freezes the trainer
deep in gen-wait (minimal live activations).

## (c) Per-cycle param_sync verdict: PASS

25/25 verifiable cycles: a param_sync completed after every trainer resume (versions 5–29,
strictly sequential; controller `cycle_verified` on each). 30 param_syncs total, mean
2.95s, max 3.75s — indistinguishable from baseline (attempt-1: 2.97s mean). No sync ever
started while frozen; no post-resume sync missed its 900s deadline.

## (d) Step-time distribution

| | M0 baseline | M2 attempt 1 | M1 run4 (manual) | **M2 run2 (autonomous)** | best-case reconstruction |
|---|---|---|---|---|---|
| Steady mean step (steps 3–29) | 619.0s | 615.6s | 553.1s | **520.0s** (incl. 3 pre-live steps) | — |
| **Fully-live steps 6–29 (n=24)** | — | — | — | **502.7s** (stdev 24.7, 431.6–535.4) | **501.8s** |
| gen-wait / step | 316.0s | 313.2s | — | **220.4s** | — |
| Trainer idle ratio (verl) | 0.511 | 0.508 | — | **0.420** | — |
| Trainer GPU (GPU 1) mean util / idle≤15% | ~46% / 53.7% | ~43% / 57.0% | — | **67.1% / 28.3%** | — |
| Rollout GPU (GPU 0) mean util | 99.6% | 98.3% | — | **98.2%** | — |

- **Regret vs best case: +0.9s/step (0.2%)** — the attempt-1 regret of 113.8s/step (18.5%)
  is fully recovered. The best-case reconstruction used run4's slower switch latencies;
  with this run's faster switches the controller is effectively at the model's optimum.
- vs M0: **−116.3s/step (−18.8%)**; vs M1 manual: −50.4s/step (−9.1%) with 26 cycles vs
  run4's 4.
- R2 harvest: 26 windows, mean 109.3s, total 2,843s of extra rollout service on the
  trainer GPU (17.5% of the run span).

## (e) Rewards / drops: healthy

Steady score mean 0.191 (band 0.131–0.261) — inside the ~0.2 band, same spread as M0
(0.196, 0.150–0.238) and attempt 1 (0.192). response_length mean 11,396 (≈ M0's 11,439).
`dropped_stale_samples` = 0 and `mq_queue_size` = 0 at every step; mq `dropped_samples` = 0
throughout; reward curve continuous across all 26 cycles (R2 serves version-0 weights with
current-version tagging — the documented M1/M2 shortcut; no observable quality drag at this
run length).

## M2 verdict: PASS

- ≥10 steady live steps under autonomous control: **24 fully-live steps, 26 cycles** ✓
- Switch-in inside a gen-wait block: **26/26** ✓
- Trainer wake ≤30s after batch-ready: **26/26** (never late; mean 34s early) ✓
- param_sync after every trainer resume: **25/25 verifiable** ✓ (26th preempted by planned
  run end)
- No-harm: every no-action tick logged with gate values + first failing reason; guards
  (min-dwell, one-pair-per-block, staleness, warmup) all exercised or correctly quiescent ✓

The M2 milestone is complete: the ETA-gated, predictively-switched-back policy controller
autonomously converts 18.8% of step time into R2 rollout service with zero correctness
cost, matching the theoretical harvest bound. Ready for M3 (regime-shift matrix:
natural drift + forced 8K↔16K flip + train-heavy no-harm control).

## Artifacts (this directory)

- `full_pod.log` — complete pod stdout (controller `[policy]` lines included — hardening #3;
  all 4 insurance dumps present, md5s match)
- `train.log`, `experiment.log` — training stdout (steps 2–29)
- `decisions.jsonl` — 6,438 records, md5 `a2ac02caa573cd8d559f463f5fee4ad4`
- `switch_timings.jsonl` — 53 ops (1 auto-park + 26+26), md5 `63cf08f2ded74c2c1d87a54d4637b23e`
- `policy_controller.log` — controller log (in-pod copy), md5 `c2e930a46196f436de8a06c226b91c9d`
- `gpu_util.csv` — 100ms 2-GPU trace (116,637 samples/GPU), md5 `2cfed52b981839145b7c08955827178d`
- `snapshot_agent.log` — fresh-agent log covering the whole run
- `steps_metrics.json`, `summary.json` — per-step metrics + summary
- `incremental.log` — mid-run operator snapshots (hardening #6)
- `start_ts`, `end_ts`, `verl_commit`, `train_rc`

Run provenance note: the job was deployed at 23:03:34Z from ConfigMaps byte-identical to
the hardened working-tree sources (verified by md5 against `m2/policy_controller.py` and
by diff of the live `run_elastic_m2.sh` against `m2/k8s-job-m2.yaml`) while this session
was completing pre-flight; the running job was verified identical to the intended spec and
adopted rather than relaunched.

Cluster cleanup: job `elastic-m2` + ConfigMaps `code-rlvr-m2`/`elastic-m2-code` deleted
after md5-verified collection; probe pod removed; trb7 left with no default-ns pods;
registry updated (trb7 → RESERVED, M2 attempt 2 SUCCESS).
