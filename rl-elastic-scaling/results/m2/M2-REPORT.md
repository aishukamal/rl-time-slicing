# M2 — Elastic Policy Controller Autonomous Run (elastic-rl-poc)

**Question (M2):** does the closed-loop policy controller (`m2/policy_controller.py`)
autonomously harvest the trainer's gen-wait idle blocks — dry-run alignment first, then
≥10 live steps of correct switch-in/switch-back decisions, no operator switch commands?

**Answer: YES — PASS on every bar.** Over a 4.5h run the controller made **26 fully
autonomous switch pairs in 26 consecutive steps** (one pair per step, zero operator
commands after the one-time pod relabel), 25 param_sync-verified (the 26th cut mid-step by
the planned timeout, both switches clean). Live steady step time **502.7s = −18.8% vs the
M0 619.0s baseline** and −9.1% vs M1 run4's manually-driven 553.1s. Every switch-in landed
inside a gen-wait block; every switch-back was early-or-on-time (trainer never woke after
batch-ready). Regret vs a perfect-harvest reconstruction: **5.8%**.

## Result run facts (attempt 3, the successful run)

- Job `elastic-m2`, pod `elastic-m2-v785d`, node `*-trb7`, image `verl-cr-shim:m1`, s=8,
  RUN_SECONDS=16200; verl `983cb0f2` (same as M0/M1); snapshot-agent recycled pre-launch
  (fresh state machine — see attempt 2)
- Phase 5 start 2026-08-06 23:04Z → SIGINT 03:34:45Z (planned, rc=124); 28 steps logged
  (2–29), 30 param_syncs (versions 0–29), zero RayActorError, zero mq drops, GPUs left clean
- Controller: launched by the job itself at +120s, `--dry-run-steps 2 --auto-live`;
  6,438 decision records in `decisions.jsonl`
- Timeline: steady-state gate cleared 23:33:56 (3rd param_sync) → agent gate cleared in
  **0.03s** (attempt-1 fix verified) → dry-run steps 4–5 → **LIVE 23:56:20** (`via=auto`,
  alignment 2/2) → 26 cycles → planned timeout

## Controller decision quality

**Dry-run (no-harm validation, 2 steps):** exactly one would-switch pair per step, both
would-ins inside open gen-wait blocks (collected 33 and 31, ETAs 152s/424s > 128s gate),
both would-backs predictive at collected 58. Steps 3–5 (pre-live) ran at baseline pace
(629–685s) — actions-disabled mode provably does nothing. Auto-live engaged only after the
alignment check (≥1 would-in) passed.

**Live switch-ins: 26/26 correct.** All fired with `blocked=True` (in-block by
construction — the gate's first conjunct — and re-checked from a fresh log parse in the
firing tick). Fired at collected 8–47/64, gate ETAs 148–676s vs threshold
`1.5×(rt_in+rt_out)` ≈ 128s seeded → ~98s once online estimates converged. Zero spurious
switches: 4,003 `not_blocked` no-action ticks (update_actor phases), 830
`eta_above_trigger`, 812 `min_dwell`, 275 `already_switched_this_block`, 379
`fill_ema_warmup`, 2 transient `signal_error_mq` — every no-action logged with its reason.

**Live switch-backs: 24/26 predictive** (`ETA ≤ rt_out + 15s margin`, typically fired at
collected 56–58), 2 hard-collect failsafes (collected 61/60 — ETA estimate briefly lagged a
fast tail). **Trainer wake vs batch-ready: never late.** Wake−ready gap mean −33.7s
(median −25.1s, range −104.2 to −2.0s): the trainer always woke *before* the batch
completed, satisfying the ≤30s-after-ready target on all 26 cycles with zero late wakes.
The −104s tail is the cost of the failsafe/margin conservatism — bounded and visible in
`decisions.jsonl` for M3 tuning.

**Adaptivity:** round-trip estimates updated online from this run's own ops:
rt_in 37.8s (seed) → 28.3s, rt_out 47.8s → 37.4s; the switch-in threshold self-tightened
128.4s → ~98s. R2-active windows averaged 109s (60–162s).

**param_sync verdict: 25/25 verified cycles** (each within its step's normal cadence; the
26th pair completed cleanly at 03:30:05 but its step's param_sync fell after the planned
SIGINT — not a failure). Versions 0→29 strictly sequential.

## Performance

| | M0 baseline | attempt 1 (0 switches) | M1 run4 (manual) | **M2 live (this run)** |
|---|---|---|---|---|
| Steady mean step | 619.0s | 615.6s | 553.1s | **502.7s (−18.8% vs M0)** |
| Range (stdev) | 571–675 | 558–692 (37.4) | — | **431.6–535.4 (25.2)** |
| gen-wait / step | 316.0s | 313.2s | — | **205.0s (−35%)** |
| update_actor / step | 280.4s | 282.1s | — | 281.5s (unchanged, as designed) |
| Trainer-GPU idle (<10%, 100ms trace) | 53.7% | 57.0% | — | **19.6%** |
| R1 GPU util | 99.6% | 98.3% | — | 99.7% |
| Switch latencies (mean) | — | — | in 37.8 / out 47.8 | **in 27.9 / out 37.4** |

- Live window = steps 6–29 (n=24 ≥ 10 required). Time-to-24-steps: 201 min vs 248 min at
  M0 pace — 47 min saved.
- Rewards healthy: live score mean ≈0.19 (band 0.131–0.261), same as M0 (0.196,
  0.150–0.238); response_length ≈11.4K; `dropped_stale=0`, `mq_queue_size=0`,
  `dropped_samples=0` at every step → staleness budget never touched, KV/weight hygiene
  held (no reward collapse after any of the 26 R2 windows).
- Switch latencies beat run4 by ~10s/side (warm PID caches: `get_ce_worker_pids` amortized,
  `set_global_steps` no longer on a cold path).

**Regret estimate:** best-case reconstruction (every gen-wait block harvested with this
run's measured latencies, R1+R2 filling the boosted window: best step ≈ update 281.5s +
sync 2.9s + (316 + 65.3)/2 ≈ 475.0s) → achieved 502.7s = **+27.7s/step regret (5.8%)**,
down from 18.5% forgone in the no-controller run. Residual regret decomposes into
early-wake margin (~15–34s mean) + first-block-of-step gate latency; both are logged
per-tick in `decisions.jsonl` (the M3 input).

**Fill-rate caveat for M3:** the controller's fill-EMA (reset at each regime change,
tau=60s vs 109s mean R2 windows) is a poor estimator of R2's marginal fill — it read
*lower* during R2-active than baseline (0.107 vs 0.127/s) even though gen-wait fell 35%.
ETA-driven switch-back still worked (24/26 predictive, none late) because relative ETA
matters more than absolute fill; M3 should use a produced-count regression per window
instead.

## Attempt history (this milestone took 3 runs)

1. **Attempt 1 (pod `elastic-m2-b4d2r`, NO RESULT):** controller stalled its entire 4h
   window in `wait_agent_trainer_running` — `pb.JobState.Name(str)` TypeError; the
   timeslice client returns enum *names* and the `str` fallback cycles.sh had was dropped
   in the port. 968 status polls, 0 decisions. Training itself was healthy (615.6s ≡ M0),
   bounding the forgone benefit at ~114s/step. Full post-mortem preserved at
   `attempt1/M2-REPORT-attempt1.md`; artifacts in `attempt1/`. Fixes: `state_name()`
   fallback (+ pre-flight probe of the exact parse path), startup-gate hard timeouts,
   controller launched by the job with `--auto-live` + alignment check, stdout heartbeats,
   gzip+base64 insurance dumps in the epilogue.
2. **Attempt 2 (pod `elastic-m2-crlkn` relaunch, failed in init):** auto-park rejected —
   `cannot snapshot job elastic-r2 in state JOB_STATE_SAVED`: the snapshot-agent kept
   attempt-1's in-memory `elastic-r2=SAVED` across job deletion (the known sticky-state
   landmine). Fix: **recycle the agent pod before every launch** (fresh state machine);
   now a standing launch step.
3. **Attempt 3 (pod `elastic-m2-v785d`): the result above.**

## Cluster state left

- Job `elastic-m2` + ConfigMaps `code-rlvr-m2`/`elastic-m2-code` deleted after collection;
  pre-flight probe pod removed; `nvidia-smi` clean (no processes) before teardown
- trb7: only DaemonSet pods remain (snapshot-agent `timeslice-snapshot-agent-npxgp`
  healthy, DRA plugin); no default-namespace pods; node released in the registry
- Protected nodes/pods untouched; kubectl context never switched (per-command `--context`)

## Artifacts (`run2/`)

`decisions.jsonl` (6,438 records — per-tick signals/gates/actions, the M3 regret input),
`policy_controller.log`, `switch_timings.jsonl` (53 ops: 1 park + 26+26 switches),
`train.log` (steps 2–29), `gpu_util.csv` (100ms, both GPUs, md5-consistent with in-pod
copy), `full_pod.log`, `snapshot_agent.log`, `nvidia_smi_final.txt`, `train_rc` (124),
`start_ts`/`end_ts`, `verl_commit`.
