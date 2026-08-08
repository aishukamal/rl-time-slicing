# M2 attempt 1 — post-mortem (NO RESULT; superseded by ../M2-REPORT.md)

**Question (M2):** does the closed-loop policy controller (`m2/policy_controller.py`)
autonomously harvest the trainer's gen-wait idle blocks — dry-run alignment, then ≥10 live
steps of correct switch-in/switch-back decisions?

**Answer: NO RESULT — the controller never made a single decision.** It started, reached
steady state on schedule, then spent its entire 4-hour live window stuck in the
`wait_agent_trainer_running` startup gate because of a one-line porting bug (proto-enum
parse without the string fallback `cycles.sh` had). Zero dry-run decisions, zero live
switches, `decisions.jsonl` never got past `controller_start`. The training run itself was
healthy end-to-end and lands exactly on the M0 no-elastic baseline (615.6s vs 619.0s mean
step), which cleanly bounds the forgone benefit: **~114s/step (18.5%) of step time left on
the table** against a best-case reconstruction with the measured switch latencies.

## Run facts

- Job `elastic-m2`, pod `elastic-m2-b4d2r`, node `*-trb7`, image `verl-cr-shim:m1`,
  s=8, RUN_SECONDS=16200 (4.5h) + 4h keep-alive sleep
- Pod ran 2026-08-06 06:41:35Z → 15:12:55Z, Completed rc=0; training stopped by the planned
  16200s timeout (expected end state); 0 restarts
- verl commit `983cb0f24443f87b3d161fad318445130a620b07` (same as M0/M1)
- 24 steps completed (2–25), 26 param_syncs (versions 0–25), steady window = steps 3–25
  (23 steps, same skip-first-logged-step convention as M0)
- snapshot-agent (same image as M1 run4; pod restarted 06:21Z, fresh state)

## Timeline (reconstructed from pod stdout, snapshot-agent log, GKE audit log)

| Time (UTC) | Event |
|---|---|
| 06:41:35 | Pod start; phases 1–4 fast (warm caches) |
| 06:44:12 | Agent: `elastic-r2` RUNNING (R1 engine holds GPU) |
| 06:45:40–06:46:08 | **Auto-park of R2**: snapshot 28.08s (the only switch op of the entire run) |
| 06:46:44 | Operator relabel → `timeslice.io/job-id=elastic-trainer` (audit: pods.patch) |
| 06:46:45 | Agent: `elastic-trainer` RUNNING (pids 939949, 941596, 948716) |
| 06:46:46 | Exec #1 (audit): controller nohup launch |
| 06:48:58 | Exec #2 (audit): launch verification — **last exec ever**; prior agent stalled here |
| 07:12:01 | Controller passes `wait_steady_state` (3rd param_sync) and starts polling agent status every 15s |
| 07:12:01–11:13:49 | **968 status polls, one every 15.0s, none ever confirms RUNNING** (see root cause) |
| 11:13:49–50 | RUN_SECONDS timeout → SIGINT to driver → Ray GCS dies → controller process dies (polls stop at the same second) |
| 11:13:50+ | Run script epilogue: per-GPU summary, switch_timings cat, GPUCSV dump |
| 15:12:55 | 4h sleep ends, pod Completed; kubelet destroys the results emptyDir |

## Root cause: controller stuck in its startup gate for the whole run

`policy_controller.py` `wait_agent_trainer_running()` parsed agent job states as:

```python
st = {j.job_id: pb.JobState.Name(j.state) for j in c.status().job_statuses}
```

The timeslice python client returns `JobStatus.state` as the enum **name string**
(`'JOB_STATE_RUNNING'`), not the proto int. `pb.JobState.Name(str)` raises
`TypeError: Enum value for JobState must be an int, but got <class 'str'>` — verified live
against the same agent from a probe pod on trb7:

```
StatusResponse(job_statuses=[JobStatus(job_id='elastic-r2', state='JOB_STATE_SAVED'),
                             JobStatus(job_id='elastic-trainer', state='JOB_STATE_RUNNING')], ...)
TypeError: Enum value for JobState must be an int, but got <class 'str'> 'JOB_STATE_SAVED'.
```

M1's `cycles.sh` (run4, 4/4 cycles PASS, same day, same agent image) used the **same
expression wrapped in a per-job try/except with `str(j.state)` fallback** — the fallback was
silently load-bearing, and it was dropped in the port to `policy_controller.py`. The
controller's outer `except` turned the TypeError into a `"agent status error: ..."` retry
every 15s (logged only to the lost in-pod controller log), forever. The agent-side state was
correct (RUNNING from 06:46:45); the gate could never observe it.

Evidence chain:
- Agent log: exactly 968 `Status called` entries at 15.0s intervals, first 07:12:01
  (matches 3rd param_sync ≈ steady-state exit), last 11:13:49 (matches driver SIGINT —
  the controller was a Ray driver; GCS death killed it mid-loop)
- Agent log: **one** Snapshot (the 06:45:40 auto-park), zero restores, zero further ops
- `switch_timings.jsonl`: single entry, `initial-park(auto)`, 28.08s
- Audit log: no exec after 06:48:58 → live flag was never touched either (moot — the
  controller never reached the dry-run loop where `maybe_go_live` runs, and it defaulted to
  flag-gated, not `--auto-live`)

**Fix applied** to `m2/policy_controller.py` (state_name() helper with the str fallback,
matching cycles.sh). One line of behavior; ready for an M2 re-run.

## Recovery completeness

The pod completed its 4h keep-alive and the kubelet destroyed the results emptyDir at
15:12:55 before any collection happened. Recovered:

| Artifact | Status |
|---|---|
| `full_pod.log` | Complete (single log file on node, no rotation — verified against `/var/log/pods`) |
| `train.log` | Complete, extracted from tee'd stdout (24 steps, all metric lines) |
| `gpu_util.csv` | Complete via GPUCSV stdout dump; **md5 `032c65d114dc909a5600a0a61592d3b4` verified** against the GPUCSV-BEGIN header; 236,488 samples (118,244/GPU @100ms, 16,206s span) |
| `switch_timings.jsonl` | Complete via stdout cat (1 entry: auto-park) |
| `snapshot_agent.log` | Complete (agent pod up since 06:21Z, covers whole run) |
| `decisions.jsonl` | **Lost with the emptyDir — but provably near-empty**: the controller never exited its startup gate, so it contained only the `controller_start` record (written before `wait_steady_state`). No decision data ever existed. |
| controller stdout log | Lost with the emptyDir; content reconstructible: "waiting for steady state" → "steady state: 3 param syncs" → "waiting for agent job elastic-trainer RUNNING" → 968× "agent status error: Enum value for JobState must be an int…" |
| `start_ts` | Recovered from first GPU-trace sample (1785998569313) |

Nothing of analytic value is missing.

## M2 report spec — item by item

**(a) Dry-run phase verdict:** NOT REACHED. The controller never entered the dry-run loop,
so no `would_*` decisions exist to compare against actual gen-wait blocks. Feasibility
reconstruction from the training data: all 23 steady gen-wait phases (mean 313.2s, min
252.1s) far exceed the switch-in gate threshold (c·(rt_in+rt_out) = 1.5 × 85.6 = 128.4s), so
a functioning controller would have had a clearable gate in every single steady step.

**(b) Live phase:** NOT REACHED. Switches per step: 0. No switch-in triggers, no
switch-backs, no spurious/missed-switch log records (none were ever written), no
min-dwell/no-harm activations. The one recorded op is the pre-controller auto-park
(28.08s, clean).

**(c) Per-cycle param_sync verdict:** no live cycles to verify. The trainer itself was never
frozen, and all 26 param_syncs succeeded (mean 2.97s, max 3.75s, versions 0→25 strictly
sequential) — the baseline sync path is healthy.

**(d) Step-time distribution vs M0/M1:**

| | M0 (baseline) | **M2 (this run)** | M1 run4 (manual cycles) |
|---|---|---|---|
| Steady mean step | 619.0s | **615.6s** | 553.1s |
| Range | 571–675s | 558–692s (stdev 37.4) | — |
| gen-wait / step | 316.0s | **313.2s** | — |
| update_actor / step | 280.4s | **282.1s** | — |
| Trainer idle ratio (verl) | 0.511 | **0.508** | — |
| Trainer GPU idle (<10%, 100ms trace) | 53.7% | **57.0%** | — |
| Rollout GPU util | 99.6% | **98.3%** | — |

M2 ≡ M0 to −0.6%: with zero switches the run degenerates to the no-elastic baseline,
confirming (i) the elastic scaffolding itself (auto-park, parked R2, C/R shim env) costs
nothing measurable, and (ii) all of M1 run4's 66s/step improvement came from the switches.

**(e) Rewards / drops:** healthy. Steady score mean 0.192 (band 0.110–0.232) — inside the
~0.2 band, same spread as M0 (0.196, 0.150–0.238). response_length mean 11,456 (≈ M0's
11,439). `dropped_stale_samples` = 0 and `mq_queue_size` = 0 at every step; no mq drops.

**(f) Regret estimate:** trainer GPU (GPU 1) had 27 idle blocks ≥60s totaling 9,036s of the
16,206s window (55.8%), mean 334.6s, max 611.6s — all fully wasted. Best-case
reconstruction: harvest every steady gen-wait block W with the M1-run4-measured latencies
(rt_in 37.8s + rt_out 47.8s), R1+R2 both filling during the boosted window
(W′ = (W + 85.6)/2 per block):

- Best-case mean step: **501.8s**
- Achieved: 615.6s → **regret ≈ 113.8s/step (18.5%)**, ≈ 2,617s (43.6 min) over the 23
  steady steps
- Cross-check: M1 run4's manual driving (4 cycles over ~10 steps) achieved 553.1s, between
  the two, as expected for partial-coverage switching

## M2 attempt-1 verdict

**Controller decision quality: unmeasurable (0 decisions).** Correctness of everything the
controller touched before stalling was fine (clean auto-park, correct steady-state
detection at the 3rd param_sync, correct 15s poll cadence, and it never took an unsafe
action — the no-harm property trivially held). Training correctness: PASS (M0-equivalent).
Step time: no improvement (615.6s ≈ M0), regret 18.5% vs best case. Re-run executed with
the fixed controller — see `../M2-REPORT.md` (PASS: 26 autonomous cycles, 502.7s mean).

## Artifacts (this directory)

- `full_pod.log` — complete pod stdout (all phases + GPUCSV dump)
- `train.log` — training section of stdout (steps 2–25)
- `gpu_util.csv` — decoded 100ms 2-GPU trace, md5-verified
- `switch_timings.jsonl` — 1 entry (auto-park, 28.08s)
- `snapshot_agent.log` — agent log covering the full run (968 status polls, 1 snapshot)
- `steps_metrics.json`, `summary.json` — per-step metrics + steady-state/regret summary
- `start_ts`, `verl_commit`
- `early-partial-copies/` — first-pass log fetches made before full recovery
