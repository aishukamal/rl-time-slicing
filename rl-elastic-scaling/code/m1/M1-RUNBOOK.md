# M1 Runbook — manual trainer↔R2 switch demo

Operator procedure for the M1 milestone (PLAN.md §7): N scripted switch cycles
in both directions on the code-RLVR fully-async run, no divergence, producing
the **C2 switch-latency breakdown** and the **R2 marginal fill rate**
(go/no-go on the R2 story, PLAN.md §6).

Everything here is driven from a laptop shell with `kubectl exec` into the job
pod; the pod is the Ray cluster. Never change the active kubectl context — use
`--context` per command.

## 0. Topology recap (what you should see)

| Component | Where | Process(es) |
|---|---|---|
| Trainer (FSDP, 1.5B) | physical GPU **T** | `DetachActorWorker` Ray actor process |
| R1 (vLLM standalone, in-tree) | physical GPU **R** | vLLM mp-executor subprocs under `vllm_server_0_0` |
| R2 (vLLM standalone, out-of-tree) | GPU **T** (time-sliced) | vLLM mp-executor subprocs under `vllm_server_1_0` |

GPU T/R assignment is decided at runtime: the driver reads R1's
`CUDA_VISIBLE_DEVICES` after rollouter init and puts R2 (and, by Ray placement
arithmetic, the trainer) on the other GPU. The driver log prints
`R1 is on physical GPU x; R2 (and later the trainer) -> GPU y`.

R2's weights: **version-0 disk weights for the entire M1 run** (R2 is outside
the frozen NCCL sync group and `save_freq=-1` means `--reload-weights` has
nothing newer to load). R2 samples are *tagged* with the current param version
(`set_global_steps(k)` on resume) so staleness accounting doesn't drop them —
this is the documented M1 shortcut; training correctness rests on
`use_rollout_log_probs=True` + `rollout_correction.bypass_mode=True` (config
defaults at pin 983cb0f2). Watch the reward curve accordingly (§5).

## 1. Prerequisites (verify before launch)

1. snapshot-agent serving on node `...-trb7` at `<nodeIP>:9001`
   (DaemonSet in `timeslice-system` or standalone binary — separate deploy
   task). Health probe runs in job Phase 1c and prints `agent health: SERVING`.
2. ConfigMaps applied:
   ```
   cd ~/workspaces/GPU-CR/elastic-rl-poc/m1
   kubectl --context <ctx> create configmap elastic-m1-code \
     --from-file=fully_async_main_elastic.py --from-file=elastic_trainer.py \
     --from-file=r2_lifecycle.py -n default --dry-run=client -o yaml \
     | kubectl --context <ctx> apply -f -
   kubectl --context <ctx> apply -f k8s-job-m1.yaml
   ```
3. Node is otherwise idle (this job assumes exclusive use of both H100s).

## 2. Launch and reach steady state

```
POD=$(kubectl --context <ctx> get pod -l app=elastic-m1 -o name | head -1)
kubectl --context <ctx> logs -f $POD
```

Expected init sequence in the log (deviations = abort):

1. `Rollouter created and initialized successfully` — R1 engine up.
2. `R1 is on physical GPU x; R2 ... -> GPU y`.
3. `Launching R2 vLLM server actor 'vllm_server_1_0' on GPU y` then
   `R2 warmup ok`.
4. `Auto-parking R2 ...` → `initial-park(auto) timing breakdown` →
   `R2 parked (auto). GPU freed for trainer init.`
   Verify: `kubectl exec $POD -- nvidia-smi` shows GPU y near-empty before
   trainer init lines appear. (If `ELASTIC_AUTO_PARK=0` or the agent probe
   failed, run `python3 /workspace/m1/r2_lifecycle.py park-r2` in the pod now.)
5. `Checkpoint manager initialized ... excluded replica_ranks=[1]` (from
   ElasticFullyAsyncTrainer; with out-of-tree R2 the exclusion list is a
   no-op — the important part is R2's address is absent).
6. `Param sync before fit..` completes (first and freezing build of the
   trainer+R1 NCCL group).
7. `elastic_controller_handles ready ... trainer worker pids=[...]`.
8. Training steps flow: `[FullyAsyncTrainer] Requesting 64 samples from queue`,
   `sample collected i/64`, `update_actor`, `timing_s/param_sync`.

**Steady state** = at least 3 completed param versions and a stable per-step
rhythm. Record baseline fill rate now:

```
kubectl --context <ctx> exec -it $POD -- python3 /workspace/m1/r2_lifecycle.py watch-mq --interval 2 --count 60 \
  | tee /tmp/mq_baseline.log   # runs in the pod; ~2 min of samples
```

## 3. Switch cycles

All commands run **inside the pod**:
`kubectl --context <ctx> exec -it $POD -- bash`, then
`cd /workspace/m1`.

### When to trigger (M1 manual gate)

`switch-to-rollout` only **during a trainer gen-wait block**: after the trainer
log prints `Requesting 64 samples from queue` and while `sample collected i/64`
is trickling (mq_len low, collection slow). On code-RLVR these blocks average
~4 min, so there is ample margin.

Safety property if mistimed: the trainer step stalls inside the frozen
worker's first CUDA call and simply completes after resume — `param_sync`
cannot start while frozen (fit_step ordering, fully_async_trainer.py:584-594).
Cost of mistiming is wasted wall-clock, not corruption.

### One full cycle

```
# A. trainer blocked in gen-wait  ->  give its GPU to R2
python3 r2_lifecycle.py switch-to-rollout
# phases: get_trainer_pids, get_ce_worker_pids, nccl_suspend_signal,
#         nccl_suspend_confirm, cuda_snapshot_trainer, cuda_restore_r2,
#         set_global_steps, clear_kv_cache, resume_generation,
#         lb_add_servers, raise_concurrency
# NCCL shim signals are AUTOMATIC and TWO-SIDED (run4 fix): suspend-trainer
# sends SIGRTMIN+1 (ncclCommSuspend) to all trainer worker PIDs AND R1's
# CheckpointEngineWorker PIDs — the checkpoint-engine comm is a 2-rank
# group (trainer rank 0 <-> CE worker rank 1) and suspend is collective;
# one-sided signaling blocks forever (run3 abort). The old fixed 2 s settle
# is replaced by CONFIRMATION: poll each signaled process's stderr
# (/proc/<pid>/fd/2) for the shim's 'suspend done' marker + rc=0 per-comm
# lines (timeout ELASTIC_NCCL_CONFIRM_TIMEOUT, default 30 s) before the
# cuda-checkpoint freeze. CE workers are signal-only — NEVER checkpointed;
# R1 keeps serving throughout. resume-trainer mirrors this: SIGRTMIN+2 to
# both sides after restore + 'resume done' confirmation.

# B. R2 active for the FULL gen-wait block (cycles.sh holds until collected
#    nears batch-ready; fill-rate capture is diagnostics only — the payoff
#    metric is per-step wall time vs the M0 baseline, 619.0s mean steps 3-11)
python3 r2_lifecycle.py watch-mq --interval 2 --count 30

# C. when the mq nears batch-ready, reverse
python3 r2_lifecycle.py switch-to-trainer
# phases: lb_remove_servers, restore_base_concurrency, abort_all_requests,
#         wait_for_requests_to_drain, lb_inflight_settle, discover_gpu_pids,
#         cuda_snapshot_r2, cuda_restore_trainer, nccl_resume_signal,
#         nccl_resume_confirm

# D. confirm the trainer step completes: log shows collection finishing,
#    then update_actor + param_sync for that version.
```

Run **N ≥ 4 cycles** (minimum for the C2 table), spread across different param
versions. Between cycles run `python3 r2_lifecycle.py status` and archive its
output.

Timings accumulate in `/workspace/results/switch_timings.jsonl` (one JSON
record per operation with per-phase seconds).

## 4. What to measure

1. **C2 per-phase switch latencies** — `switch_timings.jsonl`. Headline:
   `switch-to-rollout` total and `switch-to-trainer` total; target round-trip
   ≪ 245 s mean gen-wait blocks (goal < 30 s).
2. **R2 marginal fill rate** — fill_rate from `watch-mq` with R2 active (step B)
   vs baseline (§2). This is the §6 go/no-go: if rollout is throughput-bound,
   fill rate should approach 2×; if it's long-tail decode-bound, it won't.
   Also check the `[rollouter] active_tasks` line rises toward the doubled
   `max_concurrent` after `raise_concurrency`.
3. **Reward curve continuity + hygiene** — from `train.log`: per-version reward
   metrics must stay within seed noise across switch cycles;
   `dropped_samples` in mq stats stays 0; `dropped_stale_samples` doesn't jump;
   no `RayActorError` anywhere.
4. **GPU trace** — `gpu_util.csv` (100 ms cadence) shows GPU T alternating
   trainer-busy / R2-busy with short C/R gaps; GPU R pinned busy throughout.

## 5. Abort criteria (stop switching, collect artifacts, keep the run alive)

- Any phase > 60 s, or `snapshot`/`restore` returns `OPERATION_STATUS_FAILED`.
- `RayActorError` in train.log after a switch (engine hard-death poisons the
  run — landmine #3; there is no per-sample retry except aborts).
- Trainer does not resume stepping within 2 min of `switch-to-trainer`
  (suspect: NCCL comm in the trainer worker did not survive C/R — the frozen
  trainer+R1 checkpoint-engine group is the known-fragile piece; requires the
  ncclCommSuspend/TCP-transport patched stack).
- `dropped_samples > 0` in mq stats (queue overflow) or reward collapses to
  0/NaN for a whole version after R2 served traffic (suspect KV/weight
  hygiene; check `clear_kv_cache` phase ran).
- R2 generates garbage after resume (check first `watch-mq` samples' reward in
  log): weights were lost — R2 must be restored via cuda backend, never
  resumed with `SUSPEND_MODE_DISCARD` semantics.

After abort: `r2_lifecycle.py status`, `nvidia-smi`, copy
`/workspace/results/{train.log,switch_timings.jsonl,gpu_util.csv}` off the pod.

## 6. Wrap-up

- Let the run hit the `RUN_SECONDS` timeout (expected end state rc=124) or
  stop switching and let it train normally to the end.
- Collect: `experiment.log`, `train.log`, `gpu_util.csv`,
  `switch_timings.jsonl`, `verl_commit`, the mq baseline/active captures, and
  `status` snapshots.
- Deliverable readouts: C2 table (mean/max per phase over N cycles), R2
  marginal fill-rate ratio, reward-continuity verdict — feed PLAN.md §6 C2/C4
  and the M1 go/no-go.

## Appendix: known M1 shortcuts and their owners

| Shortcut | Consequence | Fixed in |
|---|---|---|
| R2 serves version-0 weights, samples tagged current version | metrics-only staleness lie; policy-quality drag on R2 samples grows with run length | M2 (PLAN §3 staging buffer / reload path) |
| Out-of-tree R2 bypasses rollouter concurrency accounting | handled via `elastic_set_max_concurrent_samples` lever (base ↔ 2×base) | M2 controller owns the lever |
| Manual gen-wait gating | operator timing; mistiming wastes wall-clock only | M2 ETA gate |
| `--reload-weights` requires `save_freq>0` and an unverified zero-arg vLLM `reload_weights` RPC | stub only | M2 weight path |
