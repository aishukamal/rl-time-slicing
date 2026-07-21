# Two-job GPU time-slicing experiment — evidence summary
Date: 2026-07-20 22:15 – 2026-07-21 00:03 UTC
Cluster: verl-research-cluster-west, node gke-verl-research-clus-h100-2gpu-spot-8cf82ce9-v0cd (H100 80GB, DRA claim shared-gpu-claim, 1 GPU shared by both jobs)
Platform: orchestrator+snapshot-agent image main-11a661f; workload gcr.io/aishuk-test/timeslice-verl-poc@sha256:100e7d95 (verl 0.9.0.dev GRPO, Qwen2.5-0.5B, GSM8K, sync_timesliced, 12 steps, single GPU)

## Runs
- Run 1 (verl-job-a / verl-job-b): A trained 12/12 steps SOLO (B started too late). Solo baseline.
  Then B's arrival triggered a snapshot of already-completed A -> "no GPU PIDs found" -> A FAULTED -> group faulted -> B died. (Bug #2)
  A's first ACQUIRE also deadlocked 16.6 min until a manual CUDA keepalive was exec'd. (Bug #1)
- Run 2 (verl-job-a2/b2, keepalive built in): true alternation from startup; both reached step ~6;
  b2's 4th snapshot failed (cuda-checkpoint lock "initialization error" on vLLM worker pid) -> cascade FAULTED both. (Bug #3/#4)
- Run 3 (verl-job-a3/b3, + NCCL_CUMEM_ENABLE=0): FULL demonstration.
  a3: 12/12 steps. b3: 11/12 steps. 23 alternating turns, 22 clean handoffs, zero out-of-order grants.
  Endgame: a3's post-completion teardown raced its final snapshot -> a3 FAULTED -> group faulted -> b3's last acquire failed. (Bug #4)

## Run 3 quantitative results (from orchestrator/agent logs; see timeline-run3.txt)
- First turns (incl. model+vLLM init under lock): a3 123.9s, b3 127.1s
- Steady-state turn (one GRPO step): 19.7–20.5s (mean ~20.0s)
- Solo baseline step turn (run 1): 17.6–19.8s (mean ~18.8s)
- Switch overhead (yield X -> grant Y): 14.1–15.0s (mean ~14.6s), of which
  snapshot cuda-checkpoint ~9.6–10.4s, restore cuda-checkpoint ~3.0s, orchestration/polling ~1–2s
- Effective per-step cost under contention: ~20.0s step + ~14.6s switch = ~34.6s vs ~19.9s solo cycle
- Strict alternation: A3,B3,A3,B3,... 23 turns, no violations (see analyze.py output / timeline)
- snapshot_deferred=False on every contended RELEASE (both jobs, pending_waiters=1);
  run-1 solo RELEASEs show snapshot_deferred=True, pending_waiters=0
- ACQUIRE examples: a3 first waited=2001ms; b3 first waited=137000ms (blocked across a3's whole init+step1 turn); steady-state waits ~13-15s (context_restored=True)

## Training integrity across suspensions (critic/rewards/mean per step)
run1 A solo:  .008 .000 .008 .016 .023 .016 .016 .039 .070 .117 .055 .117
run3 a3:      .016 .008 .016 .016 .039 .063 .039 .055 .039 .094 .070 .141
run3 b3:      .008 .016 .047 .016 .023 .016 .063 .102 .063 .117 .063 (step 12 lost to endgame fault)
Reward trends match the solo baseline; no NaNs/corruption after ~12 suspend/restore cycles per job.

## GPU occupancy trace (nvidia-smi 2s sampling, nvidia-smi.log / gpu-occupancy-run3.txt)
- Mid-turn: only the ACTIVE job's vLLM worker pid exists on the GPU and util >0 from its pids;
  the suspended job's vLLM worker (up to ~18.6GB during generation) is evicted each switch.
- Residual note: the suspended job's keepalive (612MB) + one FSDP WorkerDict (~3.4GB) remain
  NVML-resident between turns (~4GB); total GPU usage stayed 11–27GB of 80GB. Compute alternation
  is strict; memory eviction is partial (vLLM yes, small residuals no).

## Platform bugs found (deployed build main-11a661f)
1. First-ACQUIRE deadlock: orchestrator grants Acquire only when agent reports job RUNNING (GPU pids),
   but the sync_timesliced adapter blocks in Acquire BEFORE any GPU touch. Fresh jobs deadlock.
   Workaround: background "kick" process creating a CUDA context in each job pod.
2. Snapshot of a completed pod fails hard: "no GPU PIDs found" -> job FAULTED -> whole group faulted
   (integ commit f65cbfd already fixes this as graceful no-op, but is NOT in the deployed image).
   Workaround: wrapper keeps pod+keepalive alive after run_job.sh exits.
3. cuda-checkpoint lock flake: "Could not lock on process ID <vllm worker>: initialization error"
   (~1 in ~20 ops; run 2). No retry in agent -> FAULTED. NCCL_CUMEM_ENABLE=0 run completed 23/24
   step-turns, but the endgame hit the same lock error during process teardown (run 3).
4. No rollback/no retry on failed switch: after a failed snapshot of X, the controller marks X FAULTED,
   then still attempts restore of Y onto a GPU with X's memory resident -> restore fails -> Y FAULTED too.
   One flake collapses the whole group ("requires human intervention").
5. Ghost lock/state leaks: failed/killed acquirers leave group.spec.lockingJob set with no janitor
   (recovered manually via Yield RPC as the ghost job id); agent job state (FAULTED) is sticky per
   job-id forever (no pod-delete handling) -> forced new job IDs per iteration (a2/b2, a3/b3).
   Also: orchestrator pod was replaced at 23:21 during run 2 (old pod exited/Completed; contributed to run-2 cascade timing analysis but not the root cause).

## Verdict
Time-slicing of two real verl GRPO jobs on one H100 via orchestrated cuda-checkpoint suspend/resume
IS demonstrated: 22 clean automatic handoffs, strict per-step alternation, real learning on both jobs
across ~12 suspend/restore cycles each, deterministic ~14.6s switch cost. It is NOT yet robust:
first-acquire semantics, completed-job handling, cuda-checkpoint flakes, and missing rollback make
unattended runs fail; b3 lost its final step to the endgame race.
