# PoC Report: Disaggregated Two-Pool Time-Slicing of veRL Workloads

**Date:** 2026-07-21/22 · **Cluster:** verl-research-cluster-west (2× H100-2GPU spot nodes) · **Verdict: cross-pipelining demonstrated; full two-job completion blocked by platform robustness defects** (precisely diagnosed below). Companion to the sync-colocated report (`POC-REPORT.md`), which proved the mechanism; this experiment targets the utilization value.

## Objective

Demonstrate the utilization value proposition: two veRL jobs, each disaggregated across a shared **trainers pool** (node 1) and a shared **samplers pool** (node 2), cross-pipelining so that both pools stay busy concurrently — job A training while job B generates, and vice versa — via veRL's `separate_async` mode wrapped by our `separate_async_timesliced` trainer (yield-on-starvation at the sample wait, dual-lock weight sync, trainer-first lock ordering).

## Topology and stack

- Two pools on **separate nodes** (required: the snapshot agent checkpoints by job-id label only — pod-utils selects `timeslice.io/job-id` regardless of group — so a node hosting two roles of one job would freeze both).
- Per job, 3 pods: CPU head/driver, trainer worker (trainers group + shared claim), sampler worker (samplers group + shared claim). Startup order pins pools to nodes.
- Package `timeslice_verl` 0.2.0: `RoleLocks` (enforced trainer-first order) + `PhaseTransitions`; `separate_async_timesliced` mode; 12/12 lock-protocol unit tests. Image v2.2 (`gcr.io/aishuk-test/timeslice-verl-poc:v2.2`; v2.1 added `cupy-cuda13x` — without it veRL's NCCL checkpoint engine silently no-ops; v2.2 fixed the sampler join gate for placement-group-reserved GPUs).
- Cross-pool weight sync uses veRL's NCCL checkpoint engine with `rebuild_group=true`: the cross-pool communicator exists **only inside** the dual-lock sync span — no live collective during any C/R window, by construction.

## Results — what was demonstrated

- **Cross-pipelining, measured** (attempt a9/b9): **302.3 s of concurrent two-pool execution**, including one continuous 294 s interval of *A-training-on-trainers ‖ B-generating-on-samplers* (and the mirror-image interval earlier). This is the value signature: two pools, two jobs, both busy at once.
- **Pool utilization under contention**: trainers pool **99.4% busy** (A 67.7% / B 31.7%), samplers pool **99.1%** (A 23.5% / B 77.4%) across the ~11-minute contended window.
- **Zero compute penalty for the completing job**: dis-a9 finished **12/12 GRPO steps at 11.6–12.2 s/step contended vs 11.5–12.0 s solo** — overhead lives entirely in lock waits and switches, not in training.
- **Cross-pool handoffs**: 4 clean suspend/restore handoffs, ~10–13 s each (snapshot 9.4–9.9 s, restore 0.5–3.1 s).
- **Correct deferral semantics**: 14 sampler releases with no waiter → `snapshot_deferred=True` (no physical snapshot); the 2 contended trainer releases → `deferred=False`.
- **Learning under C/R**: dis-a9 rewards 0.016→0.156 over 12 steps, matching solo.
- **Topology validated solo**: a single disaggregated job (dis-a7) completed 12/12 cleanly across the 3-pod/2-pool layout.

**Not achieved**: no attempt completed *both* jobs. Every two-job run ended on one of the defects below (≈10 attempts).

## Failure taxonomy — what blocks the full demo

1. **Agent state-model wedge (new platform defect — blocks every clean-slate two-job start).** The agent auto-transitions any job-labeled pod with live CUDA activity to RUNNING — and the CUDA keepalive that platform bug #1 *requires* makes every waiting pod look RUNNING. Two RUNNING jobs on one node → the orchestrator reconciler errors `impossible state: multiple jobs running on node` and requeues forever; no suspend is ever issued, so the state never heals. Root cause: the state model lacks a "registered but not granted" state — keepalive-bearing pods are indistinguishable from running jobs. *Working SOP*: pre-park all GPU worker pods via direct agent `Snapshot` RPCs before any driver acquires.
2. **Completion-boundary race (platform; ended the best run).** At job exit, the atexit lock release triggers suspend(finisher)+restore(waiter) while the finisher's Ray CUDA processes are dying; the restore fired 1.0 s into a ~10 s snapshot → `cuda-checkpoint toggle failed` → group FAULTED, both jobs, no rollback. This is the known completed-job-handling gap plus the no-rollback gap, now reproduced with exact logs.
3. **Hybrid-replica routing (app-level, fixable in our package/config).** Before step 1, generation requests are load-balanced across BOTH vLLM servers — including the hybrid replica living on the *trainer* pod; freezing the trainers pool kills its in-flight requests → `EngineDeadError`. The disagg design requires routing exclusively to the standalone sampler (or sleeping the hybrid replica pre-step-1).
4. Minor/anomaly: one restore placed ~42.7 GiB onto the wrong GPU of the samplers node (device-selection on restore worth investigating).

## Conclusions

1. **The utilization value is real and now measured**: with two jobs sharing two GPUs (naively needing four), both pools ran at ~99% during contention and the completing job paid zero compute penalty — training time identical to solo, all overhead in the ~10–13 s switches and queue waits.
2. **The mechanics scale to the harder topology**: cross-node pools, cross-pool NCCL weight sync under dual locks, yield-on-starvation — all functioned as designed.
3. **Platform robustness is now the demonstrated critical path.** The sync run suggested it; this run proves it: two-pool operation multiplies exposure to the known defects and surfaced a fifth (the state-model wedge). The fixes — cold-start grant semantics, a registered-vs-running state distinction, completion-boundary handling, snapshot retry-with-rollback — are the difference between "value measured in fragments" and "value demonstrated end to end."
4. One app-side item is ours: enforce standalone-sampler-only routing in the disagg mode.

## Evidence

`poc-evidence/disagg/`: `RUN-SUMMARY.md` (experiment agent's writeup), `timeline-a9b9.txt` (merged two-pool event timeline), `analysis-a9b9.txt` (overlap/duty accounting), `full-evidence-disagg.tar.gz` (complete 910 MB set, compressed to 120 MB — all pod/orchestrator/agent logs across ~10 attempts, nvidia-smi traces from both nodes, manifests, the pre-park tooling; kept locally, not in the repo). Images `timeslice-verl-poc:v2.1`/`:v2.2` pushed; node pool resized back to 1; platform, claims, and labels left in place.
