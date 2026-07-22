# Disaggregated two-pool verl time-slicing (Track E) — run record 2026-07-21/22

Work dir / evidence: /tmp/tsrun-disagg-1784595836

## Cluster mutations (all reverted or intentionally left)
1. h100-2gpu-spot resized 1->2 (2026-07-21 ~03:44Z), resized BACK 2->1 at end. Nodes used:
   v0cd = trainers node (existing), pxjj = samplers node (new, spot fulfilled in ~3.5 min).
2. Node labels: v0cd lost group.timeslice.io/shared-gpu, gained group.timeslice.io/trainers=true;
   pxjj gained group.timeslice.io/samplers=true + timeslice.io/enabled=true.  [LEFT IN PLACE]
3. ResourceClaims default/shared-trainers-gpu-claim, default/shared-samplers-gpu-claim (1 GPU, v1
   shape identical to shared-gpu-claim). [LEFT IN PLACE; shared-gpu-claim untouched]
4. timeslice-system: snapshot-agent DS pods and orchestrator pod restarted numerous times to clear
   sticky job state (final restart leaves them clean; lock CM empty). Deployment/DS specs untouched.
5. Images pushed (Cloud Build, project aishuk-test):
   - gcr.io/aishuk-test/timeslice-verl-poc:v2.1@sha256:beba66ef... = v2 + cupy-cuda13x
     (verl nccl checkpoint engine silently unregistered without cupy -> "Checkpoint engine nccl
     not registered" at actor init).
   - gcr.io/aishuk-test/timeslice-verl-poc:v2.2@sha256:1918cdfc... = v2.1 + fixed sampler join gate
     in run_job_disagg.sh (ray status reports PG-reserved GPUs as "0.0/1.0 (1.0 reserved in
     placement groups)"; the old used==1.0 gate never fired).
6. All experiment pods (dis-a*/dis-b* head/trainer/sampler + svcs), tsdisagg-util, node-debugger
   pods: DELETED at end.

## Attempts
- dis-a1: aborted pre-start (image entrypoint is NVIDIA banner; command must be set explicitly).
- dis-a2: FAIL trainer actor init: model path /tmp/models/... only exists on head pod ->
  worker pods now prefetch the model before joining ray.
- dis-a3: FAIL "Checkpoint engine nccl not registered" (missing cupy) -> image v2.1.
- dis-a4/dis-a5/dis-a6: FAIL "Total available GPUs 0 < 1": sampler join gate never fired
  (PG reservation not detected by gate regex) -> image v2.2. (dis-a6 head pod also lost to
  pathways-cpu-pool spot preemption -> heads moved to samplers GPU node, no job-id label.)
- dis-a7 SOLO on v2.2: SUCCESS 12/12 steps. Init->locks 1m35s, init 2m20s under dual lock,
  steps 11.5-12.0s compute each, rewards 0.031->0.078 (GSM8K, 0.5B, 12 steps).
- dis-a8+dis-b8 (concurrent): WEDGE, no grant ever completed: agent auto-marks any labeled pod
  with live CUDA (the keepalive!) RUNNING; two RUNNING jobs per node -> orchestrator reconciler
  "impossible state: multiple jobs running on node" -> permanent requeue loop, no suspends issued.
- dis-a9+dis-b9 (staged: B created after A's grant): same wedge when B's keepalives appeared;
  UNBLOCKED MANUALLY by calling the snapshot-agent Snapshot RPC directly (park dis-b9) ->
  platform then ran the full two-pool choreography (see evidence below). A completed 12/12.
  At A's completion the atexit release triggered suspend(A)+restore(B) overlapped 1s apart on
  trainers while A's ray CUDA procs were dying -> cuda-checkpoint toggle failed -> trainers group
  FAULTED both jobs; B died pre-step-1.
- dis-a10+dis-b10 (pre-parked all 4 workers before heads): reached split-pool state quickly, but
  both jobs died pre-step-1: vLLM EngineDeadError on the HYBRID replica living on the TRAINER pod
  (agent-loop routes pre-step-1 generation to BOTH vllm servers; freezing the trainers pool kills
  in-flight requests on that engine), plus restore flake
  'cuda-checkpoint toggle: "OS call failed or operation not supported on this OS"' -> trainers
  FAULTED (dis-a10), B blocked, both drivers exited rc=1.

## Cross-pipelining evidence (dis-a9/dis-b9, 2026-07-21 22:42-23:20Z)
- timeline-a9b9.txt: interleaved per-pool grants/releases/snapshots/restores/steps.
- 2 measured cross intervals (different jobs concurrently RUNNING on the two pools):
  22:59:07-22:59:15 (8.0s, B-trainers || A-samplers) and
  23:02:39-23:07:33 (294.3s, A-trainers || B-samplers). Total 302.3s.
- 4 clean pool handoffs (suspend+restore pairs): snapshots 9.4-9.9s, restores 0.5-3.1s
  => switch cost ~10-13s per handoff (matches sync-run's ~14.6s incl orchestration).
- Duty cycle in contended window 22:58:55-23:09:51: trainers 99.4% busy (A 67.7%/B 31.7%),
  samplers 99.1% busy (A 23.5%/B 77.4%).
- Effective step time: A contended steps 2-12 compute 11.6-12.2s vs solo 11.5-12.0s (=no compute
  penalty; overhead is switch latency + lock waits: A trainer re-acquire waited 938s across the
  wedge+B's init turn; A sampler acquire waited 289s across B's samplers hold).
- GPU occupancy (contended window): trainers node GPU0 100% occupied (max 30.1 GiB), GPU1 idle;
  samplers node GPU0 100% occupied (max 65.9 GiB), GPU1 briefly 9.8% (42.7 GiB — restore placement
  anomaly worth investigating).
- Rewards under C/R (dis-a9, 12 steps): 0.016 -> 0.156 trend consistent with solo run.
- snapshot_deferred stats (dis-a9, deduped): 14x sampler RELEASE pending_waiters=0 deferred=True;
  2x trainer RELEASE pending_waiters=1 deferred=False (immediate snapshot; correct).

## Files
analysis-a9b9.txt, timeline-a9b9.txt, steps-summary.txt, duty-occupancy.txt, lock-tally-a9b9.txt,
group-status.log (2s polls both groups), poller.log (1s pod poll), orchestrator.log (streamed),
agent-{trainers,samplers}.log (streamed; contain follower duplication), final-*.log (end-state
snapshots), nvidia-smi-{trainers,samplers}.log, dis-a*/dis-b*-{head,trainer,sampler}.log,
manifests dis-*.yaml + gen-job.sh + claims.yaml + util-pod.yaml, image-v21/ image-v22/ build dirs,
agent_call.py (direct snapshot-agent RPC tool), analyze_disagg.py.
