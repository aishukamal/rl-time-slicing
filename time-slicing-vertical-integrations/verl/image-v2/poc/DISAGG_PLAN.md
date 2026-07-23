# Disaggregated two-pool verl time-slicing PoC — manifest/topology plan

Input for the experiment track. Image: `gcr.io/aishuk-test/timeslice-verl-poc:v2`.
Mode: `separate_async_timesliced` (registered by `timeslice_verl` 0.2.0).
NO cluster changes were made while preparing this plan.

## 1. Topology verdict: pools MUST be on separate nodes

The snapshot agent checkpoints per (node, jobID) — NOT per (node, jobID, group).
At llm-d-rl-time-slicing @ main:

* `server.go` k8s CUDA path: `resolvePIDs(bgCtx, jobID, explicitPIDs)` →
  `podutils.GetLocalPods(ctx, jobID)` with
  `LabelSelector: fmt.Sprintf("%s=%s", JobIDLabel, jobID)` — job-id label only,
  the `timeslice.io/group` label is never used for PID selection.
* `state-machine/state-manager.go`: `jobs map[string]*Job` keyed by jobID only;
  `Group` is a stored/logged field. A job cannot be RUNNING on one group and
  SNAPSHOTTED on another on the same node.

So if a job had a sampler pod AND a trainer pod on one node, snapshotting its
"samplers" group would freeze BOTH pods. Hence:

* trainers pool = the trainers node (all jobs' trainer pods)
* samplers pool = the samplers node (all jobs' sampler pods)

## 2. Node capacity (checked read-only 2026-07-20)

* `h100-2gpu-spot` (a3-highgpu-2g, 2x H100-80GB, spot, zone us-west1-b):
  1 node Ready, **autoscaling DISABLED** (no autoscaling block, initialNodeCount 1).
  A second node will NOT appear on its own — the experiment track must run:
  `gcloud container clusters resize verl-research-cluster-west --node-pool h100-2gpu-spot --num-nodes 2 --zone us-west1-c --project aishuk-test`
  (spot capacity risk in us-west1-b; verify the node schedules).
* Designate: node 1 = TRAINERS node, node 2 = SAMPLERS node. Each node has
  2 GPUs = one 1-GPU claim per job per pool (job-a + job-b fit both pools).
  Snapshot-agent DaemonSet must run on both nodes.

## 3. Per-job layout (plain pods + `ray start`, no KubeRay)

Each verl job is its own 3-pod Ray cluster, all running image :v2 with
`/opt/poc/run_job_disagg.sh` as entrypoint (ROLE env selects behavior):

| pod                 | node          | GPU | ROLE           | labels |
|---------------------|---------------|-----|----------------|--------|
| `<job>-head`        | CPU node      | 0   | `head-driver`   | none required (holds no GPU state) |
| `<job>-trainer`     | trainers node | 1   | `trainer-worker`| `timeslice.io/job-id=<job>`, `timeslice.io/group=<TRAINER_GROUP>` |
| `<job>-sampler`     | samplers node | 1   | `sampler-worker`| `timeslice.io/job-id=<job>`, `timeslice.io/group=<SAMPLER_GROUP>` |

* Pin pods with nodeSelector/nodeName to the designated nodes. hostIPC not
  required; pods need the GPU resource claim (1 each).
* A per-job headless Service (or pod IP) for the ray head; set
  `RAY_HEAD_ADDR=<head-svc>:6379` on the worker pods. The two jobs' Ray
  clusters are fully independent (different heads/ports/pods).
* Env on ALL three pods of a job: `TIMESLICE_JOB_ID` (FRESH id per attempt),
  `TIMESLICE_ORCH_ADDR` (cluster-reachable orchestrator Service — the lock
  holder is the driver actor, which is not on a GPU node),
  `TIMESLICE_TRAINER_GROUP`, `TIMESLICE_SAMPLER_GROUP`.
  Orchestrator group config: group `<TRAINER_GROUP>` = trainers node GPUs,
  group `<SAMPLER_GROUP>` = samplers node GPUs, two member jobs each.
* The driver propagates TIMESLICE_*/NCCL_CUMEM_ENABLE=0/VERL_USE_EXTERNAL_MODULES
  to every Ray actor via `ray_kwargs.ray_init.runtime_env.env_vars` (baked in
  run_job_disagg.sh) — required because TaskRunnerV1 (the trainer object and
  lock holder) is a CPU Ray actor that may be scheduled on any pod.

## 4. Startup order (placement is enforced by ordering, not by Ray config)

Ray cannot pin a 1-GPU bundle to a chosen node. `run_job_disagg.sh` enforces:

1. Start `<job>-head` (ray head, data/model prep, driver launch).
2. Start `<job>-trainer` (joins Ray with 1 GPU). The trainer worker-group
   placement group (trainer.nnodes=1 x n_gpus_per_node=1) takes this GPU —
   it is the only one in the cluster.
3. `<job>-sampler` polls `ray status` and joins ONLY once it sees GPU
   `1.0/1.0` (trainer GPU claimed). The standalone rollout pool
   (rollout.nnodes=1 x rollout.n_gpus_per_node=1, created later in
   `PPOTrainerSeparateAsync._setup`) then blocks until the sampler node joins
   and lands on it deterministically.
4. Repeat for job B (independent cluster; can be concurrent with A after A's
   step 3 completes — the gate is per-job).
5. VERIFY placement before calling the run valid: on each GPU node,
   `nvidia-smi` PIDs must map to the expected pod (trainer pod PIDs on the
   trainers node, vLLM PIDs on the samplers node). If inverted: delete the
   job's pods and retry with a FRESH TIMESLICE_JOB_ID.

Both jobs' first acquire happens in trainer `__init__` (before any CUDA work);
the baked CUDA keepalive process per GPU pod prevents the first-acquire
deadlock (snapshot agent needs a live CUDA PID to transition the job RUNNING).

## 5. Lock phase model (what the experiment should observe)

Global order: TRAINER before SAMPLER (never request TRAINER holding SAMPLER).

| phase                            | trainers lock | samplers lock |
|----------------------------------|---------------|----------------|
| init + initial weight sync       | HELD          | HELD           |
| feed (prompt submission)         | HELD          | –              |
| sample-wait (generation exec)    | – (yielded)   | HELD           |
| train sub-phases (logprob/adv/update) | HELD     | –              |
| per-step weight sync (on_step_end)| HELD         | HELD (span)    |
| validation (disabled in PoC)     | HELD          | HELD           |

Cross-pipelining signature in logs (`[timeslice]` lines are always printed):
job A `ACQUIRE role=trainer` overlapping job B `ACQUIRE role=sampler`, i.e.
A trains while B samples, alternating. Conditional-release: if A's replay
buffer already holds a full batch at on_sample_begin, A keeps TRAINER and
skips the SAMPLER acquire for that step (logged as "batch already buffered").

## 6. Config baked into run_job_disagg.sh (head-driver role)

Qwen2.5-0.5B-Instruct, GSM8K, TOTAL_TRAIN_STEPS=12, TRAIN_BATCH_SIZE=32 =
PARAM_SYNC_STEP(2) x PPO_MINI_BATCH_SIZE(16), gpu_memory_utilization=0.25,
NCCL_CUMEM_ENABLE=0, checkpoint_engine backend=nccl with
`engine_kwargs.nccl.rebuild_group=True` (the cross-pool NCCL group is created
and destroyed inside each dual-lock weight-sync span — no persistent NCCL
communicator between the pools between syncs), validation disabled,
max_off_policy_threshold=100 (no drop churn under C/R freezes in 12 steps).

## 7. Open risks (ranked)

1. Spot capacity for the 2nd a3-highgpu-2g node in us-west1-b (pool has no
   autoscaling; manual resize; may not fulfill).
2. Frozen-sampler HTTP edges: agent-loop actors keep HTTP requests open to
   the standalone vLLM server while it is checkpointed. TCP resets on
   restore could surface as request failures rather than retriable aborts;
   partial-rollout retry covers aborts, not necessarily connection errors.
3. ZMQ metadata channel of the NCCL checkpoint engine: PUB socket lives on the
   actor rank-0 (trainer pod) across syncs; both endpoints get frozen/restored
   between syncs. SUB side reconnects per sync (init_process_group), PUB
   socket survival after C/R is unproven.
4. TaskRunnerV1 (CPU driver actor) placement is Ray's choice; it is immune to
   GPU C/R (no CUDA context) but must reach the orchestrator — hence the
   cluster-reachable TIMESLICE_ORCH_ADDR requirement.
5. Natural last-step exit skips on_train_end (verl early-return); the TRAINER
   lock is released by the RoleLocks atexit hook when python exits. The
   stay-alive wrapper keeps the POD alive, not python — do not `kill -9` the
   driver before atexit runs.
6. Trainer-pool memory: the trainer GPU hosts FSDP actor + a sleeping hybrid
   vLLM replica (separate_async always builds it). Fine for 0.5B; watch for
   OOM on first validation-free run; hybrid replica is slept after step 1.
7. In-flight generation for FUTURE steps continues on the sampler pool after
   sample() returns; when the other job takes the pool those requests freeze
   mid-token. Expect elevated trajectory_spans/staleness metrics — this is
   measurement signal, not a bug.
