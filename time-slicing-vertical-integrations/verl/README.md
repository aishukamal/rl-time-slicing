# veRL × llm-d-rl-time-slicing — vertical integration PoC

Two real, unmodified veRL GRPO workloads time-slicing one physical H100 (22 automatic
cuda-checkpoint handoffs, both jobs learning across ~12 suspensions each) with **zero
veRL source changes** — the integration is a pip package + env vars + one config value
+ pod labels. Full results: [`../POC-REPORT.md`](../POC-REPORT.md); strategy:
[`../ONE-PAGER.md`](../ONE-PAGER.md).

## Layout

| Dir | Contents |
|---|---|
| `package/timeslice_verl/` | The integration package: `@register_trainer("sync_timesliced")` extending veRL's stock `PPOTrainerSync` (whole-step turn-taking), `PhaseLocks` helper, registered via a `verl.plugins` entry point so it loads inside Ray actor processes |
| `image/` | Workload image build (Cloud Build): `verlai/verl:vllm023.dev1` base + veRL pinned `6a6242f3` + `timeslice` client + this package; build-time gates verify trainer-mode registration and hydra config composition. Published as `gcr.io/aishuk-test/timeslice-verl-poc:v1` |
| `image/poc/run_job.sh` | Job entrypoint: GSM8K prep + Qwen2.5-0.5B download + `main_ppo` launch with `trainer.use_v1=True trainer.v1.trainer_mode=sync_timesliced`; env knobs `TIMESLICE_JOB_ID/ORCH_ADDR/GROUP`, `TOTAL_TRAIN_STEPS` |
| `deploy/` | Platform-deploy extras used for the PoC: helm values overlay (`values-poc.yaml`), the platform-image Cloud Build recipe, and the smoke-test pod that validated agent discovery |
| `manifests/` | The two-job k8s Jobs (a3/b3 = the successful run 3; a/b and a2/b2 = earlier iterations kept for the workaround history) + util pod (ghost-lock clearing) |
| `evidence/` | Experiment writeup, merged run-3 event timeline, and the turn/switch accounting script |

## How the run works

1. Platform (accelerator orchestrator + snapshot-agent DaemonSet + NVIDIA DRA driver)
   helm-deployed; one GPU node labeled into group `shared-gpu`; one shared 1-GPU
   `ResourceClaim` that **both** jobs reference (DRA oversubscription → same physical GPU).
2. Each Job pod: `timeslice.io/job-id` + `group` labels (must equal `TIMESLICE_JOB_ID`),
   group nodeSelector, tolerations, the shared claim — no `nvidia.com/gpu` limits.
3. The trainer mode acquires the group lock before touching the GPU, holds it for one
   full training step (generate → train → weight sync), releases, re-queues. The
   orchestrator snapshots the yielder and restores the next holder via cuda-checkpoint.

## Known platform workarounds baked into the manifests

(All four are platform bugs with fixes pending upstream — see POC-REPORT.md findings.)
- CUDA keepalive process per pod (first-acquire grant requires visible GPU PIDs)
- stay-alive wrapper after training exits (snapshotting a finished job faults the group)
- fresh job ids per attempt (agent/orchestrator state is sticky per job id)
- `NCCL_CUMEM_ENABLE=0`

Status: mechanism demonstrated (sync colocated = simplest end-to-end validation).
The disaggregated topologies (async/sync disagg — where the utilization value lives)
are in preparation; this directory will grow `separate_async` support next.
