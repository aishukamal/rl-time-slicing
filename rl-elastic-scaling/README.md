# Elastic RL Scaling PoC — Trainer↔Rollout GPU Time-Slicing via Transparent C/R

Proof-of-concept for elastic GPU reassignment between the trainer and rollout roles of an
RL post-training job, using transparent checkpoint/restore — the snapshot-agent's cuda backend, with the
[multi-gpu-cr-poc](../multi-gpu-cr-poc/) NCCL-suspend shim (`universal_cr_shim.c`) and
TCP-transport NCCL configuration making NCCL-holding processes checkpointable — under a
completely **unmodified verl** fully-async recipe (DeepSeek-R1-Distill-Qwen-1.5B, code-RLVR,
Eurus-2-RL code prompts, s=8). When the run is generation-bound the rollout engine borrows the
trainer's GPU (the trainer is suspended and evicted to host memory); when a training batch is
ready the GPU is switched back. M1 proves the mechanism with manually scripted switch cycles;
M2 closes the loop with an autonomous policy controller driven by queue depth and phase signals;
M3 runs the comparison matrix against colocated and static baselines plus dynamism/no-harm arms.

## Headline results

- **Same 2 GPUs:** elastic switching delivers **+23% samples/GPU-hour vs the static 1 trainer : 1
  rollout baseline** (M0 619.0s → M2 502.7s wall for the same workload).
- **Vs 3-GPU static 1:2:** the 2-GPU elastic run reaches **95.8% of the per-GPU-hour efficiency of
  the 3-GPU static 1 trainer : 2 rollout arm — with 1/3 fewer GPUs**.

### Results ladder (same workload throughout)

| Arm | GPUs | Wall time | Notes |
|---|---|---|---|
| M0 static 1:1 baseline | 2 | 619.0 s | unmodified verl fully-async |
| M1 manual switch cycles | 2 | 553.1 s | scripted trainer↔rollout switches |
| M2 autonomous controller | 2 | 502.7 s | closed-loop policy controller (best case 501.8 s) |
| M3 colocated (verl native) | 2 | 522.7 s | trainer+rollout sharing GPUs, no C/R |
| M3 static 1:2 | 3 | 321.0 s | upper bound on throughput, +50% GPUs |

## Contents

| Path | Description |
|---|---|
| [research/rl-infra-landscape.md](research/rl-infra-landscape.md) | RL infra landscape survey motivating the PoC |
| [PLAN.md](PLAN.md) | PoC plan: milestones M0–M3, hypotheses, measurement design |
| [verl-integration-notes.md](verl-integration-notes.md) | Notes on verl fully-async internals and integration points |
| **code/** | |
| [code/m0/k8s-job-m0-baseline.yaml](code/m0/k8s-job-m0-baseline.yaml) | M0 static 1:1 baseline job spec |
| [code/m1/](code/m1/) | M1 manual-switch demo: [M1-RUNBOOK.md](code/m1/M1-RUNBOOK.md), `elastic_trainer.py`, `fully_async_main_elastic.py`, `r2_lifecycle.py`, `cycles.sh`, `k8s-job-m1.yaml`, `image/` (Dockerfile, cloudbuild, C/R shim sources) |
| [code/m2/](code/m2/) | M2 autonomous controller: `policy_controller.py`, `k8s-job-m2.yaml`, `preflight-probe.yaml` |
| [code/m3/](code/m3/) | M3 matrix: `regime_flip.py`, `launch-m3b.sh`, job specs for colocated / regime-shift / armed / no-harm / static-1:2 arms, preflight probes |
| **results/** | |
| [results/m0/](results/m0/) | [M0-REPORT.md](results/m0/M0-REPORT.md), train/pod logs, `gpu_util.csv`, `summary.json`, `analyze_run.py` |
| [results/m1/](results/m1/) | run1/run2 pod logs; `run3/` and `run4/` full cycle artifacts (`switch_timings.jsonl`, per-cycle logs, `train.log`, `gpu_util.csv`) |
| [results/m2/](results/m2/) | [M2-REPORT.md](results/m2/M2-REPORT.md); `attempt1/` ([report](results/m2/attempt1/M2-REPORT-attempt1.md)) and `run2/` ([report](results/m2/run2/M2-RUN2-REPORT.md), `decisions.jsonl`, `switch_timings.jsonl`, `steps_metrics.json`, `summary.json`, logs) |
| [results/m3/](results/m3/) | [M3-RUNS-REPORT.md](results/m3/M3-RUNS-REPORT.md); arms: `colocated/`, `colocated-b64/`, `regime-shift/`, `no-harm/` (armed + control), `static12/` (head + worker) |

## Artifact notes (exclusions from the raw capture)

All markdown reports, `decisions.jsonl` / `switch_timings.jsonl`, and summary/metrics JSON files
are included uncompressed and unmodified. The following raw artifacts were excluded because they
exactly duplicate files already present:

- `results/m3/**/results*.tgz` (7 archives) — tarballs whose contents are the already-extracted
  files sitting next to them (`colocated/`, `colocated-b64/`, `regime-shift/`, `no-harm/armed/`,
  `no-harm/control/`, `static12/results-head.tgz`, `static12/results-worker.tgz`).
- `results/m2/attempt1/early-partial-copies/` — a byte-identical partial copy of
  `attempt1/full_pod.log` plus an empty `gpu_util.csv`.
- `results/m2/run2/experiment.log` — identical to `run2/full_pod.log` minus its 28-line header;
  `full_pod.log` (the superset) is kept.
- `__pycache__/` directories under `code/m1`, `code/m2`, `code/m3`.

No files were compressed (nothing exceeded the 20 MB threshold).
