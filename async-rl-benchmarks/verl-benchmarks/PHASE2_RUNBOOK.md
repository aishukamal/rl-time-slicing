# Phase 2 Runbook — resume state (2026-07-26)

## Where things stand

- **Phase 1 DONE** (3 recipes × Mode 4 defaults, 29-39 syncs each). Results in
  `phase1_results/dapo{44,88,412}_mode4/`, plots in `plots/*_mode4_{separate,overlay}.png`,
  section appended to RESULTS.md.
- **Phase 2 batch A RUNNING**: mode1 (on-policy) × 3 recipes, launched ~2026-07-26,
  pods `p1-44`, `p1-88-head`, `p1-88-worker`, `p1-412-head`, `p1-412-worker` on the
  5 pinned `h100-mega-8gpu-spot-a` nodes. Experiment names: `dapo{44,88,412}_mode1`.
- **Batches B and C pending** (launch after A extracts).

## Cluster / access

- Context (ALWAYS pass --context, never switch global): `gke_aishuk-test_us-west1-c_verl-research-cluster-west`
- Pool: `h100-mega-8gpu-spot-a` (5 nodes). Resize (zone us-west1-c despite pool in west1-a):
  `echo "Y" | gcloud container clusters resize verl-research-cluster-west --node-pool=h100-mega-8gpu-spot-a --num-nodes=N --zone=us-west1-c`
- **Do not touch** `h100-mega-8gpu-spot-b` (another agent's).
- Autoscaler reclaims idle nodes in ~3 min → ALWAYS create pods first (Pending pins), then scale up.

## Batch launch commands (from repo dir)

```bash
cd ~/workspaces/GPU-CR/async-rl-timeslicing
# Batch B (mode2 stream off-policy):
MODE_NAME=mode2 STALENESS=0   SYNC_STEP=4 PARTIAL=False bash launch_phase2.sh
# Batch C (mode3 async+stale):
MODE_NAME=mode3 STALENESS=0.1 SYNC_STEP=4 PARTIAL=False bash launch_phase2.sh
```
**The launcher does NOT delete pods** (kubectl apply can't mutate pod specs) — delete
p1-44 p1-88-head p1-88-worker p1-412-head p1-412-worker first, wait until fully gone,
then run the launcher (new pods re-pin the nodes before autoscaler reclaim if done
within ~3 min). Only do this after the previous batch is extracted.
Mode 4 (already done) was: STALENESS=0.1 SYNC_STEP=4 PARTIAL=True.
Mode 1 (batch A, running) is: STALENESS=0 SYNC_STEP=1 PARTIAL=False.

## Monitoring (cron was session-only; recreate if lost)

Every ~25 min:
```bash
KUBECTL="/opt/homebrew/bin/kubectl --context gke_aishuk-test_us-west1-c_verl-research-cluster-west"
for pod in p1-44 p1-88-head p1-412-head; do
  $KUBECTL logs $pod 2>/dev/null | grep -oP 'current_param_version:[0-9.]+' | tail -1
  $KUBECTL logs $pod 2>/dev/null | grep -c '=== DONE'
done
```
Done when `=== DONE` appears. Targets: 44/88 → 29 syncs; 412 → ~39 (mini_bsz=24 → smaller cycles).
Mode 1 note: sync_step=1 → syncs every fetch, so param_version counts will be ~4× higher (~120).

## Extraction per completed batch

```bash
BASE=~/workspaces/GPU-CR/async-rl-timeslicing/phase1_results
mkdir -p $BASE/dapo44_MODEN $BASE/dapo88_MODEN $BASE/dapo412_MODEN
$KUBECTL cp p1-44:/workspace/results/gpu_util_head.csv   $BASE/dapo44_MODEN/gpu_util_head.csv
$KUBECTL cp p1-44:/workspace/results/train.log           $BASE/dapo44_MODEN/train.log
$KUBECTL cp p1-88-head:/workspace/results/gpu_util_head.csv     $BASE/dapo88_MODEN/gpu_util_head.csv
$KUBECTL cp p1-88-worker:/workspace/results/gpu_util_worker.csv $BASE/dapo88_MODEN/gpu_util_worker.csv
$KUBECTL cp p1-412-head:/workspace/results/gpu_util_head.csv     $BASE/dapo412_MODEN/gpu_util_head.csv
$KUBECTL cp p1-412-worker:/workspace/results/gpu_util_worker.csv $BASE/dapo412_MODEN/gpu_util_worker.csv
```
(also grab train.log from 88/412 heads)

## Plots + stats + report

- Plots (aligned separate + overlay, auto role classification):
  `/usr/bin/python3 plot_phase1.py` — walks all `phase1_results/*/`, writes `plots/<run>_{separate,overlay}.png`.
  Role classifier: sampler iff nonzero-mem median within 3GB of 65,500 MiB AND p95-p50 < 1.5GB (vLLM pool @ util 0.80); else trainer. Validated on all 3 topologies incl. mixed 6T+2S nodes.
- Stats table: `/usr/bin/python3 analyze_gaps.py` — prints the full markdown table row
  (train window, trainer/sampler active%, trainer/sampler gaps ≥5s, weight-sync duration
  from train.log) and writes `plots/<run>_gaps.png` (histogram + recurrence scatter).
  All stats are trimmed to the training window (first→last GPU activity) — plot_phase1.py
  does the same trim. Both scripts walk ALL run dirs, so old-mode plots regenerate too.
- Report structure (2026-07-26 restructure): RESULTS.md is READER-FACING — Objective /
  Setup / Recipes / 4 modes / Results with one subsection per mode (Mode 4 done; fill the
  "Mode N (PENDING/RUNNING)" placeholder subsections as batches extract) / cross-mode
  comparison / Conclusion / Artifacts. All incremental history (1-step survey, pidfd,
  0.5B + 7B steady-state) lives in worklog.md — do NOT re-add it to RESULTS.md.
- Report: fill the corresponding mode subsection in RESULTS.md (mirror Mode 4's format), then
  `/usr/bin/python3 md_to_html.py && open RESULTS.html` (self-contained, base64 images).

## After batch C completes

1. Extract + plot + append mode2/mode3 sections and a 3×4 cross-mode comparison.
2. Delete pods `p1-*`, configmap `phase1-scripts`, services `p1-88-head-svc p1-412-head-svc`.
3. Scale pool to 0.

## Critical environment facts (cost a week to learn)

- **pidfd_getfd**: verl weight sync needs pod `hostPID: true` + cap SYS_PTRACE + seccomp Unconfined, else first sync dies/hangs with misleading vLLM EngineCore collective_rpc errors.
- **OOM on H100**: recipes tuned for H20 96GB; fix is `fsdp_size=4` (not 2). No offloading needed.
- **Divisibility**: trajectories (mini_bsz × require_batches × n) % trainer-GPU-count == 0; 4_12 (12 GPUs) needs mini_bsz=24.
- Working stack: `verlai/verl:vllm020.dev2` + `pip install verl@git+main --no-deps` + `cupy-cuda12x` + checkpoint_engine.backend=nccl.
- Qwen2.5-Math-7B requires config.json `max_position_embeddings: 32768` patch (recipe instruction).
- Dataset: `BytedTsinghua-SIA/DAPO-Math-17k` HF dataset, file `data/dapo-math-17k.parquet`, already has `prompt`/`reward_model`/`data_source` fields.
