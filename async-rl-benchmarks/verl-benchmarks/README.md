# veRL Async RL Time-Slicing Benchmarks

Does GPU time-slicing (checkpoint/restore) apply to **async RL**? Full 12-run matrix:
3 standard veRL `fully_async_policy` recipes (`dapo_7b_math_fsdp2_{4_4,8_8,4_12}`,
Qwen2.5-Math-7B + DAPO-Math-17k) × all 4 async modes, on H100-Mega nodes (GKE).

**Answer: yes, in every configuration.** The 4 async modes collapse into 2 regimes —
sync-every-fetch (Mode 1: trainers 35-60% active) vs sync-every-4 (Modes 2/3/4,
identical within ±4%: trainers 63-89% active). Staleness allowance and partial
rollout barely move GPU duty cycles; sync cadence and the trainer:sampler GPU split
are the only levers. Idle gaps run 5-170s and recur every sync cycle, vs 2-3s C/R cost.

## Report

- **[RESULTS.md](RESULTS.md)** — curated report (renders on GitHub with all plots)
- **[RESULTS.html](https://aishukamal.github.io/rl-time-slicing/async-rl-benchmarks/verl-benchmarks/RESULTS.html)** — same report, self-contained HTML via GitHub Pages
- [worklog.md](worklog.md) — chronological log of every incremental run and blocker
  (1-step survey, `pidfd_getfd` root cause, 0.5B/7B steady-state validations)

## Contents

| Path | What |
|------|------|
| `plots/` | All figures: aligned trainer/sampler panels, overlays, gap histograms + recurrence scatters, cross-mode summary |
| `phase1_results/dapo{44,88,412}_mode{1,2,3,4}/` | Raw traces per run: `gpu_util_*.csv.gz` (nvidia-smi @ 100ms, all GPUs/nodes), `train.log.gz` (veRL metrics incl. param_sync timings) |
| `multistep_results/` | Steady-state validation traces (0.5B 64-step, 7B exact-recipe) |
| `full_results/` | June 1-step survey traces (superseded; see worklog) |
| `launch_phase1.sh` / `launch_phase2.sh` | K8s launchers (pods + configmap; full veRL arg list, pidfd securityContext) |
| `plot_phase1.py` | Aligned/overlay plots; trainer-vs-sampler role classification by vLLM memory-pool signature |
| `analyze_gaps.py` | Idle-gap detection (window-trimmed), histogram/recurrence figures, stats tables |
| `gpu_monitor.sh` | 100ms nvidia-smi sampler |
| `md_to_html.py` | RESULTS.md → self-contained RESULTS.html |
| `PHASE2_RUNBOOK.md` | Operational runbook (launch/monitor/extract/cleanup, env gotchas) |

CSV columns: `timestamp_ms,gpu_index,gpu_util_pct,mem_util_pct,mem_used_mib,power_w`.
`gunzip` the traces to reproduce: `python3 plot_phase1.py && python3 analyze_gaps.py`.
