#!/usr/bin/env python3
"""
Phase-1 plots: trainer vs sampler GPU duty cycle with a SHARED absolute time axis.

For each run directory (phase1_results/<run>/ containing gpu_util_head.csv and
optionally gpu_util_worker.csv):
  1. <run>_separate.png — trainer panel + sampler panel, identical x-limits
  2. <run>_overlay.png  — both on one axis

Role assignment is data-driven via memory footprint: vLLM (sampler) GPUs hold a
large constant allocation (~0.8 × 80GB); FSDP trainer GPUs fluctuate lower.

Usage: python3 plot_phase1.py [results_base]
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAINER_C = "#E53935"
SAMPLER_C = "#1E88E5"


def load(csv_path):
    """Return {gpu_index: {'ts': [...], 'util': [...], 'mem': [...]}} (absolute ms)."""
    traces = defaultdict(lambda: {"ts": [], "util": [], "mem": []})
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            gi = int(row["gpu_index"])
            traces[gi]["ts"].append(int(row["timestamp_ms"]))
            traces[gi]["util"].append(float(row["gpu_util_pct"]))
            traces[gi]["mem"].append(float(row.get("mem_used_mib", 0) or 0))
    return dict(traces)


def classify_roles(all_gpus):
    """Split (node, gpu_idx) keys into trainer/sampler by memory signature.

    vLLM samplers allocate a fixed pool = gpu_memory_utilization (0.80) × 81GB
    ≈ 65,500 MiB and stay flat (spread < ~1.5GB). FSDP trainers' reserved memory
    fluctuates and typically grows past the pool (72GB+). Classify as sampler iff
    the nonzero-memory median sits within 3GB of the pool AND p95-p50 < 1.5GB.
    """
    trainers, samplers = [], []
    POOL = 65500
    for key, tr in all_gpus.items():
        mem = sorted(v for v in tr["mem"] if v > 1000)
        if not mem:
            trainers.append(key)
            continue
        n = len(mem)
        p50, p95 = mem[n // 2], mem[int(n * 0.95)]
        if abs(p50 - POOL) < 3000 and (p95 - p50) < 1500:
            samplers.append(key)
        else:
            trainers.append(key)
    return trainers, samplers


def group_series(all_gpus, keys):
    """Average util across a set of (node,gpu) keys onto the first key's timeline."""
    if not keys:
        return [], []
    base = all_gpus[keys[0]]
    ts = base["ts"]
    out = []
    idx = {k: 0 for k in keys}
    for i, t in enumerate(ts):
        vals = []
        for k in keys:
            tr = all_gpus[k]
            j = min(i, len(tr["ts"]) - 1)
            vals.append(tr["util"][j])
        out.append(sum(vals) / len(vals))
    return ts, out


def plot_run(run_dir: Path, out_dir: Path):
    all_gpus = {}
    for csv_name, node in [("gpu_util_head.csv", "head"), ("gpu_util_worker.csv", "worker")]:
        p = run_dir / csv_name
        if p.exists():
            for gi, tr in load(p).items():
                all_gpus[(node, gi)] = tr
    if not all_gpus:
        print(f"  no CSVs in {run_dir}")
        return

    trainers, samplers = classify_roles(all_gpus)
    print(f"  {run_dir.name}: trainers={sorted(trainers)} samplers={sorted(samplers)}")

    t_tr, u_tr = group_series(all_gpus, trainers)
    t_sa, u_sa = group_series(all_gpus, samplers)

    # SHARED absolute time base: common t0 across both roles, trimmed to the
    # training window (first→last nonzero activity) so monitor startup and the
    # post-training idle tail don't pad the axis or dilute the stats.
    nz_ts = [t for ts, us in [(t_tr, u_tr), (t_sa, u_sa)]
             for t, u in zip(ts, us) if u > 0]
    t0, tmax = min(nz_ts), max(nz_ts)
    tr_w = [(t, u) for t, u in zip(t_tr, u_tr) if t0 <= t <= tmax]
    sa_w = [(t, u) for t, u in zip(t_sa, u_sa) if t0 <= t <= tmax]
    x_tr = [(t - t0) / 1000 for t, _ in tr_w]
    u_tr = [u for _, u in tr_w]
    x_sa = [(t - t0) / 1000 for t, _ in sa_w]
    u_sa = [u for _, u in sa_w]
    xlim = (0, (tmax - t0) / 1000)

    def stats(u):
        nz = [v for v in u if v > 0]
        return (100 * len(nz) / len(u) if u else 0, sum(nz) / len(nz) if nz else 0)

    a_tr, m_tr = stats(u_tr)
    a_sa, m_sa = stats(u_sa)
    name = run_dir.name

    # ---- separate panels, identical x-axis ----------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(20, 8), sharex=True)
    axes[0].plot(x_tr, u_tr, color=TRAINER_C, lw=0.6)
    axes[0].fill_between(x_tr, 0, u_tr, color=TRAINER_C, alpha=0.25)
    axes[0].set_title(f"Trainer GPUs ({len(trainers)}, avg) — active {a_tr:.0f}%, {m_tr:.0f}% when active",
                      loc="left", fontweight="bold")
    axes[1].plot(x_sa, u_sa, color=SAMPLER_C, lw=0.6)
    axes[1].fill_between(x_sa, 0, u_sa, color=SAMPLER_C, alpha=0.25)
    axes[1].set_title(f"Sampler GPUs ({len(samplers)}, avg) — active {a_sa:.0f}%, {m_sa:.0f}% when active",
                      loc="left", fontweight="bold")
    for ax in axes:
        ax.set_ylabel("GPU util %")
        ax.set_xlim(*xlim)
        ax.set_ylim(-2, 105)
        ax.grid(alpha=0.25)
    axes[1].set_xlabel("Time (s) — shared axis, same t0 for both panels")
    fig.suptitle(f"{name} — trainer vs sampler, aligned time axis", fontweight="bold")
    plt.tight_layout()
    p1 = out_dir / f"{name}_separate.png"
    plt.savefig(p1, dpi=130, bbox_inches="tight")
    plt.close()

    # ---- overlay -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(20, 5.5))
    ax.plot(x_tr, u_tr, color=TRAINER_C, lw=0.7, label=f"Trainer ({len(trainers)} GPUs avg)")
    ax.fill_between(x_tr, 0, u_tr, color=TRAINER_C, alpha=0.18)
    ax.plot(x_sa, u_sa, color=SAMPLER_C, lw=0.7, label=f"Sampler ({len(samplers)} GPUs avg)")
    ax.fill_between(x_sa, 0, u_sa, color=SAMPLER_C, alpha=0.18)
    ax.set_xlim(*xlim)
    ax.set_ylim(-2, 105)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("GPU util %")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    ax.set_title(f"{name} — overlay (same axis)", fontweight="bold", loc="left")
    plt.tight_layout()
    p2 = out_dir / f"{name}_overlay.png"
    plt.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {p1.name}, {p2.name}")


def main():
    base = Path(sys.argv[1] if len(sys.argv) > 1 else
                "/Users/aishuk/workspaces/GPU-CR/async-rl-timeslicing/phase1_results")
    out = Path("/Users/aishuk/workspaces/GPU-CR/async-rl-timeslicing/plots")
    out.mkdir(exist_ok=True)
    for run_dir in sorted(base.iterdir()):
        if run_dir.is_dir():
            plot_run(run_dir, out)


if __name__ == "__main__":
    main()
