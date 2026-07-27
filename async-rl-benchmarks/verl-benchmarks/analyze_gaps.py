#!/usr/bin/env python3
"""
Idle-gap analysis for phase runs: histogram + recurrence view, table stats.

For each run dir under phase1_results/:
  - detect idle gaps (group-average util == 0, duration >= MIN_GAP_S) for
    trainer and sampler groups separately
  - parse weight-sync durations (timing_s/param_sync) from train.log
  - write plots/<run>_gaps.png:
      left  = histogram of gap durations (trainer red / sampler blue)
      right = gap start time vs duration scatter (recurrence over the run)
  - print a markdown table row with trainer gaps, sampler gaps, sync duration

Usage: python3 analyze_gaps.py [results_base]
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_phase1 import load, classify_roles, group_series, TRAINER_C, SAMPLER_C

MIN_GAP_S = 5.0


def find_gaps(ts, util, min_gap_s=MIN_GAP_S):
    """Return [(start_s_rel_t0_ms, duration_s)] of contiguous util==0 runs."""
    gaps = []
    start = None
    for i, (t, u) in enumerate(zip(ts, util)):
        if u == 0:
            if start is None:
                start = t
        else:
            if start is not None:
                dur = (t - start) / 1000
                if dur >= min_gap_s:
                    gaps.append((start, dur))
                start = None
    if start is not None:
        dur = (ts[-1] - start) / 1000
        if dur >= min_gap_s:
            gaps.append((start, dur))
    return gaps


def sync_durations(run_dir: Path):
    """All param_sync timings (s) from any train*.log in the run dir."""
    durs = []
    for logf in run_dir.glob("train*.log"):
        text = logf.read_text(errors="replace")
        durs += [float(x) for x in re.findall(r"timing_s/param_sync: ([0-9.]+) seconds", text)]
    return durs


def training_window(series_by_role):
    """[first nonzero ts, last nonzero ts] across roles — trims monitor
    startup (model download etc.) and the idle tail after training ends."""
    starts, ends = [], []
    for ts, util in series_by_role.values():
        nz = [t for t, u in zip(ts, util) if u > 0]
        if nz:
            starts.append(nz[0])
            ends.append(nz[-1])
    return min(starts), max(ends)


def analyze(run_dir: Path, out_dir: Path):
    all_gpus = {}
    for csv_name, node in [("gpu_util_head.csv", "head"), ("gpu_util_worker.csv", "worker")]:
        p = run_dir / csv_name
        if p.exists():
            for gi, tr in load(p).items():
                all_gpus[(node, gi)] = tr
    if not all_gpus:
        return None

    trainers, samplers = classify_roles(all_gpus)
    series = {role: group_series(all_gpus, keys)
              for role, keys in [("trainer", trainers), ("sampler", samplers)]}
    w0, w1 = training_window(series)

    results, active = {}, {}
    for role, (ts, util) in series.items():
        keep = [(t, u) for t, u in zip(ts, util) if w0 <= t <= w1]
        ts_w = [t for t, _ in keep]
        ut_w = [u for _, u in keep]
        gaps = find_gaps(ts_w, ut_w)
        results[role] = [((s - w0) / 1000, d) for s, d in gaps]
        nz = [u for u in ut_w if u > 0]
        active[role] = (100 * len(nz) / len(ut_w) if ut_w else 0,
                        sum(nz) / len(nz) if nz else 0)

    syncs = sync_durations(run_dir)
    name = run_dir.name
    dur_min = (w1 - w0) / 60000

    # ---- figure: histogram + recurrence scatter -----------------------------
    fig, (axh, axr) = plt.subplots(1, 2, figsize=(16, 5), width_ratios=[1, 1.6])
    bins = [5, 10, 20, 30, 45, 60, 80, 100, 130]
    for role, color in [("trainer", TRAINER_C), ("sampler", SAMPLER_C)]:
        durs = [d for _, d in results[role]]
        axh.hist(durs, bins=bins, alpha=0.55, color=color,
                 label=f"{role} ({len(durs)} gaps, {sum(durs):.0f}s total)",
                 edgecolor="white")
    axh.set_xlabel("Idle gap duration (s)")
    axh.set_ylabel("Number of gaps")
    axh.set_title("Gap duration distribution (gaps ≥5s)", loc="left", fontweight="bold")
    axh.legend()
    axh.grid(alpha=0.25)

    for role, color, marker in [("trainer", TRAINER_C, "o"), ("sampler", SAMPLER_C, "^")]:
        xs = [s / 60 for s, _ in results[role]]
        ys = [d for _, d in results[role]]
        axr.scatter(xs, ys, s=22, color=color, alpha=0.7, marker=marker, label=role)
    axr.set_xlabel("Gap start time (min into run)")
    axr.set_ylabel("Gap duration (s)")
    axr.set_title("When gaps occur — recurrence across the whole run",
                  loc="left", fontweight="bold")
    axr.legend()
    axr.grid(alpha=0.25)
    fig.suptitle(f"{name} — idle gaps ≥{MIN_GAP_S:.0f}s", fontweight="bold")
    plt.tight_layout()
    out = out_dir / f"{name}_gaps.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()

    def fmt(role):
        durs = sorted((d for _, d in results[role]), reverse=True)
        if not durs:
            return "none"
        top = "/".join(f"{d:.0f}s" for d in durs[:3])
        return f"{len(durs)} gaps, top {top}, total {sum(durs):.0f}s"

    sync_str = (f"{sum(syncs)/len(syncs):.1f}s avg ({min(syncs):.1f}-{max(syncs):.1f}, n={len(syncs)})"
                if syncs else "n/a")
    a_tr, m_tr = active["trainer"]
    a_sa, m_sa = active["sampler"]
    print(f"| {name} | {dur_min:.0f} min | {a_tr:.0f}% ({m_tr:.0f}% util) | {a_sa:.0f}% ({m_sa:.0f}% util) "
          f"| {fmt('trainer')} | {fmt('sampler')} | {sync_str} |")
    return results, syncs


def main():
    base = Path(sys.argv[1] if len(sys.argv) > 1 else
                "/Users/aishuk/workspaces/GPU-CR/async-rl-timeslicing/phase1_results")
    out = Path("/Users/aishuk/workspaces/GPU-CR/async-rl-timeslicing/plots")
    out.mkdir(exist_ok=True)
    print("| Run | Train window | Trainer active | Sampler active | Trainer idle gaps ≥5s | Sampler idle gaps ≥5s | Weight sync |")
    print("|-----|--------------|----------------|----------------|----------------------|----------------------|-------------|")
    for run_dir in sorted(base.iterdir()):
        if run_dir.is_dir():
            analyze(run_dir, out)


if __name__ == "__main__":
    main()
