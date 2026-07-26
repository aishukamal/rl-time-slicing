#!/usr/bin/env python3
"""Long-CoT sweep charts: timelines, idle comparison, and combined regime map."""
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DIR = "/Users/aishuk/workspaces/GPU-CR/async-longcot"
RES = f"{DIR}/results"

SURFACE = "#fcfcfb"
TEXT_1 = "#0b0b0b"
TEXT_2 = "#52514e"
TRAINER = "#2a78d6"   # categorical slot 1 (blue)
ROLLOUT = "#eb6834"   # categorical slot 2 (orange)
GRID = "#e5e4e0"

RUNS = [
    ("run_s0", "staleness = 0 (on-policy)"),
    ("run_s8", "staleness = 8"),
    ("run_sinf", "staleness = ∞ (unbounded, 10000)"),
]

def load(run):
    t, u = defaultdict(list), defaultdict(list)
    with open(f"{RES}/{run}_gpu_util.csv") as f:
        for row in csv.DictReader(f):
            try:
                gi = int(row["gpu_index"])
                t[gi].append(int(row["timestamp_ms"]))
                u[gi].append(float(row["gpu_util_pct"]))
            except (ValueError, KeyError):
                continue
    return t, u

def smooth(ts, us, t0, win_s=5.0):
    b = defaultdict(list)
    for ms, util in zip(ts, us):
        b[int((ms - t0) / 1000 / win_s)].append(util)
    xs = sorted(b)
    return [x * win_s / 60 for x in xs], [sum(b[x]) / len(b[x]) for x in xs]

# ---------- Chart 1: small-multiple timelines ----------
fig, axes = plt.subplots(len(RUNS), 1, figsize=(15, 8))
fig.patch.set_facecolor(SURFACE)
for (run, label), ax in zip(RUNS, axes):
    t, u = load(run)
    t0 = min(min(v) for v in t.values())
    ax.set_facecolor(SURFACE)
    xmax = 0
    for gi, color in ((0, TRAINER), (1, ROLLOUT)):
        xs, ys = smooth(t[gi], u[gi], t0)
        ax.plot(xs, ys, color=color, linewidth=1.4)
        xmax = max(xmax, max(xs))
    ax.set_ylim(0, 105)
    ax.set_xlim(0, xmax)
    ax.set_ylabel("util %", color=TEXT_2, fontsize=9)
    ax.set_title(label, loc="left", fontsize=11, color=TEXT_1)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)
axes[-1].set_xlabel("minutes", color=TEXT_2, fontsize=10)
fig.suptitle(
    "veRL fully-async disagg — long-CoT math RL (DAPO-Math-17k, R1-Distill-Qwen-1.5B, ~6K-token responses)\n"
    "1 trainer + 1 rollout H100 — GPU utilization vs staleness_threshold — 5s bins",
    color=TEXT_1, fontsize=12, y=0.995,
)
fig.legend(
    handles=[mpatches.Patch(color=TRAINER, label="GPU 0 — trainer (FSDP)"),
             mpatches.Patch(color=ROLLOUT, label="GPU 1 — rollout (vLLM)")],
    loc="upper right", fontsize=10, frameon=False, bbox_to_anchor=(0.99, 0.98),
)
plt.tight_layout(rect=(0, 0, 1, 0.93))
plt.savefig(f"{DIR}/longcot_timelines.png", dpi=110, facecolor=SURFACE)

# ---------- Chart 2: combined regime map (both sweeps) ----------
# HotpotQA (train-heavy) from async-multiturn sweep + long-CoT (gen-heavy)
hp = {r["run"]: r for r in csv.DictReader(open("/Users/aishuk/workspaces/GPU-CR/async-multiturn/results/sweep_summary.csv"))}
lc = {r["config"]: r for r in csv.DictReader(open(f"{RES}/sweep_summary.csv"))}

groups = [
    ("Train-heavy workload\n(HotpotQA multi-turn, gen 30s / train 70s)",
     [("s=0", float(hp["s0"]["gpu0_idle_pct"]), float(hp["s0"]["gpu1_idle_pct"])),
      ("s=8", float(hp["s8"]["gpu0_idle_pct"]), float(hp["s8"]["gpu1_idle_pct"])),
      ("s=∞", float(hp["sinf"]["gpu0_idle_pct"]), float(hp["sinf"]["gpu1_idle_pct"]))]),
    ("Gen-heavy workload\n(long-CoT math, gen 210s / train 125s)",
     [("s=0", float(lc["s0"]["tr_idle_pct"]), float(lc["s0"]["ro_idle_pct"])),
      ("s=8", float(lc["s8"]["tr_idle_pct"]), float(lc["s8"]["ro_idle_pct"])),
      ("s=∞", float(lc["sinf"]["tr_idle_pct"]), float(lc["sinf"]["ro_idle_pct"]))]),
]

fig2, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
fig2.patch.set_facecolor(SURFACE)
for ax, (gtitle, rows) in zip(axs, groups):
    ax.set_facecolor(SURFACE)
    x = range(len(rows))
    w = 0.38
    b0 = ax.bar([i - w / 2 for i in x], [r[1] for r in rows], w, color=TRAINER)
    b1 = ax.bar([i + w / 2 for i in x], [r[2] for r in rows], w, color=ROLLOUT)
    for bars in (b0, b1):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                    f"{b.get_height():.0f}%", ha="center", color=TEXT_1, fontsize=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels([r[0] for r in rows], color=TEXT_1, fontsize=10)
    ax.set_title(gtitle, color=TEXT_1, fontsize=10.5, loc="left")
    ax.set_ylim(0, 82)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
axs[0].set_ylabel("% of time idle (<10% util)", color=TEXT_2, fontsize=10)
fig2.suptitle("Where the harvestable idle lives: workload balance × staleness policy",
              color=TEXT_1, fontsize=13, y=0.99)
fig2.legend(
    handles=[mpatches.Patch(color=TRAINER, label="trainer GPU"),
             mpatches.Patch(color=ROLLOUT, label="rollout GPU")],
    loc="upper right", fontsize=10, frameon=False, bbox_to_anchor=(0.99, 0.94),
)
plt.tight_layout(rect=(0, 0, 1, 0.9))
plt.savefig(f"{DIR}/regime_map.png", dpi=110, facecolor=SURFACE)
print("wrote longcot_timelines.png, regime_map.png")
