#!/usr/bin/env python3
"""Staleness sweep charts: per-config GPU util timelines + idle duty-cycle comparison."""
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DIR = "/Users/aishuk/workspaces/GPU-CR/async-multiturn"
RES = f"{DIR}/results"

SURFACE = "#fcfcfb"
TEXT_1 = "#0b0b0b"
TEXT_2 = "#52514e"
TRAINER = "#2a78d6"   # categorical slot 1 (blue)
ROLLOUT = "#eb6834"   # categorical slot 2 (orange)
GRID = "#e5e4e0"

RUNS = [
    ("run_s0_final", "staleness = 0 (on-policy)"),
    ("run_s1", "staleness = 1"),
    ("run_s4", "staleness = 4"),
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
fig, axes = plt.subplots(len(RUNS), 1, figsize=(15, 12), sharex=False)
fig.patch.set_facecolor(SURFACE)
for (run, label), ax in zip(RUNS, axes):
    t, u = load(run)
    t0 = min(min(v) for v in t.values())
    ax.set_facecolor(SURFACE)
    for gi, color in ((0, TRAINER), (1, ROLLOUT)):
        xs, ys = smooth(t[gi], u[gi], t0)
        ax.plot(xs, ys, color=color, linewidth=1.4)
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0, right=max(max(smooth(t[gi], u[gi], t0)[0]) for gi in t))
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
    "veRL fully-async disagg — multi-turn search agent (HotpotQA, Qwen2.5-3B, 1 trainer + 1 rollout H100)\n"
    "GPU utilization vs staleness_threshold — 5s bins",
    color=TEXT_1, fontsize=12, y=0.995,
)
fig.legend(
    handles=[mpatches.Patch(color=TRAINER, label="GPU 0 — trainer (FSDP)"),
             mpatches.Patch(color=ROLLOUT, label="GPU 1 — rollout (vLLM)")],
    loc="upper right", fontsize=10, frameon=False, bbox_to_anchor=(0.99, 0.985),
)
plt.tight_layout(rect=(0, 0, 1, 0.955))
plt.savefig(f"{DIR}/sweep_timelines.png", dpi=110, facecolor=SURFACE)

# ---------- Chart 2: idle duty-cycle comparison ----------
rows = list(csv.DictReader(open(f"{RES}/sweep_summary.csv")))
labels = ["s=0\n(on-policy)", "s=1", "s=4", "s=8", "s=∞\n(unbounded)"]
g0 = [float(r["gpu0_idle_pct"]) for r in rows]
g1 = [float(r["gpu1_idle_pct"]) for r in rows]

fig2, ax = plt.subplots(figsize=(11, 5.5))
fig2.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
x = range(len(rows))
w = 0.38
b0 = ax.bar([i - w / 2 for i in x], g0, w, color=TRAINER, label="GPU 0 — trainer (FSDP)")
b1 = ax.bar([i + w / 2 for i in x], g1, w, color=ROLLOUT, label="GPU 1 — rollout (vLLM)")
for bars in (b0, b1):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.0f}%",
                ha="center", color=TEXT_1, fontsize=10)
ax.set_xticks(list(x))
ax.set_xticklabels(labels, color=TEXT_1, fontsize=10)
ax.set_ylabel("% of time idle (<10% util)", color=TEXT_2, fontsize=10)
ax.set_ylim(0, 80)
ax.set_title("Harvestable idle time per GPU vs staleness — same workload, no throttling",
             color=TEXT_1, fontsize=12, loc="left")
ax.grid(axis="y", color=GRID, linewidth=0.7)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=TEXT_2, labelsize=9)
ax.legend(frameon=False, fontsize=10, loc="upper right")
plt.tight_layout()
plt.savefig(f"{DIR}/sweep_idle_comparison.png", dpi=110, facecolor=SURFACE)
print("wrote sweep_timelines.png, sweep_idle_comparison.png")
