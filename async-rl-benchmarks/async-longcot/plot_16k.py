#!/usr/bin/env python3
"""16K validation run: GPU utilization timeline + duty-cycle summary."""
import csv
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DIR = "/Users/aishuk/workspaces/GPU-CR/async-longcot"

SURFACE = "#fcfcfb"
TEXT_1 = "#0b0b0b"
TEXT_2 = "#52514e"
TRAINER = "#2a78d6"
ROLLOUT = "#eb6834"
GRID = "#e5e4e0"

t, u = defaultdict(list), defaultdict(list)
with open(f"{DIR}/results/run_s8_16k_gpu_util.csv") as f:
    for row in csv.DictReader(f):
        try:
            gi = int(row["gpu_index"])
            t[gi].append(int(row["timestamp_ms"]))
            u[gi].append(float(row["gpu_util_pct"]))
        except (ValueError, KeyError):
            continue
t0 = min(min(v) for v in t.values())

def smooth(gi, win_s=5.0):
    b = defaultdict(list)
    for ms, util in zip(t[gi], u[gi]):
        b[int((ms - t0) / 1000 / win_s)].append(util)
    xs = sorted(b)
    return [x * win_s / 60 for x in xs], [sum(b[x]) / len(b[x]) for x in xs]

s = json.load(open(f"{DIR}/results/run_s8_16k_summary.json"))

fig, axes = plt.subplots(2, 1, figsize=(15, 6.5), sharex=True)
fig.patch.set_facecolor(SURFACE)
labels = {
    0: ("GPU 0 — trainer (FSDP): 59.6% idle, blocks mean 201s / max 286s, one per step", TRAINER),
    1: ("GPU 1 — rollout (vLLM): 99.5% util, never pauses", ROLLOUT),
}
for gi, ax in zip((0, 1), axes):
    xs, ys = smooth(gi)
    label, color = labels[gi]
    ax.set_facecolor(SURFACE)
    ax.fill_between(xs, ys, color=color, alpha=0.65, linewidth=0)
    ax.plot(xs, ys, color=color, linewidth=1.2)
    ax.set_ylim(0, 105)
    ax.set_xlim(0, max(xs))
    ax.set_ylabel("util %", color=TEXT_2, fontsize=9)
    ax.set_title(label, loc="left", fontsize=11, color=TEXT_1)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)
axes[1].set_xlabel("minutes", color=TEXT_2, fontsize=10)
fig.suptitle(
    "16K validation — long-CoT math RL, staleness=8, max_response=16384 (R1-Distill-1.5B, 1+1 H100)\n"
    "step cadence 407s = gen ~210s wait + train 185s · trainer idles in 3.3-4.8 min contiguous blocks",
    color=TEXT_1, fontsize=12, y=0.99,
)
plt.tight_layout(rect=(0, 0, 1, 0.92))
plt.savefig(f"{DIR}/run_s8_16k_timeline.png", dpi=110, facecolor=SURFACE)
print("wrote run_s8_16k_timeline.png")
