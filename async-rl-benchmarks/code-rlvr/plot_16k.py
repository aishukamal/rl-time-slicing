#!/usr/bin/env python3
"""16K validation run: GPU utilization timeline, same format as the sweep timelines."""
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DIR = "/Users/aishuk/workspaces/GPU-CR/code-rlvr"

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

fig, ax = plt.subplots(1, 1, figsize=(15, 4.2))
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
xmax = 0
for gi, color in ((0, TRAINER), (1, ROLLOUT)):
    xs, ys = smooth(gi)
    ax.plot(xs, ys, color=color, linewidth=1.4)
    xmax = max(xmax, max(xs))
ax.set_ylim(0, 105)
ax.set_xlim(0, xmax)
ax.set_ylabel("util %", color=TEXT_2, fontsize=9)
ax.set_title("staleness = 8, max_response_length = 16384", loc="left", fontsize=11, color=TEXT_1)
ax.grid(axis="y", color=GRID, linewidth=0.7)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(GRID)
ax.tick_params(colors=TEXT_2, labelsize=8)
ax.set_xlabel("minutes", color=TEXT_2, fontsize=10)
fig.suptitle(
    "veRL fully-async disagg — long-CoT code RLVR at 16K response cap (Eurus-2 code, live test-execution rewards) (R1-Distill-Qwen-1.5B, 1 trainer + 1 rollout H100)\n"
    "GPU utilization — 5s bins · trainer 52.8% idle in ~4-6.7 min blocks · rollout 99.6% util",
    color=TEXT_1, fontsize=12, y=0.99,
)
fig.legend(
    handles=[mpatches.Patch(color=TRAINER, label="GPU 0 — trainer (FSDP)"),
             mpatches.Patch(color=ROLLOUT, label="GPU 1 — rollout (vLLM)")],
    loc="upper right", fontsize=10, frameon=False, bbox_to_anchor=(0.99, 0.97),
)
plt.tight_layout(rect=(0, 0, 1, 0.86))
plt.savefig(f"{DIR}/run_s8_16k_timeline.png", dpi=110, facecolor=SURFACE)
print("wrote run_s8_16k_timeline.png")
