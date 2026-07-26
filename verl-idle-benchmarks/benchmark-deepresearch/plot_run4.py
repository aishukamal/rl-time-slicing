#!/usr/bin/env python3
"""Plot run-4 colocated multi-turn GPU trace: util timeline + phase structure."""
import csv
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = "/Users/aishuk/workspaces/GPU-CR/benchmark-deepresearch"

# --- load trace ---
t = defaultdict(list)
u = defaultdict(list)
with open(f"{DIR}/gpu_util_run4_final.csv") as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            gi = int(row["gpu_index"])
            t[gi].append(int(row["timestamp_ms"]))
            u[gi].append(float(row["gpu_util_pct"]))
        except (ValueError, KeyError):
            continue

t0 = min(min(v) for v in t.values())

def smooth(ts, us, win_s=10.0):
    """Bucket into win_s-second bins, mean per bin."""
    buckets = defaultdict(list)
    for ms, util in zip(ts, us):
        buckets[int((ms - t0) / 1000 / win_s)].append(util)
    xs = sorted(buckets)
    return [x * win_s / 60 for x in xs], [sum(buckets[x]) / len(buckets[x]) for x in xs]

# --- parse step boundaries from training log ---
# gen/train phase durations per step from the metric lines
gens, updates, steps_total = [], [], []
with open(f"{DIR}/train_run4_final.log", errors="replace") as f:
    log = f.read()
for m in re.finditer(r"timing_s/gen:([\d.]+).*?timing_s/update_actor:([\d.]+).*?timing_s/step:([\d.]+)", log):
    gens.append(float(m.group(1)))
    updates.append(float(m.group(2)))
    steps_total.append(float(m.group(3)))

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
for gi, ax in zip(sorted(t), axes):
    xs, ys = smooth(t[gi], u[gi])
    ax.fill_between(xs, ys, color="tab:blue" if gi == 0 else "tab:orange", alpha=0.6)
    ax.set_ylabel(f"GPU {gi} util %")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
axes[0].set_title(
    "Run 4 — Multi-turn deep-research RL, colocated 2xH100 TP=2 (16 GRPO steps, ~%d min)\n"
    "gen phase %.0f-%.0fs (multi-turn rollout, long-tail), train phase ~%.0fs steady"
    % ((max(max(v) for v in t.values()) - t0) / 60000, min(gens), max(gens), sum(updates) / len(updates) + 34)
)
axes[1].set_xlabel("minutes")
plt.tight_layout()
plt.savefig(f"{DIR}/run4_gpu_timeline.png", dpi=110)

# --- phase structure bar chart ---
fig2, ax = plt.subplots(figsize=(14, 4))
left = 0.0
for i, (g, up, st) in enumerate(zip(gens, updates, steps_total)):
    other = st - g - up  # logprob+ref+adv+reward (+validation on steps 4/8/12/16)
    ax.barh(0, g / 60, left=left / 60, color="tab:orange", edgecolor="white")
    left += g
    ax.barh(0, (up + other) / 60, left=left / 60, color="tab:blue", edgecolor="white")
    left += up + other
ax.barh(1, 0, color="tab:orange")
ax.set_yticks([])
ax.set_xlabel("minutes")
ax.set_title("Per-step phase structure: generation (orange) vs training+logprob/ref/val (blue)")
import matplotlib.patches as mpatches
ax.legend(handles=[mpatches.Patch(color="tab:orange", label="multi-turn generation (trainer would idle in disagg)"),
                   mpatches.Patch(color="tab:blue", label="training / logprob / ref / validation")],
          loc="upper right")
plt.tight_layout()
plt.savefig(f"{DIR}/run4_phase_structure.png", dpi=110)

n = len(gens)
gen_tot, tr_tot = sum(gens), sum(s - g for s, g in zip(steps_total, gens))
print(f"steps={n} gen: min={min(gens):.0f}s max={max(gens):.0f}s mean={gen_tot/n:.0f}s")
print(f"train-side per step mean={tr_tot/n:.0f}s (update_actor mean={sum(updates)/len(updates):.0f}s)")
print(f"gen fraction of wall time: {gen_tot/(gen_tot+tr_tot):.1%}")
print("wrote run4_gpu_timeline.png, run4_phase_structure.png")
