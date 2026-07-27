#!/usr/bin/env python3
"""
Plot overlaid trainer vs rollouter GPU duty cycle from raw nvidia-smi CSVs.
Handles both single-node (4_4: GPUs 0-3=trainer, 4-7=rollouter on one CSV)
and multi-node (8_8/4_12: head CSV + worker CSV) recipes.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_trace(csv_path):
    traces = defaultdict(lambda: {"ts": [], "util": []})
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            idx = int(row["gpu_index"])
            traces[idx]["ts"].append(int(row["timestamp_ms"]))
            traces[idx]["util"].append(float(row["gpu_util_pct"]))
    return dict(traces)


def rolling_duty_cycle(timestamps_ms, util_pct, window_s=3.0):
    if not timestamps_ms:
        return [], []
    t0 = timestamps_ms[0]
    window_ms = int(window_s * 1000)
    times, duty = [], []
    for i in range(len(timestamps_ms)):
        center = timestamps_ms[i]
        start, end = center - window_ms // 2, center + window_ms // 2
        window_vals = [util_pct[j] for j in range(len(timestamps_ms))
                       if start <= timestamps_ms[j] <= end]
        if window_vals:
            # duty cycle = fraction of window with ANY kernel running
            dc = sum(1 for v in window_vals if v > 0) / len(window_vals) * 100
            times.append((center - t0) / 1000.0)
            duty.append(dc)
    return times, duty


def aggregate_gpu_group(traces, gpu_indices, window_s=3.0):
    """Average the duty cycle across a group of GPUs."""
    all_series = []
    for gi in gpu_indices:
        if gi in traces:
            t, d = rolling_duty_cycle(traces[gi]["ts"], traces[gi]["util"], window_s)
            if t:
                all_series.append((t, d))
    if not all_series:
        return [], []
    # Use the first series' timestamps, average duty cycles
    ref_t = all_series[0][0]
    avg_d = []
    for i in range(len(ref_t)):
        vals = [s[1][i] for s in all_series if i < len(s[1])]
        avg_d.append(sum(vals) / len(vals) if vals else 0)
    return ref_t, avg_d


def plot_recipe(recipe_name, modes, results_base, out_dir, window_s=3.0):
    fig, axes = plt.subplots(len(modes), 1, figsize=(18, 4 * len(modes)), squeeze=False)

    mode_labels = {
        "mode1_on_policy": "Mode 1: On-Policy Pipeline",
        "mode2_stream_offpolicy": "Mode 2: Stream Off-Policy",
        "mode3_async_stale": "Mode 3: Async + Stale Samples",
        "mode4_async_partial": "Mode 4: Async + Partial Rollout",
    }

    recipe_gpu_config = {
        "dapo_7b_4_4": {"type": "single_node", "trainer_gpus": [0,1,2,3], "rollouter_gpus": [4,5,6,7]},
        "dapo_7b_8_8": {"type": "multi_node"},
        "dapo_7b_4_12": {"type": "multi_node"},
    }

    config = recipe_gpu_config.get(recipe_name, {"type": "multi_node"})
    plotted_any = False

    for i, mode in enumerate(modes):
        ax = axes[i][0]
        run_name = f"{recipe_name}_{mode}"
        run_dir = results_base / run_name

        if not run_dir.exists():
            ax.set_title(f"{mode_labels.get(mode, mode)} — NO DATA", fontsize=10, loc="left")
            ax.text(0.5, 0.5, "Run missing", transform=ax.transAxes, ha="center", fontsize=14, color="gray")
            continue

        if config["type"] == "single_node":
            head_csv = run_dir / "gpu_util_head.csv"
            if not head_csv.exists():
                ax.set_title(f"{mode_labels.get(mode, mode)} — NO CSV", fontsize=10, loc="left")
                continue
            traces = load_trace(str(head_csv))
            t_train, d_train = aggregate_gpu_group(traces, config["trainer_gpus"], window_s)
            t_roll, d_roll = aggregate_gpu_group(traces, config["rollouter_gpus"], window_s)
        else:
            head_csv = run_dir / "gpu_util_head.csv"
            worker_csv = run_dir / "gpu_util_worker.csv"
            t_train, d_train, t_roll, d_roll = [], [], [], []

            # Head node — could be either trainer or rollouter depending on Ray placement
            # Worker node — the other
            if head_csv.exists():
                traces = load_trace(str(head_csv))
                all_gpus = sorted(traces.keys())
                t_h, d_h = aggregate_gpu_group(traces, all_gpus, window_s)

            if worker_csv.exists():
                traces_w = load_trace(str(worker_csv))
                all_gpus_w = sorted(traces_w.keys())
                t_w, d_w = aggregate_gpu_group(traces_w, all_gpus_w, window_s)
            else:
                t_w, d_w = [], []

            # Assign: whichever node has FEWER nonzero samples is likely the trainer
            # (trainer has lower duty cycle due to waiting for samples)
            if head_csv.exists() and t_w:
                head_nz = sum(1 for v in d_h if v > 0)
                worker_nz = sum(1 for v in d_w if v > 0)
                if head_nz <= worker_nz:
                    t_train, d_train = t_h, d_h
                    t_roll, d_roll = t_w, d_w
                else:
                    t_train, d_train = t_w, d_w
                    t_roll, d_roll = t_h, d_h
            elif head_csv.exists():
                # Only head data — label as "head" generically
                t_train, d_train = t_h, d_h

        # Plot
        if t_train:
            ax.plot(t_train, d_train, color="#E53935", alpha=0.85, linewidth=1.2, label="Trainer")
            ax.fill_between(t_train, 0, d_train, color="#E53935", alpha=0.15)
            plotted_any = True
        if t_roll:
            ax.plot(t_roll, d_roll, color="#1E88E5", alpha=0.85, linewidth=1.2, label="Rollouter")
            ax.fill_between(t_roll, 0, d_roll, color="#1E88E5", alpha=0.15)

        # Stats annotation
        if t_train and d_train:
            nz_train = sum(1 for v in d_train if v > 0) / len(d_train) * 100 if d_train else 0
            nz_roll = sum(1 for v in d_roll if v > 0) / len(d_roll) * 100 if d_roll else 0
            stats = f"trainer active: {nz_train:.0f}%"
            if d_roll:
                stats += f"  |  rollouter active: {nz_roll:.0f}%"
            ax.text(0.99, 0.95, stats, transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color="gray", fontstyle="italic")

        label = mode_labels.get(mode, mode)
        ax.set_title(label, fontsize=11, loc="left", fontweight="bold")
        ax.set_ylabel("Duty Cycle %", fontsize=9)
        ax.set_ylim(-2, 105)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(loc="upper right", fontsize=10)

    axes[-1][0].set_xlabel("Time (seconds)", fontsize=11)

    gpu_desc = {"dapo_7b_4_4": "8 GPU (4T+4R, 1 node)",
                "dapo_7b_8_8": "16 GPU (8T+8R, 2 nodes)",
                "dapo_7b_4_12": "16 GPU (12T+4R, 2 nodes)"}
    fig.suptitle(f"GPU Duty Cycle: Trainer vs Rollouter — {recipe_name}\n"
                 f"{gpu_desc.get(recipe_name, '')}  |  Qwen2.5-7B  |  {window_s:.0f}s window",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = out_dir / f"overlay_{recipe_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()
    return plotted_any


def main():
    results_base = Path("/Users/aishuk/workspaces/GPU-CR/async-rl-timeslicing/full_results")
    out_dir = Path("/Users/aishuk/workspaces/GPU-CR/async-rl-timeslicing/plots")
    out_dir.mkdir(exist_ok=True)

    modes = ["mode1_on_policy", "mode2_stream_offpolicy", "mode3_async_stale", "mode4_async_partial"]

    for recipe in ["dapo_7b_4_4", "dapo_7b_8_8", "dapo_7b_4_12"]:
        print(f"\n=== {recipe} ===")
        plot_recipe(recipe, modes, results_base, out_dir)


if __name__ == "__main__":
    main()
