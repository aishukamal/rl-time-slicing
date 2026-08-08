#!/usr/bin/env python3
"""M3 RUN A analyzer: colocated sleep/wake arm (stock verl sync hybrid).

Parses train.log step lines (step:N - key:value ...) and gpu_util.csv.
Outputs steady-state step time, phase decomposition (gen / old_log_prob /
update_actor / update_weights / reward), rewards, response lengths, per-GPU
utilization, and tokens/GPU-hour vs the reference ladder.
"""

import csv
import json
import re
import sys
from collections import defaultdict

TRAIN_LOG = sys.argv[1] if len(sys.argv) > 1 else "train.log"
GPU_CSV = sys.argv[2] if len(sys.argv) > 2 else "gpu_util.csv"
SKIP_FIRST = 2  # steady window convention: skip first 2 logged steps

KV_RE = re.compile(r"([\w/@.-]+):([-+eE\d.]+)")


def parse_steps(path):
    steps = []
    with open(path, errors="replace") as f:
        for line in f:
            if not re.search(r"step:\d+ - ", line):
                continue
            kv = dict()
            for m in KV_RE.finditer(line):
                try:
                    kv[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
            if "step" in kv and "timing_s/step" in kv:
                steps.append(kv)
    return steps


def stat(xs):
    if not xs:
        return None
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n if n > 1 else 0.0
    return {"n": n, "mean": round(mean, 1), "min": round(min(xs), 1),
            "max": round(max(xs), 1), "stdev": round(var ** 0.5, 1)}


def main():
    steps = parse_steps(TRAIN_LOG)
    if not steps:
        print(json.dumps({"error": "no step lines parsed"}))
        return
    steady = steps[SKIP_FIRST:] if len(steps) > SKIP_FIRST + 2 else steps

    def col(key):
        return [s[key] for s in steady if key in s]

    out = {
        "steps_logged": [int(s["step"]) for s in steps],
        "steady_window": [int(s["step"]) for s in steady],
        "step_time": stat(col("timing_s/step")),
        "gen": stat(col("timing_s/gen")),
        "old_log_prob": stat(col("timing_s/old_log_prob")),
        "update_actor": stat(col("timing_s/update_actor")),
        "update_weights": stat(col("timing_s/update_weights")),
        "reward": stat(col("timing_s/reward")),
        "adv": stat(col("timing_s/adv")),
        "score_mean_per_step": [round(s.get("critic/score/mean", float("nan")), 3) for s in steady],
        "response_length_mean": stat(col("response_length/mean")),
        "timing_keys_seen": sorted({k for s in steps for k in s if k.startswith("timing_s/")}),
    }

    # tokens/GPU-hour: response tokens produced per step / (step_time * n_gpus)
    if out["response_length_mean"] and out["step_time"]:
        toks_per_step = out["response_length_mean"]["mean"] * 64  # 64 trajectories/step
        out["resp_tokens_per_gpu_hour"] = round(
            toks_per_step / (out["step_time"]["mean"] * 2) * 3600, 0)

    try:
        traces = defaultdict(list)
        with open(GPU_CSV) as f:
            for row in csv.DictReader(f):
                traces[int(row["gpu_index"])].append(float(row["gpu_util_pct"]))
        out["gpu"] = {}
        for gi in sorted(traces):
            u = traces[gi]
            out["gpu"][gi] = {
                "samples": len(u),
                "mean_util": round(sum(u) / len(u), 1),
                "idle_lt10": round(sum(1 for v in u if v < 10) / len(u), 3),
                "busy_ge50": round(sum(1 for v in u if v >= 50) / len(u), 3),
            }
    except OSError as e:
        out["gpu"] = f"unavailable: {e}"

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
