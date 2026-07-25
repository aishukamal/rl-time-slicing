#!/usr/bin/env python3
"""Extract demo dashboard data from real run logs.

Outputs two JSON files:
  verl_demo_data.json  - from the veRL PoC exports (baseline + timeslice)
  slime_demo_data.json - from the Slime T1 validation run orchestrator log
"""
import csv
import json
import statistics as st
from collections import defaultdict
from datetime import datetime, timezone

VERL_TS = "/Users/aishuk/Downloads/e2e_rl/verl/rl_logs_export_timeslice_20260427_092932/rl_logs_export_timeslice_20260427_092932"
VERL_BL = "/Users/aishuk/Downloads/e2e_rl/verl/rl_logs_export_baseline_20260427_173147/rl_logs_export_baseline_20260427_173147"
SLIME_LOG = "/Users/aishuk/workspaces/GPU-CR/rl-timeslicing-validation/logs/t1/orchestrator.log"
OUT = "/Users/aishuk/workspaces/GPU-CR/timeslice-demos"


def downsample_duty(path, bucket_sec=60):
    """Average util per GPU into time buckets; returns rel-time series."""
    rows = list(csv.DictReader(open(path)))
    t0 = float(rows[0]["ts"])
    buckets = defaultdict(list)
    for r in rows:
        rel = float(r["ts"]) - t0
        buckets[int(rel // bucket_sec)].append(float(r["util_pct"]))
    series = [
        {"t": k * bucket_sec, "util": round(st.mean(v), 1)}
        for k, v in sorted(buckets.items())
    ]
    return series


def duty_pct(path, busy_threshold=5):
    rows = list(csv.DictReader(open(path)))
    by_gpu = defaultdict(list)
    for r in rows:
        by_gpu[r["gpu"]].append(float(r["util_pct"]))
    per_gpu = {g: sum(1 for u in v if u > busy_threshold) / len(v) * 100 for g, v in by_gpu.items()}
    return round(st.mean(per_gpu.values()), 1)


def verl():
    # --- lock events -> alternation timeline ---
    events = []
    t0 = None
    for line in open(f"{VERL_TS}/rl_metrics.jsonl"):
        e = json.loads(line)
        if t0 is None:
            t0 = e["ts"]
        if e["type"] in ("acquire", "yield"):
            job = "job-a" if "job-a" in e["workload_id"] else "job-b"
            events.append({
                "t": round(e["ts"] - t0, 1),
                "type": e["type"],
                "job": job,
                "pool": e["pool"],
                "wait_ms": e.get("wait_ms"),
                "restore_ms": e.get("restore_ms"),
                "evict_ms": e.get("evict_ms"),
            })

    # --- swap latency stats ---
    evict = [e["evict_ms"] for e in events if e["type"] == "yield"]
    restore = [e["restore_ms"] for e in events if e["type"] == "acquire"]
    waits = [e["wait_ms"] for e in events if e["type"] == "acquire"]

    # --- lock hold intervals per job+pool (acquire -> next yield) ---
    intervals = []
    open_acq = {}
    for e in events:
        key = (e["job"], e["pool"])
        if e["type"] == "acquire":
            open_acq[key] = e["t"]
        elif e["type"] == "yield" and key in open_acq:
            intervals.append({
                "job": e["job"], "pool": e["pool"],
                "start": open_acq.pop(key), "end": e["t"],
            })

    # --- rewards ---
    rewards = {}
    for j in ("a", "b"):
        rewards[f"job-{j}"] = [
            {"step": m["step"], "reward": round(m["mean_reward"], 4)}
            for m in map(json.loads, open(f"{VERL_TS}/metrics_job-{j}.jsonl"))
        ]
    # baseline single job reward for convergence comparison
    rewards["baseline"] = [
        {"step": m["step"], "reward": round(m["mean_reward"], 4)}
        for m in map(json.loads, open(f"{VERL_BL}/metrics_job-a.jsonl"))
    ]

    data = {
        "meta": {
            "framework": "veRL (GRPO)",
            "model": "Qwen2.5-0.5B-Instruct on GSM8K",
            "hardware": "2x NVIDIA H100 80GB (GKE a3-highgpu-2g)",
            "source": "rl_logs_export_{baseline,timeslice}_20260427",
        },
        "kpis": {
            "duty_baseline_pct": duty_pct(f"{VERL_BL}/gpu_duty_cycle.csv"),
            "duty_timeslice_pct": duty_pct(f"{VERL_TS}/gpu_duty_cycle.csv"),
            "median_restore_ms": round(st.median(restore)),
            "median_evict_ms": round(st.median(evict)),
            "lock_cycles": sum(1 for e in events if e["type"] == "acquire"),
        },
        "duty_baseline": downsample_duty(f"{VERL_BL}/gpu_duty_cycle.csv"),
        "duty_timeslice": downsample_duty(f"{VERL_TS}/gpu_duty_cycle.csv"),
        "intervals": intervals,
        "rewards": rewards,
        "swap": {"evict_ms": sorted(evict), "restore_ms": sorted(restore), "wait_ms": sorted(waits)},
    }
    json.dump(data, open(f"{OUT}/verl_demo_data.json", "w"), indent=1)
    print("verl:", json.dumps(data["kpis"]))
    print(f"  intervals={len(intervals)} reward_steps a={len(rewards['job-a'])} b={len(rewards['job-b'])}")


def slime():
    acquires, yields, snaps, restores = [], [], [], []
    t0 = None
    for line in open(SLIME_LOG):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = datetime.fromisoformat(e["time"].replace("Z", "+00:00")).timestamp()
        if t0 is None and e.get("msg") == "Acquire succeeded, job loaded and lock held":
            t0 = ts
        msg = e.get("msg", "")
        if msg == "Acquire succeeded, job loaded and lock held":
            acquires.append({"t": ts, "job": e["JobID"], "pool": e["GroupID"]})
        elif msg == "Yield called":
            yields.append({"t": ts, "job": e["JobID"], "pool": e["GroupID"]})
        elif msg == "Triggering snapshot for job":
            snaps.append({"t": ts, "job": e["jobID"], "pool": e["GroupID"]})
        elif msg == "Triggering restore for active job":
            restores.append({"t": ts, "job": e["jobID"], "pool": e["GroupID"]})

    if t0 is None:
        raise SystemExit("no acquire events found")

    def rel(evts):
        return [{**e, "t": round(e["t"] - t0, 1)} for e in evts if e["t"] - t0 > -60]

    acquires, yields, snaps, restores = map(rel, (acquires, yields, snaps, restores))

    # hold intervals: acquire -> matching yield per (job, pool)
    intervals = []
    open_acq = {}
    evs = sorted(
        [{**a, "type": "acquire"} for a in acquires] + [{**y, "type": "yield"} for y in yields],
        key=lambda e: e["t"],
    )
    for e in evs:
        key = (e["job"], e["pool"])
        if e["type"] == "acquire":
            open_acq[key] = e["t"]
        elif key in open_acq:
            intervals.append({"job": e["job"], "pool": e["pool"], "start": open_acq.pop(key), "end": e["t"]})
    # close any dangling holds at last event time
    t_end = max(e["t"] for e in evs)
    for (job, pool), start in open_acq.items():
        intervals.append({"job": job, "pool": pool, "start": start, "end": t_end})

    data = {
        "meta": {
            "framework": "Slime (sync GRPO)",
            "model": "Qwen2.5-0.5B on 2x H100 (dual-group: trainers + samplers)",
            "hardware": "GKE verl-research-cluster, 1 node, 2x H100",
            "source": "rl-timeslicing-validation T1 run 2026-07-16",
        },
        "kpis": {
            "acquires": len(acquires),
            "yields": len(yields),
            "snapshots": len(snaps),
            "restores": len(restores),
            "wallclock_min": round((t_end) / 60, 1),
        },
        "intervals": intervals,
        "snapshots": snaps,
        "restores": restores,
    }
    json.dump(data, open(f"{OUT}/slime_demo_data.json", "w"), indent=1)
    print("slime:", json.dumps(data["kpis"]))
    print(f"  intervals={len(intervals)}")


if __name__ == "__main__":
    verl()
    slime()
