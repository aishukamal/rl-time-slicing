#!/usr/bin/env python3
"""Build the 'See Time-Slicing in Action' replay page from real run data.

Timeline convention: t=0 is each side's job-active start (first GPU-busy sample).
Time-slice lock/agent events are clock-aligned to the utilization scraper before
shifting (the two telemetry sources start ~15 min apart).
"""
import json
import statistics as st
from collections import defaultdict
from datetime import datetime

BASE = "/Users/aishuk/workspaces/GPU-CR/timeslice-demos"
VERL_TS = "/Users/aishuk/Downloads/e2e_rl/verl/rl_logs_export_timeslice_20260427_092932/rl_logs_export_timeslice_20260427_092932"
SLIME_LOG = "/Users/aishuk/workspaces/GPU-CR/rl-timeslicing-validation/logs/t1/orchestrator.log"


# ------------------------------------------------------------------ veRL data
def verl_data():
    demo = json.load(open(f"{BASE}/verl_demo_data.json"))
    sp = demo["spans_sec"]
    ts_shift = sp["ts_lo"]   # first GPU-busy sample on the lock timeline
    bl_shift = sp["bl_lo"]

    # events with real per-op durations, shifted to active start
    events, t0 = [], None
    for line in open(f"{VERL_TS}/rl_metrics.jsonl"):
        e = json.loads(line)
        t0 = t0 if t0 is not None else e["ts"]
        if e["type"] not in ("acquire", "yield"):
            continue
        job = "A" if "job-a" in e["workload_id"] else "B"
        pool = e["pool"]
        t = e["ts"] - t0 - ts_shift
        if e["type"] == "acquire":
            r = e.get("restore_ms", 0)
            if r > 0:
                events.append({"t": round(t - r / 1000, 1), "kind": "restoring", "job": job, "pool": pool})
                events.append({"t": round(t, 1), "kind": "restore_done", "job": job, "pool": pool, "ms": r})
            events.append({"t": round(t, 1), "kind": "acquired", "job": job, "pool": pool,
                           "wait_ms": e.get("wait_ms", 0), "restore_ms": r, "step": e.get("step", 0)})
        else:
            ck = e.get("evict_ms", 0)
            events.append({"t": round(t, 1), "kind": "yielded", "job": job, "pool": pool, "ckpt_ms": ck})
            if ck > 0:
                events.append({"t": round(t, 1), "kind": "checkpointing", "job": job, "pool": pool})
                events.append({"t": round(t + ck / 1000, 1), "kind": "ckpt_done", "job": job, "pool": pool, "ms": ck})
    events.sort(key=lambda e: e["t"])

    windows = defaultdict(list)
    for v in demo["intervals"]:
        windows[v["pool"]].append({"t0": round(v["start"] - ts_shift, 1),
                                   "t1": round(v["end"] - ts_shift, 1),
                                   "job": "A" if v["job"] == "job-a" else "B"})

    def shifted(side, shift):
        return {g: [[p[0] - shift, p[1]] for p in v["series"] if p[0] >= shift]
                for g, v in demo["per_gpu"][side].items()}

    duty_ts = shifted("timeslice", ts_shift)
    duty_bl = shifted("baseline", bl_shift)

    last_yield = max(w["t1"] for ws in windows.values() for w in ws)
    dur_ts = round(last_yield)                  # replay ends at the final yield
    dur_bl = round(sp["bl_hi"] - sp["bl_lo"])   # baseline active span

    ck = [e["ms"] for e in events if e["kind"] == "ckpt_done"]
    rs = [e["ms"] for e in events if e["kind"] == "restore_done"]
    pg = demo["per_gpu"]

    return {
        "mode": "race",
        "label": "veRL · sync disaggregated · 2 RL jobs vs baseline",
        "meta": {
            "framework": "veRL",
            "mode": "sync disaggregated",
            "algo": "GRPO",
            "model": "Qwen2.5-0.5B-Instruct (GSM8K)",
            "hardware": "2×H100 80GB",
        },
        "pool_map": demo["pool_map"],
        "baseline": {"duration_s": dur_bl, "duty_gpus": duty_bl, "windows": {}, "events": []},
        "timeslice": {"duration_s": dur_ts, "duty_gpus": duty_ts,
                      "windows": {k: sorted(v, key=lambda w: w["t0"]) for k, v in windows.items()},
                      "events": sorted(
                          [{**e, "t": min(e["t"], dur_ts)} for e in events if e["t"] <= dur_ts + 60],
                          key=lambda e: e["t"])},
        "rewards": demo["rewards"],
        "summary": {
            "duty_base": demo["kpis"]["duty_baseline_pct"],
            "duty_ts": demo["kpis"]["duty_timeslice_pct"],
            "duty_gpu": {g: [pg["baseline"][g]["duty_pct"], pg["timeslice"][g]["duty_pct"]]
                         for g in pg["baseline"]},
            "swaps": len(ck),
            "med_ckpt_ms": round(st.median(ck)) if ck else 0,
            "med_restore_ms": round(st.median(rs)) if rs else 0,
            "base_min": round(dur_bl / 60), "ts_min": round(dur_ts / 60),
            "speedup": round(2 * dur_bl / dur_ts, 2),
            "cost_saving_pct": round((1 - dur_ts / (2 * dur_bl)) * 100),
        },
        "foot": "All series and events reconstructed from recorded run telemetry: per-GPU utilization samples "
                "(both runs), orchestrator acquire/yield events, and measured evict/restore durations. "
                "Duty cycle = share of minutes with GPU activity (util >5%) within each run's active window; "
                "curves show per-minute mean utilization. t=0 is each run's first GPU activity; the replay "
                "ends at the final lock yield.",
    }


# ----------------------------------------------------------------- Slime data
def slime_data():
    # Held back from the dashboard until Slime runs report the same metrics as veRL
    # (per-GPU utilization curves + real C/R durations).
    acq, yld, ops = [], [], []
    t0 = None
    for line in open(SLIME_LOG):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = datetime.fromisoformat(e["time"].replace("Z", "+00:00")).timestamp()
        msg = e.get("msg", "")
        if msg == "Acquire succeeded, job loaded and lock held":
            if t0 is None:
                t0 = ts
            acq.append({"t": ts - t0, "job": "A" if "slime-a" in e["JobID"] else "B", "pool": e["GroupID"]})
        elif msg == "Yield called" and t0 is not None:
            yld.append({"t": ts - t0, "job": "A" if "slime-a" in e["JobID"] else "B", "pool": e["GroupID"]})
        elif msg in ("Triggering snapshot for job", "Triggering restore for active job") and t0 is not None:
            ops.append({"t": ts - t0, "job": "A" if "slime-a" in e["jobID"] else "B", "pool": e["GroupID"],
                        "kind": "ckpt_done" if "snapshot" in msg else "restore_done", "ms": 0})

    events = sorted(
        [{**a, "kind": "acquired", "wait_ms": 0, "restore_ms": 0, "step": 0} for a in acq]
        + [{**y, "kind": "yielded", "ckpt_ms": 0} for y in yld]
        + ops, key=lambda e: e["t"])
    for e in events:
        e["t"] = round(e["t"], 1)

    windows, open_acq = defaultdict(list), {}
    for e in events:
        if e["kind"] == "acquired":
            open_acq[(e["job"], e["pool"])] = e["t"]
        elif e["kind"] == "yielded" and (e["job"], e["pool"]) in open_acq:
            windows[e["pool"]].append({"t0": open_acq.pop((e["job"], e["pool"])), "t1": e["t"], "job": e["job"]})
    t_end = max(e["t"] for e in events) + 20
    for (job, pool), start in open_acq.items():
        windows[pool].append({"t0": start, "t1": t_end, "job": job})

    return {
        "mode": "pools",
        "label": "Slime · sync · dual-pool lock alternation",
        "meta": {
            "framework": "Slime",
            "mode": "sync",
            "algo": "GRPO",
            "model": "Qwen2.5-0.5B",
            "hardware": "2×H100",
        },
        "timeslice": {"duration_s": t_end,
                      "windows": {k: sorted(v, key=lambda w: w["t0"]) for k, v in windows.items()},
                      "events": events},
        "summary": {
            "handoffs": len(acq),
            "snapshots": sum(1 for o in ops if o["kind"] == "ckpt_done"),
            "restores": sum(1 for o in ops if o["kind"] == "restore_done"),
            "wall_min": round(t_end / 60, 1),
        },
        "foot": "Reconstructed from the orchestrator lock events of a validated integration run. Agent "
                "snapshot/restore ops completed as no-ops in this run — both jobs fit in VRAM simultaneously, "
                "so the validation exercises lock-level alternation.",
    }


TEMPLATE = open(f"{BASE}/replay_template.html").read()

# Slime run held back until it reports the same metrics as veRL
data = {"runs": [verl_data()]}
html = TEMPLATE.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
open(f"{BASE}/timeslice-replay.html", "w").write(html)
print(f"wrote timeslice-replay.html ({len(html)//1024} KiB)")
