#!/usr/bin/env python3
"""Bake replay data for the verl ease-of-adoption demo dashboard.

Parses ../evidence/timeline-run3.txt (real orchestrator/agent events from the
2026-07-20 two-job sync PoC, run 3) into a compact JSON structure the dashboard
replays: GPU turn segments, switch windows, and a curated event feed.
Rewards per step come from EVIDENCE-SUMMARY.md (critic/rewards/mean).
"""
import json
import re
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
TIMELINE = HERE.parent / "evidence" / "timeline-run3.txt"

REWARDS = {
    "a3": [0.016, 0.008, 0.016, 0.016, 0.039, 0.063, 0.039, 0.055, 0.039, 0.094, 0.070, 0.141],
    "b3": [0.008, 0.016, 0.047, 0.016, 0.023, 0.016, 0.063, 0.102, 0.063, 0.117, 0.063],
}

def parse():
    rows = []
    for line in TIMELINE.read_text().splitlines():
        m = re.match(r"(\S+)\s+(ORC|AGT)\s+(\S+)\s+(.*)", line.strip())
        if not m:
            continue
        ts, src, job, msg = m.groups()
        if not job.endswith("3"):  # run 3 only (a3/b3)
            continue
        rows.append((datetime.fromisoformat(ts), src, job[-2:], msg))
    rows.sort(key=lambda r: r[0])
    t0 = rows[0][0]

    def rel(ts):
        return round((ts - t0).total_seconds(), 2)

    segments = []   # {job, start, end} — lock held (GPU turn)
    switches = []   # {start, end, snap_s, restore_s, from, to}
    events = []     # curated feed
    held = {}       # job -> acquire-succeeded time
    steps = {"a3": [], "b3": []}  # completion times (rel s), one per turn end
    turn_count = {"a3": 0, "b3": 0}
    pending_switch = None

    for ts, src, job, msg in rows:
        jid = "a3" if job == "a3" else "b3"
        r = rel(ts)
        if "Acquire succeeded" in msg:
            held[jid] = r
            if pending_switch is not None:
                pending_switch["end"] = r
                pending_switch["to"] = jid
                switches.append(pending_switch)
                pending_switch = None
            events.append({"t": r, "src": "ORC", "job": jid, "kind": "grant",
                           "msg": "Lock granted — context restored, training resumes"})
        elif "Yield called" in msg and src == "ORC":
            if jid in held:
                seg = {"job": jid, "start": held.pop(jid), "end": r}
                segments.append(seg)
                turn_count[jid] += 1
                n = turn_count[jid]
                if n <= len(REWARDS[jid]):
                    steps[jid].append({"t": r, "step": n, "reward": REWARDS[jid][n - 1]})
            pending_switch = {"start": r, "from": jid, "snap_s": None, "restore_s": None}
            events.append({"t": r, "src": "ORC", "job": jid, "kind": "yield",
                           "msg": "Step done — yielding GPU"})
        elif m2 := re.search(r"snapshot cuda-checkpoint done \(([\d.]+)s\)", msg):
            if pending_switch is not None:
                pending_switch["snap_s"] = float(m2.group(1))
            events.append({"t": r, "src": "AGT", "job": jid, "kind": "snap",
                           "msg": f"Snapshot complete ({m2.group(1)}s) — GPU state saved"})
        elif m2 := re.search(r"restore cuda-checkpoint done \(([\d.]+)s\)", msg):
            if pending_switch is not None:
                pending_switch["restore_s"] = float(m2.group(1))
            events.append({"t": r, "src": "AGT", "job": jid, "kind": "restore",
                           "msg": f"Restore complete ({m2.group(1)}s) — back on GPU"})
        elif "Acquire called" in msg and jid not in held:
            events.append({"t": r, "src": "ORC", "job": jid, "kind": "acquire",
                           "msg": "Requesting GPU lock…"})

    total = rel(rows[-1][0])
    clean = [s for s in switches if s.get("snap_s") and s.get("restore_s")]
    stats = {
        "turns": len(segments),
        "handoffs": len(switches),
        "switch_mean_s": round(sum(s["end"] - s["start"] for s in clean) / max(len(clean), 1), 1),
        "snap_mean_s": round(sum(s["snap_s"] for s in clean) / max(len(clean), 1), 1),
        "restore_mean_s": round(sum(s["restore_s"] for s in clean) / max(len(clean), 1), 1),
        "total_s": total,
        "steps_a": len(steps["a3"]),
        "steps_b": len(steps["b3"]),
    }
    return {"segments": segments, "switches": switches, "events": events,
            "steps": steps, "stats": stats, "t0": rows[0][0].isoformat()}


if __name__ == "__main__":
    data = parse()
    out = HERE / "replay_data.json"
    out.write_text(json.dumps(data, indent=1))
    s = data["stats"]
    print(f"turns={s['turns']} handoffs={s['handoffs']} switch~{s['switch_mean_s']}s "
          f"(snap {s['snap_mean_s']} + restore {s['restore_mean_s']}) "
          f"steps a3={s['steps_a']} b3={s['steps_b']} span={s['total_s']}s events={len(data['events'])}")
