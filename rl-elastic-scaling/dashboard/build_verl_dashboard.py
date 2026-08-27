#!/usr/bin/env python3
"""verl PoC dashboard: all headline metrics/graphs in one self-contained HTML."""
import csv, json, statistics, datetime

B = "/Users/aishuk/workspaces/GPU-CR/elastic-rl-poc"

# ---------- helpers ----------
def jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try: out.append(json.loads(line))
                except json.JSONDecodeError: pass
    return out

def gpu_csv(path):
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out.setdefault(row["gpu_index"], []).append((int(row["timestamp_ms"])/1000.0, float(row["gpu_util_pct"])))
    return out

def bucket(pts, t0, w=15.0):
    out, cur, edge = [], [], None
    for t, u in pts:
        if edge is None: edge = t
        if t - edge >= w:
            if cur: out.append((round((edge-t0+w/2)/60.0,3), round(statistics.fmean(cur),1)))
            cur, edge = [], t
        cur.append(u)
    if cur: out.append((round((edge-t0)/60.0,3), round(statistics.fmean(cur),1)))
    return out

def ts(s):  # ISO -> epoch
    return datetime.datetime.fromisoformat(s.replace("Z","+00:00")).timestamp()

D = {}  # all data blobs for the template

# ---------- A: results ladder (canonical numbers from M0/M2/M3 reports) ----------
D["ladder"] = {
    "labels": ["(1) static 1:1<br>2 GPUs", "(2) colocated<br>2 GPUs", "(3) ELASTIC<br>2 GPUs", "(5) static 1:2<br>3 GPUs"],
    "step": [619.0, 522.7, 502.7, 321.0],
    "eff": [186.1, 220.4, 229.2, 239.2],
}

# ---------- B: duty cycles M0 vs M2 ----------
for tag, path in [("m0", f"{B}/m0-results/gpu_util.csv"), ("m2", f"{B}/m2-results/run2/gpu_util.csv")]:
    data = gpu_csv(path)
    t0 = min(v[0][0] for v in data.values())
    stats = {g: (statistics.fmean([x[1] for x in v]), sum(1 for x in v if x[1] < 5)/len(v)) for g, v in data.items()}
    D[tag] = {"t0": t0, "series": {g: bucket(v, t0) for g, v in data.items()}, "stats": {g: (round(m,1), round(i,2)) for g,(m,i) in stats.items()}}
D["m0_trainer"] = max(D["m0"]["stats"], key=lambda g: D["m0"]["stats"][g][1])
D["m2_shared"]  = max(D["m2"]["stats"], key=lambda g: D["m2"]["stats"][g][1])

# ---------- C: switch anatomy from M2 switch_timings ----------
sw = jsonl(f"{B}/m2-results/run2/switch_timings.jsonl")
phase_agg = {}
op_totals = {}
for rec in sw:
    op = rec.get("operation")
    if op not in ("switch-to-rollout", "switch-to-trainer"): continue
    op_totals.setdefault(op, []).append(rec["total_seconds"])
    for ph in rec.get("phases", []):
        phase_agg.setdefault(op, {}).setdefault(ph["phase"], []).append(ph["seconds"])
D["anatomy"] = {op: {ph: round(statistics.fmean(v),2) for ph, v in phs.items()} for op, phs in phase_agg.items()}
D["op_totals"] = {op: (round(statistics.fmean(v),1), round(statistics.pstdev(v),1), len(v)) for op, v in op_totals.items()}

# ---------- D: M2 controller behavior ----------
dec = jsonl(f"{B}/m2-results/run2/decisions.jsonl")
t0d = ts(dec[0]["ts"])
D["events"] = {"in": [], "back": [], "rt_in": [], "rt_out": [], "gaps": []}
for r in dec:
    m = (ts(r["ts"]) - t0d)/60.0
    if r.get("action") == "switch_to_rollout": D["events"]["in"].append(round(m,2))
    elif r.get("action") == "switch_to_trainer": D["events"]["back"].append(round(m,2))
    elif r.get("event") == "switch_in_done": D["events"]["rt_in"].append((round(m,2), r["total_seconds"]))
    elif r.get("event") == "switch_back_done": D["events"]["rt_out"].append((round(m,2), r["total_seconds"]))
    elif r.get("event") == "cycle_verified" and r.get("wake_minus_batch_ready_s") is not None:
        D["events"]["gaps"].append(r["wake_minus_batch_ready_s"])

# ---------- E: regime shift ----------
rdec = jsonl(f"{B}/m3-results/regime-shift/decisions.jsonl")
rt0 = ts(rdec[0]["ts"])
flip = jsonl(f"{B}/m3-results/regime-shift/regime_flip.jsonl")
D["rs"] = {"in": [round((ts(r["ts"])-rt0)/60.0,2) for r in rdec if r.get("action")=="switch_to_rollout"],
           "flip_min": round((flip[0].get("ts_epoch", ts(flip[0]["ts"]) if "ts" in flip[0] else rt0)-rt0)/60.0,2) if flip else None,
           "steps": {"pre_switching": 506.8, "post_with_switching": 261.8, "post_without": 288.2}}

# ---------- F: no-harm ----------
ndec = jsonl(f"{B}/m3-results/no-harm/armed/decisions.jsonl")
reasons = {}
for r in ndec:
    if r.get("action") == "none":
        reasons[r.get("reason","?")] = reasons.get(r.get("reason","?"), 0) + 1
D["noharm"] = {"reasons": dict(sorted(reasons.items(), key=lambda kv:-kv[1])[:6]),
               "armed": 42.58, "armed_sd": 4.1, "control": 42.24, "control_sd": 4.0,
               "switches": sum(1 for r in ndec if r.get("action")=="switch_to_rollout")}

# ---------- G: run D ----------
rd = gpu_csv(f"{B}/m3-results/static12/head/gpu_util.csv")
rdw = gpu_csv(f"{B}/m3-results/static12/worker/gpu_util_worker.csv")
t0r = min(min(v[0][0] for v in rd.values()), min(v[0][0] for v in rdw.values()))
D["rund"] = {"head": {g: bucket(v, t0r) for g, v in rd.items()},
             "worker": {g: bucket(v, t0r) for g, v in rdw.items()}}

json.dump(D, open(f"{B}/dashboard_data.json","w"))
print("data built:", {k: (len(v) if hasattr(v,'__len__') else v) for k,v in D.get("op_totals",{}).items()},
      "| m2 live in/back:", len(D["events"]["in"]), len(D["events"]["back"]),
      "| gaps:", len(D["events"]["gaps"]), "| rs switch-ins:", len(D["rs"]["in"]), "| noharm none-reasons:", D["noharm"]["reasons"])
