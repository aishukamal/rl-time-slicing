#!/usr/bin/env python3
"""Generate an animated baseline-vs-time-slicing replay (self-contained HTML).

The replay "plays back" a run like a video: duty-cycle curves draw in behind a
moving playhead in two synchronized columns — baseline (1 job, idle gaps) on the
left, time-slicing (2 jobs interleaved via C/R) on the right — while orchestrator
checkpoint/restore events appear in a live feed.

Data sources:
  - <run>/baseline/tpu_duty_cycle_{sampler,trainer}.csv        REAL scraper data
    (baseline has no restores, so the v1 duty-freeze bug doesn't affect it)
  - <run>/baseline/orchestrator.log                            REAL events (windows)
  - <run>/timeslice_synthetic/tpu_duty_cycle_{sampler,trainer}.csv  synthetic
    reconstruction (baseline per-step patterns masked by real orch windows), same
    series the approved dashboards use — v1 metric server reports stale duty
    after restore, so raw timeslice scraper data is unusable
  - <run>/timeslice/orchestrator.log                           REAL events + C/R ms

Usage:
  python3 generate_replay.py <run_dir> [-o out.html]

Stdlib only. Output is a single self-contained HTML file.
"""
import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone

TS_FMT = "%Y-%m-%d %H:%M:%S,%f"

RE_ACQUIRED = re.compile(
    r"^(?P<ts>[\d-]+ [\d:,]+) \[orchestrator\] INFO Acquired (?P<wl>job-[ab])-(?P<pool>sampler|trainer) "
    r"step=(?P<step>\d+) wait_ms=(?P<wait>\d+) restore_ms=(?P<restore>\d+)")
RE_YIELDED = re.compile(
    r"^(?P<ts>[\d-]+ [\d:,]+) \[orchestrator\] INFO Yielded (?P<wl>job-[ab])-(?P<pool>sampler|trainer) "
    r"checkpoint_ms=(?P<ckpt>\d+)")
RE_CHECKPOINTING = re.compile(
    r"^(?P<ts>[\d-]+ [\d:,]+) \[orchestrator\] INFO Checkpointing (?P<wl>job-[ab])-(?P<pool>sampler|trainer)")
RE_RESTORING = re.compile(
    r"^(?P<ts>[\d-]+ [\d:,]+) \[orchestrator\] INFO Restoring (?P<wl>job-[ab])-(?P<pool>sampler|trainer)")
RE_CKPT_DONE = re.compile(
    r"^(?P<ts>[\d-]+ [\d:,]+) \[orchestrator\] INFO checkpoint via workload (?P<wl>job-[ab])-(?P<pool>sampler|trainer): "
    r"(?P<ms>\d+)ms")
RE_RESTORE_DONE = re.compile(
    r"^(?P<ts>[\d-]+ [\d:,]+) \[orchestrator\] INFO restore via workload (?P<wl>job-[ab])-(?P<pool>sampler|trainer): "
    r"(?P<ms>\d+)ms")


def parse_ts(s):
    # Orchestrator logs use the pod clock (UTC), matching the scraper's epoch column.
    return datetime.strptime(s, TS_FMT).replace(tzinfo=timezone.utc).timestamp()


def load_duty(path, bin_s=2.0):
    """Chip-0 duty series, binned to bin_s seconds. Returns [[epoch, duty], ...]."""
    pts = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["chip"] != "0":
                continue
            pts.append((float(row["ts"]), float(row["duty_cycle_pct"])))
    pts.sort()
    binned, cur_bin, acc = [], None, []
    for t, v in pts:
        b = int(t // bin_s)
        if cur_bin is None:
            cur_bin = b
        if b != cur_bin:
            binned.append([cur_bin * bin_s, max(acc)])
            cur_bin, acc = b, []
        acc.append(v)
    if acc:
        binned.append([cur_bin * bin_s, max(acc)])
    return binned


def load_events(path):
    """Parse orchestrator.log into events + ownership windows per pool."""
    events = []
    windows = {"sampler": [], "trainer": []}
    open_win = {}

    with open(path) as f:
        for line in f:
            m = RE_ACQUIRED.match(line)
            if m:
                t = parse_ts(m["ts"])
                job = m["wl"][-1].upper()
                events.append({"t": t, "kind": "acquired", "job": job, "pool": m["pool"],
                               "step": int(m["step"]), "wait_ms": int(m["wait"]),
                               "restore_ms": int(m["restore"])})
                pool = m["pool"]
                if pool in open_win:
                    t0, j0 = open_win.pop(pool)
                    windows[pool].append({"t0": t0, "t1": t, "job": j0})
                open_win[pool] = (t, job)
                continue
            m = RE_YIELDED.match(line)
            if m:
                t = parse_ts(m["ts"])
                pool = m["pool"]
                events.append({"t": t, "kind": "yielded", "job": m["wl"][-1].upper(),
                               "pool": pool, "ckpt_ms": int(m["ckpt"])})
                if pool in open_win:
                    t0, j0 = open_win.pop(pool)
                    windows[pool].append({"t0": t0, "t1": t, "job": j0})
                continue
            m = RE_CHECKPOINTING.match(line)
            if m:
                events.append({"t": parse_ts(m["ts"]), "kind": "checkpointing",
                               "job": m["wl"][-1].upper(), "pool": m["pool"]})
                continue
            m = RE_RESTORING.match(line)
            if m:
                events.append({"t": parse_ts(m["ts"]), "kind": "restoring",
                               "job": m["wl"][-1].upper(), "pool": m["pool"]})
                continue
            m = RE_CKPT_DONE.match(line)
            if m:
                events.append({"t": parse_ts(m["ts"]), "kind": "ckpt_done",
                               "job": m["wl"][-1].upper(), "pool": m["pool"], "ms": int(m["ms"])})
                continue
            m = RE_RESTORE_DONE.match(line)
            if m:
                events.append({"t": parse_ts(m["ts"]), "kind": "restore_done",
                               "job": m["wl"][-1].upper(), "pool": m["pool"], "ms": int(m["ms"])})

    if events:
        t_end = max(e["t"] for e in events)
        for pool, (t0, j0) in open_win.items():
            windows[pool].append({"t0": t0, "t1": t_end, "job": j0})
    events.sort(key=lambda e: e["t"])
    return events, windows


def clip(events, windows, t0, t1, margin=60):
    events = [e for e in events if t0 - margin <= e["t"] <= t1 + margin]
    cw = {}
    for pool, ws in windows.items():
        out = []
        for w in ws:
            if w["t1"] < t0 - margin or w["t0"] > t1 + margin:
                continue
            out.append({"t0": max(w["t0"], t0), "t1": min(w["t1"], t1), "job": w["job"]})
        cw[pool] = out
    return events, cw


def build_side(run, sub, duty_sub):
    sampler = load_duty(os.path.join(run, duty_sub, "tpu_duty_cycle_sampler.csv"))
    trainer = load_duty(os.path.join(run, duty_sub, "tpu_duty_cycle_trainer.csv"))
    events, windows = load_events(os.path.join(run, sub, "orchestrator.log"))
    t0 = min(sampler[0][0], trainer[0][0])
    t1 = max(sampler[-1][0], trainer[-1][0])
    events, windows = clip(events, windows, t0, t1)
    if not events:
        sys.exit(f"no orchestrator events inside duty window for {sub}")
    # Work span: first acquire -> last yield/acquire (excludes scraper tails).
    work_end = max(e["t"] for e in events)
    work_start = min(e["t"] for e in events)

    # Per-pool average duty over the work span. In baseline each pool sits idle
    # during the other phase; time-slicing fills those windows with the other job.
    def avg_duty(series):
        vals = [v for t, v in series if work_start <= t <= work_end]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    def rel(series):
        return [[round(t - t0, 1), round(v, 1)] for t, v in series]

    return {
        "duty_sampler": avg_duty(sampler),
        "duty_trainer": avg_duty(trainer),
        "t0": t0,
        "t0_wall": datetime.fromtimestamp(t0, tz=timezone.utc).strftime("%H:%M:%S UTC"),
        "duration_s": round(t1 - t0, 1),
        "work_s": round(work_end - work_start, 1),
        "sampler": rel(sampler),
        "trainer": rel(trainer),
        "events": [dict(e, t=round(e["t"] - t0, 1)) for e in events],
        "windows": {p: [{"t0": round(w["t0"] - t0, 1), "t1": round(w["t1"] - t0, 1),
                         "job": w["job"]} for w in ws] for p, ws in windows.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    run = args.run_dir.rstrip("/")
    out = args.out or os.path.join(run, "replay_timeslice.html")

    base = build_side(run, "baseline", "baseline")
    ts = build_side(run, "timeslice", "timeslice_synthetic")

    ckpts = [e["ckpt_ms"] for e in ts["events"] if e["kind"] == "yielded" and e["ckpt_ms"] > 0]
    restores = [e["restore_ms"] for e in ts["events"] if e["kind"] == "acquired" and e["restore_ms"] > 0]
    speedup = (2 * base["work_s"]) / ts["work_s"]

    data = {
        "run": os.path.basename(os.path.abspath(run)),
        "baseline": base,
        "timeslice": ts,
        "summary": {
            "swaps": len(ckpts),
            "avg_ckpt_ms": round(sum(ckpts) / len(ckpts)) if ckpts else 0,
            "avg_restore_ms": round(sum(restores) / len(restores)) if restores else 0,
            "base_min": round(base["work_s"] / 60, 1),
            "ts_min": round(ts["work_s"] / 60, 1),
            "speedup": round(speedup, 2),
            "duty_base": round((base["duty_sampler"] + base["duty_trainer"]) / 2, 1),
            "duty_ts": round((ts["duty_sampler"] + ts["duty_trainer"]) / 2, 1),
            "duty_detail": {
                "sampler": [base["duty_sampler"], ts["duty_sampler"]],
                "trainer": [base["duty_trainer"], ts["duty_trainer"]],
            },
            "cost_saving_pct": round((1 - ts["work_s"] / (2 * base["work_s"])) * 100),
        },
    }

    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out}  ({os.path.getsize(out)//1024} KiB)")
    print(f"  baseline {data['summary']['base_min']} min work, timeslice {data['summary']['ts_min']} min, "
          f"speedup {data['summary']['speedup']}x, {data['summary']['swaps']} C/R swaps")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TPU RL Time-Slicing — Baseline vs Time-Slicing Replay</title>
<style>
  .viz-root {
    color-scheme: light;
    --surface-1: #fcfcfb; --page: #f9f9f7;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
    --grid: #e1e0d9; --baseline: #c3c2b7; --border: rgba(11,11,11,0.10);
    --job-a: #2a78d6; --job-b: #eb6834; --idle: #c3c2b7;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) .viz-root {
      color-scheme: dark;
      --surface-1: #1a1a19; --page: #0d0d0d;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
      --job-a: #3987e5; --job-b: #d95926; --idle: #383835;
    }
  }
  * { box-sizing: border-box; margin: 0; }
  body.viz-root { background: var(--page); color: var(--text-primary);
    font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; padding: 20px; }
  .wrap { max-width: 1320px; margin: 0 auto; }
  header { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; margin-bottom: 4px; }
  h1 { font-size: 19px; font-weight: 650; }
  .sub { color: var(--text-secondary); font-size: 13px; }
  .controls { display: flex; align-items: center; gap: 10px; margin: 12px 0; flex-wrap: wrap; }
  button.play { width: 88px; padding: 7px 0; border: 1px solid var(--border); border-radius: 8px;
    background: var(--surface-1); color: var(--text-primary); font: inherit; font-weight: 600; cursor: pointer; }
  button.play:hover { border-color: var(--muted); }
  select { padding: 6px 8px; border: 1px solid var(--border); border-radius: 8px;
    background: var(--surface-1); color: var(--text-primary); font: inherit; }
  .clock { font-variant-numeric: tabular-nums; color: var(--text-secondary); }
  .legend { display: flex; gap: 16px; margin-left: auto; align-items: center; font-size: 13px; }
  .key { display: inline-block; width: 14px; height: 4px; border-radius: 2px; vertical-align: middle; margin-right: 6px; }
  .cols { display: grid; grid-template-columns: 1fr 1fr 270px; gap: 14px; align-items: start; }
  .colhead { font-size: 14px; font-weight: 650; margin-bottom: 8px; }
  .colhead .tag { font-weight: 400; color: var(--text-secondary); }
  .panelcard, .side { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 12px 14px; }
  .panelcard { margin-bottom: 12px; }
  .panelcard h2 { font-size: 12.5px; font-weight: 600; color: var(--text-secondary); margin-bottom: 4px; }
  canvas { display: block; width: 100%; }
  .tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 2px; }
  .tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 9px 12px; }
  .tile .lbl { font-size: 11.5px; color: var(--muted); }
  .tile .val { font-size: 20px; font-weight: 650; }
  .side h2 { font-size: 13px; font-weight: 600; color: var(--text-secondary); margin-bottom: 4px; }
  .side .hint { font-size: 11.5px; color: var(--muted); margin-bottom: 8px; }
  .feed { list-style: none; display: flex; flex-direction: column; gap: 6px; max-height: 480px; overflow-y: auto; }
  .feed li { padding: 6px 10px; border-left: 3px solid var(--baseline); background: color-mix(in srgb, var(--surface-1) 88%, var(--page));
    border-radius: 0 8px 8px 0; font-size: 12.5px; opacity: 0; transform: translateY(-4px); transition: opacity .25s, transform .25s; }
  .feed li.show { opacity: 1; transform: none; }
  .feed li .t { color: var(--muted); font-variant-numeric: tabular-nums; margin-right: 6px; }
  .feed li.jobA { border-left-color: var(--job-a); }
  .feed li.jobB { border-left-color: var(--job-b); }
  .feed li.pending { font-style: italic; }
  .endcard { position: fixed; inset: 0; display: none; align-items: center; justify-content: center;
    background: color-mix(in srgb, var(--page) 82%, transparent); backdrop-filter: blur(2px); }
  .endcard .box { background: var(--surface-1); border: 1px solid var(--border); border-radius: 14px;
    padding: 30px 38px; text-align: center; max-width: 520px; }
  .endcard .big { font-size: 46px; font-weight: 650; }
  .endcard p { color: var(--text-secondary); margin-top: 8px; }
  .endcard button { margin-top: 18px; }
  details.tbl { margin-top: 14px; color: var(--text-secondary); font-size: 13px; }
  details.tbl table { border-collapse: collapse; margin-top: 8px; font-variant-numeric: tabular-nums; }
  details.tbl th, details.tbl td { text-align: left; padding: 3px 12px 3px 0; border-bottom: 1px solid var(--grid); }
  .foot { margin-top: 14px; color: var(--muted); font-size: 12px; }
  .tip { position: fixed; pointer-events: none; background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 8px; padding: 6px 9px; font-size: 12px; display: none; box-shadow: 0 2px 10px rgba(0,0,0,.12); }
  @media (max-width: 1000px) { .cols { grid-template-columns: 1fr; } }
</style>
</head>
<body class="viz-root">
<div class="wrap">
  <header>
    <h1>TPU RL Time-Slicing — Baseline vs Time-Slicing</h1>
    <span class="sub" id="subtitle"></span>
  </header>
  <div class="controls">
    <button class="play" id="btn">&#9654; Play</button>
    <label class="sub">Speed <select id="speed">
      <option value="30">30&times;</option>
      <option value="60" selected>60&times;</option>
      <option value="90">90&times;</option>
      <option value="120">120&times;</option>
      <option value="240">240&times;</option>
    </select></label>
    <span class="clock" id="clock"></span>
    <div class="legend">
      <span><span class="key" style="background:var(--job-a)"></span>Job A</span>
      <span><span class="key" style="background:var(--job-b)"></span>Job B</span>
      <span><span class="key" style="background:var(--idle)"></span>idle / no holder</span>
    </div>
  </div>

  <div class="cols">
    <div>
      <div class="colhead">Baseline <span class="tag">— 1 job alone on the TPU</span></div>
      <div class="panelcard"><h2>Sampler — vLLM rollouts — duty cycle %</h2><canvas id="bs" height="150"></canvas></div>
      <div class="panelcard"><h2>Trainer — JAX GRPO — duty cycle %</h2><canvas id="bt" height="150"></canvas></div>
      <div class="tiles" style="grid-template-columns: repeat(2, 1fr);">
        <div class="tile"><div class="lbl" id="b_done_lbl">Job A running&hellip;</div><div class="val" id="b_done">–</div></div>
        <div class="tile"><div class="lbl">2 jobs back-to-back would take</div><div class="val" id="b_proj">–</div></div>
      </div>
    </div>
    <div>
      <div class="colhead">Time-slicing <span class="tag">— 2 jobs share it via checkpoint/restore</span></div>
      <div class="panelcard"><h2>Sampler pool — vLLM rollouts — duty cycle %</h2><canvas id="ts" height="150"></canvas></div>
      <div class="panelcard"><h2>Trainer pool — JAX GRPO — duty cycle %</h2><canvas id="tt" height="150"></canvas></div>
      <div class="tiles">
        <div class="tile"><div class="lbl">Elapsed</div><div class="val" id="t_el">0:00</div></div>
        <div class="tile"><div class="lbl">C/R swaps</div><div class="val" id="t_sw">0</div></div>
        <div class="tile"><div class="lbl">Avg checkpoint</div><div class="val" id="t_ck">–</div></div>
        <div class="tile"><div class="lbl">Avg restore</div><div class="val" id="t_rs">–</div></div>
      </div>
    </div>
    <aside class="side">
      <h2>Orchestrator events</h2>
      <div class="hint">Window management: which job is granted / yields each pool.</div>
      <ul class="feed" id="feed_orch" style="max-height:210px"></ul>
      <h2 style="margin-top:14px">Snapshot agent events</h2>
      <div class="hint">C/R execution: an in-progress entry updates in place with its duration. Scroll for history.</div>
      <ul class="feed" id="feed_agent" style="max-height:210px"></ul>
    </aside>
  </div>

  <details class="tbl"><summary>All time-slice events (table view)</summary><div id="tblwrap"></div></details>
  <p class="foot">Baseline panels: real scraper data. Time-slice events, ownership windows and C/R timings: real
     orchestrator log. Time-slice duty series: synthetic reconstruction (baseline per-step patterns masked by real
     orchestrator windows) — the v1 libtpu metric server reports stale duty after restore.</p>
</div>
<div class="endcard" id="end"><div class="box">
  <div class="big" id="endbig"></div>
  <p id="endduty" style="font-size:16px; color:var(--text-primary); font-weight:600;"></p>
  <p id="endstats"></p>
  <p id="endcost"></p>
  <p>Checkpoint/restore keeps both RL jobs resident — each swap parks one job&rsquo;s full TPU state
     in host RAM and hands the chips to the other, so the idle half of each RL phase is filled
     by the other job&rsquo;s work.</p>
  <button class="play" id="replay">&#8635; Replay</button>
</div></div>
<div class="tip" id="tip"></div>

<script>
const D = /*__DATA__*/;

const css = n => getComputedStyle(document.body).getPropertyValue(n).trim();
const jobColor = j => j === 'A' ? css('--job-a') : css('--job-b');
const fmtT = s => { s = Math.max(0, s|0); const m = (s/60)|0; return m + ':' + String(s%60).padStart(2,'0'); };
const fmtMs = ms => ms >= 1000 ? (ms/1000).toFixed(1) + 's' : ms + 'ms';
const tip = document.getElementById('tip');

const DUR = Math.max(D.baseline.duration_s, D.timeslice.duration_s);

function makePanel(canvas, side, pool) {
  const series = side[pool === 'sampler' ? 'sampler' : 'trainer'];
  const wins = side.windows[pool];
  const sideDur = side.duration_s;
  const ctx = canvas.getContext('2d');
  const P = { l: 32, r: 8, t: 8, b: 18, strip: 8 };
  let W, H, dpr;
  function size() {
    dpr = window.devicePixelRatio || 1;
    W = canvas.clientWidth; H = canvas.clientHeight = 150;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  size(); addEventListener('resize', () => { size(); draw(lastNow); });

  // Shared x-scale across both columns: x maps [0, DUR] so durations compare visually.
  const X = t => P.l + (t / DUR) * (W - P.l - P.r);
  const plotH = () => H - P.t - P.b - P.strip;
  const Y = v => P.t + (1 - v / 105) * plotH();
  const owner = t => { for (const w of wins) if (t >= w.t0 && t < w.t1) return w.job; return null; };

  let lastNow = 0;
  function draw(now) {
    lastNow = now;
    const local = Math.min(now, sideDur);
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = css('--grid'); ctx.lineWidth = 1;
    for (const g of [0, 50, 100]) {
      ctx.beginPath(); ctx.moveTo(P.l, Y(g)); ctx.lineTo(W - P.r, Y(g)); ctx.stroke();
      ctx.fillStyle = css('--muted'); ctx.font = '10px system-ui';
      ctx.textAlign = 'right'; ctx.fillText(g, P.l - 5, Y(g) + 3);
    }
    ctx.textAlign = 'center';
    for (let m = 0; m * 1200 <= DUR; m++) {
      ctx.fillText(m * 20 + 'm', X(m * 1200), H - P.b + 12);
    }
    const stripY = P.t + plotH() + 3;
    for (const w of wins) {
      if (w.t0 > local) break;
      ctx.fillStyle = jobColor(w.job);
      ctx.beginPath();
      ctx.roundRect(X(w.t0), stripY, Math.max(1, X(Math.min(w.t1, local)) - X(w.t0)), P.strip, 2);
      ctx.fill();
    }
    let seg = [], segOwner;
    const flush = () => {
      if (seg.length < 2) { seg = []; return; }
      const col = segOwner ? jobColor(segOwner) : css('--idle');
      ctx.beginPath();
      ctx.moveTo(seg[0][0], Y(0));
      for (const [x, y] of seg) ctx.lineTo(x, y);
      ctx.lineTo(seg[seg.length-1][0], Y(0)); ctx.closePath();
      ctx.globalAlpha = 0.10; ctx.fillStyle = col; ctx.fill(); ctx.globalAlpha = 1;
      ctx.beginPath(); ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      ctx.strokeStyle = col; ctx.lineWidth = 2;
      for (let i = 0; i < seg.length; i++) i ? ctx.lineTo(seg[i][0], seg[i][1]) : ctx.moveTo(seg[i][0], seg[i][1]);
      ctx.stroke();
      seg = [];
    };
    let lastPt = null;
    for (const [t, v] of series) {
      if (t > local) break;
      const o = owner(t);
      if (seg.length && o !== segOwner) { seg.push([X(t), Y(v)]); flush(); }
      if (!seg.length) segOwner = o;
      seg.push([X(t), Y(v)]);
      lastPt = [X(t), Y(v), o];
    }
    flush();
    if (lastPt && local < sideDur) {
      ctx.beginPath(); ctx.arc(lastPt[0], lastPt[1], 6, 0, 7);
      ctx.fillStyle = css('--surface-1'); ctx.fill();
      ctx.beginPath(); ctx.arc(lastPt[0], lastPt[1], 4, 0, 7);
      ctx.fillStyle = lastPt[2] ? jobColor(lastPt[2]) : css('--idle'); ctx.fill();
    }
    for (const e of side.events) {
      if (e.t > local) break;
      if (e.pool !== pool || (e.kind !== 'yielded' && e.kind !== 'acquired')) continue;
      if (!e.ckpt_ms && !e.restore_ms) continue;   // baseline acquires aren't C/R
      ctx.fillStyle = jobColor(e.job);
      ctx.beginPath();
      const x = X(e.t);
      ctx.moveTo(x, P.t); ctx.lineTo(x + 4, P.t + 6); ctx.lineTo(x, P.t + 12); ctx.lineTo(x - 4, P.t + 6);
      ctx.closePath(); ctx.fill();
    }
    if (local >= sideDur && sideDur < DUR) {
      // side finished before the global clock: mark completion
      const x = X(sideDur);
      ctx.strokeStyle = css('--muted'); ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(x, P.t); ctx.lineTo(x, P.t + plotH()); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = css('--muted'); ctx.textAlign = 'left'; ctx.font = '11px system-ui';
      ctx.fillText('done', x + 5, P.t + 12);
    } else {
      const px = X(local);
      ctx.strokeStyle = css('--baseline'); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(px, P.t); ctx.lineTo(px, P.t + plotH() + P.strip + 3); ctx.stroke();
    }
  }
  canvas.addEventListener('mousemove', ev => {
    const r = canvas.getBoundingClientRect();
    const t = (ev.clientX - r.left - P.l) / (W - P.l - P.r) * DUR;
    if (t < 0 || t > Math.min(lastNow, sideDur)) { tip.style.display = 'none'; return; }
    let best = null;
    for (const [ts, v] of series) { if (ts > t) break; best = [ts, v]; }
    if (!best) return;
    const o = owner(best[0]);
    tip.style.display = 'block';
    tip.style.left = (ev.clientX + 14) + 'px'; tip.style.top = (ev.clientY + 10) + 'px';
    tip.innerHTML = `<b>${fmtT(best[0])}</b> &middot; duty ${best[1]}%` + (o ? ` &middot; Job ${o}` : ' &middot; idle');
  });
  canvas.addEventListener('mouseleave', () => tip.style.display = 'none');
  return { draw };
}

const S = D.summary;
document.getElementById('subtitle').textContent =
  `${D.run} — same 5-step GRPO recipe on one v5e-8 · baseline start ${D.baseline.t0_wall} · timeslice start ${D.timeslice.t0_wall}`;
let baseDone = false;
function baselineTiles() {
  if (!baseDone && now >= D.baseline.duration_s) {
    baseDone = true;
    document.getElementById('b_done_lbl').textContent = 'Job A finished in';
    document.getElementById('b_done').textContent = S.base_min + ' min';
    document.getElementById('b_proj').textContent = (2 * S.base_min).toFixed(0) + ' min';
  }
}

const panels = [
  makePanel(document.getElementById('bs'), D.baseline, 'sampler'),
  makePanel(document.getElementById('bt'), D.baseline, 'trainer'),
  makePanel(document.getElementById('ts'), D.timeslice, 'sampler'),
  makePanel(document.getElementById('tt'), D.timeslice, 'trainer'),
];

const feedOrch = document.getElementById('feed_orch');
const feedAgent = document.getElementById('feed_agent');
const btn = document.getElementById('btn');
const endcard = document.getElementById('end');
let now = 0, playing = false, raf, lastFrame, evIdx = 0, ckSum = 0, ckN = 0, rsSum = 0, rsN = 0;
const pending = {};   // "job:pool" -> li for in-place completion updates (agent feed)

function addEntry(feed, e, txt, cls) {
  const li = document.createElement('li');
  li.className = 'job' + e.job + (cls ? ' ' + cls : '');
  li.innerHTML = `<span class="t">${fmtT(e.t)}</span>${txt}`;
  feed.prepend(li);
  requestAnimationFrame(() => li.classList.add('show'));
  while (feed.children.length > 40) feed.lastChild.remove();
  return li;
}

function pushFeed(e) {
  const key = e.job + ':' + e.pool;
  switch (e.kind) {
    // --- orchestrator feed: window management only -------------------------
    case 'acquired':
      addEntry(feedOrch, e,
        `Acquired: Job ${e.job} ${e.pool} (step ${e.step}` +
        (e.wait_ms > 0 ? `, waited ${fmtMs(e.wait_ms)}` : '') + ')');
      if (e.restore_ms > 0) { rsSum += e.restore_ms; rsN++; }
      break;
    case 'yielded':
      addEntry(feedOrch, e, `Yielded: Job ${e.job} ${e.pool}`);
      if (e.ckpt_ms > 0) { ckSum += e.ckpt_ms; ckN++; }
      break;
    // --- snapshot agent feed: C/R execution --------------------------------
    case 'checkpointing':
      pending[key] = addEntry(feedAgent, e, `&#9208; Snapshotting Job ${e.job} ${e.pool}…`, 'pending');
      break;
    case 'restoring':
      pending[key] = addEntry(feedAgent, e, `&#9654; Restoring Job ${e.job} ${e.pool}…`, 'pending');
      break;
    case 'ckpt_done':
      if (pending[key]) {
        const li = pending[key]; delete pending[key];
        li.classList.remove('pending');
        li.innerHTML = `<span class="t">${fmtT(e.t)}</span>&#10003; Snapshot Job ${e.job} ${e.pool}: ${fmtMs(e.ms)} — chips released`;
      } else {
        addEntry(feedAgent, e, `&#10003; Snapshot Job ${e.job} ${e.pool}: ${fmtMs(e.ms)}`);
      }
      break;
    case 'restore_done':
      if (pending[key]) {
        const li = pending[key]; delete pending[key];
        li.classList.remove('pending');
        li.innerHTML = `<span class="t">${fmtT(e.t)}</span>&#10003; Restore Job ${e.job} ${e.pool}: ${fmtMs(e.ms)} — state back on TPU`;
      } else {
        addEntry(feedAgent, e, `&#10003; Restore Job ${e.job} ${e.pool}: ${fmtMs(e.ms)}`);
      }
      break;
  }
}

function tiles() {
  document.getElementById('t_el').textContent = fmtT(Math.min(now, D.timeslice.duration_s));
  document.getElementById('t_sw').textContent = ckN;
  document.getElementById('t_ck').textContent = ckN ? fmtMs(Math.round(ckSum / ckN)) : '–';
  document.getElementById('t_rs').textContent = rsN ? fmtMs(Math.round(rsSum / rsN)) : '–';
  document.getElementById('clock').textContent = `${fmtT(now)} / ${fmtT(DUR)} run time`;
}

function frame(tsNow) {
  if (!playing) return;
  const dt = (tsNow - lastFrame) / 1000; lastFrame = tsNow;
  now = Math.min(now + dt * +document.getElementById('speed').value, DUR);
  const tsEvents = D.timeslice.events;
  while (evIdx < tsEvents.length && tsEvents[evIdx].t <= now) pushFeed(tsEvents[evIdx++]);
  for (const p of panels) p.draw(now);
  tiles(); baselineTiles();
  if (now >= DUR) { playing = false; btn.innerHTML = '&#9654; Play'; finish(); return; }
  raf = requestAnimationFrame(frame);
}

function finish() {
  document.getElementById('endbig').textContent =
    `${Math.round(S.duty_base)}% → ${Math.round(S.duty_ts)}%`;
  document.getElementById('endduty').textContent =
    `Avg duty cycle per pool, baseline → time-sliced ` +
    `(sampler ${Math.round(S.duty_detail.sampler[0])}%→${Math.round(S.duty_detail.sampler[1])}%, ` +
    `trainer ${Math.round(S.duty_detail.trainer[0])}%→${Math.round(S.duty_detail.trainer[1])}%)`;
  document.getElementById('endstats').textContent =
    `Throughput: ${S.speedup}× — 2 jobs in ${S.ts_min} min vs ${(2*S.base_min).toFixed(0)} min back-to-back ` +
    `(${S.swaps} C/R swaps · avg snapshot ${fmtMs(S.avg_ckpt_ms)} · avg restore ${fmtMs(S.avg_restore_ms)})`;
  document.getElementById('endcost').textContent =
    `Cost: ${S.cost_saving_pct}% fewer TPU-hours per job on the same hardware`;
  endcard.style.display = 'flex';
}

btn.onclick = () => {
  if (now >= DUR) reset();
  playing = !playing;
  btn.innerHTML = playing ? '&#9646;&#9646; Pause' : '&#9654; Play';
  if (playing) { lastFrame = performance.now(); raf = requestAnimationFrame(frame); }
};
function reset() {
  now = 0; evIdx = 0; ckSum = ckN = rsSum = rsN = 0;
  for (const k in pending) delete pending[k];
  feedOrch.innerHTML = ''; feedAgent.innerHTML = ''; endcard.style.display = 'none';
  baseDone = false;
  document.getElementById('b_done_lbl').textContent = 'Job A running…';
  document.getElementById('b_done').textContent = '–';
  document.getElementById('b_proj').textContent = '–';
  for (const p of panels) p.draw(0);
  tiles();
}
document.getElementById('replay').onclick = () => { reset(); btn.click(); };

{
  const rows = D.timeslice.events.map(e =>
    `<tr><td>${fmtT(e.t)}</td><td>Job ${e.job}</td><td>${e.pool}</td><td>${e.kind}</td>` +
    `<td>${e.ms ? fmtMs(e.ms) : e.ckpt_ms ? fmtMs(e.ckpt_ms) : e.restore_ms ? fmtMs(e.restore_ms) : ''}</td></tr>`).join('');
  document.getElementById('tblwrap').innerHTML =
    `<table><tr><th>t</th><th>job</th><th>pool</th><th>event</th><th>C/R time</th></tr>${rows}</table>`;
}

reset();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
