#!/usr/bin/env python3
"""Build vd-campaign-plots.html: self-contained SVG dashboard for the verl DAPO
32-GPU elastic campaign (genbound2). Python stdlib only, no external deps.

Truthful-plotting contract: every mark is parsed from the local measured
artifacts under genbound2/results/ (trajectory.tsv, *_switch_timings.jsonl,
*_window_invariants.jsonl, *_decisions.jsonl, *_train.log wait lines). The only
report-sourced numbers are the GPU-duty means (raw 1Hz CSVs live on-node only)
and they are labeled as such. Nothing is interpolated or fabricated.
"""
import html
import json
import math
import re
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent          # .../genbound2
RES = BASE / "results"
OUT = Path(__file__).resolve().parent / "vd-campaign-plots.html"

# ---------------------------------------------------------------- palette ----
# Google Material palette per campaign convention. CVD-validated (OKLab dE*100,
# Machado 100% severity sims): arms trio {red, blue, light-blue} min pair 11.8;
# regime trio {gray, #188038, blue} min pair 11.5 (all >= 8 target).
RED = "#EA4335"      # trainer / baseline
BLUE = "#4285F4"     # elastic (S4b, headline arm)
LBLUE = "#8AB4F8"    # elastic first arm (S4a)
GREEN = "#34A853"    # samplers
DGREEN = "#188038"   # Math-7B in regime panel (darker green: tritan-safe vs blue)
YELLOW = "#FBBC04"   # highlights
DYELLOW = "#F9AB00"  # highlight lines / failsafe
GRAY = "#9AA0A6"     # neutral
INK = "#202124"
INK2 = "#5F6368"
GRID = "#E8EAED"

TRAJ_COLS = ["ver", "len", "step_s", "gen_s", "upd_s", "tr_idle", "ro_idle",
             "wait_s", "mfu", "reward", "ratio"]

# ---------------------------------------------------------------- parsers ----

def parse_traj(path):
    """Parse a trajectory.tsv into a list of runs; each run = list of row dicts
    (ver >= 1 only; ver 0 is the pre-train placeholder row)."""
    runs, cur = [], None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("---"):
            continue
        parts = line.split("\t")
        if parts[0] == "ver":                    # header starts a new run
            cur = []
            runs.append(cur)
            continue
        if cur is None:                          # headerless file (s3/s4/s4b)
            cur = []
            runs.append(cur)
        if parts[0] == "0":
            continue
        row = {}
        for k, v in zip(TRAJ_COLS, parts):
            try:
                row[k] = float(v)
            except ValueError:
                row[k] = None
        cur.append(row)
    return runs


WAIT_RE = re.compile(
    r"Loop collection completed: [\d/]+ samples, total wait time: ([\d.]+) seconds")

def parse_waits(log_path):
    """Per-micro-step trainer collection waits from a train.log."""
    out = []
    with open(log_path, errors="replace") as f:
        for line in f:
            m = WAIT_RE.search(line)
            if m:
                out.append(float(m.group(1)))
    return out


def parse_switch(path):
    ins, backs = [], []
    for line in open(path):
        d = json.loads(line)
        rec = {"total": d["total_seconds"],
               "phases": {p["phase"]: p["seconds"] for p in d["phases"]},
               "loaded": d.get("loaded"), "window": d.get("window")}
        (ins if d["operation"] == "switch-to-spares" else backs).append(rec)
    return ins, backs


def parse_invariants(path):
    ok = tot = 0
    for line in open(path):
        d = json.loads(line)
        tot += 1
        ok += bool(d.get("invariant_ok"))
    return ok, tot


def parse_close_reasons(path):
    counts = {}
    max_dropped = 0
    for line in open(path):
        d = json.loads(line)
        if d.get("action") == "switch_to_trainer":
            r = d.get("reason", "?")
            counts[r] = counts.get(r, 0) + 1
        sig = d.get("signals")
        if sig and isinstance(sig.get("dropped_samples"), (int, float)):
            max_dropped = max(max_dropped, sig["dropped_samples"])
    return counts, max_dropped


def parse_s1_7b(path):
    """s1-7b/trajectory_and_waits.txt = trajectory block + unlabeled block of
    per-micro-step waits (10 blocks x 4 micro-steps; cross-checked against
    VD-STATE.md 'p50 micro-step wait 4s, p90 24s')."""
    traj, waits = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("ver"):
            continue
        parts = line.split("\t")
        if len(parts) >= 11:
            if parts[0] == "0":
                continue
            traj.append({k: float(v) if v != "None" else None
                         for k, v in zip(TRAJ_COLS, parts)})
        elif len(parts) == 1:
            try:
                waits.append(float(parts[0]))
            except ValueError:
                pass
    return traj, waits


# ------------------------------------------------------------------ stats ----

def p50(v):
    return statistics.median(v)

def p90(v):
    """Nearest-rank p90 (matches the campaign report's convention closely)."""
    s = sorted(v)
    return s[math.ceil(0.9 * len(s)) - 1]

def fmt(v, nd=1):
    return f"{v:,.{nd}f}".rstrip("0").rstrip(".") if nd else f"{v:,.0f}"

def pct(new, old):
    return (new - old) / old * 100.0

def fpct(x, nd=1):
    sign = "−" if x < 0 else "+"
    return f"{sign}{abs(x):.{nd}f}%"

def esc(s):
    return html.escape(str(s), quote=True)


# ------------------------------------------------------------ svg helpers ----

def T(x, y, s, size=11, fill=INK2, anchor="start", weight=None, rotate=None,
      mono=False, opacity=None):
    style = f'font-size:{size}px;fill:{fill}'
    if weight:
        style += f';font-weight:{weight}'
    if mono:
        style += ";font-family:'SF Mono',Menlo,Consolas,monospace"
    extra = ""
    if rotate is not None:
        extra += f' transform="rotate({rotate} {x:.1f} {y:.1f})"'
    if opacity is not None:
        extra += f' opacity="{opacity}"'
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'style="{style}"{extra}>{esc(s)}</text>')


def bar_path(x, y, w, h, color, title=None, r=3.0):
    """Bar with rounded top corners, anchored to the baseline."""
    r = max(0.0, min(r, w / 2, h))
    d = (f"M{x:.2f},{y + h:.2f} v{-(h - r):.2f} q0,{-r:.2f} {r:.2f},{-r:.2f} "
         f"h{w - 2 * r:.2f} q{r:.2f},0 {r:.2f},{r:.2f} v{h - r:.2f} z")
    t = f"<title>{esc(title)}</title>" if title else ""
    return f'<path d="{d}" fill="{color}">{t}</path>'


def polyline(pts, color, width=2, dash=None):
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    dd = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<polyline points="{d}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linejoin="round" '
            f'stroke-linecap="round"{dd}/>')


def dot(x, y, color, r=3.0, title=None, ring=True):
    t = f"<title>{esc(title)}</title>" if title else ""
    stroke = ' stroke="#fff" stroke-width="2"' if ring else ""
    # oversized transparent hit target + visible mark
    return (f'<g>{t}<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="transparent"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}"{stroke}/></g>')


def spread(ys, min_gap=16.0):
    """Nudge label y-positions apart so right-edge series labels never collide."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    adj = [ys[i] for i in order]
    for k in range(1, len(adj)):
        if adj[k] - adj[k - 1] < min_gap:
            adj[k] = adj[k - 1] + min_gap
    out = [0.0] * len(ys)
    for k, i in enumerate(order):
        out[i] = adj[k]
    return out


def legend(x, y, entries, size=11):
    out, cx = [], x
    for label, color in entries:
        out.append(f'<rect x="{cx}" y="{y - 9}" width="10" height="10" rx="2" fill="{color}"/>')
        out.append(T(cx + 15, y, label, size=size, fill=INK2))
        cx += 15 + len(label) * size * 0.58 + 22
    return "".join(out)


def y_grid(x0, x1, vals, vmax, vmin, py0, py1, fmt_fn=None, unit=""):
    """Horizontal gridlines + tick labels. py0 = y at vmin, py1 = y at vmax."""
    out = []
    for v in vals:
        y = py0 + (v - vmin) / (vmax - vmin) * (py1 - py0)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" '
                   f'stroke="{GRID}" stroke-width="1"/>')
        lbl = fmt_fn(v) if fmt_fn else fmt(v, 0)
        out.append(T(x0 - 8, y + 4, f"{lbl}{unit}", size=10.5, anchor="end"))
    return "".join(out)


def svg_open(w, h):
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" '
            f'style="max-width:{w}px;display:block" role="img" '
            f'xmlns="http://www.w3.org/2000/svg">')


# ------------------------------------------------------------- load data -----

s3 = parse_traj(RES / "s3" / "trajectory.tsv")[0]        # 12 rows, v1..v12
s4 = parse_traj(RES / "s4" / "trajectory.tsv")[0]
s4b = parse_traj(RES / "s4b" / "trajectory.tsv")[0]
s130_runs = parse_traj(RES / "s1-30b" / "trajectory.tsv")  # [S1 (2v), S1b (3v)]
s17_traj, s17_waits = parse_s1_7b(RES / "s1-7b" / "trajectory_and_waits.txt")

w_s3 = parse_waits(RES / "s3" / "s3_train.log")
w_s4 = parse_waits(RES / "s4" / "s4_train.log")
w_s4b = parse_waits(RES / "s4b" / "s4b_train.log")

sw4_in, sw4_back = parse_switch(RES / "s4" / "s4_switch_timings.jsonl")
sw4b_in, sw4b_back = parse_switch(RES / "s4b" / "s4b_switch_timings.jsonl")

inv4 = parse_invariants(RES / "s4" / "s4_window_invariants.jsonl")
inv4b = parse_invariants(RES / "s4b" / "s4b_window_invariants.jsonl")

close4, drop4 = parse_close_reasons(RES / "s4" / "s4_decisions.jsonl")
close4b, drop4b = parse_close_reasons(RES / "s4b" / "s4b_decisions.jsonl")

NV = 12
step3 = [r["step_s"] for r in s3]
step4 = [r["step_s"] for r in s4]
step4b = [r["step_s"] for r in s4b]
tot3, tot4, tot4b = sum(step3), sum(step4), sum(step4b)
SAMPLES = 12 * 4 * 2048          # 48 micro-steps x 2048 responses = 98304
GPUS = 32
sph3 = SAMPLES / (GPUS * tot3 / 3600)
sph4 = SAMPLES / (GPUS * tot4 / 3600)
sph4b = SAMPLES / (GPUS * tot4b / 3600)

wait3 = [r["wait_s"] for r in s3]
wait4 = [r["wait_s"] for r in s4]
wait4b = [r["wait_s"] for r in s4b]

srcbase = "results/"
CARD = []      # accumulated card html
NOTES = []     # data-quality notes


def card(title, source, body, caption):
    CARD.append(
        f'<section class="card"><h2>{esc(title)}</h2>'
        f'<div class="src">source: {esc(source)}</div>'
        f'{body}<p class="cap">{caption}</p></section>')


# =================================================================== chart 1 =
def chart_step_bars():
    W, H = 1080, 400
    L, Rm, Tm, Bm = 58, 14, 40, 44
    x0, x1, py0, py1 = L, W - Rm, H - Bm, Tm + 14
    vmax = 900.0
    parts = [svg_open(W, H)]
    parts.append(legend(L, Tm - 22 + 14, [
        ("baseline (S3)", RED), ("elastic S4a", LBLUE), ("elastic S4b", BLUE)]))
    parts.append(y_grid(x0, x1, [0, 150, 300, 450, 600, 750, 900], vmax, 0, py0, py1))
    parts.append(T(16, (py0 + py1) / 2, "seconds", size=11, anchor="middle", rotate=-90))
    slot = (x1 - x0) / NV
    bw = min(24.0, (slot - 14) / 3 - 2)
    for i in range(NV):
        gx = x0 + i * slot + (slot - (3 * bw + 4)) / 2
        vals = [(step3[i], RED, "baseline", None),
                (step4[i], LBLUE, "S4a", pct(step4[i], step3[i])),
                (step4b[i], BLUE, "S4b", pct(step4b[i], step3[i]))]
        for j, (v, c, name, d) in enumerate(vals):
            bx = gx + j * (bw + 2)
            bh = v / vmax * (py0 - py1)
            by = py0 - bh
            tt = f"v{i+1} {name}: {v:.1f} s" + (f" ({fpct(d)})" if d is not None else "")
            parts.append(bar_path(bx, by, bw, bh, c, title=tt))
            if d is not None:
                parts.append(T(bx + bw / 2 + 3.5, by - 5, fpct(d), size=9,
                               fill=INK2, rotate=-90))
        lbl = f"v{i+1}" + ("*" if i == NV - 1 else "")
        parts.append(T(gx + (3 * bw + 4) / 2, py0 + 16, lbl, size=11, anchor="middle"))
    parts.append(f'<line x1="{x0}" y1="{py0}" x2="{x1}" y2="{py0}" stroke="{GRAY}" stroke-width="1"/>')
    parts.append(T(x1, py0 + 32, "* v12 is the standard termination flush in all arms (partial block)",
                   size=10, anchor="end", fill=INK2))
    parts.append("</svg>")
    card("1 · Per-sync-block step time, baseline vs both elastic arms",
         "results/{s3,s4,s4b}/trajectory.tsv (col step_s); deltas computed vs baseline",
         "".join(parts),
         f"Elastic time-slicing cuts every block's wall: totals v1-12 are baseline {fmt(tot3,0)} s "
         f"vs S4a {fmt(tot4,0)} s ({fpct(pct(tot4, tot3), 2)}) vs S4b {fmt(tot4b,0)} s "
         f"({fpct(pct(tot4b, tot3), 2)}); best single block {fpct(min(pct(step4[i],step3[i]) for i in range(NV)),1)} (S4a v10). "
         f"S4a's controller was killed ~3 min early by an operator error, so its block 12 ran windowless (conservative bias).")


# =================================================================== chart 2 =
def chart_cumulative():
    W, H = 1080, 380
    L, Rm, Tm, Bm = 66, 320, 34, 40
    x0, x1, py0, py1 = L, W - Rm, H - Bm, Tm
    vmax = 9000.0
    parts = [svg_open(W, H)]
    parts.append(y_grid(x0, x1, [0, 1500, 3000, 4500, 6000, 7500, 9000], vmax, 0, py0, py1,
                        fmt_fn=lambda v: fmt(v, 0)))
    parts.append(T(16, (py0 + py1) / 2, "cumulative seconds", size=11, anchor="middle", rotate=-90))
    def xs(i):  # i = 0..12
        return x0 + i / NV * (x1 - x0)
    for i in range(NV + 1):
        parts.append(T(xs(i), py0 + 16, "v%d" % i if i else "0", size=10, anchor="middle"))
    series = [("baseline (S3)", step3, RED, tot3, sph3),
              ("elastic S4a", step4, LBLUE, tot4, sph4),
              ("elastic S4b", step4b, BLUE, tot4b, sph4b)]
    end_labels = []
    for name, steps, color, tot, sph in series:
        cum, acc = [0.0], 0.0
        for v in steps:
            acc += v
            cum.append(acc)
        pts = [(xs(i), py0 - c / vmax * (py0 - py1)) for i, c in enumerate(cum)]
        parts.append(polyline(pts, color))
        for i, (px, py) in enumerate(pts):
            if i:
                parts.append(dot(px, py, color, r=3,
                                 title=f"{name} through v{i}: {fmt(cum[i],0)} s"))
        end_labels.append((pts[-1][1],
                           f"{name}: {fmt(tot,0)} s · {fmt(sph,0)} samples/GPU-hr", color))
    for (ey, lab, color), ly in zip(end_labels, spread([e[0] for e in end_labels])):
        parts.append(f'<line x1="{x1+2}" y1="{ey:.1f}" x2="{x1+8}" y2="{ly:.1f}" '
                     f'stroke="{color}" stroke-width="1.5"/>')
        parts.append(T(x1 + 12, ly + 4, lab, size=11.5, fill=INK, weight=600))
    parts.append(f'<line x1="{x0}" y1="{py0}" x2="{x1}" y2="{py0}" stroke="{GRAY}"/>')
    parts.append("</svg>")
    card("2 · Cumulative wall-clock to 12 param syncs, and samples/GPU-hour",
         "results/{s3,s4,s4b}/trajectory.tsv (cumulative step_s); samples/GPU-hr = 98,304 responses / 32 GPUs / wall",
         "".join(parts),
         f"Same work (48 micro-steps x 2048 responses) finishes {fmt(tot3-tot4b,0)} s sooner on S4b: throughput "
         f"{fmt(sph3,0)} → {fmt(sph4,0)} ({fpct(pct(sph4,sph3),2)}, S4a) → {fmt(sph4b,0)} samples/GPU-hr "
         f"({fpct(pct(sph4b,sph3),2)}, S4b), reproducing ~+11% across two independent elastic arms.")


# =================================================================== chart 3 =
def chart_idle():
    W, H = 1080, 360
    L, Rm, Tm, Bm = 58, 190, 34, 40
    x0, x1, py0, py1 = L, W - Rm, H - Bm, Tm
    vmax = 50.0
    parts = [svg_open(W, H)]
    parts.append(y_grid(x0, x1, [0, 10, 20, 30, 40, 50], vmax, 0, py0, py1, unit="%"))
    def xs(i):
        return x0 + i / (NV - 1) * (x1 - x0)
    for i in range(NV):
        parts.append(T(xs(i), py0 + 16, f"v{i+1}" + ("*" if i == NV - 1 else ""),
                       size=10, anchor="middle"))
    series = [("baseline (S3)", s3, RED), ("elastic S4a", s4, LBLUE),
              ("elastic S4b", s4b, BLUE)]
    end_labels = []
    for name, rows, color in series:
        vals = [r["tr_idle"] * 100 for r in rows]
        pts = [(xs(i), py0 - v / vmax * (py0 - py1)) for i, v in enumerate(vals)]
        parts.append(polyline(pts, color))
        for i, (px, py) in enumerate(pts):
            parts.append(dot(px, py, color, r=3, title=f"{name} v{i+1}: {vals[i]:.1f}%"))
        end_labels.append((pts[-1][1], name, color))
    for (ey, lab, color), ly in zip(end_labels, spread([e[0] for e in end_labels])):
        parts.append(f'<line x1="{x1+2}" y1="{ey:.1f}" x2="{x1+8}" y2="{ly:.1f}" '
                     f'stroke="{color}" stroke-width="1.5"/>')
        parts.append(T(x1 + 12, ly + 4, lab, size=11.5, fill=INK, weight=600))
    b = [r["tr_idle"] * 100 for r in s3]
    parts.append(T(xs(0) + 8, py0 - b[0] / vmax * (py0 - py1) - 8, f"{b[0]:.1f}%",
                   size=11, fill=RED, weight=600))
    parts.append(T(xs(10), py0 - b[10] / vmax * (py0 - py1) - 10, f"{b[10]:.1f}%",
                   size=11, fill=RED, weight=600, anchor="middle"))
    parts.append(f'<line x1="{x0}" y1="{py0}" x2="{x1}" y2="{py0}" stroke="{GRAY}"/>')
    parts.append(T(x1, py0 + 32, "* v12 = termination flush (no gen-wait tail)", size=10, anchor="end"))
    parts.append("</svg>")
    m3 = statistics.mean(r["tr_idle"] for r in s3[:11]) * 100
    m4b = statistics.mean(r["tr_idle"] for r in s4b[:11]) * 100
    card("3 · Trainer idle share per sync block: the narrowing harvest pool",
         "results/{s3,s4,s4b}/trajectory.tsv (col tr_idle)",
         "".join(parts),
         f"The baseline's harvestable idle narrows from {b[0]:.1f}% (v1) toward ~{b[10]:.0f}% as DAPO's response "
         f"length falls; the elastic arms run below it because windows convert part of that idle into serving "
         f"(mean v1-11: baseline {m3:.1f}% → S4b {m4b:.1f}%). Elastic lines show residual idle, not harvested idle.")


# =================================================================== chart 4 =
def strip_panel(title_txt, rows, xmax, w=520, unit_note=None):
    """rows = [(label, values, color)]"""
    H = 96 + 64 * len(rows)
    L, Rm = 96, 20
    x0, x1 = L, w - Rm
    parts = [svg_open(w, H)]
    parts.append(T(x0, 22, title_txt, size=12.5, fill=INK, weight=600))
    ticks = [0, xmax / 4, xmax / 2, 3 * xmax / 4, xmax]
    base_y = 44
    bot = base_y + 64 * len(rows)
    for tv in ticks:
        tx = x0 + tv / xmax * (x1 - x0)
        parts.append(f'<line x1="{tx:.1f}" y1="{base_y}" x2="{tx:.1f}" y2="{bot}" stroke="{GRID}"/>')
        parts.append(T(tx, bot + 16, fmt(tv, 1), size=10, anchor="middle"))
    parts.append(T((x0 + x1) / 2, bot + 32, "seconds", size=10.5, anchor="middle"))
    for r_i, (label, vals, color) in enumerate(rows):
        cy = base_y + 64 * r_i + 32
        parts.append(T(x0 - 8, cy + 4, f"{label} (n={len(vals)})", size=11, anchor="end", fill=INK))
        for i, v in enumerate(sorted(vals)):
            jy = cy + ((i % 7) - 3) * 5.2
            px = x0 + min(v, xmax) / xmax * (x1 - x0)
            parts.append(dot(px, jy, color, r=4, title=f"{label} window: {v:.2f} s"))
        m, p9 = p50(vals), p90(vals)
        mx = x0 + m / xmax * (x1 - x0)
        px9 = x0 + p9 / xmax * (x1 - x0)
        parts.append(f'<line x1="{mx:.1f}" y1="{cy-22}" x2="{mx:.1f}" y2="{cy+22}" stroke="{INK}" stroke-width="2"/>')
        parts.append(T(mx + 4, cy - 24, f"p50 {m:.2f}s", size=10, fill=INK, weight=600))
        parts.append(f'<line x1="{px9:.1f}" y1="{cy-18}" x2="{px9:.1f}" y2="{cy+18}" stroke="{INK2}" stroke-width="1.5" stroke-dasharray="3,3"/>')
        parts.append(T(px9 + 4, cy + 26, f"p90 {p9:.2f}s", size=10, fill=INK2))
    parts.append("</svg>")
    return "".join(parts)


def chart_switch():
    a = strip_panel("Switch-in wall (trainer → spares serve)",
                    [("S4a", [r["total"] for r in sw4_in], LBLUE),
                     ("S4b", [r["total"] for r in sw4b_in], BLUE)], 4.0)
    b = strip_panel("Switch-back wall (spares park → trainer)",
                    [("S4a", [r["total"] for r in sw4_back], LBLUE),
                     ("S4b", [r["total"] for r in sw4b_back], BLUE)], 20.0)
    body = f'<div class="row">{a}{b}</div>'
    card("4 · Switch walls across all recorded elastic windows",
         "results/s4/s4_switch_timings.jsonl (27 windows), results/s4b/s4b_switch_timings.jsonl (26 windows)",
         body,
         "Entering a window costs ~1.3-3.2 s (the ~2.4 s cluster = windows that also load a fresh weight "
         "version); leaving costs ~1-3 s typically, with a long tail up to 18.7 s from engine-side aborts of "
         "long in-flight sequences (tokens retained by partial rollout). Percentiles recomputed from raw logs; "
         "the report quotes S4b p50 1.46 / 2.98 s under its own percentile convention.")


# ================================================================== chart 4b =
PHASE_ORDER = ["trainer_hold", "trainer_release_cache", "spares_wake",
               "spares_load_wcache", "spares_stamp_version",
               "spares_resume_generation", "lb_add_spares", "raise_concurrency",
               "lb_remove_spares", "restore_concurrency",
               "abort_spares_partial_rollout", "spares_drain", "spares_sleep_L1",
               "trainer_release"]
# one distinct color per phase (a phase name appears once in the legend)
PHASE_COLOR = {
    "trainer_hold": GRAY, "trainer_release_cache": RED,
    "spares_wake": GREEN, "spares_load_wcache": DYELLOW,
    "spares_stamp_version": "#CEEAD6", "spares_resume_generation": "#A8DAB5",
    "lb_add_spares": "#669DF6", "raise_concurrency": "#AECBFA",
    "lb_remove_spares": "#174EA6", "restore_concurrency": "#D2E3FC",
    "abort_spares_partial_rollout": "#A50E0E", "spares_drain": YELLOW,
    "spares_sleep_L1": DGREEN, "trainer_release": "#DADCE0",
}

def chart_phases():
    rows = []
    loaded = [r for r in sw4b_in if r["loaded"]]
    stamp = [r for r in sw4b_in if not r["loaded"]]
    rows.append((f"switch-in, new weights (n={len(loaded)})", loaded))
    rows.append((f"switch-in, stamp-only (n={len(stamp)})", stamp))
    rows.append((f"switch-back (n={len(sw4b_back)})", sw4b_back))
    W, H = 1080, 66 + 46 * len(rows) + 60
    L, Rm = 250, 130
    x0, x1 = L, W - Rm
    xmax = 4.0
    parts = [svg_open(W, H)]
    for tv in [0, 1, 2, 3, 4]:
        tx = x0 + tv / xmax * (x1 - x0)
        parts.append(f'<line x1="{tx:.1f}" y1="30" x2="{tx:.1f}" y2="{30+46*len(rows)}" stroke="{GRID}"/>')
        parts.append(T(tx, 30 + 46 * len(rows) + 16, str(tv), size=10, anchor="middle"))
    parts.append(T((x0 + x1) / 2, 30 + 46 * len(rows) + 32, "median seconds per phase", size=10.5, anchor="middle"))
    used_phases = []
    for r_i, (label, recs) in enumerate(rows):
        y = 38 + 46 * r_i
        parts.append(T(x0 - 10, y + 15, label, size=11, anchor="end", fill=INK))
        cx = x0
        phases = [p for p in PHASE_ORDER if any(p in r["phases"] for r in recs)]
        totm = p50([r["total"] for r in recs])
        for ph in phases:
            vals = [r["phases"].get(ph, 0.0) for r in recs]
            m = p50(vals)
            if m <= 0:
                continue
            wpx = m / xmax * (x1 - x0)
            parts.append(f'<rect x="{cx:.1f}" y="{y}" width="{max(wpx-1.2,0.8):.1f}" height="22" rx="2" '
                         f'fill="{PHASE_COLOR.get(ph, GRAY)}">'
                         f'<title>{esc(ph)}: median {m:.2f} s</title></rect>')
            if ph not in [u[0] for u in used_phases]:
                used_phases.append((ph, PHASE_COLOR.get(ph, GRAY)))
            cx += wpx
        parts.append(T(cx + 8, y + 15, f"∑ median total {totm:.2f}s", size=10.5, fill=INK2))
    ly = 30 + 46 * len(rows) + 52
    cx = x0 - 190
    for ph, color in used_phases:
        parts.append(f'<rect x="{cx}" y="{ly-9}" width="10" height="10" rx="2" fill="{color}"/>')
        lab = ph.replace("_", " ")
        parts.append(T(cx + 14, ly, lab, size=9.5))
        cx += 14 + len(lab) * 5.6 + 16
        if cx > W - 200:
            cx = x0 - 190
            ly += 18
    parts_svg = "".join(parts) + "</svg>"
    card("4b · Where the switch walls go: median phase breakdown (S4b)",
         "results/s4b/s4b_switch_timings.jsonl (per-phase walls)",
         parts_svg,
         "The stamp-only switch-in is dominated by trainer cache release + vLLM wake_up (~1.4 s); windows that "
         "land a new param version add ~2.4 s of wcache load. Switch-back medians are drain + sleep; its p90 "
         "tail (chart 4) comes from aborting long in-flight sequences.")


# =================================================================== chart 5 =
def chart_reward():
    W, H = 1080, 360
    L, Rm, Tm, Bm = 66, 190, 34, 40
    x0, x1, py0, py1 = L, W - Rm, H - Bm, Tm
    vmin, vmax = -0.25, 0.50
    parts = [svg_open(W, H)]
    parts.append(y_grid(x0, x1, [-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5], vmax, vmin,
                        py0, py1, fmt_fn=lambda v: f"{v:+.1f}" if v else "0"))
    zy = py0 + (0 - vmin) / (vmax - vmin) * (py1 - py0)
    parts.append(f'<line x1="{x0}" y1="{zy:.1f}" x2="{x1}" y2="{zy:.1f}" stroke="{GRAY}" stroke-width="1"/>')
    def xs(i):
        return x0 + i / (NV - 1) * (x1 - x0)
    for i in range(NV):
        parts.append(T(xs(i), py0 + 16, f"v{i+1}", size=10, anchor="middle"))
    end_labels = []
    for name, rows, color in [("baseline (S3)", s3, RED), ("elastic S4a", s4, LBLUE),
                              ("elastic S4b", s4b, BLUE)]:
        vals = [r["reward"] for r in rows]
        pts = [(xs(i), py0 + (v - vmin) / (vmax - vmin) * (py1 - py0)) for i, v in enumerate(vals)]
        parts.append(polyline(pts, color))
        for i, (px, py) in enumerate(pts):
            parts.append(dot(px, py, color, r=3, title=f"{name} v{i+1}: reward {vals[i]:+.3f}"))
        end_labels.append((pts[-1][1], name, color))
    for (ey, lab, color), ly in zip(end_labels, spread([e[0] for e in end_labels])):
        parts.append(f'<line x1="{x1+2}" y1="{ey:.1f}" x2="{x1+8}" y2="{ly:.1f}" '
                     f'stroke="{color}" stroke-width="1.5"/>')
        parts.append(T(x1 + 12, ly + 4, lab, size=11.5, fill=INK, weight=600))
    parts.append(T(16, (py0 + py1) / 2, "mean reward", size=11, anchor="middle", rotate=-90))
    parts.append("</svg>")
    card("5 · Reward trajectory overlay: the no-harm evidence",
         "results/{s3,s4,s4b}/trajectory.tsv (col reward)",
         "".join(parts),
         f"All three arms climb the same band ({s3[0]['reward']:+.2f} → {max(r['reward'] for r in s3):+.2f} "
         f"baseline; {s4[0]['reward']:+.2f} / {s4b[0]['reward']:+.2f} → "
         f"{max(r['reward'] for r in s4):+.2f} / {max(r['reward'] for r in s4b):+.2f} elastic): serving spare "
         "windows off truthfully-stamped weight versions does not perturb learning.")


# =================================================================== chart 6 =
def mini_bars(title_txt, groups, unit, w=352, vmax=None, ref=None, nd=1,
              not_logged=None):
    """groups = [(label, color, mean, [per-version values])]"""
    H = 300
    L, Rm, Tm, Bm = 58, 12, 40, 58
    x0, x1, py0, py1 = L, w - Rm, H - Bm, Tm + 6
    if vmax is None:
        vmax = max([g[2] for g in groups if g[2] is not None] +
                   [v for g in groups if g[3] for v in g[3]] + ([ref] if ref else [])) * 1.18
    parts = [svg_open(w, H)]
    parts.append(T(x0, 22, title_txt, size=12.5, fill=INK, weight=600))
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        v = frac * vmax
        y = py0 - frac * (py0 - py1)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{GRID}"/>')
        parts.append(T(x0 - 6, y + 4, fmt(v, nd if vmax < 100 else 0), size=9.5, anchor="end"))
    slot = (x1 - x0) / len(groups)
    bw = min(64, slot - 30)
    for i, (label, color, meanv, pervals) in enumerate(groups):
        cxm = x0 + i * slot + slot / 2
        if meanv is None:
            parts.append(T(cxm, (py0 + py1) / 2, not_logged or "not logged", size=10,
                           anchor="middle", fill=INK2))
            parts.append(T(cxm, (py0 + py1) / 2 + 14, "in artifacts", size=10,
                           anchor="middle", fill=INK2))
        else:
            bh = meanv / vmax * (py0 - py1)
            parts.append(bar_path(cxm - bw / 2, py0 - bh, bw, bh, color,
                                  title=f"{label}: {fmt(meanv, 2)} {unit}"))
            top = max([meanv] + (pervals or []))
            parts.append(T(cxm, py0 - top / vmax * (py0 - py1) - 9, fmt(meanv, nd),
                           size=11, anchor="middle", fill=INK, weight=600))
            if pervals:
                for j, v in enumerate(pervals):
                    jx = cxm + ((j % 5) - 2) * 6
                    jy = py0 - v / vmax * (py0 - py1)
                    parts.append(dot(jx, jy, INK, r=2.2, ring=False,
                                     title=f"{label} per-version: {fmt(v, 2)} {unit}"))
        for li, ln in enumerate(label.split("\n")):
            parts.append(T(cxm, py0 + 16 + 13 * li, ln, size=10.5, anchor="middle", fill=INK))
    if ref is not None:
        ry = py0 - ref / vmax * (py0 - py1)
        parts.append(f'<line x1="{x0}" y1="{ry:.1f}" x2="{x1}" y2="{ry:.1f}" '
                     f'stroke="{DYELLOW}" stroke-width="2" stroke-dasharray="5,4"/>')
        parts.append(T(x0 + 4, ry - 20, "switch round-trip", size=9.5, fill="#B06000", weight=600))
        parts.append(T(x0 + 4, ry - 8, f"p50 ≈{ref:.1f}s", size=9.5, fill="#B06000", weight=600))
    parts.append(f'<line x1="{x0}" y1="{py0}" x2="{x1}" y2="{py0}" stroke="{GRAY}"/>')
    parts.append("</svg>")
    return "".join(parts)


def chart_regime():
    s1b30 = s130_runs[1]                      # S1b probe, v1-3
    idle30 = [r["tr_idle"] * 100 for r in s1b30]
    idle7 = [r["tr_idle"] * 100 for r in s17_traj]
    idleR = [r["tr_idle"] * 100 for r in s3[:11]]      # scored baseline v1-11
    len30 = [r["len"] for r in s1b30]
    len7 = [r["len"] for r in s17_traj]
    lenR = [r["len"] for r in s3[:11]]
    toll = p50([r["total"] for r in sw4b_in]) + p50([r["total"] for r in sw4b_back])
    g_idle = mini_bars("Trainer idle share (%)", [
        ("30B-A3B base\n(S1b, v1-3)", GRAY, statistics.mean(idle30), idle30),
        ("Qwen2.5-Math-7B\n(S1b, v1-9)", DGREEN, statistics.mean(idle7), idle7),
        ("R1-Distill-7B\n(S3, v1-11)", BLUE, statistics.mean(idleR), idleR)], "%", vmax=50)
    g_wait = mini_bars("Median micro-step gen-wait (s)", [
        ("30B-A3B base", GRAY, None, None),
        ("Qwen2.5-Math-7B\n(40 micro-steps)", DGREEN, p50(s17_waits), None),
        ("R1-Distill-7B\n(48 micro-steps)", BLUE, p50(w_s3), None)],
        "s", vmax=48, ref=toll, not_logged="micro-step waits")
    g_len = mini_bars("Mean response length (tokens)", [
        ("30B-A3B base", GRAY, statistics.mean(len30), len30),
        ("Qwen2.5-Math-7B", DGREEN, statistics.mean(len7), len7),
        ("R1-Distill-7B", BLUE, statistics.mean(lenR), lenR)], "tok", nd=0, vmax=8000)
    body = f'<div class="row">{g_idle}{g_wait}{g_len}</div>'
    card("6 · Regime study: which workloads are elastic-eligible at a shipped 16:16 split",
         "results/s1-30b/trajectory.tsv (S1b run), results/s1-7b/trajectory_and_waits.txt, results/s3/{trajectory.tsv,s3_train.log}; toll from s4b_switch_timings.jsonl",
         body,
         f"Eligibility needs both a real idle pool AND waits that dwarf the ~{toll:.1f} s switch round-trip: "
         "the 30B recipe's warmup regime is train-bound (idle 3-8%, ~890-token responses, micro-step waits not "
         "logged because there was no pool), stock Math-7B has idle but fragmented ~4 s waits, and long-CoT "
         "R1-Distill-7B has both (idle 32% mean, median wait ~40 s, ~6.6K-token responses): it passed the "
         "registered gate (idle ≥ 30%, median wait ≥ 30 s) and became the scored workload. "
         "Dots = per-version measurements behind each mean.")


# =================================================================== chart 7 =
def chart_fragmentation():
    body = strip_panel(
        "Per-micro-step trainer gen-wait, stock Math-7B vs R1-Distill-7B",
        [("Math-7B", s17_waits, DGREEN),
         ("R1-Distill", w_s3, BLUE)], 240.0, w=1060)
    card("7 · Wait fragmentation: why Math-7B fails the gate and R1-Distill passes",
         "results/s1-7b/trajectory_and_waits.txt (waits block, 40 micro-steps v1-10) and results/s3/s3_train.log (collection-wait lines, 48 micro-steps)",
         body,
         f"Math-7B's waits are fragmented: p50 {p50(s17_waits):.1f} s / p90 {p90(s17_waits):.1f} s, one modest "
         f"pre-sync wait per block and ~4 s otherwise, so a ~4.4 s switch round-trip eats the gain. R1-Distill "
         f"consolidates: p50 {p50(w_s3):.1f} s / p90 {p90(w_s3):.1f} s with a 100-235 s first-micro-step wait "
         "every block, leaving room to serve inside the window.")


# =================================================================== chart 8 =
def chart_wait_conversion():
    W, H = 1080, 380
    L, Rm, Tm, Bm = 58, 14, 40, 44
    x0, x1, py0, py1 = L, W - Rm, H - Bm, Tm + 14
    vmax = 330.0
    parts = [svg_open(W, H)]
    parts.append(legend(L, Tm - 8, [
        ("baseline (S3)", RED), ("elastic S4a", LBLUE), ("elastic S4b", BLUE)]))
    parts.append(y_grid(x0, x1, [0, 60, 120, 180, 240, 300], vmax, 0, py0, py1))
    parts.append(T(16, (py0 + py1) / 2, "exposed gen-wait (s)", size=11, anchor="middle", rotate=-90))
    slot = (x1 - x0) / NV
    bw = min(24.0, (slot - 14) / 3 - 2)
    for i in range(NV):
        gx = x0 + i * slot + (slot - (3 * bw + 4)) / 2
        for j, (v, c, name) in enumerate([(wait3[i], RED, "baseline"),
                                          (wait4[i], LBLUE, "S4a"),
                                          (wait4b[i], BLUE, "S4b")]):
            bh = v / vmax * (py0 - py1)
            parts.append(bar_path(gx + j * (bw + 2), py0 - bh, bw, bh, c,
                                  title=f"v{i+1} {name}: {v:.1f} s exposed gen-wait"))
        parts.append(T(gx + (3 * bw + 4) / 2, py0 + 16,
                       f"v{i+1}" + ("*" if i == NV - 1 else ""), size=11, anchor="middle"))
    parts.append(f'<line x1="{x0}" y1="{py0}" x2="{x1}" y2="{py0}" stroke="{GRAY}"/>')
    parts.append("</svg>")
    sB, sA, sBb = sum(wait3[1:11]), sum(wait4[1:11]), sum(wait4b[1:11])
    card("8 · Exposed trainer gen-wait per block: what the windows actually converted",
         "results/{s3,s4,s4b}/trajectory.tsv (col wait_s)",
         "".join(parts),
         f"Over the window-active range v2-11 the exposed wait pool shrinks from {fmt(sB,0)} s (baseline) to "
         f"{fmt(sA,0)} s (S4a, {fpct(pct(sA,sB),0)}) and {fmt(sBb,0)} s (S4b, {fpct(pct(sBb,sB),0)}); the "
         "remainder is pre-fire collection (windows open at median ~52/128 samples collected) plus finite spare "
         "capacity. The report's 2348 → 1478 s figure uses on-node accounting; see data notes.")


# =================================================================== chart 9 =
def chart_close_reasons():
    REASONS = [("predictive_eta", "predictive ETA", BLUE),
               ("hard_collect_failsafe", "hard-collect failsafe", DYELLOW),
               ("batch_completed_late", "batch completed late", GRAY)]
    W, H = 1080, 190
    L, Rm = 150, 40
    x0 = L
    total_max = max(sum(close4.values()), sum(close4b.values()))
    xw = (W - L - Rm) / total_max
    parts = [svg_open(W, H)]
    parts.append(legend(x0, 24, [(lbl, c) for _, lbl, c in REASONS]))
    for r_i, (arm, counts) in enumerate([("S4a", close4), ("S4b", close4b)]):
        y = 48 + r_i * 52
        parts.append(T(x0 - 10, y + 17, f"{arm} (n={sum(counts.values())})",
                       size=11.5, anchor="end", fill=INK, weight=600))
        cx = x0
        for key, lbl, color in REASONS:
            n = counts.get(key, 0)
            if not n:
                continue
            wpx = n * xw
            parts.append(f'<rect x="{cx:.1f}" y="{y}" width="{wpx-2:.1f}" height="26" rx="3" '
                         f'fill="{color}"><title>{arm}: {lbl} × {n}</title></rect>')
            tcol = "#fff" if color != DYELLOW else INK
            if wpx > 22:
                parts.append(T(cx + wpx / 2 - 1, y + 17, str(n), size=11, anchor="middle",
                               fill=tcol, weight=600))
            cx += wpx
    parts.append("</svg>")
    card("9 · Window close reasons: what the S4a→S4b controller tuning changed",
         "results/{s4,s4b}/*_decisions.jsonl (action=switch_to_trainer, field reason)",
         "".join(parts),
         "S4a closed almost every window on the hard-collect failsafe (fill-EMA underestimating the "
         "spare-boosted fill); S4b's registered tuning (VD-6 hold/release guard, hard-collect 124, post-reset "
         "fill warmup 5) flipped the mix to mostly predictive-ETA closes, including one batch-completed-late "
         "close handled by a 7.1 s stall-not-crash hold.")


# ------------------------------------------------------------------ tiles ----
def tiles():
    t = []
    def tile(value, label, sub, color=INK):
        t.append(f'<div class="tile"><div class="tv" style="color:{color}">{value}</div>'
                 f'<div class="tl">{label}</div><div class="ts">{sub}</div></div>')
    tile(fpct(pct(tot4b, tot3), 1), "step time, v1-12 (S4b vs baseline)",
         f"S4a {fpct(pct(tot4, tot3),1)} · computed from trajectory.tsv", BLUE)
    tile(f"{fmt(sph3,0)} → {fmt(sph4b,0)}",
         "samples/GPU-hour (" + fpct(pct(sph4b, sph3), 1) + ")",
         f"S4a {fmt(sph4,0)} ({fpct(pct(sph4,sph3),1)}) · 98,304 responses / 32 GPUs", BLUE)
    tile(f"{inv4[0]+inv4b[0]}/{inv4[1]+inv4b[1]}",
         "window invariants clean (both arms)",
         f"S4a {inv4[0]}/{inv4[1]} · S4b {inv4b[0]}/{inv4b[1]} · *_window_invariants.jsonl", GREEN)
    tile(f"{int(max(drop4, drop4b))}", "dropped samples across all elastic windows",
         "max of signals.dropped_samples in *_decisions.jsonl", GREEN)
    tile("56% → 72%", "trainer-node GPU duty (whole-arm mean)",
         "source: VERL-DAPO-REPORT.md §3 only (raw 1Hz CSVs are on-node, not local)", INK2)
    return '<div class="tiles">' + "".join(t) + "</div>"


# ------------------------------------------------------------------ table ----
def data_table():
    rows = []
    for i in range(NV):
        rows.append(
            f"<tr><td>v{i+1}</td><td>{step3[i]:.1f}</td><td>{step4[i]:.1f}</td>"
            f"<td>{fpct(pct(step4[i], step3[i]))}</td><td>{step4b[i]:.1f}</td>"
            f"<td>{fpct(pct(step4b[i], step3[i]))}</td>"
            f"<td>{wait3[i]:.1f}</td><td>{wait4b[i]:.1f}</td>"
            f"<td>{s3[i]['reward']:+.3f}</td><td>{s4[i]['reward']:+.3f}</td>"
            f"<td>{s4b[i]['reward']:+.3f}</td></tr>")
    rows.append(
        f"<tr class='tot'><td>∑ v1-12</td><td>{tot3:.0f}</td><td>{tot4:.0f}</td>"
        f"<td>{fpct(pct(tot4, tot3),2)}</td><td>{tot4b:.0f}</td><td>{fpct(pct(tot4b, tot3),2)}</td>"
        f"<td>{sum(wait3):.0f}</td><td>{sum(wait4b):.0f}</td><td></td><td></td><td></td></tr>")
    return ("<details class='card'><summary>Data table: per-block measurements behind charts 1, 2, 3, 5, 8 "
            "(results/{s3,s4,s4b}/trajectory.tsv)</summary><table><thead><tr>"
            "<th>block</th><th>baseline step (s)</th><th>S4a step (s)</th><th>S4a Δ</th>"
            "<th>S4b step (s)</th><th>S4b Δ</th><th>baseline wait (s)</th><th>S4b wait (s)</th>"
            "<th>reward S3</th><th>reward S4a</th><th>reward S4b</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></details>")


# ------------------------------------------------------------------- page ----
def build():
    chart_step_bars()
    chart_cumulative()
    chart_idle()
    chart_switch()
    chart_phases()
    chart_reward()
    chart_regime()
    chart_fragmentation()
    chart_wait_conversion()
    chart_close_reasons()

    NOTES.extend([
        "S4a operator error: a wrong pgrep pattern killed the S4a controller ~3 minutes early, so its sync "
        "block 12 ran windowless (conservative bias for S4a in charts 1, 2, 8). S4a artifacts nonetheless "
        f"record {len(sw4_in)} switch-in cycles / {inv4[1]} window-invariant checks vs S4b's {len(sw4b_in)}/"
        f"{inv4b[1]}; the report's text counts 26 verified S4a cycles.",
        "Percentile convention: p50 here is the interpolated median and p90 is nearest-rank, recomputed from "
        "the raw switch-timing logs; the report quotes S4b switch p50 1.46 s (in) / 2.98 s (back) and p90 "
        "2.50 / 11.03 s under its own convention. Raw dots are plotted, so nothing hides in the convention.",
        "Wait-conversion accounting: the report's “2348 → 1478 s (37% converted)” comes from "
        f"on-node accounting. Local artifacts give v2-11 sums of {sum(wait3[1:11]):.0f} → "
        f"{sum(wait4b[1:11]):.0f} s from trajectory.tsv wait_s and {sum(w_s3[4:44]):.0f} → "
        f"{sum(w_s4b[4:44]):.0f} s from the train.log collection waits: same story, slightly different "
        "measure boundaries.",
        "results/s1-30b/trajectory.tsv contains TWO probe runs (S1: v1-2, including a 73.8 s v2 that is an "
        "MQ-drain artifact; S1b: v1-3). Chart 6 uses the S1b run. The S1 run's values (idle 7.2%/1.8%, len "
        "837/1526) are consistent with the train-bound verdict.",
        "The waits block in results/s1-7b/trajectory_and_waits.txt is unlabeled in the file; it parses as 40 "
        f"per-micro-step waits (10 blocks × 4) with p50 {p50(s17_waits):.2f} s / p90 "
        f"{p90(s17_waits):.1f} s, matching VD-STATE.md's logged “p50 micro-step wait 4s, p90 24s”, "
        "which fixes the interpretation. The trajectory block covers v1-9 (killed at v9), so the 10th wait "
        "group is the in-flight v10 block.",
        "v12 in every arm is the standard termination flush (a partial block, ~150-160 s); it is plotted but "
        "starred wherever it appears.",
    ])
    not_plotted = [
        "Trainer/rollout-node GPU duty over time (claim-3 evidence): the 1Hz gpu_util CSVs live on the head "
        "node only, not in local results/; only the whole-arm means from VERL-DAPO-REPORT.md §3 are shown, "
        "as a labeled stat tile.",
        "wcache dump + prefetch wall per param sync: the scored-arm local artifacts do not carry a per-sync "
        "series (the report quotes the 18.4-23.1 s range; s2 shakedown logs exist but are out of scope for "
        "scored charts).",
        "30B micro-step wait distribution (chart 6, middle panel): not logged in results/s1-30b/; only "
        "per-block wait_s exists (0.9-40.5 s), consistent with “no harvest pool”.",
        "Validation-window economics (report §9 item 3): no val fell inside the scored N=12 range, so "
        "there is nothing measured to plot.",
    ]

    css = """
    :root{color-scheme:light}
    *{box-sizing:border-box}
    body{margin:0;background:#F8F9FA;color:#202124;
         font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif}
    .wrap{max-width:1140px;margin:0 auto;padding:28px 20px 60px}
    h1{font-size:24px;margin:0 0 4px;font-weight:700}
    h2{font-size:16px;margin:0 0 2px;font-weight:600}
    .sub{color:#5F6368;font-size:13px;margin-bottom:6px}
    .srcs{color:#5F6368;font-size:12px;margin-bottom:22px;
          font-family:'SF Mono',Menlo,Consolas,monospace}
    .card{background:#fff;border-radius:12px;padding:18px 20px 14px;margin:0 0 18px;
          box-shadow:0 1px 2px rgba(60,64,67,.14),0 1px 3px 1px rgba(60,64,67,.08)}
    .src{font-family:'SF Mono',Menlo,Consolas,monospace;font-size:11px;color:#80868B;margin:0 0 10px}
    .cap{color:#5F6368;font-size:12.5px;margin:8px 0 2px;max-width:1000px}
    .row{display:flex;flex-wrap:wrap;gap:8px;align-items:flex-start}
    .tiles{display:flex;flex-wrap:wrap;gap:14px;margin:0 0 18px}
    .tile{background:#fff;border-radius:12px;padding:14px 18px;flex:1 1 190px;
          box-shadow:0 1px 2px rgba(60,64,67,.14),0 1px 3px 1px rgba(60,64,67,.08)}
    .tv{font-size:23px;font-weight:700;letter-spacing:-.3px}
    .tl{font-size:12.5px;color:#202124;margin-top:2px;font-weight:600}
    .ts{font-size:11px;color:#80868B;margin-top:3px}
    ul{margin:6px 0;padding-left:20px}
    li{margin:6px 0;color:#3C4043;font-size:13px}
    details.card summary{cursor:pointer;font-weight:600;font-size:14px;color:#202124}
    table{border-collapse:collapse;margin-top:12px;font-size:12.5px;width:100%}
    th,td{border-bottom:1px solid #E8EAED;padding:5px 8px;text-align:right;
          font-variant-numeric:tabular-nums}
    th:first-child,td:first-child{text-align:left}
    th{color:#5F6368;font-weight:600;border-bottom:2px solid #DADCE0}
    tr.tot td{font-weight:700;border-top:2px solid #DADCE0}
    """
    head = (
        '<header><h1>verl DAPO 32-GPU elastic campaign: measured plots</h1>'
        '<div class="sub">Elastic time-slicing of 16 trainer GPUs on the stock verl fully-async DAPO recipe '
        '(dapo_7b_math_fsdp2_16_16 + DeepSeek-R1-Distill-Qwen-7B), 4×8 H200, runs of 2026-09-03: '
        'S3 baseline vs S4a/S4b elastic arms, N=12 sync blocks each. Every mark is parsed from local measured '
        'artifacts; report-sourced numbers are labeled. Hover any mark for its exact value.</div>'
        '<div class="srcs">data: elastic-rl-poc/mlperf-replica/genbound2/results/{s3,s4,s4b,s1-30b,s1-7b}/ '
        '&nbsp;·&nbsp; context: VERL-DAPO-REPORT.md, VD-STATE.md &nbsp;·&nbsp; generator: plots/build_vd_plots.py'
        '</div></header>')
    notes_html = ('<section class="card"><h2>Data notes (read before quoting)</h2><ul>'
                  + "".join(f"<li>{esc(n)}</li>" for n in NOTES) + "</ul></section>")
    np_html = ('<section class="card"><h2>Not plotted (wanted, but no local measured data)</h2><ul>'
               + "".join(f"<li>{esc(n)}</li>" for n in not_plotted) + "</ul></section>")
    page = ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>verl DAPO 32-GPU elastic campaign: plots</title>"
            f"<style>{css}</style></head><body><div class='wrap'>"
            + head + tiles() + "".join(CARD) + data_table() + notes_html + np_html
            + "</div></body></html>")
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(page):,} bytes)")
    # sanity: totals must match the report's headline table
    assert abs(tot3 - 8420.4) < 0.5 and abs(tot4 - 7644.1) < 1.0 and abs(tot4b - 7589.6) < 1.0, \
        (tot3, tot4, tot4b)
    print(f"check: totals {tot3:.1f}/{tot4:.1f}/{tot4b:.1f} s; "
          f"samples/GPU-hr {sph3:.0f}/{sph4:.0f}/{sph4b:.0f}; "
          f"S4b switch p50 in {p50([r['total'] for r in sw4b_in]):.2f}s back "
          f"{p50([r['total'] for r in sw4b_back]):.2f}s")


if __name__ == "__main__":
    build()
