"""scraper_dashboard.py — Dashboard from tpu-info scraper CSVs (1s resolution)

Plots duty_cycle_pct and mem_used_mib from tpu_duty_cycle_sampler.csv
and tpu_duty_cycle_trainer.csv for baseline vs timeslice comparison.

Usage:
  python3 scraper_dashboard.py \
    --baseline-dir runs/full_5step/baseline \
    --timeslice-dir runs/full_5step/timeslice \
    --output runs/full_5step/scraper_dashboard.html
"""

import argparse
import base64
import csv
import io
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def img_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def load_scraper(log_dir, role, chip='0'):
    path = os.path.join(log_dir, f'tpu_duty_cycle_{role}.csv')
    if not os.path.exists(path):
        return [], [], []
    rows = [r for r in csv.DictReader(open(path)) if r['chip'] == chip]
    ts = [float(r['ts']) for r in rows]
    duty = [float(r['duty_cycle_pct']) for r in rows]
    mem = [float(r['mem_used_mib']) for r in rows]
    return ts, duty, mem


def load_scraper_all_chips(log_dir, role):
    path = os.path.join(log_dir, f'tpu_duty_cycle_{role}.csv')
    if not os.path.exists(path):
        return {}
    by_chip = defaultdict(lambda: {'ts': [], 'duty': [], 'mem': []})
    for r in csv.DictReader(open(path)):
        c = r['chip']
        by_chip[c]['ts'].append(float(r['ts']))
        by_chip[c]['duty'].append(float(r['duty_cycle_pct']))
        by_chip[c]['mem'].append(float(r['mem_used_mib']))
    return {c: d for c, d in by_chip.items() if any(v > 0 for v in d['duty']) or any(v > 0 for v in d['mem'])}


def plot_duty_cycle(baseline_dir, timeslice_dir):
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    chip_colors = ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#8E24AA', '#F57C00', '#0097A7', '#546E7A']

    for idx, (log_dir, label) in enumerate([(baseline_dir, 'Baseline'), (timeslice_dir, 'Timeslice')]):
        ax = axes[idx]
        s_windows, t_windows = load_orch_windows(log_dir)
        all_starts = []
        for role in ['sampler', 'trainer']:
            chips = load_scraper_all_chips(log_dir, role)
            for cdata in chips.values():
                if cdata['ts']:
                    all_starts.append(cdata['ts'][0])
        t0 = min(all_starts) if all_starts else 0

        for role, windows in [('sampler', s_windows), ('trainer', t_windows)]:
            chips = load_scraper_all_chips(log_dir, role)
            for chip_id in sorted(chips.keys(), key=int):
                cdata = chips[chip_id]
                duty = cdata['duty']
                ts = cdata['ts']
                if windows:
                    duty = mask_by_orch(ts, duty, windows)
                elapsed_min = [(t - t0) / 60 for t in ts]
                ax.plot(elapsed_min, duty, label=f'{role} chip {chip_id}',
                        color=chip_colors[int(chip_id) % len(chip_colors)],
                        linewidth=0.8, alpha=0.7,
                        linestyle='-' if role == 'sampler' else '--')

        ax.set_title(f'{label} — TPU Duty Cycle (tpu-info, 1s resolution, masked by orch)')
        ax.set_ylabel('Duty Cycle (%)')
        ax.set_xlabel('Elapsed (minutes)')
        ax.set_ylim(-2, 55)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper right', ncol=2)

    plt.tight_layout()
    return img_to_b64(fig)


def plot_memory(baseline_dir, timeslice_dir):
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    GIB = 1024
    chip_colors = ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#8E24AA', '#F57C00', '#0097A7', '#546E7A']

    for idx, (log_dir, label) in enumerate([(baseline_dir, 'Baseline'), (timeslice_dir, 'Timeslice')]):
        ax = axes[idx]
        all_starts = []
        for role in ['sampler', 'trainer']:
            chips = load_scraper_all_chips(log_dir, role)
            for cdata in chips.values():
                if cdata['ts']:
                    all_starts.append(cdata['ts'][0])
        t0 = min(all_starts) if all_starts else 0

        for role in ['sampler', 'trainer']:
            chips = load_scraper_all_chips(log_dir, role)
            for chip_id in sorted(chips.keys(), key=int):
                cdata = chips[chip_id]
                elapsed_min = [(t - t0) / 60 for t in cdata['ts']]
                mem_gib = [m / GIB for m in cdata['mem']]
                ax.plot(elapsed_min, mem_gib, label=f'{role} chip {chip_id}',
                        color=chip_colors[int(chip_id) % len(chip_colors)],
                        linewidth=0.8, alpha=0.7,
                        linestyle='-' if role == 'sampler' else '--')

        ax.set_title(f'{label} — TPU HBM Usage (tpu-info, 1s resolution, chip 0)')
        ax.set_ylabel('HBM Used (GiB)')
        ax.set_xlabel('Elapsed (minutes)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    plt.tight_layout()
    return img_to_b64(fig)


def load_orch_windows(log_dir):
    """Parse orchestrator events to get active windows per pool."""
    path = os.path.join(log_dir, 'rl_metrics.jsonl')
    if not os.path.exists(path):
        return {}, {}
    sampler_windows = []
    trainer_windows = []
    sampler_start = {}
    trainer_start = {}
    with open(path) as f:
        for line in f:
            try:
                e = json.loads(line)
            except:
                continue
            t = e.get('type', '')
            wl = e.get('workload_id', '')
            ts = e.get('ts', 0)
            pool = e.get('pool', '')
            if t == 'acquire':
                if pool == 'sampler':
                    sampler_start[wl] = ts
                elif pool == 'trainer':
                    trainer_start[wl] = ts
            elif t == 'yield':
                if pool == 'sampler' and wl in sampler_start:
                    sampler_windows.append((sampler_start.pop(wl), ts))
                elif pool == 'trainer' and wl in trainer_start:
                    trainer_windows.append((trainer_start.pop(wl), ts))
    return sampler_windows, trainer_windows


def mask_by_orch(timestamps, values, windows):
    """Zero out values outside orchestrator-defined active windows."""
    masked = []
    for ts, val in zip(timestamps, values):
        active = any(start <= ts <= end for start, end in windows)
        masked.append(val if active else 0.0)
    return masked


def plot_duty_overlay(baseline_dir, timeslice_dir):
    """Overlay sampler + trainer chip 0 on same timeline to show interleaving."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))

    for idx, (log_dir, label) in enumerate([(baseline_dir, 'Baseline'), (timeslice_dir, 'Timeslice')]):
        ax = axes[idx]
        ts_s, duty_s, _ = load_scraper(log_dir, 'sampler')
        ts_t, duty_t, _ = load_scraper(log_dir, 'trainer')
        s_windows, t_windows = load_orch_windows(log_dir)

        all_starts = []
        if ts_s: all_starts.append(ts_s[0])
        if ts_t: all_starts.append(ts_t[0])
        t0 = min(all_starts) if all_starts else 0

        if ts_s:
            elapsed_s = [(t - t0) / 60 for t in ts_s]
            if s_windows:
                duty_s = mask_by_orch(ts_s, duty_s, s_windows)
            ax.fill_between(elapsed_s, duty_s, alpha=0.4, color='#4285F4', label='Sampler (chip 0)')

        if ts_t:
            elapsed_t = [(t - t0) / 60 for t in ts_t]
            if t_windows:
                duty_t = mask_by_orch(ts_t, duty_t, t_windows)
            ax.fill_between(elapsed_t, duty_t, alpha=0.4, color='#EA4335', label='Trainer (chip 0)')

        ax.set_title(f'{label} — Duty Cycle Overlay (masked by orchestrator)')
        ax.set_ylabel('Duty Cycle (%)')
        ax.set_xlabel('Elapsed (minutes)')
        ax.set_ylim(-2, 55)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    return img_to_b64(fig)


def compute_summary(log_dir, label):
    stats = {}
    for role in ['sampler', 'trainer']:
        chips = load_scraper_all_chips(log_dir, role)
        if not chips:
            continue
        all_duty = []
        all_mem = []
        all_ts = []
        for cdata in chips.values():
            all_duty.extend(cdata['duty'])
            all_mem.extend(cdata['mem'])
            all_ts.extend(cdata['ts'])
        non_zero = [d for d in all_duty if d > 0]
        stats[role] = {
            'samples': len(all_duty),
            'chips': len(chips),
            'duration_min': (max(all_ts) - min(all_ts)) / 60 if len(all_ts) > 1 else 0,
            'avg_duty_active': np.mean(non_zero) if non_zero else 0,
            'active_pct': 100 * len(non_zero) / len(all_duty) if all_duty else 0,
            'avg_mem_gib': np.mean(all_mem) / 1024,
            'max_mem_gib': max(all_mem) / 1024,
        }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-dir', required=True)
    parser.add_argument('--timeslice-dir', required=True)
    parser.add_argument('--output', default='scraper_dashboard.html')
    args = parser.parse_args()

    img_duty = plot_duty_cycle(args.baseline_dir, args.timeslice_dir)
    img_mem = plot_memory(args.baseline_dir, args.timeslice_dir)
    img_overlay = plot_duty_overlay(args.baseline_dir, args.timeslice_dir)

    bs = compute_summary(args.baseline_dir, 'Baseline')
    ts = compute_summary(args.timeslice_dir, 'Timeslice')

    # Job metrics
    job_html = ''
    for label, log_dir in [('Baseline', args.baseline_dir), ('Timeslice', args.timeslice_dir)]:
        job_html += f'<h4>{label}</h4><table style="width:100%; border-collapse:collapse; font-size:14px;">'
        job_html += '<tr style="background:#ecf0f1;"><th style="padding:6px;">Job</th><th>Step</th><th>Gen (s)</th><th>Train (s)</th><th>Reward</th><th>Correct</th><th>Acq Samp (s)</th><th>Acq Train (s)</th></tr>'
        for f in sorted(os.listdir(log_dir)):
            if f.startswith('metrics_job') and f.endswith('.jsonl'):
                job = f.replace('metrics_', '').replace('.jsonl', '')
                data = []
                with open(os.path.join(log_dir, f)) as fh:
                    for line in fh:
                        try:
                            data.append(json.loads(line))
                        except:
                            pass
                for d in sorted(data, key=lambda x: x.get('step', 0)):
                    job_html += f'<tr style="border-bottom:1px solid #eee;"><td style="padding:4px;">{job}</td><td>{d["step"]}</td><td>{d["gen_ms"]/1000:.0f}</td><td>{d["train_ms"]/1000:.0f}</td><td>{d["mean_reward"]:.3f}</td><td>{d["correct_rate"]:.2%}</td><td>{d.get("acquire_sampler_ms",0)/1000:.0f}</td><td>{d.get("acquire_trainer_ms",0)/1000:.0f}</td></tr>'
        job_html += '</table>'

    def row(label, bs_val, ts_val, fmt='%.1f'):
        return f'<tr style="border-bottom:1px solid #eee;"><td style="padding:8px;">{label}</td><td style="padding:8px;">{fmt % bs_val if bs_val is not None else "N/A"}</td><td style="padding:8px;">{fmt % ts_val if ts_val is not None else "N/A"}</td></tr>'

    summary_html = '<table style="width:100%; border-collapse:collapse; font-size:15px;">'
    summary_html += '<tr style="background:#ecf0f1;border-bottom:2px solid #bdc3c7;"><th style="padding:10px;">Metric (all active chips)</th><th style="padding:10px;">Baseline</th><th style="padding:10px;">Timeslice</th></tr>'

    for role in ['sampler', 'trainer']:
        summary_html += f'<tr style="background:#fcfcfc;"><td colspan="3" style="padding:8px;font-weight:bold;color:#7f8c8d;">{role.title()} Pool</td></tr>'
        b = bs.get(role, {})
        t = ts.get(role, {})
        summary_html += row(f'  <b>Duty Cycle (% time active)</b>', b.get('active_pct'), t.get('active_pct'), '<b>%.1f%%</b>')
        summary_html += row(f'  Avg Duty When Active', b.get('avg_duty_active'), t.get('avg_duty_active'), '%.1f%%')
        summary_html += row(f'  Avg HBM', b.get('avg_mem_gib'), t.get('avg_mem_gib'), '%.1f GiB')
        summary_html += row(f'  Max HBM', b.get('max_mem_gib'), t.get('max_mem_gib'), '%.1f GiB')
        summary_html += row(f'  Duration', b.get('duration_min'), t.get('duration_min'), '%.1f min')

    summary_html += '</table>'

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>TPU Time-Slicing — Scraper Metrics Dashboard</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; margin: 40px; background: #f8f9fa; color: #333; }}
        .container {{ max-width: 1400px; margin: auto; background: white; padding: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-radius: 8px; }}
        h1, h2 {{ color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
        .card {{ margin-top: 30px; border: 1px solid #ddd; border-radius: 5px; padding: 20px; background: #fff; }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
<div class="container">
    <h1>TPU Time-Slicing: Scraper Metrics (tpu-info, 1s resolution)</h1>

    <div class="card" style="border-left: 4px solid #4A9EE0;">
        <h2>Summary (chip 0 only)</h2>
        {summary_html}
    </div>

    <div class="card">
        <h2>1. Duty Cycle Overlay</h2>
        <p>Baseline should show alternating sampler/trainer (square wave). Timeslice should show both active simultaneously.</p>
        <img src="data:image/png;base64,{img_overlay}">
    </div>

    <div class="card">
        <h2>2. Duty Cycle Time Series</h2>
        <p>Full 1-second resolution duty cycle for chip 0. Note: tpu-info duty_cycle_pct can be sticky/cached in some cases.</p>
        <img src="data:image/png;base64,{img_duty}">
    </div>

    <div class="card">
        <h2>3. HBM Memory Usage</h2>
        <p>1-second resolution memory tracking. Sampler HBM stays constant (~15 GiB) even during C/R. Trainer memory fluctuates with training activity.</p>
        <img src="data:image/png;base64,{img_mem}">
    </div>

    <div class="card">
        <h2>4. Per-Step Metrics</h2>
        {job_html}
    </div>
</div>
</body>
</html>"""

    with open(args.output, 'w') as f:
        f.write(html)
    print(f"Scraper dashboard: {args.output}")


if __name__ == '__main__':
    main()
