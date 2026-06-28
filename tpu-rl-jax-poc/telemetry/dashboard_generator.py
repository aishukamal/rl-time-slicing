#!/usr/bin/env python3
import os
import sys
import re
import csv
import json
import base64
import argparse
import subprocess
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

def get_base64_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def read_jsonl(path):
    data = []
    if not os.path.exists(path): return data
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try: data.append(json.loads(line))
                except: pass
    return data

def parse_rl_metrics(path, remove_datapoints=0):
    data = read_jsonl(path)
    
    overhead_series = defaultdict(lambda: defaultdict(list))
    events = defaultdict(list)
    wall_clocks = defaultdict(list)
    active_acquires = {}
    acquire_counts = defaultdict(int)
    
    for d in data:
        wid = d.get('workload_id')
        if not wid: continue
            
        if d['type'] == 'acquire':
            acquire_counts[wid] += 1
            if acquire_counts[wid] <= remove_datapoints:
                continue # Skip warmup cycles
                
            wait_s = d.get('wait_ms', 0) / 1000.0
            restore_s = d.get('restore_ms', 0) / 1000.0
            
            req_ts = d['ts'] - wait_s - restore_s
            active_acquires[wid] = (req_ts, d['pool'])
            
            overhead_series['wait'][wid].append(wait_s)
            overhead_series['restore'][wid].append(restore_s)
            
        elif d['type'] == 'yield':
            if acquire_counts[wid] <= remove_datapoints:
                continue # Skip corresponding yield for warmup
                
            evict_s = d.get('evict_ms', 0) / 1000.0
            overhead_series['evict'][wid].append(evict_s)
            
            if wid in active_acquires:
                req_ts, pool = active_acquires[wid]
                wall_clock_s = d['ts'] - req_ts
                wall_clocks[pool].append(wall_clock_s)
                del active_acquires[wid]
                
        elif d['type'] == 'train':
            events[d.get('step', 0)].append(d['ts'])

    return {
        'overheads': overhead_series,
        'wall_clocks': wall_clocks
    }

def parse_job_metrics(log_dir, remove_datapoints=0):
    gen_ms, sync_ms, train_ms = [], [], []
    for f in os.listdir(log_dir):
        if f.startswith('metrics_') and f.endswith('.jsonl'):
            path = os.path.join(log_dir, f)
            data = read_jsonl(path)
            
            # Sort by step to ensure chronological order
            data = sorted([x for x in data if 'step' in x], key=lambda x: x['step'])
            # Drop warmup steps
            if len(data) > remove_datapoints:
                data = data[remove_datapoints:]
            elif remove_datapoints > 0:
                data = []
                
            for d in data:
                if 'gen_ms' in d: gen_ms.append(d['gen_ms'])
                if 'sync_ms' in d: sync_ms.append(d['sync_ms'])
                if 'train_ms' in d: train_ms.append(d['train_ms'])
    return gen_ms, sync_ms, train_ms

def extract_duty_cycle(log_dir):
    csv_path = os.path.join(log_dir, "tpu_duty_cycle.csv")
    if not os.path.exists(csv_path): return 0, 0
    
    utils = defaultdict(list)
    utils_with_ts = defaultdict(list)
    non_zero_ts = []
    
    try:
        with open(csv_path, 'r') as f:
            for row in csv.DictReader(f):
                if 'duty_cycle_pct' in row and 'chip' in row and 'ts' in row:
                    ts = float(row['ts'])
                    u = float(row['duty_cycle_pct'])
                    tpu = row['chip']
                    utils[tpu].append(u)
                    utils_with_ts[tpu].append((ts, u))
                    if u > 0:
                        non_zero_ts.append(ts)
    except:
        pass
        
    start_ts = min(non_zero_ts) if non_zero_ts else None
    end_ts = max(non_zero_ts) if non_zero_ts else None
    
    duty_cycles = []
    active_duty_cycles = []
    
    for tpu, vals in utils.items():
        if vals:
            duty_cycles.append(sum(1 for u in vals if u > 0) / len(vals) * 100)
            
    for tpu, ts_vals in utils_with_ts.items():
        if start_ts is not None and end_ts is not None:
            active_vals = [u for t, u in ts_vals if start_ts <= t <= end_ts]
            if active_vals:
                active_duty_cycles.append(sum(1 for u in active_vals if u > 0) / len(active_vals) * 100)
                
    overall_duty = np.mean(duty_cycles) if duty_cycles else 0
    active_duty = np.mean(active_duty_cycles) if active_duty_cycles else 0
    return overall_duty, active_duty

def compute_sliding_windows(log_dir, window_mins=10):
    import datetime
    csv_path = os.path.join(log_dir, "tpu_duty_cycle.csv")
    if not os.path.exists(csv_path): return []
    
    ts_util = defaultdict(int)
    tpu_data = defaultdict(lambda: {'ts': [], 'util': []})

    try:
        with open(csv_path, 'r') as f:
            for row in csv.DictReader(f):
                if 'duty_cycle_pct' in row and 'chip' in row and 'ts' in row:
                    ts = float(row['ts'])
                    u = float(row['duty_cycle_pct'])
                    tpu = int(row['chip'])
                    ts_util[ts] += u
                    tpu_data[tpu]['ts'].append(ts)
                    tpu_data[tpu]['util'].append(u)
    except:
        return []
        
    unique_ts = sorted(ts_util.keys())
    if not unique_ts: return []
    
    non_zero_ts = [ts for ts in unique_ts if ts_util[ts] > 0]
    if not non_zero_ts: return []
    
    t_first = non_zero_ts[0]
    t_last = non_zero_ts[-1]
    
    def ts_to_dt(ts):
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone(datetime.timedelta(hours=-7)))
        return dt.strftime('%H:%M')

    windows = []
    t = t_first
    while t + window_mins * 60 <= t_last:
        et = t + window_mins * 60
        window_unique_ts = [ts for ts in unique_ts if t <= ts <= et]
        
        tpu_duties = {}
        for tpu, data in tpu_data.items():
            bounded_indices = [i for i, ts in enumerate(data['ts']) if t <= ts <= et]
            if bounded_indices:
                active = sum(1 for i in bounded_indices if data['util'][i] > 0)
                tpu_duties[tpu] = active / len(bounded_indices) * 100
            else:
                tpu_duties[tpu] = 0
                
        sys_duty = sum(tpu_duties.values()) / len(tpu_duties) if tpu_duties else 0
                
        windows.append({
            'start': ts_to_dt(t),
            'end': ts_to_dt(et),
            'sys': sys_duty,
            'g0': tpu_duties.get(0, 0),
            'g1': tpu_duties.get(1, 0)
        })
        t += window_mins * 60
        
    return windows

def extract_avg_util(log_dir):
    csv_path = os.path.join(log_dir, "tpu_duty_cycle.csv")
    if not os.path.exists(csv_path): return 0
    utils = defaultdict(list)
    try:
        with open(csv_path, 'r') as f:
            for row in csv.DictReader(f):
                if 'duty_cycle_pct' in row and 'chip' in row:
                    val = float(row['duty_cycle_pct'])
                    utils[row['chip']].append(val)
    except:
        pass
    avg_utils = []
    for tpu, vals in utils.items():
        if vals:
             avg_utils.append(np.mean(vals))
    return np.mean(avg_utils) if avg_utils else 0

def get_total_time(log_dir):
    min_ts, max_ts = float('inf'), 0
    for f in os.listdir(log_dir):
        if f.endswith('.jsonl'):
            data = read_jsonl(os.path.join(log_dir, f))
            for d in data:
                if 'ts' in d:
                    min_ts = min(min_ts, d['ts'])
                    max_ts = max(max_ts, d['ts'])
    if min_ts == float('inf'): return 0
    return max_ts - min_ts

def filter_zeros(vals):
    return [v for v in vals if v > 0]

def extract_convergence_metrics(log_dir):
    overall_res = {'first_10': defaultdict(list), 'last_10': defaultdict(list)}
    jobs_res = {}
    
    for f in os.listdir(log_dir):
        if f.startswith('metrics_job') and f.endswith('.jsonl'):
            job_id = f.replace("metrics_job-", "").replace(".jsonl", "")
            if job_id not in jobs_res: jobs_res[job_id] = {'first_10': defaultdict(list), 'last_10': defaultdict(list)}
            
            d = read_jsonl(os.path.join(log_dir, f))
            d = sorted([x for x in d if 'step' in x], key=lambda x: x['step'])
            if not d: continue
            for x in d:
                if 'acc' not in x and 'correct_rate' in x: x['acc'] = x['correct_rate']
                if 'kl' not in x and 'kl_loss' in x: x['kl'] = x['kl_loss']
            first_10 = d[:10]
            last_10 = d[-10:]

            for key in ['mean_reward', 'acc', 'kl', 'loss']:
                f10 = [x[key] for x in first_10 if key in x]
                if f10:
                    jobs_res[job_id]['first_10'][key].append(np.mean(f10))
                    overall_res['first_10'][key].append(np.mean(f10))
                    
                l10 = [x[key] for x in last_10 if key in x]
                if l10:
                    jobs_res[job_id]['last_10'][key].append(np.mean(l10))
                    overall_res['last_10'][key].append(np.mean(l10))
                
    def _compute_means(res_dict):
        out = {'first_10': {}, 'last_10': {}}
        for period in ['first_10', 'last_10']:
            for key in ['mean_reward', 'acc', 'kl', 'loss']:
                out[period][key] = np.mean(res_dict[period][key]) if res_dict[period][key] else 0
        return out

    final_res = {'overall': _compute_means(overall_res)}
    for jid, jdata in jobs_res.items():
        final_res[jid] = _compute_means(jdata)
        
    return final_res


def plot_overheads_over_time(ts_metrics):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    overheads = ts_metrics['overheads']
    
    for idx, (metric_name, title) in enumerate([('wait', 'Queue Wait Time'), ('evict', 'Eviction Time'), ('restore', 'Restore Time')]):
        ax = axes[idx]
        for wid, vals in overheads[metric_name].items():
            if not vals: continue
            plot_vals = vals
            x = list(range(1, len(plot_vals)+1))
            ax.plot(x, plot_vals, marker='o', label=wid, linewidth=2, markersize=6, alpha=0.8)
            
        ax.set_title(f"{title} per Job Over Time")
        ax.set_xlabel("Event Sequence")
        ax.set_ylabel("Seconds")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    return get_base64_image(fig)

def plot_wall_clocks(bs_metrics, ts_metrics, bs_job_metrics, ts_job_metrics, time_bs, time_ts):
    bs_gen, bs_sync, bs_train = bs_job_metrics
    ts_gen, ts_sync, ts_train = ts_job_metrics
    
    bs_pool_clocks = bs_metrics['wall_clocks']
    ts_pool_clocks = ts_metrics['wall_clocks']
    
    # Raw Mathematical Times
    raw_cat = ['Baseline', 'Timeslice']
    gen_avgs = [np.mean(bs_gen)/1000.0 if bs_gen else 0, np.mean(ts_gen)/1000.0 if ts_gen else 0]
    sync_avgs = [np.mean(bs_sync)/1000.0 if bs_sync else 0, np.mean(ts_sync)/1000.0 if ts_sync else 0]
    train_avgs = [np.mean(bs_train)/1000.0 if bs_train else 0, np.mean(ts_train)/1000.0 if ts_train else 0]
    
    # For baseline, actual observed wall-clock via pools is 0 since acquire doesn't block. 
    # Fallback entirely to raw time for baseline.
    avg_bs_sampler = gen_avgs[0]
    avg_bs_trainer = train_avgs[0]
    
    avg_ts_sampler = np.mean(filter_zeros(ts_pool_clocks.get('sampler', [0]))) if filter_zeros(ts_pool_clocks.get('sampler', [0])) else 0
    avg_ts_trainer = np.mean(filter_zeros(ts_pool_clocks.get('trainer', [0]))) if filter_zeros(ts_pool_clocks.get('trainer', [0])) else 0

    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    
    p1 = ax[0].bar(raw_cat, gen_avgs, color='#4285F4', label='Sampling/Gen (Raw)')
    p2 = ax[0].bar(raw_cat, train_avgs, bottom=gen_avgs, color='#FBBC05', label='Training (Raw)')
    p3 = ax[0].bar(raw_cat, sync_avgs, bottom=np.array(gen_avgs)+np.array(train_avgs), color='#34A853', label='Weight Sync')
    ax[0].set_title('Raw execution time per phase (No Overheads)')
    ax[0].set_ylabel('Seconds')
    ax[0].legend()

    # Wall Clock Times
    wall_cat = ['Baseline', 'Timeslice']
    samp_clocks = [avg_bs_sampler, avg_ts_sampler]
    train_clocks = [avg_bs_trainer, avg_ts_trainer]
    
    p1_wall = ax[1].bar(wall_cat, samp_clocks, color='#4285F4', label='Sampling Phase')
    p2_wall = ax[1].bar(wall_cat, train_clocks, bottom=samp_clocks, color='#FBBC05', label='Training Phase')
    ax[1].set_title('Wall Clock Time per Phase (acquire -> yield)')
    ax[1].set_ylabel('Seconds')
    ax[1].legend()

    # Total Execution Times
    total_cat = ['Baseline', 'Timeslice']
    total_times_mins = [(time_bs * 2) / 60.0, time_ts / 60.0]
    
    ax[2].bar(total_cat, total_times_mins, color=['#4285F4', '#FBBC05'], label='Total Execution Time (2 Jobs)')
    ax[2].set_title('Total E2E Execution Time')
    ax[2].set_ylabel('Minutes')
    ax[2].legend()

    plt.tight_layout()
    return get_base64_image(fig), gen_avgs, sync_avgs, train_avgs, samp_clocks, train_clocks

def plot_convergence(baseline_dir, timeslice_dir):
    fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
    
    def _normalize(records):
        for x in records:
            if 'acc' not in x and 'correct_rate' in x: x['acc'] = x['correct_rate']
            if 'kl' not in x and 'kl_loss' in x: x['kl'] = x['kl_loss']
        return records

    # Parse all jobs in baseline
    bs_files = sorted([f for f in os.listdir(baseline_dir) if f.startswith('metrics_job') and f.endswith('.jsonl')])
    for f in bs_files:
        d = _normalize(read_jsonl(os.path.join(baseline_dir, f)))
        d = sorted([x for x in d if 'step' in x], key=lambda x: x['step'])
        job_id = f.replace("metrics_", "").replace(".jsonl", "")
        steps = [x['step'] for x in d]
        rwds = [x['mean_reward'] for x in d if 'mean_reward' in x]
        losses = [x['loss'] for x in d if 'loss' in x]
        accs = [x['acc'] for x in d if 'acc' in x]
        kls = [x['kl'] for x in d if 'kl' in x]

        if rwds: axes[0].plot(steps[:len(rwds)], rwds, label=f'Baseline {job_id}', color='#4285F4', linewidth=2.5, marker='o', linestyle='--')
        if accs: axes[1].plot(steps[:len(accs)], accs, label=f'Baseline {job_id}', color='#4285F4', linewidth=2.5, marker='o', linestyle='--')
        if kls: axes[2].plot(steps[:len(kls)], kls, label=f'Baseline {job_id}', color='#4285F4', linewidth=2.5, marker='o', linestyle='--')
        if losses: axes[3].plot(steps[:len(losses)], losses, label=f'Baseline {job_id}', color='#4285F4', linewidth=2.5, marker='o', linestyle='--')

    # Parse all jobs in timeslice
    ts_files = sorted([f for f in os.listdir(timeslice_dir) if f.startswith('metrics_job') and f.endswith('.jsonl')])
    colors = ['#FBBC05', '#34A853', '#EA4335', '#4285F4']
    markers = ['x', '^', 's', 'd']

    for idx, f in enumerate(ts_files):
        d = _normalize(read_jsonl(os.path.join(timeslice_dir, f)))
        d = sorted([x for x in d if 'step' in x], key=lambda x: x['step'])
        job_id = f.replace("metrics_", "").replace(".jsonl", "")
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]

        steps = [x['step'] for x in d]
        rwds = [x['mean_reward'] for x in d if 'mean_reward' in x]
        losses = [x['loss'] for x in d if 'loss' in x]
        accs = [x['acc'] for x in d if 'acc' in x]
        kls = [x['kl'] for x in d if 'kl' in x]
        
        if rwds: axes[0].plot(steps[:len(rwds)], rwds, label=f'Timeslice {job_id}', color=c, linewidth=2, marker=m, alpha=0.7)
        if accs: axes[1].plot(steps[:len(accs)], accs, label=f'Timeslice {job_id}', color=c, linewidth=2, marker=m, alpha=0.7)
        if kls: axes[2].plot(steps[:len(kls)], kls, label=f'Timeslice {job_id}', color=c, linewidth=2, marker=m, alpha=0.7)
        if losses: axes[3].plot(steps[:len(losses)], losses, label=f'Timeslice {job_id}', color=c, linewidth=2, marker=m, alpha=0.7)

    titles = ["Reward Convergence", "Accuracy Convergence", "KL Divergence", "Training Loss"]
    ylabels = ["Mean Reward", "Accuracy", "KL Divergence (<= 0.1)", "Loss"]
    
    for i in range(4):
        axes[i].set_title(titles[i])
        axes[i].set_ylabel(ylabels[i])
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        
    axes[-1].set_xlabel("RL Step")
    
    plt.tight_layout()
    return get_base64_image(fig)

def run_duty_cycle(log_dir, workspace):
    script_path = os.path.join(workspace, 'tpu_duty_cycle.py')
    if not os.path.exists(script_path): return None, None
    try:
        env = os.environ.copy()
        env['LOG_DIR'] = log_dir
        
        img_duty_path = os.path.join(log_dir, 'tpu_duty_cycle.png')
        img_mem_path = os.path.join(log_dir, 'tpu_mem_cycle.png')
        if os.path.exists(img_duty_path): os.remove(img_duty_path)
        if os.path.exists(img_mem_path): os.remove(img_mem_path)
            
        result = subprocess.run([sys.executable, script_path, '--plot-only'], env=env, capture_output=True)
        img_duty_b64, img_mem_b64 = None, None
        
        if result.returncode == 0:
            if os.path.exists(img_duty_path):
                with open(img_duty_path, 'rb') as f:
                    img_duty_b64 = base64.b64encode(f.read()).decode('utf-8')
            if os.path.exists(img_mem_path):
                with open(img_mem_path, 'rb') as f:
                    img_mem_b64 = base64.b64encode(f.read()).decode('utf-8')
            return img_duty_b64, img_mem_b64
        else:
            print(f"tpu_duty_cycle.png not generated. Exit Code: {result.returncode}")
            if result.stdout: print(f"Stdout:\n{result.stdout.decode('utf-8')}")
            if result.stderr: print(f"Stderr:\n{result.stderr.decode('utf-8')}")
            return None, None
    except subprocess.CalledProcessError as e:
        print(f"subprocess failed: {e}")
        if e.stdout: print(f"Stdout:\n{e.stdout.decode('utf-8')}")
        if e.stderr: print(f"Stderr:\n{e.stderr.decode('utf-8')}")
    except Exception as e:
        print(f"Failed to run duty cycle: {e}")
    return None, None

def load_cloud_metrics(log_dir, container):
    path = os.path.join(log_dir, f"cloud_metrics_{container.replace('-', '_')}.csv")
    if not os.path.exists(path):
        return {}
    data = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for row in csv.DictReader(f):
            chip = row['chip']
            metric = row['metric']
            data[metric][chip].append({
                'time': row['time'],
                'value': float(row['value']),
            })
    return data


def plot_cloud_duty_cycle(baseline_dir, timeslice_dir):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    for col, (log_dir, label) in enumerate([(baseline_dir, 'Baseline'), (timeslice_dir, 'Timeslice')]):
        for row, container in enumerate(['sampler_a', 'trainer_a']):
            ax = axes[row][col]
            data = load_cloud_metrics(log_dir, container)
            dc = data.get('duty_cycle', {})

            if not dc:
                ax.text(0.5, 0.5, 'No Cloud Monitoring data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{label} — {container.replace("_", "-")}')
                continue

            for chip, points in sorted(dc.items()):
                points = sorted(points, key=lambda p: p['time'])
                if not any(p['value'] > 0 for p in points):
                    continue
                times = list(range(len(points)))
                vals = [p['value'] for p in points]
                ax.plot(times, vals, label=f'chip {chip}', linewidth=1.5, alpha=0.8)
                t0_label = points[0]['time'][11:16] if points else ''
                t1_label = points[-1]['time'][11:16] if points else ''
                ax.set_xlabel(f'Minutes ({t0_label} — {t1_label} UTC)')

            ax.set_title(f'{label} — {container.replace("_", "-")} Duty Cycle')
            ax.set_ylabel('Duty Cycle (%)')
            ax.set_ylim(-5, 105)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

    plt.tight_layout()
    return get_base64_image(fig)


def plot_cloud_memory(baseline_dir, timeslice_dir):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    GIB = 1024 ** 3

    for col, (log_dir, label) in enumerate([(baseline_dir, 'Baseline'), (timeslice_dir, 'Timeslice')]):
        for row, container in enumerate(['sampler_a', 'trainer_a']):
            ax = axes[row][col]
            data = load_cloud_metrics(log_dir, container)
            mem = data.get('memory_used', {})

            if not mem:
                ax.text(0.5, 0.5, 'No Cloud Monitoring data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{label} — {container.replace("_", "-")}')
                continue

            for chip, points in sorted(mem.items()):
                points = sorted(points, key=lambda p: p['time'])
                if not any(p['value'] > 0.001 for p in points):
                    continue
                times = list(range(len(points)))
                vals = [p['value'] / GIB for p in points]
                ax.plot(times, vals, label=f'chip {chip}', linewidth=1.5, alpha=0.8)
                t0_label = points[0]['time'][11:16] if points else ''
                t1_label = points[-1]['time'][11:16] if points else ''
                ax.set_xlabel(f'Minutes ({t0_label} — {t1_label} UTC)')

            ax.set_title(f'{label} — {container.replace("_", "-")} HBM Used')
            ax.set_ylabel('HBM Used (GiB)')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

    plt.tight_layout()
    return get_base64_image(fig)


def extract_cloud_metrics_summary(log_dir):
    result = {}
    GIB = 1024 ** 3
    for container in ['sampler_a', 'trainer_a']:
        data = load_cloud_metrics(log_dir, container)
        dc = data.get('duty_cycle', {})
        mem = data.get('memory_used', {})
        mem_total = data.get('memory_total', {})

        # Only include chips that have non-zero activity
        dc_vals = []
        for chip, points in dc.items():
            chip_vals = [p['value'] for p in points]
            if any(v > 0 for v in chip_vals):
                dc_vals.extend(chip_vals)

        mem_vals = []
        for chip, points in mem.items():
            chip_vals = [p['value'] / GIB for p in points]
            if any(v > 0.001 for v in chip_vals):
                mem_vals.extend(chip_vals)

        total_gib = 0
        for chip, points in mem_total.items():
            if points:
                total_gib = points[0]['value'] / GIB
                break

        if dc_vals or mem_vals:
            non_zero_dc = [v for v in dc_vals if v > 0]
            result[container] = {
                'avg_duty': np.mean(dc_vals) if dc_vals else 0,
                'avg_duty_active': np.mean(non_zero_dc) if non_zero_dc else 0,
                'max_duty': max(dc_vals) if dc_vals else 0,
                'active_pct': 100 * len(non_zero_dc) / len(dc_vals) if dc_vals else 0,
                'avg_mem_gib': np.mean(mem_vals) if mem_vals else 0,
                'max_mem_gib': max(mem_vals) if mem_vals else 0,
                'total_mem_gib': total_gib,
            }
    return result


def load_scraper_chip0(log_dir, role):
    path = os.path.join(log_dir, f'tpu_duty_cycle_{role}.csv')
    if not os.path.exists(path):
        return [], [], []
    rows = [r for r in csv.DictReader(open(path)) if r['chip'] == '0']
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
    # Only return chips with any activity
    return {c: d for c, d in by_chip.items() if any(v > 0 for v in d['duty']) or any(v > 0 for v in d['mem'])}


def load_orch_windows(log_dir):
    path = os.path.join(log_dir, 'rl_metrics.jsonl')
    if not os.path.exists(path):
        return [], []
    sampler_windows, trainer_windows = [], []
    sampler_start, trainer_start = {}, {}
    for line in open(path):
        try:
            e = json.loads(line)
        except:
            continue
        t, wl, ts, pool = e.get('type',''), e.get('workload_id',''), e.get('ts',0), e.get('pool','')
        if t == 'acquire':
            if pool == 'sampler': sampler_start[wl] = ts
            elif pool == 'trainer': trainer_start[wl] = ts
        elif t == 'yield':
            if pool == 'sampler' and wl in sampler_start: sampler_windows.append((sampler_start.pop(wl), ts))
            elif pool == 'trainer' and wl in trainer_start: trainer_windows.append((trainer_start.pop(wl), ts))
    return sampler_windows, trainer_windows


def mask_by_orch(timestamps, values, windows):
    return [v if any(s <= t <= e for s, e in windows) else 0.0 for t, v in zip(timestamps, values)]


def plot_scraper_duty_cycle(baseline_dir, timeslice_dir):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    chip_colors = ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#8E24AA', '#F57C00', '#0097A7', '#546E7A']

    for col, (log_dir, label) in enumerate([(baseline_dir, 'Baseline'), (timeslice_dir, 'Timeslice')]):
        s_windows, t_windows = load_orch_windows(log_dir)
        all_starts = []
        for role in ['sampler', 'trainer']:
            chips = load_scraper_all_chips(log_dir, role)
            for cdata in chips.values():
                if cdata['ts']:
                    all_starts.append(cdata['ts'][0])
        t0 = min(all_starts) if all_starts else 0

        for row, (role, windows) in enumerate([('sampler', s_windows), ('trainer', t_windows)]):
            ax = axes[row][col]
            chips = load_scraper_all_chips(log_dir, role)
            if not chips:
                ax.text(0.5, 0.5, 'No scraper data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{label} — {role}')
                continue
            for chip_id in sorted(chips.keys(), key=int):
                cdata = chips[chip_id]
                duty = cdata['duty']
                ts = cdata['ts']
                if windows:
                    duty = mask_by_orch(ts, duty, windows)
                elapsed = [(t - t0) / 60 for t in ts]
                ax.plot(elapsed, duty, color=chip_colors[int(chip_id) % len(chip_colors)],
                        linewidth=0.8, alpha=0.7, label=f'chip {chip_id}')
            ax.set_title(f'{label} — {role} Duty Cycle')
            ax.set_ylabel('Duty Cycle (%)')
            ax.set_xlabel('Elapsed (minutes)')
            ax.set_ylim(-2, 105)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    return get_base64_image(fig)


def plot_scraper_memory(baseline_dir, timeslice_dir):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    GIB = 1024
    chip_colors = ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#8E24AA', '#F57C00', '#0097A7', '#546E7A']

    for col, (log_dir, label) in enumerate([(baseline_dir, 'Baseline'), (timeslice_dir, 'Timeslice')]):
        all_starts = []
        for role in ['sampler', 'trainer']:
            chips = load_scraper_all_chips(log_dir, role)
            for cdata in chips.values():
                if cdata['ts']:
                    all_starts.append(cdata['ts'][0])
        t0 = min(all_starts) if all_starts else 0

        for row, role in enumerate(['sampler', 'trainer']):
            ax = axes[row][col]
            chips = load_scraper_all_chips(log_dir, role)
            if not chips:
                ax.text(0.5, 0.5, 'No scraper data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{label} — {role}')
                continue
            for chip_id in sorted(chips.keys(), key=int):
                cdata = chips[chip_id]
                elapsed = [(t - t0) / 60 for t in cdata['ts']]
                mem_gib = [m / GIB for m in cdata['mem']]
                ax.plot(elapsed, mem_gib, color=chip_colors[int(chip_id) % len(chip_colors)],
                        linewidth=0.8, alpha=0.7, label=f'chip {chip_id}')
            ax.set_title(f'{label} — {role} HBM Used')
            ax.set_ylabel('HBM Used (GiB)')
            ax.set_xlabel('Elapsed (minutes)')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    return get_base64_image(fig)


def extract_scraper_summary(log_dir):
    result = {}
    for role in ['sampler', 'trainer']:
        chips = load_scraper_all_chips(log_dir, role)
        if not chips:
            continue
        all_duty = []
        all_mem = []
        for cdata in chips.values():
            all_duty.extend(cdata['duty'])
            all_mem.extend(cdata['mem'])
        non_zero = [d for d in all_duty if d > 0]
        container = f'{role}_a'
        result[container] = {
            'avg_duty': np.mean(all_duty) if all_duty else 0,
            'active_pct': 100 * len(non_zero) / len(all_duty) if all_duty else 0,
            'avg_duty_active': np.mean(non_zero) if non_zero else 0,
            'avg_mem_gib': np.mean(all_mem) / 1024 if all_mem else 0,
            'max_mem_gib': max(all_mem) / 1024 if all_mem else 0,
        }
    return result


def extract_configs_from_logs(log_dir):
    config_dict = {}
    
    for filename in os.listdir(log_dir):
        if (filename.startswith('rl-job-') or filename.startswith('rl-loop-')) and filename.endswith('.log'):
            log_file = os.path.join(log_dir, filename)
            with open(log_file, 'r') as f:
                content = f.read()
                
            match = re.search(r'steps=(\d+)\s*\|\s*prompts=(\d+)\s*\|\s*G=(\d+)\s*\|\s*max_tokens=(\d+)', content) or \
                    re.search(r'Steps:\s*(\d+),\s*Prompts/step:\s*(\d+),\s*G=(\d+),\s*max_tokens=(\d+)', content)
            if match:
                config_dict['N_RL_STEPS'] = match.group(1)
                config_dict['PROMPTS_PER_STEP'] = match.group(2)
                config_dict['GROUP_SIZE'] = match.group(3)
                config_dict['MAX_NEW_TOKENS'] = match.group(4)
                
            # Search for LR setting manually if present e.g. "LR=5e-6"
            match_lr = re.search(r'(?:LR|lr)=([\d\.e-]+)', content)
            if match_lr:
                config_dict['LR'] = match_lr.group(1)
                
            if config_dict:
                break
                
    if not config_dict:
        return "<p>No config parameters parsed.</p>"
        
    html = "<ul>"
    for k, v in config_dict.items():
        html += f"<li><strong>{k}:</strong> {v}</li>"
    html += "</ul>"
    return html

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--baseline-dir', required=True)
  parser.add_argument('--timeslice-dir', required=True)
  parser.add_argument('--output', default='dashboard.html')
  parser.add_argument('--remove-datapoints', type=int, default=2, help='Number of initial datapoints/steps to remove from the beginning')
  parser.add_argument('--use-scraper', action='store_true', help='Use tpu-info scraper data for duty cycle and HBM plots instead of Cloud Monitoring')
  args = parser.parse_args()

  workspace_path = os.path.dirname(os.path.abspath(__file__))

  bs_metrics = parse_rl_metrics(os.path.join(args.baseline_dir, 'rl_metrics.jsonl'), args.remove_datapoints)
  ts_metrics = parse_rl_metrics(os.path.join(args.timeslice_dir, 'rl_metrics.jsonl'), args.remove_datapoints)

  bs_job_metrics = parse_job_metrics(args.baseline_dir, args.remove_datapoints)
  ts_job_metrics = parse_job_metrics(args.timeslice_dir, args.remove_datapoints)

  time_bs = get_total_time(args.baseline_dir)
  time_ts = get_total_time(args.timeslice_dir)

  img_overheads = plot_overheads_over_time(ts_metrics)
  img_clocks, gen_avgs, sync_avgs, train_avgs, samp_clocks, train_clocks = plot_wall_clocks(
        bs_metrics, ts_metrics, bs_job_metrics, ts_job_metrics, time_bs, time_ts
    )
  img_convergence = plot_convergence(args.baseline_dir, args.timeslice_dir)

  img_bs_duty, img_bs_mem = run_duty_cycle(args.baseline_dir, workspace_path)
  img_ts_duty, img_ts_mem = run_duty_cycle(args.timeslice_dir, workspace_path)

  # TPU duty cycle + memory plots
  if args.use_scraper:
      img_cloud_dc = plot_scraper_duty_cycle(args.baseline_dir, args.timeslice_dir)
      img_cloud_mem = plot_scraper_memory(args.baseline_dir, args.timeslice_dir)
      cloud_bs = extract_scraper_summary(args.baseline_dir)
      cloud_ts = extract_scraper_summary(args.timeslice_dir)
      metrics_source = "tpu-info scraper, all active chips, 1s resolution"
  else:
      img_cloud_dc = plot_cloud_duty_cycle(args.baseline_dir, args.timeslice_dir)
      img_cloud_mem = plot_cloud_memory(args.baseline_dir, args.timeslice_dir)
      cloud_bs = extract_cloud_metrics_summary(args.baseline_dir)
      cloud_ts = extract_cloud_metrics_summary(args.timeslice_dir)
      metrics_source = "GKE Cloud Monitoring API (1-minute resolution)"

  # Calculate Summary Table Metrics
  duty_bs_overall, duty_bs_active = extract_duty_cycle(args.baseline_dir)
  duty_ts_overall, duty_ts_active = extract_duty_cycle(args.timeslice_dir)
  util_bs = extract_avg_util(args.baseline_dir)
  util_ts = extract_avg_util(args.timeslice_dir)

  bs_windows = compute_sliding_windows(args.baseline_dir, window_mins=10)
  ts_windows = compute_sliding_windows(args.timeslice_dir, window_mins=10)

  def build_window_table(title, windows):
    html = f"<div style='flex: 1;'><h3 style='margin-top: 0;'>{title}</h3>"
    html += "<table style='width:100%; border-collapse: collapse; text-align: left; font-size: 13px;'>"
    html += "<tr style='background-color: #ecf0f1; border-bottom: 2px solid #bdc3c7;'><th style='padding: 8px;'>Window (10m)</th><th style='padding: 8px;'>System</th><th style='padding: 8px;'>TPU 0</th><th style='padding: 8px;'>TPU 1</th></tr>"
    for w in windows:
      html += f"<tr style='border-bottom: 1px solid #ecf0f1;'>"
      html += f"<td style='padding: 8px;'>{w['start']} - {w['end']}</td>"
      html += f"<td style='padding: 8px; font-weight: bold;'>{w['sys']:.1f}%</td>"
      html += f"<td style='padding: 8px;'>{w['g0']:.1f}%</td>"
      html += f"<td style='padding: 8px;'>{w['g1']:.1f}%</td>"
      html += "</tr>"
    html += "</table></div>"
    return html

  window_html_bs = build_window_table("Baseline Windows", bs_windows)
  window_html_ts = build_window_table("Timeslice Windows", ts_windows)

  # Calculate Overheads per Pool
  ts_trainer_evict, ts_trainer_restore = [], []
  ts_sampler_evict, ts_sampler_restore = [], []

  overheads = ts_metrics['overheads']
  for wid, vals in overheads['evict'].items():
    if 'trainer' in wid: ts_trainer_evict.extend(vals)
    if 'sampler' in wid: ts_sampler_evict.extend(vals)

  for wid, vals in overheads['restore'].items():
    if 'trainer' in wid: ts_trainer_restore.extend(vals)
    if 'sampler' in wid: ts_sampler_restore.extend(vals)

  avg_trainer_evict = np.mean(ts_trainer_evict) if ts_trainer_evict else 0
  avg_trainer_restore = np.mean(ts_trainer_restore) if ts_trainer_restore else 0
  avg_sampler_evict = np.mean(ts_sampler_evict) if ts_sampler_evict else 0
  avg_sampler_restore = np.mean(ts_sampler_restore) if ts_sampler_restore else 0

  summary_html = f"""
    <table style="width:100%; border-collapse: collapse; text-align: left; font-size: 15px;">
        <tr style="background-color: #ecf0f1; border-bottom: 2px solid #bdc3c7;">
            <th style="padding: 10px;">Metric</th>
            <th style="padding: 10px;">Baseline (1 RL Job)</th>
            <th style="padding: 10px;">TimeSliced (2 RL Jobs)</th>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1; background-color: #fcfcfc;">
            <td colspan="3" style="padding: 10px; font-weight: bold; color: #7f8c8d;">TPU Duty Cycle ({metrics_source})</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1;">
            <td style="padding: 10px; padding-left: 20px;">Sampler Avg Duty Cycle</td>
            <td style="padding: 10px; font-weight: bold;">{"%.1f%%" % cloud_bs.get('sampler_a', {}).get('avg_duty', cloud_bs.get('sampler_a', {}).get('avg_duty_active', 0)) if cloud_bs.get('sampler_a') else "N/A"}</td>
            <td style="padding: 10px; font-weight: bold;">{"%.1f%%" % cloud_ts.get('sampler_a', {}).get('avg_duty', cloud_ts.get('sampler_a', {}).get('avg_duty_active', 0)) if cloud_ts.get('sampler_a') else "N/A"}</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1;">
            <td style="padding: 10px; padding-left: 20px;">Trainer Avg Duty Cycle</td>
            <td style="padding: 10px; font-weight: bold;">{"%.1f%%" % cloud_bs.get('trainer_a', {}).get('avg_duty', cloud_bs.get('trainer_a', {}).get('avg_duty_active', 0)) if cloud_bs.get('trainer_a') else "N/A"}</td>
            <td style="padding: 10px; font-weight: bold;">{"%.1f%%" % cloud_ts.get('trainer_a', {}).get('avg_duty', cloud_ts.get('trainer_a', {}).get('avg_duty_active', 0)) if cloud_ts.get('trainer_a') else "N/A"}</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1; background-color: #fcfcfc;">
            <td colspan="3" style="padding: 10px; font-weight: bold; color: #7f8c8d;">TPU HBM Usage ({metrics_source})</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1;">
            <td style="padding: 10px; padding-left: 20px;">Sampler Avg HBM</td>
            <td style="padding: 10px;">{"%.1f GiB" % cloud_bs.get('sampler_a', {}).get('avg_mem_gib', 0) if cloud_bs.get('sampler_a') else "N/A"}</td>
            <td style="padding: 10px;">{"%.1f GiB" % cloud_ts.get('sampler_a', {}).get('avg_mem_gib', 0) if cloud_ts.get('sampler_a') else "N/A"}</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1;">
            <td style="padding: 10px; padding-left: 20px;">Trainer Avg HBM</td>
            <td style="padding: 10px;">{"%.1f GiB" % cloud_bs.get('trainer_a', {}).get('avg_mem_gib', 0) if cloud_bs.get('trainer_a') else "N/A"}</td>
            <td style="padding: 10px;">{"%.1f GiB" % cloud_ts.get('trainer_a', {}).get('avg_mem_gib', 0) if cloud_ts.get('trainer_a') else "N/A"}</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1;">
            <td style="padding: 10px; font-weight: bold;">Total Execution Time</td>
            <td style="padding: 10px;">{time_bs/60:.1f}m</td>
            <td style="padding: 10px;">{time_ts/60:.1f}m</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1; background-color: #fcfcfc;">
            <td colspan="3" style="padding: 10px; font-weight: bold; color: #7f8c8d;">Raw Mathematical Times (Averages)</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1;">
            <td style="padding: 10px; padding-left: 20px;">Sampling (Generation)</td>
            <td style="padding: 10px;">{gen_avgs[0]:.1f}s</td>
            <td style="padding: 10px;">{gen_avgs[1]:.1f}s</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1;">
            <td style="padding: 10px; padding-left: 20px;">Training</td>
            <td style="padding: 10px;">{train_avgs[0]:.1f}s</td>
            <td style="padding: 10px;">{train_avgs[1]:.1f}s</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1;">
            <td style="padding: 10px; padding-left: 20px;">Weight Sync</td>
            <td style="padding: 10px;">{sync_avgs[0]:.1f}s</td>
            <td style="padding: 10px;">{sync_avgs[1]:.1f}s</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1; background-color: #fcfcfc;">
            <td colspan="3" style="padding: 10px; font-weight: bold; color: #7f8c8d;">Wall Clock Times (acquire &rarr; yield)</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1;">
            <td style="padding: 10px; padding-left: 20px;">Sampling Phase</td>
            <td style="padding: 10px;">{samp_clocks[0]:.1f}s</td>
            <td style="padding: 10px;">{samp_clocks[1]:.1f}s</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1;">
            <td style="padding: 10px; padding-left: 20px;">Training Phase</td>
            <td style="padding: 10px;">{train_clocks[0]:.1f}s</td>
            <td style="padding: 10px;">{train_clocks[1]:.1f}s</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1; background-color: #fcfcfc;">
            <td colspan="3" style="padding: 10px; font-weight: bold; color: #7f8c8d;">System Context Switch Overheads (Averages)</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1;">
            <td style="padding: 10px; padding-left: 20px;">Trainer Evict Time</td>
            <td style="padding: 10px;">N/A</td>
            <td style="padding: 10px;">{avg_trainer_evict:.1f}s</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1;">
            <td style="padding: 10px; padding-left: 20px;">Trainer Restore Time</td>
            <td style="padding: 10px;">N/A</td>
            <td style="padding: 10px;">{avg_trainer_restore:.1f}s</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1;">
            <td style="padding: 10px; padding-left: 20px;">Sampler Evict Time</td>
            <td style="padding: 10px;">N/A</td>
            <td style="padding: 10px;">{avg_sampler_evict:.1f}s</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1;">
            <td style="padding: 10px; padding-left: 20px;">Sampler Restore Time</td>
            <td style="padding: 10px;">N/A</td>
            <td style="padding: 10px;">{avg_sampler_restore:.1f}s</td>
        </tr>
    </table>
    """

  conv_bs = extract_convergence_metrics(args.baseline_dir)
  conv_ts = extract_convergence_metrics(args.timeslice_dir)

  ts_a = conv_ts.get('a', {'first_10': {}, 'last_10': {}})
  ts_b = conv_ts.get('b', {'first_10': {}, 'last_10': {}})

  bs_acc_delta = conv_bs['overall']['last_10'].get('acc', 0) - conv_bs['overall']['first_10'].get('acc', 0)
  ts_a_acc_delta = ts_a['last_10'].get('acc', 0) - ts_a['first_10'].get('acc', 0)
  ts_b_acc_delta = ts_b['last_10'].get('acc', 0) - ts_b['first_10'].get('acc', 0)

  bs_kl_delta = conv_bs['overall']['last_10'].get('kl', 0) - conv_bs['overall']['first_10'].get('kl', 0)
  ts_a_kl_delta = ts_a['last_10'].get('kl', 0) - ts_a['first_10'].get('kl', 0)
  ts_b_kl_delta = ts_b['last_10'].get('kl', 0) - ts_b['first_10'].get('kl', 0)

  bs_loss_delta = conv_bs['overall']['last_10'].get('loss', 0) - conv_bs['overall']['first_10'].get('loss', 0)
  ts_a_loss_delta = ts_a['last_10'].get('loss', 0) - ts_a['first_10'].get('loss', 0)
  ts_b_loss_delta = ts_b['last_10'].get('loss', 0) - ts_b['first_10'].get('loss', 0)

  bs_rwd_delta = conv_bs['overall']['last_10'].get('mean_reward', 0) - conv_bs['overall']['first_10'].get('mean_reward', 0)
  ts_a_rwd_delta = ts_a['last_10'].get('mean_reward', 0) - ts_a['first_10'].get('mean_reward', 0)
  ts_b_rwd_delta = ts_b['last_10'].get('mean_reward', 0) - ts_b['first_10'].get('mean_reward', 0)

  conv_summary_html = f"""
    <p>This table compares the rolling averages of the first 10 steps versus the final 10 steps to isolate rate of improvement over time.</p>
    <table style="width:100%; border-collapse: collapse; text-align: left; font-size: 15px;">
        <tr style="background-color: #ecf0f1; border-bottom: 2px solid #bdc3c7;">
            <th style="padding: 10px;">Metric</th>
            <th style="padding: 10px;">Baseline</th>
            <th style="padding: 10px;">TimeSliced (Job A)</th>
            <th style="padding: 10px;">TimeSliced (Job B)</th>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1; background-color: #fcfcfc;">
            <td colspan="4" style="padding: 10px; font-weight: bold; color: #7f8c8d;">Accuracy</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1;">
            <td style="padding: 10px; padding-left: 20px;">Accuracy (First 10 Avg)</td>
            <td style="padding: 10px;">{conv_bs['overall']['first_10'].get('acc', 0):.2%}</td>
            <td style="padding: 10px;">{ts_a['first_10'].get('acc', 0):.2%}</td>
            <td style="padding: 10px;">{ts_b['first_10'].get('acc', 0):.2%}</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1;">
            <td style="padding: 10px; padding-left: 20px;">Accuracy (Last 10 Avg)</td>
            <td style="padding: 10px;">{conv_bs['overall']['last_10'].get('acc', 0):.2%}</td>
            <td style="padding: 10px;">{ts_a['last_10'].get('acc', 0):.2%}</td>
            <td style="padding: 10px;">{ts_b['last_10'].get('acc', 0):.2%}</td>
        </tr>
        <tr style="border-bottom: 1px solid #bdc3c7; background-color: #f9f9f9;">
            <td style="padding: 10px; padding-left: 20px; font-style: italic;">&Delta; Overall Improvement</td>
            <td style="padding: 10px; font-style: italic; font-weight: bold; color: {'#27ae60' if bs_acc_delta > 0 else '#c0392b'};">{bs_acc_delta:+.2%}</td>
            <td style="padding: 10px; font-style: italic; font-weight: bold; color: {'#27ae60' if ts_a_acc_delta > 0 else '#c0392b'};">{ts_a_acc_delta:+.2%}</td>
            <td style="padding: 10px; font-style: italic; font-weight: bold; color: {'#27ae60' if ts_b_acc_delta > 0 else '#c0392b'};">{ts_b_acc_delta:+.2%}</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1; background-color: #fcfcfc;">
            <td colspan="4" style="padding: 10px; font-weight: bold; color: #7f8c8d;">KL Divergence</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1;">
            <td style="padding: 10px; padding-left: 20px;">KL (First 10 Avg)</td>
            <td style="padding: 10px;">{conv_bs['overall']['first_10'].get('kl', 0):.3f}</td>
            <td style="padding: 10px;">{ts_a['first_10'].get('kl', 0):.3f}</td>
            <td style="padding: 10px;">{ts_b['first_10'].get('kl', 0):.3f}</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1;">
            <td style="padding: 10px; padding-left: 20px;">KL (Last 10 Avg)</td>
            <td style="padding: 10px;">{conv_bs['overall']['last_10'].get('kl', 0):.3f}</td>
            <td style="padding: 10px;">{ts_a['last_10'].get('kl', 0):.3f}</td>
            <td style="padding: 10px;">{ts_b['last_10'].get('kl', 0):.3f}</td>
        </tr>
        <tr style="border-bottom: 1px solid #bdc3c7; background-color: #f9f9f9;">
            <td style="padding: 10px; padding-left: 20px; font-style: italic;">&Delta; Overall Change</td>
            <td style="padding: 10px; font-style: italic; font-weight: bold; color: {'#c0392b' if bs_kl_delta > 0.05 else '#27ae60'};">{bs_kl_delta:+.3f}</td>
            <td style="padding: 10px; font-style: italic; font-weight: bold; color: {'#c0392b' if ts_a_kl_delta > 0.05 else '#27ae60'};">{ts_a_kl_delta:+.3f}</td>
            <td style="padding: 10px; font-style: italic; font-weight: bold; color: {'#c0392b' if ts_b_kl_delta > 0.05 else '#27ae60'};">{ts_b_kl_delta:+.3f}</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1; background-color: #fcfcfc;">
            <td colspan="4" style="padding: 10px; font-weight: bold; color: #7f8c8d;">Training Loss</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1;">
            <td style="padding: 10px; padding-left: 20px;">Loss (First 10 Avg)</td>
            <td style="padding: 10px;">{conv_bs['overall']['first_10'].get('loss', 0):.3f}</td>
            <td style="padding: 10px;">{ts_a['first_10'].get('loss', 0):.3f}</td>
            <td style="padding: 10px;">{ts_b['first_10'].get('loss', 0):.3f}</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1;">
            <td style="padding: 10px; padding-left: 20px;">Loss (Last 10 Avg)</td>
            <td style="padding: 10px;">{conv_bs['overall']['last_10'].get('loss', 0):.3f}</td>
            <td style="padding: 10px;">{ts_a['last_10'].get('loss', 0):.3f}</td>
            <td style="padding: 10px;">{ts_b['last_10'].get('loss', 0):.3f}</td>
        </tr>
        <tr style="border-bottom: 1px solid #bdc3c7; background-color: #f9f9f9;">
            <td style="padding: 10px; padding-left: 20px; font-style: italic;">&Delta; Overall Change</td>
            <td style="padding: 10px; font-style: italic; font-weight: bold; color: {'#c0392b' if bs_loss_delta > 0 else '#27ae60'};">{bs_loss_delta:+.3f}</td>
            <td style="padding: 10px; font-style: italic; font-weight: bold; color: {'#c0392b' if ts_a_loss_delta > 0 else '#27ae60'};">{ts_a_loss_delta:+.3f}</td>
            <td style="padding: 10px; font-style: italic; font-weight: bold; color: {'#c0392b' if ts_b_loss_delta > 0 else '#27ae60'};">{ts_b_loss_delta:+.3f}</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1; background-color: #fcfcfc;">
            <td colspan="4" style="padding: 10px; font-weight: bold; color: #7f8c8d;">Reward</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1;">
            <td style="padding: 10px; padding-left: 20px;">Mean Reward (First 10 Avg)</td>
            <td style="padding: 10px;">{conv_bs['overall']['first_10'].get('mean_reward', 0):.3f}</td>
            <td style="padding: 10px;">{ts_a['first_10'].get('mean_reward', 0):.3f}</td>
            <td style="padding: 10px;">{ts_b['first_10'].get('mean_reward', 0):.3f}</td>
        </tr>
        <tr style="border-bottom: 1px solid #ecf0f1;">
            <td style="padding: 10px; padding-left: 20px;">Mean Reward (Last 10 Avg)</td>
            <td style="padding: 10px;">{conv_bs['overall']['last_10'].get('mean_reward', 0):.3f}</td>
            <td style="padding: 10px;">{ts_a['last_10'].get('mean_reward', 0):.3f}</td>
            <td style="padding: 10px;">{ts_b['last_10'].get('mean_reward', 0):.3f}</td>
        </tr>
        <tr style="border-bottom: 1px solid #bdc3c7; background-color: #f9f9f9;">
            <td style="padding: 10px; padding-left: 20px; font-style: italic;">&Delta; Overall Improvement</td>
            <td style="padding: 10px; font-style: italic; font-weight: bold; color: {'#27ae60' if bs_rwd_delta > 0 else '#c0392b'};">{bs_rwd_delta:+.3f}</td>
            <td style="padding: 10px; font-style: italic; font-weight: bold; color: {'#27ae60' if ts_a_rwd_delta > 0 else '#c0392b'};">{ts_a_rwd_delta:+.3f}</td>
            <td style="padding: 10px; font-style: italic; font-weight: bold; color: {'#27ae60' if ts_b_rwd_delta > 0 else '#c0392b'};">{ts_b_rwd_delta:+.3f}</td>
        </tr>
    </table>
    """

  configs_html = f"<h4>Baseline</h4>{extract_configs_from_logs(args.baseline_dir)}"
  configs_html += f"<h4>Timeslice</h4>{extract_configs_from_logs(args.timeslice_dir)}"

  html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>GKE TPU RL Time-Slicing Dashboard</title>
        <style>
            body {{ font-family: -apple-system, sans-serif; margin: 40px; background: #f8f9fa; color: #333; }}
            .container {{ max-width: 1400px; margin: auto; background: white; padding: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-radius: 8px; }}
            h1, h2 {{ color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; margin-top: 0; }}
            .card {{ margin-top: 30px; border: 1px solid #ddd; border-radius: 5px; padding: 20px; background: #fff; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 GKE TPU Time-Slicing: Reinforcement Learning Dashboard</h1>
            
            <div class="card" style="background: #ffffff; border-left: 4px solid #4A9EE0;">
                <h2>Summary</h2>
                {summary_html}
            </div>

            <div class="card">
                <h2>1. Execution Breakdown</h2>
                <img src="data:image/png;base64,{img_clocks}">
            </div>

            <div class="card">
                <h2>2. System Overheads Over Time</h2>
                <p>Track Wait, Eviction, and Restore times per job/pool over the run length.</p>
                <img src="data:image/png;base64,{img_overheads}">
            </div>
            
            <div class="card">
                <h2>3. TPU Duty Cycle</h2>
                <p>Source: {metrics_source}.
                   Baseline should show alternating high/low between sampler and trainer (square wave).
                   Timeslice should show near-continuous utilization as jobs interleave.</p>
                <img src="data:image/png;base64,{img_cloud_dc}">
                <h3>HBM Memory Usage</h3>
                <img src="data:image/png;base64,{img_cloud_mem}">
            </div>

            <div class="card">
                <h2>4. Run Configuration (From Logs)</h2>
                {configs_html}
            </div>

            <div class="card">
                <h2>5. Model Convergence Analysis</h2>
                {conv_summary_html}
                <br>
                <img src="data:image/png;base64,{img_convergence}">
            </div>
        </div>
    </body>
    </html>
    """

  with open(args.output, 'w') as f:
    f.write(html)

  print(f"✅ Dashboard generated successfully!")

if __name__ == '__main__':
    main()