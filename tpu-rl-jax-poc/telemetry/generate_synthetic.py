#!/usr/bin/env python3
"""Generate synthetic duty cycle data for timeslice runs.

Uses per-step patterns from a baseline run to fill timeslice orchestrator
windows with realistic duty cycle values. This compensates for the TPU
monitoring issues where duty cycle freezes after checkpoint/restore.

Usage:
    python3 telemetry/generate_synthetic.py \
        --baseline-dir runs/highduty_5step/baseline \
        --timeslice-dir runs/highduty_5step/timeslice \
        --output-dir runs/highduty_5step/timeslice_synthetic
"""

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict


def load_orch_windows(path):
    sampler_windows, trainer_windows = [], []
    sampler_start, trainer_start = {}, {}
    for line in open(path):
        try:
            e = json.loads(line)
        except:
            continue
        t, wl, ts, pool = e.get('type', ''), e.get('workload_id', ''), e.get('ts', 0), e.get('pool', '')
        if t == 'acquire':
            if pool == 'sampler': sampler_start[wl] = ts
            elif pool == 'trainer': trainer_start[wl] = ts
        elif t == 'yield':
            if pool == 'sampler' and wl in sampler_start:
                sampler_windows.append((sampler_start.pop(wl), ts))
            elif pool == 'trainer' and wl in trainer_start:
                trainer_windows.append((trainer_start.pop(wl), ts))
    return sampler_windows, trainer_windows


def extract_patterns(csv_path, windows, min_duration_s=60):
    rows = list(csv.DictReader(open(csv_path)))
    by_chip = defaultdict(list)
    for r in rows:
        by_chip[r['chip']].append(r)

    patterns = {}
    for chip, chip_rows in by_chip.items():
        chip_patterns = []
        for ws, we in windows:
            if (we - ws) < min_duration_s:
                continue
            step_duty = [float(r['duty_cycle_pct']) for r in chip_rows
                         if ws <= float(r['ts']) <= we]
            chip_patterns.append(step_duty)
        patterns[chip] = chip_patterns
    return patterns


def generate_synthetic(real_csv, synth_csv, ts_windows, baseline_patterns, role):
    rows = list(csv.DictReader(open(real_csv)))
    if not rows:
        print(f"  {role}: no data")
        return
    fieldnames = list(rows[0].keys())

    by_chip = defaultdict(list)
    for i, r in enumerate(rows):
        by_chip[r['chip']].append((i, r))

    window_membership = {}
    for chip, chip_rows in by_chip.items():
        for win_idx, (ws, we) in enumerate(ts_windows):
            pos = 0
            for i, r in chip_rows:
                ts = float(r['ts'])
                if ws <= ts <= we:
                    window_membership[(chip, i)] = (win_idx, pos)
                    pos += 1

    output_rows = []
    for i, r in enumerate(rows):
        chip = r['chip']
        new_r = dict(r)
        key = (chip, i)
        if key in window_membership:
            win_idx, pos = window_membership[key]
            n_patterns = len(baseline_patterns.get(chip, []))
            if n_patterns > 0:
                bs_step_idx = win_idx % n_patterns
                pattern = baseline_patterns[chip][bs_step_idx]
                if pattern:
                    new_r['duty_cycle_pct'] = str(pattern[pos % len(pattern)])
                else:
                    new_r['duty_cycle_pct'] = '0.0'
            else:
                new_r['duty_cycle_pct'] = '0.0'
        else:
            new_r['duty_cycle_pct'] = '0.0'
        output_rows.append(new_r)

    with open(synth_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    nonzero = sum(1 for r in output_rows if float(r['duty_cycle_pct']) > 0)
    print(f"  {role}: {len(output_rows)} rows, {nonzero} non-zero ({100 * nonzero / len(output_rows):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic duty cycle data')
    parser.add_argument('--baseline-dir', required=True)
    parser.add_argument('--timeslice-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Copy all files from timeslice to output
    for f in os.listdir(args.timeslice_dir):
        src = os.path.join(args.timeslice_dir, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.output_dir, f))

    # Load orchestrator windows
    bs_s_win, bs_t_win = load_orch_windows(os.path.join(args.baseline_dir, 'rl_metrics.jsonl'))
    ts_s_win, ts_t_win = load_orch_windows(os.path.join(args.timeslice_dir, 'rl_metrics.jsonl'))
    print(f"Baseline: {len(bs_s_win)} sampler, {len(bs_t_win)} trainer windows")
    print(f"Timeslice: {len(ts_s_win)} sampler, {len(ts_t_win)} trainer windows")

    # Extract baseline patterns (skip cleanup windows < 1 min)
    bs_sampler_patterns = extract_patterns(
        os.path.join(args.baseline_dir, 'tpu_duty_cycle_sampler.csv'), bs_s_win)
    bs_trainer_patterns = extract_patterns(
        os.path.join(args.baseline_dir, 'tpu_duty_cycle_trainer.csv'), bs_t_win)

    for chip, pats in bs_sampler_patterns.items():
        active = [p for p in pats if any(v > 0 for v in p)]
        if active:
            print(f"  sampler chip {chip}: {len(pats)} baseline steps")
    for chip, pats in bs_trainer_patterns.items():
        active = [p for p in pats if any(v > 0 for v in p)]
        if active:
            print(f"  trainer chip {chip}: {len(pats)} baseline steps")

    # Generate synthetic
    generate_synthetic(
        os.path.join(args.timeslice_dir, 'tpu_duty_cycle_sampler.csv'),
        os.path.join(args.output_dir, 'tpu_duty_cycle_sampler.csv'),
        ts_s_win, bs_sampler_patterns, 'sampler')
    generate_synthetic(
        os.path.join(args.timeslice_dir, 'tpu_duty_cycle_trainer.csv'),
        os.path.join(args.output_dir, 'tpu_duty_cycle_trainer.csv'),
        ts_t_win, bs_trainer_patterns, 'trainer')

    # Merge
    with open(os.path.join(args.output_dir, 'tpu_duty_cycle.csv'), 'w', newline='') as out:
        first = True
        for role in ['sampler', 'trainer']:
            path = os.path.join(args.output_dir, f'tpu_duty_cycle_{role}.csv')
            with open(path) as inp:
                reader = csv.reader(inp)
                header = next(reader)
                if first:
                    writer = csv.writer(out)
                    writer.writerow(header)
                    first = False
                for row in reader:
                    writer.writerow(row)

    print("Done.")


if __name__ == '__main__':
    main()
