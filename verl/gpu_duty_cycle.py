#!/usr/bin/env python3
"""gpu_duty_cycle.py — Manual GPU utilization scraper for GKE time-slicing comparison.

Runs as a sidecar or standalone pod on the GPU node.
Polls nvidia-smi every POLL_INTERVAL seconds and records:
  - GPU utilization % per physical GPU
  - Memory used MiB per physical GPU
  - Timestamp

Writes to /data/rl_logs/gpu_duty_cycle.csv and generates a comparison PNG
when interrupted (or after DURATION_MINS minutes).

Usage:
  # Run continuously until Ctrl-C or timeout:
  python3 gpu_duty_cycle.py

  # Generate plot from existing CSV:
  python3 gpu_duty_cycle.py --plot-only

  # Label a phase for before/after comparison:
  PHASE=baseline  python3 gpu_duty_cycle.py
  PHASE=timeslice python3 gpu_duty_cycle.py
"""

import argparse, csv, logging, os, signal, subprocess, time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [duty_cycle] %(message)s"
)
log = logging.getLogger("duty_cycle")

POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "1"))  # seconds
DURATION_MINS = float(os.environ.get("DURATION_MINS", "0"))  # 0 = run forever
LOG_DIR = os.environ.get("LOG_DIR", "/data/rl_logs")
TITLE = os.environ.get("TITLE", "")
PHASE = os.environ.get("PHASE", "run")  # label for CSV
NVIDIA_SMI = os.environ.get("NVIDIA_SMI", "/usr/local/nvidia/bin/nvidia-smi")

CSV_PATH = Path(LOG_DIR) / "gpu_duty_cycle.csv"
PNG_PATH = Path(LOG_DIR) / "gpu_duty_cycle.png"

_running = True


def signal_handler(sig, frame):
  global _running
  _running = False


def query_gpus() -> list[dict]:
  """Query all physical GPUs via nvidia-smi.

  Returns list of {index, util_pct, mem_used_mib, mem_total_mib}.
  """
  try:
    result = subprocess.run(
        [
            NVIDIA_SMI,
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    rows = []
    for line in result.stdout.strip().splitlines():
      parts = [p.strip() for p in line.split(",")]
      if len(parts) == 4:
        rows.append({
            "gpu": int(parts[0]),
            "util_pct": int(parts[1]),
            "mem_used": int(parts[2]),
            "mem_total": int(parts[3]),
        })
    return rows
  except Exception as e:
    log.warning(f"nvidia-smi query failed: {e}")
    return []


def scrape(duration_mins: float):
  Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
  write_header = not CSV_PATH.exists()

  log.info(f"Scraping GPU duty cycle every {POLL_INTERVAL}s → {CSV_PATH}")
  log.info(
      f"Phase: {PHASE} | Duration:"
      f" {'∞' if duration_mins == 0 else f'{duration_mins}min'}"
  )

  deadline = (
      time.time() + duration_mins * 60 if duration_mins > 0 else float("inf")
  )
  n_samples = 0

  with open(CSV_PATH, "a", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "ts",
            "wall",
            "phase",
            "gpu",
            "util_pct",
            "mem_used",
            "mem_total",
        ],
    )
    if write_header:
      writer.writeheader()

    while _running and time.time() < deadline:
      ts = time.time()
      wall = datetime.utcfromtimestamp(ts).strftime("%H:%M:%S")
      gpus = query_gpus()

      for g in gpus:
        writer.writerow({
            "ts": round(ts, 3),
            "wall": wall,
            "phase": PHASE,
            **g,
        })
        if n_samples % 30 == 0:  # log every 60s
          log.info(
              f"  GPU{g['gpu']}: {g['util_pct']:3d}%  "
              f"{g['mem_used']}/{g['mem_total']} MiB"
          )

      f.flush()
      n_samples += 1
      time.sleep(POLL_INTERVAL)

  log.info(f"Scrape done — {n_samples} samples written to {CSV_PATH}")


def plot():
  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv as _csv

    rows = []
    with open(CSV_PATH) as f:
      for r in _csv.DictReader(f):
        rows.append(r)

    if not rows:
      log.warning("No data to plot")
      return

    # Group by (phase, gpu)
    from collections import defaultdict

    series = defaultdict(lambda: {"ts": [], "util": [], "mem": []})
    t0 = float(rows[0]["ts"])
    for r in rows:
      key = (r["phase"], int(r["gpu"]))
      ts_min = (float(r["ts"]) - t0) / 60
      
      # If gap is > 10 seconds, insert NaN to break the Matplotlib line
      if series[key]["ts"] and (ts_min - series[key]["ts"][-1]) > (10.0 / 60.0):
          series[key]["ts"].append(series[key]["ts"][-1] + 0.001)
          series[key]["util"].append(float('nan'))
          series[key]["mem"].append(float('nan'))
          
      series[key]["ts"].append(ts_min)
      series[key]["util"].append(float(r["util_pct"]))
      series[key]["mem"].append(float(r["mem_used"]))

    phases = sorted(set(r["phase"] for r in rows))
    gpus = sorted(set(int(r["gpu"]) for r in rows))
    
    # Pre-calculate active time window (first to last non-zero util across all GPUs) per phase
    phase_active_windows = {}
    for phase in phases:
        active_ts = [
            (float(r["ts"]) - t0) / 60 for r in rows
            if r["phase"] == phase and float(r["util_pct"]) > 0
        ]
        if active_ts:
            phase_active_windows[phase] = (min(active_ts), max(active_ts))
        else:
            phase_active_windows[phase] = (None, None)

    colors = [
        "#4285F4",
        "#34A853",
        "#FBBC05",
        "#EA4335",
        "#3DDC84",
        "#B794F4",
        "#FC8181",
        "#68D391",
    ]

    n_gpus = len(gpus)
    fig, axes = plt.subplots(n_gpus, 1, figsize=(14, 4 * n_gpus), squeeze=False)
    fig.suptitle(f"GPU Utilization {TITLE}", fontsize=13)

    phase_colors = {p: colors[i % len(colors)] for i, p in enumerate(phases)}

    for row_idx, gpu_idx in enumerate(gpus):
      ax = axes[row_idx][0]
      ax.set_title(f"GPU {gpu_idx}", fontsize=11)
      ax.set_ylabel("Utilization %")
      ax.set_ylim(0, 105)
      ax.set_xlabel("Time (min)")
      ax.grid(True, alpha=0.3)

      for phase in phases:
        key = (phase, gpu_idx)
        if key not in series:
          continue
        d = series[key]
        start_ts, end_ts = phase_active_windows[phase]

        valid_util = [u for u in d['util'] if u == u]
        avg_util = sum(valid_util) / len(valid_util) if valid_util else 0
        overall_duty = sum(1 for u in valid_util if u > 0) / len(valid_util) * 100 if valid_util else 0
        
        active_util = []
        if start_ts is not None and end_ts is not None:
             for ts, u in zip(d["ts"], d["util"]):
                 if u == u and start_ts <= ts <= end_ts:
                     active_util.append(u)
        active_duty = (sum(1 for u in active_util if u > 0) / len(active_util) * 100) if active_util else 0
        
        duration = max(d["ts"]) - min(d["ts"]) if d["ts"] else 0
        
        col = phase_colors[phase]
        
        ax.fill_between(d["ts"], d["util"], alpha=0.12, color=col, step="post")
        ax.step(
            d["ts"],
            d["util"],
            color=col,
            linewidth=1.5,
            where="post",
            label=f"{phase} (avg {avg_util:.0f}%, duty cycle (overall) {overall_duty:.0f}%, duty cycle (active) {active_duty:.0f}%, window {duration:.1f}m)",
        )

      ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(str(PNG_PATH), dpi=120)
    plt.close()
    log.info(f"Plot saved: {PNG_PATH}")

    # Generate Memory Plot
    PNG_MEM_PATH = Path(LOG_DIR) / "gpu_mem_cycle.png"
    fig_mem, axes_mem = plt.subplots(n_gpus, 1, figsize=(14, 4 * n_gpus), squeeze=False)
    fig_mem.suptitle(f"GPU Memory (GB) {TITLE}", fontsize=13)

    for row_idx, gpu_idx in enumerate(gpus):
      ax = axes_mem[row_idx][0]
      ax.set_title(f"GPU {gpu_idx}", fontsize=11)
      ax.set_ylabel("Memory (GB)")
      ax.set_ylim(0, 85)
      ax.set_xlabel("Time (min)")
      ax.grid(True, alpha=0.3)

      for phase in phases:
        key = (phase, gpu_idx)
        if key not in series:
          continue
        d = series[key]

        valid_mem = [m/1024 for m in d['mem'] if m == m]
        avg_mem = sum(valid_mem) / len(valid_mem) if valid_mem else 0
        peak_mem = max(valid_mem) if valid_mem else 0
        
        col = phase_colors[phase]
        mem_gb = [m/1024 if m == m else float('nan') for m in d['mem']]
        
        ax.fill_between(d["ts"], mem_gb, alpha=0.12, color=col, step="post")
        ax.step(
            d["ts"],
            mem_gb,
            color=col,
            linewidth=1.5,
            where="post",
            label=f"{phase} (avg {avg_mem:.1f} GB, peak {peak_mem:.1f} GB)",
        )

      ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(str(PNG_MEM_PATH), dpi=120)
    plt.close()
    log.info(f"Plot saved: {PNG_MEM_PATH}")

  except ImportError:
    log.warning("matplotlib not available")
  except Exception as e:
    log.error(f"Plot failed: {e}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--plot-only",
      action="store_true",
      help="Skip scraping, just regenerate plot from existing CSV",
  )
  args = parser.parse_args()

  signal.signal(signal.SIGINT, signal_handler)
  signal.signal(signal.SIGTERM, signal_handler)

  if args.plot_only:
    plot()
  else:
    scrape(DURATION_MINS)
    plot()
