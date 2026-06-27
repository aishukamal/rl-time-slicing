"""tpu_duty_cycle.py — TPU utilization and memory tracking via tpu-info

Polls tpu-info's get_chip_usage() every POLL_INTERVAL seconds (default 1s),
writing duty_cycle_pct and memory stats to CSV. Equivalent of gpu_duty_cycle.py
which uses nvidia-smi.

Usage:
  python tpu_duty_cycle.py                    # collect data
  python tpu_duty_cycle.py --plot-only        # generate plot from existing CSV

Env vars:
  POLL_INTERVAL   sampling interval in seconds (default: 1)
  DURATION_MINS   how long to collect (default: 0 = forever)
  PHASE           label for this collection phase (default: "baseline")
  CSV_FILE        output CSV path (default: /data/rl_logs/tpu_duty_cycle.csv)
"""

import csv
import os
import signal
import sys
import time
from datetime import datetime, timezone

POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "1"))
DURATION_MINS = float(os.environ.get("DURATION_MINS", "0"))
PHASE = os.environ.get("PHASE", "baseline")
CSV_FILE = os.environ.get("CSV_FILE", "/data/rl_logs/tpu_duty_cycle.csv")
PNG_FILE = os.environ.get("PNG_FILE", CSV_FILE.replace(".csv", ".png"))

FIELDS = ["ts", "wall", "phase", "chip", "duty_cycle_pct", "mem_used_mib", "mem_total_mib", "mem_pct"]

_stop = False


def _handle_signal(sig, frame):
    global _stop
    _stop = True


def collect():
    from tpu_info.device import get_local_chips
    from tpu_info.metrics import get_chip_usage

    chip_info = get_local_chips()
    if not chip_info:
        print("No TPU chips found")
        return

    chip_type, chip_count = chip_info
    print(f"Tracking {chip_count} {chip_type} chips, interval={POLL_INTERVAL}s, phase={PHASE}")

    os.makedirs(os.path.dirname(CSV_FILE) or ".", exist_ok=True)
    write_header = not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    deadline = time.time() + DURATION_MINS * 60 if DURATION_MINS > 0 else float("inf")

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()

        while not _stop and time.time() < deadline:
            ts = time.time()
            wall = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")

            try:
                usages = get_chip_usage(chip_type)
            except Exception as e:
                print(f"Error polling tpu-info: {e}")
                time.sleep(POLL_INTERVAL)
                continue

            for u in usages:
                mem_mib = round(u.memory_usage / (1024 * 1024))
                total_mib = round(u.total_memory / (1024 * 1024))
                mem_pct = round(100 * u.memory_usage / u.total_memory, 1) if u.total_memory > 0 else 0

                writer.writerow({
                    "ts": round(ts, 3),
                    "wall": wall,
                    "phase": PHASE,
                    "chip": u.device_id,
                    "duty_cycle_pct": round(u.duty_cycle_pct, 1),
                    "mem_used_mib": mem_mib,
                    "mem_total_mib": total_mib,
                    "mem_pct": mem_pct,
                })

            f.flush()
            time.sleep(POLL_INTERVAL)

    print(f"Collection stopped. {CSV_FILE}")


def plot():
    if not os.path.exists(CSV_FILE):
        print(f"No data file: {CSV_FILE}")
        return

    rows = []
    with open(CSV_FILE) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        print("No data rows")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    chips = sorted(set(int(r["chip"]) for r in rows))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for chip in chips[:4]:  # first 4 chips
        chip_rows = [r for r in rows if int(r["chip"]) == chip]
        ts = [float(r["ts"]) for r in chip_rows]
        t0 = ts[0] if ts else 0
        elapsed = [(t - t0) / 60 for t in ts]
        duty = [float(r["duty_cycle_pct"]) for r in chip_rows]
        mem = [float(r["mem_pct"]) for r in chip_rows]

        axes[0].plot(elapsed, duty, label=f"chip {chip}", alpha=0.7)
        axes[1].plot(elapsed, mem, label=f"chip {chip}", alpha=0.7)

    axes[0].set_ylabel("Duty Cycle (%)")
    axes[0].set_title("TPU Duty Cycle per Chip (via tpu-info)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-5, 105)

    axes[1].set_ylabel("HBM Memory Usage (%)")
    axes[1].set_xlabel("Elapsed (minutes)")
    axes[1].set_title("TPU HBM Usage per Chip")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PNG_FILE, dpi=150)
    print(f"Plot saved: {PNG_FILE}")

    # Print summary
    total_samples = len(rows)
    avg_duty = round(sum(float(r["duty_cycle_pct"]) for r in rows) / total_samples, 1) if total_samples else 0
    active_samples = sum(1 for r in rows if float(r["duty_cycle_pct"]) > 0)
    duty_cycle = round(100 * active_samples / total_samples, 1) if total_samples else 0
    avg_mem = round(sum(float(r["mem_pct"]) for r in rows) / total_samples, 1) if total_samples else 0
    print(f"Avg duty cycle: {avg_duty}%")
    print(f"Time with non-zero duty: {duty_cycle}% ({active_samples}/{total_samples} samples)")
    print(f"Avg HBM usage: {avg_mem}%")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot()
    else:
        collect()
