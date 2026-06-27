"""collect_cloud_metrics.py — Fetch TPU metrics from Cloud Monitoring API

Queries tensorcore_utilization and memory_bandwidth_utilization for
sampler-a and trainer-a containers, saves as CSV for dashboard generation.

Usage:
  python3 collect_cloud_metrics.py \
    --project aishuk-test \
    --start 2026-06-27T05:00:00Z \
    --end   2026-06-27T06:00:00Z \
    --output-dir runs/full_5step/baseline/
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime


METRICS = [
    "kubernetes.io/container/accelerator/duty_cycle",
    "kubernetes.io/container/accelerator/memory_used",
    "kubernetes.io/container/accelerator/memory_total",
]

CONTAINERS = ["sampler-a", "trainer-a"]


def get_token():
    result = subprocess.run(
        ["gcloud", "auth", "application-default", "print-access-token"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Failed to get token: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def query_metric(project, metric_type, container, start, end, token):
    import urllib.parse

    filter_str = (
        f'metric.type="{metric_type}" '
        f'AND resource.labels.container_name="{container}"'
    )
    params = urllib.parse.urlencode({
        "filter": filter_str,
        "interval.startTime": start,
        "interval.endTime": end,
    })
    url = f"https://monitoring.googleapis.com/v3/projects/{project}/timeSeries?{params}"
    result = subprocess.run(
        ["curl", "-s", "-H", f"Authorization: Bearer {token}", url],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"curl failed: {result.stderr}", file=sys.stderr)
        return {}
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="aishuk-test")
    parser.add_argument("--start", required=True, help="ISO8601 UTC start time")
    parser.add_argument("--end", required=True, help="ISO8601 UTC end time")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    token = get_token()

    for container in CONTAINERS:
        rows = []
        for metric_type in METRICS:
            metric_short = metric_type.split("/")[-1]
            data = query_metric(
                args.project, metric_type, container,
                args.start, args.end, token,
            )
            for ts in data.get("timeSeries", []):
                accel_id = ts.get("metric", {}).get("labels", {}).get("accelerator_id", "")
                chip = accel_id.split("-")[-1] if "-" in accel_id else "0"
                for point in ts.get("points", []):
                    end_time = point["interval"]["endTime"]
                    v = point["value"]
                    value = v.get("doubleValue", v.get("int64Value", 0))
                    if isinstance(value, str):
                        value = float(value)
                    rows.append({
                        "time": end_time,
                        "container": container,
                        "chip": chip,
                        "metric": metric_short,
                        "value": round(value, 2),
                    })

        if rows:
            rows.sort(key=lambda r: (r["time"], r["chip"], r["metric"]))
            out_path = os.path.join(args.output_dir, f"cloud_metrics_{container.replace('-', '_')}.csv")
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["time", "container", "chip", "metric", "value"])
                writer.writeheader()
                writer.writerows(rows)
            print(f"  {out_path}: {len(rows)} rows")

    # Also write a combined summary CSV with just the active chips
    all_rows = []
    for container in CONTAINERS:
        path = os.path.join(args.output_dir, f"cloud_metrics_{container.replace('-', '_')}.csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                if float(row["value"]) > 0:
                    all_rows.append(row)

    if all_rows:
        all_rows.sort(key=lambda r: r["time"])
        summary_path = os.path.join(args.output_dir, "cloud_metrics_summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["time", "container", "chip", "metric", "value"])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"  {summary_path}: {len(all_rows)} non-zero rows")


if __name__ == "__main__":
    main()
