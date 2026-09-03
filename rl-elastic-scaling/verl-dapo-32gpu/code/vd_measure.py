#!/usr/bin/env python3
# elastic-vd — S1/S3/S4 measurement from a fully-async train.log.
#
# Per micro-step (32 samples = 512 responses): the trainer's own console
# metrics line carries timing_s/step, timing_s/gen (the marked_timer around
# _get_samples_from_queue = EXPOSED gen wait for this step), plus
# fully_async metrics. param_sync lines close each sync block.
#
# Outputs a per-step table + steady-state aggregates:
#   exposed_gen  = timing_s/gen
#   train_wall   = timing_s/step - timing_s/gen  (compute + overhead;
#                  includes param_sync on block-closing steps — split out)
#   ratio        = sum(exposed_gen) / sum(train_wall_ex_sync)   [S1 GATE]
#   exposed share= sum(exposed_gen) / sum(step)
# Also: samples/GPU-hour = (steps*512 responses) / (32 GPUs * wall hours).

import argparse
import json
import re
import sys

RE_METRICS = re.compile(r"step:(\d+) - (.*)$")
RE_KV = re.compile(r"([\w\-/@]+):([-+eE\d.]+)")
RE_PSYNC = re.compile(r"timing_s/param_sync: ([\d.]+) seconds self\.current_param_version: (\d+)")


def parse(path):
    steps, psyncs = [], []
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode(errors="replace")
            m = RE_PSYNC.search(line)
            if m:
                psyncs.append({"wall": float(m.group(1)), "version": int(m.group(2))})
            m = RE_METRICS.search(line)
            if m:
                kv = {}
                for k, v in RE_KV.findall(m.group(2)):
                    try:
                        kv[k] = float(v)
                    except ValueError:
                        pass
                if "timing_s/step" in kv:
                    kv["step_idx"] = int(m.group(1))
                    steps.append(kv)
    return steps, psyncs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--skip", type=int, default=2, help="warmup micro-steps to exclude")
    ap.add_argument("--responses-per-step", type=int, default=512)
    ap.add_argument("--gpus", type=int, default=32)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    steps, psyncs = parse(args.log)
    if not steps:
        print("no metric step lines found yet")
        return 1

    print(f"{'idx':>4} {'step_s':>8} {'gen_s':>8} {'train_s':>8} {'psync_s':>8} {'ratio':>6}")
    rows = []
    for s in steps:
        step_w = s.get("timing_s/step", 0.0)
        gen_w = s.get("timing_s/gen", 0.0)
        ps = s.get("timing_s/param_sync", 0.0)
        train_w = max(step_w - gen_w - ps, 0.0)
        rows.append({"idx": s["step_idx"], "step": step_w, "gen": gen_w,
                     "train": train_w, "psync": ps,
                     "ratio": (gen_w / train_w) if train_w > 0 else None})
        r = rows[-1]
        print(f"{r['idx']:>4} {r['step']:>8.1f} {r['gen']:>8.1f} {r['train']:>8.1f} "
              f"{r['psync']:>8.1f} {(r['ratio'] or 0):>6.2f}")

    steady = rows[args.skip:]
    if steady:
        tg = sum(r["gen"] for r in steady)
        tt = sum(r["train"] for r in steady)
        tw = sum(r["step"] for r in steady)
        tp = sum(r["psync"] for r in steady)
        ratio = tg / tt if tt else float("inf")
        share = tg / tw if tw else 0
        sph = len(steady) * args.responses_per_step / (args.gpus * tw / 3600) if tw else 0
        summary = {
            "n_steps": len(steady), "skip": args.skip,
            "sum_step_s": round(tw, 1), "sum_gen_s": round(tg, 1),
            "sum_train_s": round(tt, 1), "sum_psync_s": round(tp, 1),
            "gen_train_ratio": round(ratio, 3),
            "exposed_gen_share": round(share, 4),
            "responses_per_gpu_hour": round(sph, 1),
            "mean_step_s": round(tw / len(steady), 1),
            "param_syncs_seen": len(psyncs),
        }
        print("\nSTEADY-STATE SUMMARY (post-skip):")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(f"\nS1 GATE (>= 1.5): gen:train = {ratio:.2f} -> {'PASS' if ratio >= 1.5 else 'FAIL'}")
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump({"rows": rows, "summary": summary, "psyncs": psyncs}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
