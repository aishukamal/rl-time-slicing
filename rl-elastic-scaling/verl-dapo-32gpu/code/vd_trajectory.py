#!/usr/bin/env python3
# elastic-vd — per-sync-block trajectory extractor for fully-async train.log.
# The console logger emits one AGGREGATED metrics line per sync block
# ("step:N - k:v - ..."), where N = param version. Keys are sums/means over
# the block's 16 micro-steps per MetricsAggregator rules.
import re
import sys

RE_METRICS = re.compile(r"step:(\d+) - (.*)$")

WANT = [
    "response_length/mean",
    "timing_s/step",
    "timing_s/gen",
    "timing_s/update_actor",
    "fully_async/trainer/idle_ratio",
    "fully_async/rollouter/idle_ratio",
    "fully_async/total_wait_time",
    "perf/mfu/actor",
    "perf/throughput",
    "critic/rewards/mean",
    "fully_async/partial/partial_ratio",
    "fully_async/count/total_generated_samples",
]


def main(path):
    blocks = {}
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode(errors="replace")
            m = RE_METRICS.search(line)
            if not m:
                continue
            v = int(m.group(1))
            kv = blocks.setdefault(v, {})
            for part in m.group(2).split(" - "):
                if ":" not in part:
                    continue
                k, _, val = part.partition(":")
                k = k.strip()
                if k in WANT:
                    try:
                        kv[k] = float(val)
                    except ValueError:
                        pass
    cols = ["ver", "len", "step_s", "gen_s", "upd_s", "tr_idle", "ro_idle", "wait_s", "mfu", "reward", "ratio"]
    print("\t".join(cols))
    for v in sorted(blocks):
        b = blocks[v]
        step = b.get("timing_s/step")
        gen = b.get("timing_s/gen")
        upd = b.get("timing_s/update_actor")
        ratio = (gen / (step - gen)) if (step and gen is not None and step > gen) else None
        row = [
            v,
            round(b.get("response_length/mean", -1), 0),
            round(step, 1) if step else None,
            round(gen, 1) if gen is not None else None,
            round(upd, 1) if upd else None,
            round(b.get("fully_async/trainer/idle_ratio", -1), 3),
            round(b.get("fully_async/rollouter/idle_ratio", -1), 3),
            round(b.get("fully_async/total_wait_time", -1), 1),
            round(b.get("perf/mfu/actor", -1), 4),
            round(b.get("critic/rewards/mean", -99), 3),
            round(ratio, 3) if ratio is not None else None,
        ]
        print("\t".join(str(x) for x in row))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/results/train.log")
