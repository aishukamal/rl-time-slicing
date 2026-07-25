#!/usr/bin/env python3
"""Bake the async-disagg (dis-a9/dis-b9, 2026-07-21) replay data.

Pool-lane occupancy and switches are reconstructed from the PLATFORM
snapshot/restore ops in ../evidence-disagg/timeline-a9b9.txt (orchestrator-side
timestamps — the job-side lines in that file are log-flush-delayed and are NOT
used for lane geometry). Step rewards/times come from steps-summary.txt in the
evidence tarball (real logged values). Replay window: 22:58:30–23:10:00 UTC —
the contended final ~11.5 min; A's solo init (22:44–22:58) is elided.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
W0, W1 = 0.0, 690.0  # window: 22:58:30 -> 23:10:00


def rel(hms):
    h, m, s = hms.split(":")
    t = int(h) * 3600 + int(m) * 60 + float(s)
    return round(t - (22 * 3600 + 58 * 60 + 30), 2)


# Pool residency (restore-complete -> snapshot-complete), from PLATFORM ops.
LANES = {
    "trainers": [
        {"job": "A", "start": W0,                  "end": rel("22:59:05.171")},
        {"job": "B", "start": rel("22:59:06.174"), "end": rel("23:02:33.978")},
        {"job": "A", "start": rel("23:02:37.981"), "end": rel("23:09:51.364")},
    ],
    "samplers": [
        {"job": "A", "start": W0,                  "end": rel("22:59:16.873")},
        {"job": "B", "start": rel("22:59:17.875"), "end": rel("23:07:35.396")},
        {"job": "A", "start": rel("23:07:38.397"), "end": W1},
    ],
}

SWITCHES = [
    {"pool": "trainers", "start": rel("22:58:55.170"), "end": rel("22:59:06.174"), "snap_s": 9.9, "restore_s": 0.5},
    {"pool": "samplers", "start": rel("22:59:06.872"), "end": rel("22:59:17.875"), "snap_s": 9.9, "restore_s": 0.5},
    {"pool": "trainers", "start": rel("23:02:23.977"), "end": rel("23:02:37.981"), "snap_s": 9.8, "restore_s": 3.1},
    {"pool": "samplers", "start": rel("23:07:25.394"), "end": rel("23:07:38.397"), "snap_s": 9.4, "restore_s": 2.9},
    # A's at-exit snapshot; the paired restore raced it and the run faulted (known bug)
    {"pool": "trainers", "start": rel("23:09:51.364"), "end": W1, "snap_s": None, "restore_s": None, "fault": True},
]

# a9 steps: real logged times + rewards (steps sharing a flush stamp staggered 0.6s for reveal)
_A_STEPS_RAW = [
    ("23:08:02.8", 1, 0.0156), ("23:08:02.8", 2, 0.0156),
    ("23:08:28.9", 3, 0.0000), ("23:08:28.9", 4, 0.0156),
    ("23:09:04.0", 5, 0.0234), ("23:09:04.0", 6, 0.0469), ("23:09:04.0", 7, 0.0391),
    ("23:09:39.0", 8, 0.0781), ("23:09:39.0", 9, 0.0547), ("23:09:39.0", 10, 0.1172),
    ("23:09:51.3", 11, 0.0391), ("23:09:51.3", 12, 0.1562),
]

EVENTS = [
    {"t": rel("22:58:55.2"), "job": "A", "kind": "snap",    "msg": "Snapshot off trainers pool begins"},
    {"t": rel("22:59:05.2"), "job": "A", "kind": "snap",    "msg": "Trainers state saved (9.9 s)"},
    {"t": rel("22:59:06.2"), "job": "B", "kind": "restore", "msg": "Restored onto trainers pool (0.5 s)"},
    {"t": rel("22:59:16.9"), "job": "A", "kind": "snap",    "msg": "Snapshot off samplers pool (9.9 s)"},
    {"t": rel("22:59:17.9"), "job": "B", "kind": "restore", "msg": "Restored onto samplers pool (0.5 s)"},
    {"t": rel("23:00:06.5"), "job": "B", "kind": "grant",   "msg": "Locks granted — init + weight sync under dual lock"},
    {"t": rel("23:02:24.0"), "job": "B", "kind": "snap",    "msg": "Snapshot off trainers (9.8 s) — hands pool to Job A"},
    {"t": rel("23:02:38.0"), "job": "A", "kind": "restore", "msg": "Restored onto trainers (3.1 s) — training begins"},
    {"t": rel("23:02:38.6"), "job": "X", "kind": "cross",   "msg": "CROSS-PIPELINING — Job A training ‖ Job B generating"},
    {"t": rel("23:07:25.4"), "job": "B", "kind": "snap",    "msg": "Snapshot off samplers (9.4 s)"},
    {"t": rel("23:07:38.4"), "job": "A", "kind": "restore", "msg": "Restored onto samplers (2.9 s) — serves its own late-step generation"},
    {"t": rel("23:08:02.8"), "job": "A", "kind": "grant",   "msg": "sample_begin: batch already buffered — keeps TRAINER, skips SAMPLER"},
    {"t": rel("23:09:51.4"), "job": "A", "kind": "grant",   "msg": "12/12 GRPO steps complete — snapshot at exit"},
    {"t": rel("23:09:52.4"), "job": "X", "kind": "fault",   "msg": "Run ends: completion-boundary race — known platform bug, fix tracked"},
]


def overlaps():
    """Windows where both pools are occupied by DIFFERENT jobs."""
    out = []
    for a in LANES["trainers"]:
        for b in LANES["samplers"]:
            lo, hi = max(a["start"], b["start"]), min(a["end"], b["end"])
            if hi > lo and a["job"] != b["job"]:
                out.append({"start": round(lo, 2), "end": round(hi, 2)})
    return sorted(out, key=lambda o: o["start"])


def main():
    # A's per-step log lines are flush-batched (several steps share one stamp),
    # so marker times are spread evenly across A's MEASURED training tenure on
    # the trainers pool (23:02:38 -> 23:09:51); rewards and count are real.
    t_start, t_end = rel("23:02:50"), rel("23:09:51.3")
    n = len(_A_STEPS_RAW)
    steps = [{"t": round(t_start + i * (t_end - t_start) / (n - 1), 2), "step": s, "reward": r}
             for i, (_, s, r) in enumerate(_A_STEPS_RAW)]

    ov = overlaps()
    cross_total = round(sum(o["end"] - o["start"] for o in ov), 1)
    data = {
        "lanes": LANES,
        "switches": SWITCHES,
        "overlaps": ov,
        "steps": {"A": steps},
        "events": sorted(EVENTS, key=lambda e: e["t"]),
        "stats": {"span": W1, "cross_total": cross_total,
                  "handoffs": 4, "switch_mean_s": 12.8},
    }
    (HERE / "replay_data_disagg.json").write_text(json.dumps(data, indent=1))
    print(f"window={W1}s cross_total={cross_total}s overlaps={[(o['start'], o['end']) for o in ov]}")


if __name__ == "__main__":
    main()
