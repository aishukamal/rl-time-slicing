#!/usr/bin/env python3
# elastic-rl-poc M3 RUN B — forced regime shift (C5): mid-run generation
# length-cap flip 16384 -> 8192, restart-free.
#
# Mechanism (verified against pinned verl 983cb0f2):
#   - Per-request max_tokens defaults to the SERVER-side rollout config:
#     vLLMHttpServer.generate() computes
#       max_tokens = min(self.config.response_length,
#                        self.config.prompt_length + self.config.response_length
#                        - len(prompt_ids))
#     when the client sampling_params carry no max_tokens — which is the case
#     for the fully-async agent-loop path (agent_loop.py:590-596 builds
#     sampling_params WITHOUT max_tokens). (vllm_async_server.py:574-590)
#   - RolloutConfig explicitly lists "response_length" in _mutable_fields
#     (verl/workers/config/rollout.py:146-156), so setattr on the live server
#     actor's config is a supported, restart-free push. OmegaConf DictConfig
#     (the R2 path, which received the raw omegaconf node) also allows setting
#     existing keys under struct mode.
#   - The flip is applied via actor.__ray_call__ (the same escape hatch
#     r2_lifecycle.py uses for PID discovery) on:
#       * every R1 standalone server (rollouter.get_replicas() -> rep.servers
#         — proven handle path from M1), and
#       * the out-of-tree R2 server (handles actor key "r2_server"). R2's
#         SERVER actor process is never frozen (only its GPU engine
#         subprocesses are cuda-checkpointed), so the call is safe while R2
#         is parked; the new cap takes effect on R2's next activation.
#   - NOT touched: AgentLoopWorker padding/truncation width
#     (rollout_config.response_length inside the rollouter process) and the
#     trainer's tensor shapes — they stay at 16384 so every batch keeps the
#     same layout across the boundary; responses simply never exceed 8192
#     after the flip. In-flight 16K requests finish at their original cap.
#
# Trigger: tails train.log until the step-FLIP_AT_STEP metrics line appears
# (i.e. step FLIP_AT_STEP completed), then flips. Every action is logged to
# /workspace/results/regime_flip.jsonl and stdout (prefix [regime-flip]).

import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import r2_lifecycle as rlc

RESULTS_DIR = "/workspace/results"
TRAIN_LOG = os.path.join(RESULTS_DIR, "train.log")
FLIP_LOG = os.path.join(RESULTS_DIR, "regime_flip.jsonl")

FLIP_AT_STEP = int(os.environ.get("ELASTIC_FLIP_AT_STEP", "12"))
NEW_LEN = int(os.environ.get("ELASTIC_FLIP_NEW_LEN", "8192"))
HARD_TIMEOUT_S = float(os.environ.get("ELASTIC_FLIP_TIMEOUT", "12000"))  # give up (log) after this

RE_STEP = re.compile(r"step:(\d+) - .*?timing_s/step:([\d.]+)")


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def say(msg):
    print(f"[{now_iso()}] [regime-flip] {msg}", flush=True)


def record(rec):
    rec = {"ts": now_iso(), **rec}
    with open(FLIP_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def wait_for_step(target):
    say(f"waiting for step {target} to complete in {TRAIN_LOG} "
        f"(hard timeout {HARD_TIMEOUT_S:.0f}s)")
    t0 = time.time()
    offset = 0
    buf = b""
    last_heartbeat = t0
    while True:
        if time.time() - t0 > HARD_TIMEOUT_S:
            record({"event": "flip_timeout", "waited_s": round(time.time() - t0, 1)})
            say(f"TIMEOUT: step {target} never appeared within {HARD_TIMEOUT_S:.0f}s — "
                f"no flip performed (run continues as a plain 16K run)")
            return None
        try:
            size = os.path.getsize(TRAIN_LOG)
        except OSError:
            time.sleep(10)
            continue
        if size > offset:
            with open(TRAIN_LOG, "rb") as f:
                f.seek(offset)
                data = f.read(size - offset)
            offset = size
            buf += data
            *lines, buf = buf.split(b"\n")
            for raw in lines:
                m = RE_STEP.search(raw.decode(errors="replace"))
                if m and int(m.group(1)) >= target:
                    say(f"step {m.group(1)} completed (timing_s/step={m.group(2)}) — trigger")
                    return int(m.group(1))
        if time.time() - last_heartbeat > 600:
            last_heartbeat = time.time()
            say(f"still waiting for step {target} (waited {time.time() - t0:.0f}s)")
        time.sleep(5)


def flip_server(ctx, name, server, new_len):
    """Set config.response_length on a live server actor; returns (old, new)."""
    def _do(self, nl=new_len):
        old = self.config.response_length
        try:
            self.config.response_length = int(nl)   # RolloutConfig: _mutable_fields
        except Exception:
            import omegaconf
            omegaconf.OmegaConf.update(self.config, "response_length", int(nl))
        return old, self.config.response_length
    old, new = ctx.ray.get(server.__ray_call__.remote(_do), timeout=60)
    say(f"{name}: response_length {old} -> {new}")
    return old, new


def main():
    triggered_step = wait_for_step(FLIP_AT_STEP)
    if triggered_step is None:
        return 1

    say("connecting to Ray / handles actor...")
    ctx = rlc.ElasticContext()

    results = {}
    # R1 standalone server(s): proven handle path (r2_lifecycle
    # discover_ce_worker_pids uses the same get_replicas()).
    replicas = ctx.ray.get(ctx.rollouter.get_replicas.remote())
    r1_servers = [s for rep in replicas for s in (getattr(rep, "servers", []) or [])]
    say(f"found {len(r1_servers)} R1 server actor(s) via rollouter.get_replicas()")
    for i, s in enumerate(r1_servers):
        results[f"r1_server_{i}"] = flip_server(ctx, f"r1_server_{i}", s, NEW_LEN)

    # R2 (out-of-tree) server actor — alive even while its engine is parked.
    try:
        results["r2_server"] = flip_server(ctx, "r2_server", ctx.r2_server, NEW_LEN)
    except Exception as e:
        say(f"WARNING: R2 flip failed ({e}) — R1 flip still in effect")
        results["r2_server"] = ("error", str(e)[:200])

    rec = record({
        "event": "regime_flip_done",
        "flip_at_step": FLIP_AT_STEP,
        "triggered_step": triggered_step,
        "new_response_length": NEW_LEN,
        "results": {k: list(v) for k, v in results.items()},
    })
    say(f"REGIME-FLIP COMPLETE: {json.dumps(rec)}")

    # Read-back verification 30s later (fresh call, catches accidental resets).
    time.sleep(30)
    try:
        readback = {}
        for i, s in enumerate(r1_servers):
            readback[f"r1_server_{i}"] = ctx.ray.get(
                s.__ray_call__.remote(lambda self: self.config.response_length), timeout=60)
        readback["r2_server"] = ctx.ray.get(
            ctx.r2_server.__ray_call__.remote(lambda self: self.config.response_length), timeout=60)
        record({"event": "flip_readback", "values": readback})
        say(f"readback: {readback}")
    except Exception as e:
        say(f"readback failed (non-fatal): {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
