#!/usr/bin/env python3
# elastic-rl-poc M2 — closed-loop policy controller for trainer <-> R2 GPU
# time-slicing. Replaces the manual/scripted trigger logic of m1/cycles.sh
# with the PLAN.md §2 policy: ETA gate for switch-in, predictive switch-back,
# min-dwell / staleness / one-pair-per-step guards, and the no-harm property
# (if the gate never clears the controller does nothing and logs why).
#
# Runs INSIDE the training pod as a single long-lived process (nohup after
# launch; it self-gates on steady state). Reuses m1/r2_lifecycle.py verbatim
# for the actual switch mechanics (two-sided NCCL suspend + confirmation,
# cuda-checkpoint via snapshot-agent) — imported, not reimplemented.
#
# Signals per poll (~2s):
#   - MessageQueue.get_statistics(): queue_size, total_produced (-> fill-rate
#     EMA over produced deltas; consumption bursts don't pollute it),
#     dropped_samples.
#   - trainer phase from /workspace/results/train.log (incremental parse):
#     'Requesting N samples from queue' opens a gen-wait block,
#     'sample collected i/N. mq_len: m' tracks progress,
#     'timing_s/param_sync ... current_param_version: k' closes a step.
#     The log lines come from the ElasticFullyAsyncTrainer ACTOR process,
#     which is never frozen (only the FSDP worker pids are), so parsing works
#     while the trainer is suspended — verified in M1 run4 (cycles.sh used
#     the same source while the trainer was frozen).
#   - FullyAsyncRollouter.get_statistics(): dropped_stale delta (staleness
#     guard), active_tasks (diagnostics).
#
# Policy (PLAN.md §2):
#   SWITCH-IN   iff trainer blocked in gen-wait AND collected >= 1 (backlog
#               burst absorbed) AND ETA > c * round_trip (c=1.5 default;
#               round_trip = rt_in_ema + rt_out_ema, seeded from run4:
#               37.8s + 47.8s ≈ 86s, updated online from this run's own
#               measured switch totals) AND guards clear.
#               ETA = (samples_needed - collected - queue_size) / fill_EMA.
#   SWITCH-BACK predictively when ETA_remaining <= rt_out_ema + margin
#               (default 15s), so the trainer wakes as the batch completes
#               (target: wake <= 30s after batch-ready). Failsafes: hard
#               collected threshold (default 60/64, improves on cycles.sh's
#               56) and a window cap (default 600s).
#   GUARDS      min-dwell 60s between switch operations; max 1 switch pair
#               per gen-wait block (block id = count of 'Requesting' lines);
#               staleness headroom (no dropped_stale delta in the last 120s,
#               mq dropped_samples == 0); live-mode + agent-job-RUNNING gates.
#   NO-HARM     every poll that takes no action logs the gate values and the
#               first failing reason to decisions.jsonl (M3 regret input).
#
# Modes: starts in DRY-RUN (actions disabled, decisions logged with
# action='would_*') for --dry-run-steps completed steps, then waits for the
# operator to touch --live-flag (or --auto-live) before enabling real
# switches. Fully autonomous thereafter — no operator switch commands.
#
# Abort criteria (M1 rules):
#   - a switch operation hangs > --op-timeout (360s): capture + exit(3)
#   - 2 consecutive failed switch-ins: stop switching, keep run alive, exit(2)
#   - any switch-back failure: state ambiguous -> capture + exit(4)
#   - no param_sync within 900s of a switch-back: dormant-NCCL suspect ->
#     capture + exit(5)
#   - misfire property: the switch-in gate REQUIRES blocked-in-gen-wait as its
#     first conjunct, re-evaluated from a fresh log parse in the same tick the
#     action fires; if the block closed mid-decision the action is skipped.

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone

import r2_lifecycle as rlc

RESULTS_DIR = "/workspace/results"
TRAIN_LOG = os.path.join(RESULTS_DIR, "train.log")
DECISIONS_FILE = os.path.join(RESULTS_DIR, "decisions.jsonl")
LIVE_FLAG = os.path.join(RESULTS_DIR, "controller_live.flag")
STOP_FLAG = os.path.join(RESULTS_DIR, "controller_stop.flag")

RE_REQUEST = re.compile(r"\[FullyAsyncTrainer\] Requesting (\d+) samples from queue")
RE_COLLECT = re.compile(r"\[FullyAsyncTrainer\] sample collected (\d+)/(\d+)\. mq_len: (\d+)")
RE_PSYNC = re.compile(r"timing_s/param_sync: ([\d.]+) seconds self\.current_param_version: (\d+)")
RE_STEP = re.compile(r"step:(\d+) - .*?timing_s/step:([\d.]+)")
RE_ACTOR_ERR = re.compile(r"RayActorError")


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# Startup-gate hardening (M2 attempt-1 post-mortem): no controller wait state
# may spin silently. Every gate logs progress and has a hard timeout that
# exits nonzero so the failure is visible in pod stdout within minutes, not
# after the 4.5h window. The steady-state gate legitimately takes ~25 min
# (3 param syncs at ~615s/step from a cold start — measured in attempt 1),
# so it gets a loud 20-min warning and a 45-min hard cap; all other gates
# get the 20-min hard cap.
GATE_WARN_S = 1200          # loud warning threshold (all gates)
GATE_TIMEOUT_S = 1200       # hard cap: agent-RUNNING gate, handles-actor gate
STEADY_TIMEOUT_S = 2700     # hard cap: steady-state gate (3 steps ~= 31 min worst case)
EXIT_GATE_TIMEOUT = 7


def state_name(s):
    # The timeslice python client returns JobStatus.state as the enum
    # NAME string (e.g. 'JOB_STATE_RUNNING'), not the proto int.
    # pb.JobState.Name(str) raises TypeError — this exact call, without
    # the str fallback cycles.sh always had, kept the M2 attempt-1
    # controller stuck in its wait loop for the entire 4h live window
    # (968 status polls, zero switches). Keep both paths. Module-level so
    # the pre-flight probe imports and exercises the EXACT code path.
    from timeslice.snapshot_agent import snapshot_agent_pb2 as pb
    try:
        return pb.JobState.Name(s)
    except Exception:
        return str(s)


def agent_job_states(agent):
    """The exact status-parse expression the controller gates on.
    Exercised verbatim by the pre-flight probe before every run."""
    from timeslice.snapshot_agent import SnapshotAgentClient
    with SnapshotAgentClient(agent) as c:
        return {j.job_id: state_name(j.state) for j in c.status().job_statuses}


class TrainLogParser:
    """Incremental parser of the driver's train.log (tee'd stdout)."""

    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.buf = b""
        # cumulative state
        self.requests = 0            # count of 'Requesting N samples' lines
        self.samples_needed = 64
        self.collected = 0           # last 'i' in current block (reset on Requesting)
        self.collected_ts = None
        self.mq_len_log = 0
        self.block_open = False      # Requesting seen, i/N not yet == N
        self.block_open_ts = None
        self.batch_ready_ts = None   # ts when collected hit N for current block
        self.param_syncs = 0
        self.param_version = -1
        self.last_psync_ts = None
        self.steps = []              # (step_idx, timing_s/step)
        self.ray_actor_error = False

    def poll(self):
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return
        if size < self.offset:  # truncated/rotated (should not happen)
            self.offset = 0
            self.buf = b""
        if size == self.offset:
            return
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            data = f.read(size - self.offset)
        self.offset = size
        self.buf += data
        *lines, self.buf = self.buf.split(b"\n")
        t = time.time()
        for raw in lines:
            line = raw.decode(errors="replace")
            m = RE_REQUEST.search(line)
            if m:
                self.requests += 1
                self.samples_needed = int(m.group(1))
                self.collected = 0
                self.block_open = True
                self.block_open_ts = t
                self.batch_ready_ts = None
                continue
            m = RE_COLLECT.search(line)
            if m:
                self.collected = int(m.group(1))
                self.samples_needed = int(m.group(2))
                self.mq_len_log = int(m.group(3))
                self.collected_ts = t
                if self.collected >= self.samples_needed:
                    self.block_open = False
                    self.batch_ready_ts = t
                continue
            m = RE_PSYNC.search(line)
            if m:
                self.param_syncs += 1
                self.param_version = int(m.group(2))
                self.last_psync_ts = t
                continue
            m = RE_STEP.search(line)
            if m:
                self.steps.append((int(m.group(1)), float(m.group(2))))
                continue
            if RE_ACTOR_ERR.search(line):
                self.ray_actor_error = True


class Ema:
    """Time-aware EMA: alpha = 1 - exp(-dt/tau)."""

    def __init__(self, tau, value=None):
        self.tau = tau
        self.value = value
        self.t = None

    def update(self, x, t=None):
        t = t if t is not None else time.time()
        if self.value is None or self.t is None:
            self.value = x
        else:
            dt = max(t - self.t, 1e-6)
            a = 1.0 - math.exp(-dt / self.tau)
            self.value = a * x + (1 - a) * self.value
        self.t = t
        return self.value

    def reset(self):
        self.value = None
        self.t = None

    # NOTE: Controller.fill_updates is reset alongside via Controller._reset_fill.


class Controller:
    def __init__(self, args):
        self.args = args
        # The handles actor appears partway through driver init; retry
        # (bounded — startup-gate hardening).
        t0 = time.time()
        while True:
            try:
                self.ctx = rlc.ElasticContext()
                break
            except Exception as e:
                waited = time.time() - t0
                if waited > GATE_TIMEOUT_S:
                    print(f"[policy] GATE-TIMEOUT: handles actor not ready after "
                          f"{waited:.0f}s (> {GATE_TIMEOUT_S}s); last error: {e} — exiting",
                          flush=True)
                    sys.exit(EXIT_GATE_TIMEOUT)
                print(f"[policy] handles actor not ready ({e}); retrying in 15s "
                      f"(waited {waited:.0f}s)", flush=True)
                time.sleep(15)
        self.log = TrainLogParser(args.train_log)
        self.decisions_path = args.decisions_file
        self.mode = "dry"
        self.dry_syncs_baseline = None      # param_syncs when dry-run window started
        self.state = "TRAINER_ACTIVE"       # TRAINER_ACTIVE | R2_ACTIVE (live truth)
        self.sim_state = "TRAINER_ACTIVE"   # dry-run shadow state
        self.sim_block = -1
        self.fill_ema = Ema(args.fill_tau)
        self.fill_updates = 0
        self.prev_produced = None
        self.prev_produced_t = None
        # round-trip estimates (seconds), seeded from m1-results/run4
        self.rt_in = 37.8
        self.rt_out = 47.8
        self.last_op_end = 0.0              # for min-dwell
        self.last_switch_block = -1         # one pair per gen-wait block
        self.switch_in_block = -1
        self.switch_in_t = 0.0
        self.consec_fail = 0
        self.pending_verify_deadline = None # param_sync must arrive by this
        self.pending_verify_syncs = None
        self.wake_ts = None                 # trainer restore completion ts
        self.prev_dropped_stale = None
        self.last_stale_incr_t = 0.0
        self.tick = 0
        self.cycles_completed = 0
        # progress + auto-live alignment tracking (attempt-1 hardening)
        self.decisions_written = 0
        self.would_in_count = 0
        self.would_back_count = 0
        self.live_in_count = 0
        self.live_back_count = 0
        self.last_action = "none-yet"
        self.last_progress_t = time.time()

    def _reset_fill(self):
        """Regime change (R2 activated/deactivated): re-learn the fill rate."""
        self.fill_ema.reset()
        self.fill_updates = 0

    # ------------------------------------------------------------------
    def record(self, rec):
        rec = {"ts": now_iso(), "tick": self.tick, "mode": self.mode, **rec}
        with open(self.decisions_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.decisions_written += 1
        a = rec.get("action") or rec.get("event")
        if a and a != "none":
            self.last_action = f"{a}@{rec['ts']}"
            if a == "would_switch_to_rollout":
                self.would_in_count += 1
            elif a == "would_switch_to_trainer":
                self.would_back_count += 1
            elif a == "switch_to_rollout":
                self.live_in_count += 1
            elif a == "switch_to_trainer":
                self.live_back_count += 1
        return rec

    def say(self, msg):
        print(f"[{now_iso()}] [policy] {msg}", flush=True)

    def maybe_progress(self, force=False):
        """One-line heartbeat to pod stdout every ~10 min (attempt-1
        hardening: the run must be diagnosable from pod logs alone)."""
        t = time.time()
        if not force and t - self.last_progress_t < 600:
            return
        self.last_progress_t = t
        self.say(
            f"PROGRESS mode={self.mode} state={self.state} sim={self.sim_state} "
            f"tick={self.tick} decisions={self.decisions_written} "
            f"would_in/back={self.would_in_count}/{self.would_back_count} "
            f"live_in/back={self.live_in_count}/{self.live_back_count} "
            f"cycles_verified={self.cycles_completed} "
            f"param_syncs={self.log.param_syncs} steps={len(self.log.steps)} "
            f"last_action={self.last_action}"
        )

    # ------------------------------------------------------------------
    def sample_signals(self):
        """One poll of all signals. Never raises."""
        sig = {}
        t = time.time()
        try:
            stats = self.ctx.ray.get(self.ctx.mq.get_statistics.remote(), timeout=10)
            sig["queue_size"] = stats["queue_size"]
            sig["total_produced"] = stats["total_produced"]
            sig["dropped_samples"] = stats.get("dropped_samples", 0)
            if self.prev_produced is not None and t > self.prev_produced_t:
                inst = (stats["total_produced"] - self.prev_produced) / (t - self.prev_produced_t)
                self.fill_ema.update(max(inst, 0.0), t)
                self.fill_updates += 1
            self.prev_produced = stats["total_produced"]
            self.prev_produced_t = t
        except Exception as e:
            sig["mq_error"] = str(e)[:200]
        try:
            rs = self.ctx.ray.get(self.ctx.rollouter.get_statistics.remote(), timeout=10)
            sig["active_tasks"] = rs.get("monitor/active_tasks_size")
            sig["dropped_stale"] = rs.get("count/dropped_stale_samples")
            sig["max_concurrent"] = rs.get("static/max_concurrent_samples")
            ds = sig["dropped_stale"]
            if ds is not None:
                if self.prev_dropped_stale is not None and ds > self.prev_dropped_stale:
                    self.last_stale_incr_t = t
                self.prev_dropped_stale = ds
        except Exception as e:
            sig["rollouter_error"] = str(e)[:200]

        self.log.poll()
        L = self.log
        sig.update(
            blocked=L.block_open,
            collected=L.collected,
            samples_needed=L.samples_needed,
            mq_len_log=L.mq_len_log,
            block_id=L.requests,
            param_syncs=L.param_syncs,
            param_version=L.param_version,
            fill_ema=round(self.fill_ema.value, 5) if self.fill_ema.value is not None else None,
        )
        return sig

    def eta_seconds(self, sig):
        """ETA to batch-ready from current progress + fill EMA."""
        need = sig.get("samples_needed", 64)
        progress = min(sig.get("collected", 0) + sig.get("queue_size", 0), need)
        remaining = need - progress
        fill = self.fill_ema.value
        if remaining <= 0:
            return 0.0
        if not fill or fill < self.args.fill_floor:
            return float("inf")
        return remaining / fill

    # ------------------------------------------------------------------
    # Gate evaluation
    # ------------------------------------------------------------------
    def eval_switch_in(self, sig, t):
        """Returns (ok, reason, gate). reason = first failing conjunct."""
        eta = self.eta_seconds(sig)
        round_trip = self.rt_in + self.rt_out
        threshold = self.args.c * round_trip
        gate = {
            "eta_s": None if math.isinf(eta) else round(eta, 1),
            "eta_inf": math.isinf(eta),
            "threshold_s": round(threshold, 1),
            "rt_in_ema": round(self.rt_in, 1),
            "rt_out_ema": round(self.rt_out, 1),
            "c": self.args.c,
        }
        if "mq_error" in sig:
            return False, "signal_error_mq", gate
        if self.fill_updates < self.args.fill_warmup:
            gate["fill_updates"] = self.fill_updates
            return False, "fill_ema_warmup", gate
        if not sig.get("blocked"):
            return False, "not_blocked", gate
        if sig.get("collected", 0) < 1:
            return False, "block_burst_not_absorbed", gate
        if sig.get("block_id", -1) == self.last_switch_block:
            return False, "already_switched_this_block", gate
        if t - self.last_op_end < self.args.min_dwell:
            gate["dwell_remaining_s"] = round(self.args.min_dwell - (t - self.last_op_end), 1)
            return False, "min_dwell", gate
        if sig.get("dropped_samples", 0) > 0:
            return False, "mq_dropped_samples", gate
        if t - self.last_stale_incr_t < self.args.stale_holdoff and self.last_stale_incr_t > 0:
            return False, "staleness_guard", gate
        if not (eta > threshold):
            return False, "eta_below_threshold", gate
        return True, "gate_clear", gate

    def eval_switch_back(self, sig, t, active_since):
        eta = self.eta_seconds(sig)
        trigger_at = self.rt_out + self.args.wake_margin
        gate = {
            "eta_s": None if math.isinf(eta) else round(eta, 1),
            "trigger_at_s": round(trigger_at, 1),
            "collected": sig.get("collected"),
            "hard_collect": self.args.hard_collect,
            "window_s": round(t - active_since, 1),
            "window_cap_s": self.args.window_cap,
        }
        if not sig.get("blocked") and sig.get("collected", 0) >= sig.get("samples_needed", 64):
            return True, "batch_completed_late", gate
        if sig.get("collected", 0) >= self.args.hard_collect:
            return True, "hard_collect_failsafe", gate
        if t - active_since >= self.args.window_cap:
            return True, "window_cap", gate
        if t - self.last_op_end < self.args.min_dwell:
            return False, "min_dwell", gate
        if eta <= trigger_at:
            return True, "predictive_eta", gate
        return False, "eta_above_trigger", gate

    # ------------------------------------------------------------------
    # Actions (live mode)
    # ------------------------------------------------------------------
    def _run_op(self, name, fn):
        """Run a switch op with a hang watchdog. Returns (ok, record|err)."""
        result = {}
        def target():
            try:
                result["rec"] = fn()
            except Exception as e:
                result["err"] = f"{type(e).__name__}: {e}"
        th = threading.Thread(target=target, daemon=True)
        t0 = time.time()
        th.start()
        th.join(self.args.op_timeout)
        if th.is_alive():
            self.record({"event": "op_hang", "op": name, "elapsed_s": round(time.time() - t0, 1)})
            self.capture_status(f"hang during {name}")
            self.say(f"ABORT: {name} hung > {self.args.op_timeout}s")
            sys.exit(3)
        if "err" in result:
            return False, result["err"]
        return True, result.get("rec")

    def do_switch_to_rollout(self):
        timer = rlc.PhaseTimer("switch-to-rollout", timings_file=rlc.DEFAULT_TIMINGS_FILE)
        def op():
            rlc.op_suspend_trainer(self.ctx, self.args.agent, timer)
            try:
                rlc.op_resume_r2(self.ctx, self.args.agent, timer, reload_weights=False)
            except Exception as first:
                # trainer frozen + R2 not serving: emergency trainer restore
                self.say("switch-in: resume-r2 FAILED; attempting emergency trainer restore")
                try:
                    et = rlc.PhaseTimer("emergency-resume-trainer",
                                        timings_file=rlc.DEFAULT_TIMINGS_FILE)
                    rlc.op_resume_trainer(self.ctx, self.args.agent, et)
                    et.finish()
                except Exception as second:
                    raise RuntimeError(
                        f"FATAL_TRAINER_FROZEN: resume-r2 failed ({first}) and emergency "
                        f"trainer restore failed ({second})"
                    ) from second
                raise
            return timer.finish()
        return self._run_op("switch-to-rollout", op)

    def do_switch_to_trainer(self):
        timer = rlc.PhaseTimer("switch-to-trainer", timings_file=rlc.DEFAULT_TIMINGS_FILE)
        def op():
            rlc.op_suspend_r2(self.ctx, self.args.agent, timer, self.args.drain_timeout)
            rlc.op_resume_trainer(self.ctx, self.args.agent, timer)
            return timer.finish()
        return self._run_op("switch-to-trainer", op)

    def capture_status(self, why):
        self.say(f"capturing status ({why})")
        path = os.path.join(RESULTS_DIR, f"status_capture_{int(time.time())}.log")
        try:
            import contextlib, io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rlc.op_status(self.ctx, self.args.agent)
            with open(path, "w") as f:
                f.write(f"# capture reason: {why}\n{buf.getvalue()}")
        except Exception as e:
            self.say(f"status capture failed: {e}")

    # ------------------------------------------------------------------
    # Startup gates
    # ------------------------------------------------------------------
    def wait_steady_state(self):
        self.say(f"waiting for steady state (>= {self.args.steady_syncs} param syncs)...")
        t0 = time.time()
        warned = False
        while True:
            self.log.poll()
            if self.log.param_syncs >= self.args.steady_syncs:
                break
            waited = time.time() - t0
            if waited > STEADY_TIMEOUT_S:
                self.record({"event": "abort", "reason": "steady_state_gate_timeout",
                             "waited_s": round(waited, 1),
                             "param_syncs": self.log.param_syncs})
                self.say(f"GATE-TIMEOUT: only {self.log.param_syncs} param syncs after "
                         f"{waited:.0f}s (> {STEADY_TIMEOUT_S}s) — exiting")
                sys.exit(EXIT_GATE_TIMEOUT)
            if waited > GATE_WARN_S and not warned:
                warned = True
                self.say(f"WARNING: still waiting for steady state after {waited:.0f}s "
                         f"({self.log.param_syncs}/{self.args.steady_syncs} param syncs); "
                         f"hard cap {STEADY_TIMEOUT_S}s")
            time.sleep(15)
        self.say(f"steady state: {self.log.param_syncs} param syncs, version={self.log.param_version}")

    def wait_agent_trainer_running(self):
        # Parsing lives in module-level state_name()/agent_job_states() —
        # the exact code path the pre-flight probe exercises against the
        # real agent before every run (attempt-1 post-mortem hardening).
        self.say("waiting for agent job elastic-trainer RUNNING (operator relabel)...")
        t0 = time.time()
        warned = False
        while True:
            try:
                st = agent_job_states(self.args.agent)
                if st.get("elastic-trainer") == "JOB_STATE_RUNNING":
                    self.say(f"agent jobs: {st} — elastic-trainer RUNNING confirmed")
                    return
                self.say(f"agent jobs: {st}")
            except Exception as e:
                self.say(f"agent status error: {e}")
            waited = time.time() - t0
            if waited > GATE_TIMEOUT_S:
                self.record({"event": "abort", "reason": "agent_running_gate_timeout",
                             "waited_s": round(waited, 1)})
                self.say(f"GATE-TIMEOUT: elastic-trainer not RUNNING after {waited:.0f}s "
                         f"(> {GATE_TIMEOUT_S}s) — exiting (attempt-1 failure mode: "
                         f"this gate must never spin silently)")
                sys.exit(EXIT_GATE_TIMEOUT)
            if waited > GATE_WARN_S / 2 and not warned:
                warned = True
                self.say(f"WARNING: agent-RUNNING gate at {waited:.0f}s; "
                         f"hard cap {GATE_TIMEOUT_S}s — check operator relabel")
            time.sleep(15)

    # ------------------------------------------------------------------
    def maybe_go_live(self):
        if self.mode != "dry":
            return
        if self.dry_syncs_baseline is None:
            return
        done = self.log.param_syncs - self.dry_syncs_baseline
        if done < self.args.dry_run_steps:
            return
        flag = os.path.exists(self.args.live_flag)
        if not (self.args.auto_live or flag):
            return
        # Alignment check (attempt-1 hardening): auto-live requires the shadow
        # policy to have demonstrably fired inside gen-wait blocks during the
        # dry window (every would_switch_to_rollout has blocked=True as its
        # first gate conjunct, so >=1 would-in proves end-to-end alignment
        # between decisions and observed gen-wait). Failsafe: if 2 extra dry
        # steps pass with no would-in, go live anyway with a loud warning —
        # the live gate is identical to the shadow gate, so no-harm holds and
        # staying dry forever would just reproduce the attempt-1 zero-result.
        aligned = self.would_in_count >= 1
        overdue = done >= self.args.dry_run_steps + 2
        if self.args.auto_live and not flag and not aligned and not overdue:
            if done > self.args.dry_run_steps:  # log once per extra step, cheap
                self.say(f"auto-live held: {done} dry steps done but 0 would-in "
                         f"decisions yet (need >=1 for alignment; failsafe at "
                         f"{self.args.dry_run_steps + 2} steps)")
            return
        self.mode = "live"
        via = "flag" if flag else ("auto" if aligned else "auto_failsafe_no_would_in")
        self.record({"event": "mode_change", "to": "live",
                     "dry_steps_observed": done,
                     "would_in_count": self.would_in_count,
                     "would_back_count": self.would_back_count,
                     "aligned": aligned,
                     "via": via})
        self.say(f"LIVE mode enabled after {done} dry-run steps "
                 f"(would_in={self.would_in_count}, would_back={self.would_back_count}, "
                 f"via={via})")

    # ------------------------------------------------------------------
    def run(self):
        os.makedirs(os.path.dirname(self.decisions_path), exist_ok=True)
        self.record({
            "event": "controller_start",
            "config": {k: getattr(self.args, k) for k in vars(self.args)},
            "seeds": {"rt_in": self.rt_in, "rt_out": self.rt_out},
        })
        self.wait_steady_state()
        self.wait_agent_trainer_running()
        self.dry_syncs_baseline = self.log.param_syncs
        self.record({"event": "dry_run_start", "param_syncs": self.log.param_syncs,
                     "dry_run_steps": self.args.dry_run_steps})
        self.say(f"entering poll loop (dry-run for {self.args.dry_run_steps} steps, "
                 f"auto_live={self.args.auto_live})")
        self.maybe_progress(force=True)

        while True:
            self.tick += 1
            t = time.time()
            self.maybe_progress()
            if os.path.exists(STOP_FLAG):
                self.record({"event": "controller_stop", "reason": "stop_flag"})
                self.say("stop flag seen; exiting cleanly")
                return 0
            sig = self.sample_signals()
            if self.log.ray_actor_error:
                self.record({"event": "abort", "reason": "RayActorError in train.log", "signals": sig})
                self.capture_status("RayActorError")
                self.say("ABORT: RayActorError observed")
                return 6

            # post-switch-back verification (param_sync within 900s)
            if self.pending_verify_deadline is not None:
                if self.log.param_syncs > self.pending_verify_syncs:
                    gap = None
                    if self.wake_ts and self.log.batch_ready_ts:
                        gap = round(self.wake_ts - self.log.batch_ready_ts, 1)
                    self.cycles_completed += 1
                    self.record({"event": "cycle_verified", "cycle": self.cycles_completed,
                                 "param_version": self.log.param_version,
                                 "wake_minus_batch_ready_s": gap, "signals": sig})
                    self.say(f"cycle {self.cycles_completed} verified: param_sync ok "
                             f"(version {self.log.param_version})")
                    self.pending_verify_deadline = None
                elif t > self.pending_verify_deadline:
                    self.record({"event": "abort", "reason": "no_param_sync_within_900s", "signals": sig})
                    self.capture_status("no param_sync after switch-back")
                    self.say("ABORT: no param_sync within 900s of trainer resume")
                    return 5

            self.maybe_go_live()

            if self.mode == "live" and self.state == "TRAINER_ACTIVE":
                ok, reason, gate = self.eval_switch_in(sig, t)
                if ok:
                    # misfire re-check in the SAME tick from a fresh parse
                    self.log.poll()
                    if not self.log.block_open:
                        self.record({"action": "skip_switch_in", "reason": "block_closed_at_fire",
                                     "signals": sig, "gate": gate})
                    else:
                        self.record({"action": "switch_to_rollout", "reason": reason,
                                     "signals": sig, "gate": gate})
                        self.say(f"SWITCH-IN firing (block {sig['block_id']}, "
                                 f"collected {sig['collected']}/{sig['samples_needed']}, "
                                 f"ETA {gate['eta_s']}s > {gate['threshold_s']}s)")
                        ok2, rec = self.do_switch_to_rollout()
                        self.last_op_end = time.time()
                        if ok2:
                            self.rt_in = 0.5 * rec["total_seconds"] + 0.5 * self.rt_in
                            self.state = "R2_ACTIVE"
                            self.switch_in_block = sig["block_id"]
                            self.last_switch_block = sig["block_id"]
                            self.switch_in_t = self.last_op_end
                            self.consec_fail = 0
                            self._reset_fill()  # regime change: re-learn boosted rate
                            self.record({"event": "switch_in_done",
                                         "total_seconds": rec["total_seconds"],
                                         "rt_in_ema": round(self.rt_in, 1)})
                        else:
                            if "FATAL_TRAINER_FROZEN" in str(rec):
                                self.record({"event": "abort",
                                             "reason": "trainer_frozen_unrecovered",
                                             "error": str(rec)})
                                self.capture_status("trainer frozen, emergency restore failed")
                                self.say("ABORT: trainer frozen and emergency restore failed")
                                return 4
                            self.consec_fail += 1
                            self.record({"event": "switch_in_failed", "error": str(rec),
                                         "consec_fail": self.consec_fail})
                            self.say(f"switch-in FAILED ({rec}); consec_fail={self.consec_fail}")
                            if self.consec_fail >= 2:
                                self.capture_status("2 consecutive switch-in failures")
                                self.say("ABORT: two consecutive failed switch-ins; controller stops")
                                return 2
                else:
                    self.record({"action": "none", "state": self.state, "reason": reason,
                                 "signals": sig, "gate": gate})

            elif self.mode == "live" and self.state == "R2_ACTIVE":
                ok, reason, gate = self.eval_switch_back(sig, t, self.switch_in_t)
                if ok:
                    self.record({"action": "switch_to_trainer", "reason": reason,
                                 "signals": sig, "gate": gate})
                    self.say(f"SWITCH-BACK firing ({reason}: collected "
                             f"{sig['collected']}/{sig['samples_needed']}, ETA {gate['eta_s']}s)")
                    ok2, rec = self.do_switch_to_trainer()
                    self.last_op_end = time.time()
                    if not ok2:
                        self.record({"event": "abort", "reason": "switch_back_failed",
                                     "error": str(rec)})
                        self.capture_status("switch-back failure")
                        self.say(f"ABORT: switch-back failed ({rec})")
                        return 4
                    self.rt_out = 0.5 * rec["total_seconds"] + 0.5 * self.rt_out
                    self.state = "TRAINER_ACTIVE"
                    self.wake_ts = self.last_op_end
                    self._reset_fill()
                    self.pending_verify_syncs = self.log.param_syncs
                    self.pending_verify_deadline = self.last_op_end + 900
                    self.record({"event": "switch_back_done",
                                 "total_seconds": rec["total_seconds"],
                                 "rt_out_ema": round(self.rt_out, 1)})
                else:
                    self.record({"action": "none", "state": self.state, "reason": reason,
                                 "signals": sig, "gate": gate})

            else:  # dry-run shadow policy
                if self.sim_state == "TRAINER_ACTIVE":
                    ok, reason, gate = self.eval_switch_in(sig, t)
                    if ok:
                        self.sim_state = "R2_ACTIVE"
                        self.sim_block = sig["block_id"]
                        self.last_switch_block = sig["block_id"]
                        self.switch_in_t = t
                        self.last_op_end = t  # simulate op duration ~ dwell basis
                        self.record({"action": "would_switch_to_rollout", "reason": reason,
                                     "signals": sig, "gate": gate})
                        self.say(f"[dry] would SWITCH-IN (block {sig['block_id']}, "
                                 f"collected {sig['collected']}, ETA {gate['eta_s']}s)")
                    else:
                        self.record({"action": "none", "state": "SIM_" + self.sim_state,
                                     "reason": reason, "signals": sig, "gate": gate})
                else:
                    ok, reason, gate = self.eval_switch_back(sig, t, self.switch_in_t)
                    if ok:
                        self.sim_state = "TRAINER_ACTIVE"
                        self.last_op_end = t
                        self.record({"action": "would_switch_to_trainer", "reason": reason,
                                     "signals": sig, "gate": gate})
                        self.say(f"[dry] would SWITCH-BACK ({reason}, "
                                 f"collected {sig['collected']}, ETA {gate['eta_s']}s)")
                    else:
                        self.record({"action": "none", "state": "SIM_" + self.sim_state,
                                     "reason": reason, "signals": sig, "gate": gate})

            time.sleep(self.args.poll)


def main(argv=None):
    p = argparse.ArgumentParser(description="elastic-rl-poc M2 policy controller")
    p.add_argument("--agent", default=rlc.DEFAULT_AGENT)
    p.add_argument("--poll", type=float, default=2.0)
    p.add_argument("--c", type=float, default=1.5, help="ETA > c*round_trip switch-in gate")
    p.add_argument("--min-dwell", type=float, default=60.0)
    p.add_argument("--wake-margin", type=float, default=15.0,
                   help="switch back when ETA <= rt_out + margin")
    p.add_argument("--hard-collect", type=int, default=60, help="failsafe switch-back threshold")
    p.add_argument("--window-cap", type=float, default=600.0)
    p.add_argument("--fill-tau", type=float, default=60.0, help="fill-rate EMA time constant")
    p.add_argument("--fill-floor", type=float, default=0.005, help="below this, ETA = inf")
    p.add_argument("--fill-warmup", type=int, default=15,
                   help="min fill-rate samples before the switch-in gate can clear")
    p.add_argument("--stale-holdoff", type=float, default=120.0)
    p.add_argument("--dry-run-steps", type=int, default=2)
    p.add_argument("--auto-live", action="store_true")
    p.add_argument("--live-flag", default=LIVE_FLAG)
    p.add_argument("--steady-syncs", type=int, default=3)
    p.add_argument("--drain-timeout", type=float, default=30.0)
    p.add_argument("--op-timeout", type=float, default=360.0)
    p.add_argument("--train-log", default=TRAIN_LOG)
    p.add_argument("--decisions-file", default=DECISIONS_FILE)
    args = p.parse_args(argv)

    ctl = Controller(args)
    try:
        return ctl.run()
    except KeyboardInterrupt:
        ctl.record({"event": "controller_stop", "reason": "SIGINT"})
        return 0


if __name__ == "__main__":
    sys.exit(main())
