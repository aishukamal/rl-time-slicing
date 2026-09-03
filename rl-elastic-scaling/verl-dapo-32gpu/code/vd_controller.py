#!/usr/bin/env python3
# elastic-vd — closed-loop policy controller for the 32-GPU DAPO showcase.
# Port of m2/policy_controller.py with the switch mechanics replaced:
#   switch-in  = trainer.elastic_offload (16-rank FSDP -> CPU, comms alive)
#                -> 4 spares wake (sleep L2 -> wake_up) -> wcache load
#                (fresh weights, MANDATORY) -> truthful set_global_steps
#                -> resume_generation -> LB add -> concurrency x2
#   switch-back= LB remove -> concurrency restore -> abort (partial rollout
#                resumes in-flight on the fleet with tokens retained: lossless
#                drain, llm_server.py partial-rollout client) -> drain ->
#                sleep L2 -> trainer.elastic_onload
# No snapshot-agent, no shim, no cuda-checkpoint anywhere.
#
# WCACHE PIPELINE (off the window clock): on every param_sync observed in
# train.log, trigger the per-TP-rank dump on fleet replica 0
# (collective_rpc callable), then HTTP-prefetch to both trainer nodes.
# GATE: switch-in requires wcache_ready_version == current_param_version
# (CORRECTNESS: fresh weights each wake; stamps truthful by construction).
#
# Policy (unchanged from M2): ETA switch-in gate (blocked in gen-wait, burst
# absorbed, ETA > c*(rt_in+rt_out)), predictive switch-back
# (ETA <= rt_out + margin), hard-collect + window-cap failsafes, min-dwell,
# one pair per gen-wait block, staleness guard, dry-run -> auto-live,
# no-harm decision logging. Seeds: rt_in=30s (offload ~15 + wake ~5 + load
# ~13, genbound switch-out-to-serving 21-29s analog), rt_out=15s (drain +
# sleep + onload ~10s genbound switch-in analog).
#
# Scale notes vs M2: samples_needed=32 (ppo_mini_batch_size x require_batches),
# 16 micro-steps per sync block; the trainer CANNOT be offloaded during any
# compute (resident guard in vd_elastic_trainer.py is the belt-and-braces;
# this controller's predictive gate is the primary).

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import ray

HANDLES_ACTOR_NAME = "elastic_controller_handles"
HANDLES_NAMESPACE = "elastic"

RESULTS_DIR = os.environ.get("VD_RESULTS_DIR", "/results")
TRAIN_LOG = os.path.join(RESULTS_DIR, "train.log")
DECISIONS_FILE = os.path.join(RESULTS_DIR, "decisions.jsonl")
TIMINGS_FILE = os.path.join(RESULTS_DIR, "switch_timings.jsonl")
INVARIANTS_FILE = os.path.join(RESULTS_DIR, "window_invariants.jsonl")
LIVE_FLAG = os.path.join(RESULTS_DIR, "controller_live.flag")
STOP_FLAG = os.path.join(RESULTS_DIR, "controller_stop.flag")

RE_REQUEST = re.compile(r"\[FullyAsyncTrainer\] Requesting (\d+) samples from queue")
RE_COLLECT = re.compile(r"\[FullyAsyncTrainer\] sample collected (\d+)/(\d+)\. mq_len: (\d+)")
RE_PSYNC = re.compile(r"timing_s/param_sync: ([\d.]+) seconds self\.current_param_version: (\d+)")
RE_ACTOR_ERR = re.compile(r"RayActorError")

GATE_WARN_S = 1200
GATE_TIMEOUT_S = 3600  # init at this scale is slow (model load + 5 engines + FSDP init)


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class PhaseTimer:
    def __init__(self, operation, timings_file=TIMINGS_FILE):
        self.operation = operation
        self.timings_file = timings_file
        self.phases = []
        self.t_start = time.time()

    @contextmanager
    def phase(self, name):
        t0 = time.time()
        print(f"[{now_iso()}] [{self.operation}] PHASE START {name}", flush=True)
        try:
            yield
        finally:
            t1 = time.time()
            self.phases.append({"phase": name, "seconds": round(t1 - t0, 4)})
            print(f"[{now_iso()}] [{self.operation}] PHASE END   {name}  ({t1 - t0:.3f}s)", flush=True)

    def finish(self, extra=None):
        total = time.time() - self.t_start
        record = {"ts": now_iso(), "operation": self.operation,
                  "total_seconds": round(total, 4), "phases": self.phases}
        if extra:
            record.update(extra)
        print(f"=== {self.operation} timing (total {total:.3f}s) ===", flush=True)
        for p in self.phases:
            print(f"  {p['phase']:<34s} {p['seconds']:>9.3f}s", flush=True)
        try:
            os.makedirs(os.path.dirname(self.timings_file), exist_ok=True)
            with open(self.timings_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            print(f"WARNING: could not write timings: {e}")
        return record


class Ctx:
    """Handles-actor accessor (M1 ElasticContext, slimmed)."""

    def __init__(self, ray_address="auto"):
        if not ray.is_initialized():
            ray.init(address=ray_address, namespace=HANDLES_NAMESPACE, ignore_reinit_error=True)
        self.handles = ray.get_actor(HANDLES_ACTOR_NAME, namespace=HANDLES_NAMESPACE)
        self._cache = {}

    def get(self, key, required=True, fresh=False):
        if fresh or key not in self._cache:
            value = ray.get(self.handles.get.remote(key))
            if value is None and required:
                raise RuntimeError(f"handles actor missing '{key}'")
            self._cache[key] = value
        return self._cache[key]

    def put(self, key, value):
        ray.get(self.handles.put.remote(key, value))
        self._cache[key] = value

    @property
    def trainer(self): return self.get("trainer")
    @property
    def rollouter(self): return self.get("rollouter")
    @property
    def lb(self): return self.get("load_balancer")
    @property
    def mq(self): return self.get("message_queue")
    @property
    def spare_servers(self): return self.get("spare_servers")
    @property
    def spare_addresses(self): return self.get("spare_addresses")


class TrainLogParser:
    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.buf = b""
        self.requests = 0
        self.samples_needed = 32
        self.collected = 0
        self.mq_len_log = 0
        self.block_open = False
        self.batch_ready_ts = None
        self.blocks_closed = 0
        self.param_syncs = 0
        self.param_version = -1
        self.last_psync_ts = None
        self.new_psync_versions = []   # queue of freshly seen versions (wcache triggers)
        self.ray_actor_error = False

    def poll(self):
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return
        if size < self.offset:
            self.offset, self.buf = 0, b""
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
                self.batch_ready_ts = None
                continue
            m = RE_COLLECT.search(line)
            if m:
                self.collected = int(m.group(1))
                self.samples_needed = int(m.group(2))
                self.mq_len_log = int(m.group(3))
                if self.collected >= self.samples_needed and self.block_open:
                    self.block_open = False
                    self.batch_ready_ts = t
                    self.blocks_closed += 1
                continue
            m = RE_PSYNC.search(line)
            if m:
                self.param_syncs += 1
                v = int(m.group(2))
                self.param_version = v
                self.last_psync_ts = t
                self.new_psync_versions.append(v)
                continue
            if RE_ACTOR_ERR.search(line):
                self.ray_actor_error = True


class Ema:
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


# ============================================================================
# Switch + wcache mechanics
# ============================================================================


class Mechanics:
    def __init__(self, ctx: Ctx, args):
        self.ctx = ctx
        self.args = args
        self.window_id = 0
        self.spare_loaded_version = 0  # spares boot with disk v0 weights
        self.wcache_ready_version = None
        self.fleet_agent = None
        self.fleet_http_url = None
        self._wcache_lock = threading.Lock()
        self._wcache_busy = False

    # ---------- wcache pipeline ----------

    def setup_fleet_agent(self):
        """Create/lookup the VDNodeAgent on fleet replica 0's node + HTTP.

        Node roles flip between runs while the agent actors are detached and
        node-pinned; a stale agent from a previous run serves the WRONG node
        (defect VD-5: 404s on prefetch). Verify the agent's node ip against
        replica0's actual node and recreate on mismatch."""
        import vd_spares
        fleet = self.ctx.get("fleet_servers")  # [(address, handle), ...]
        addr, server = fleet[0]
        node_id, node_ip = ray.get(server.__ray_call__.remote(
            lambda self: (__import__("ray").get_runtime_context().get_node_id(),
                          __import__("ray").util.get_node_ip_address())))
        for attempt in range(3):
            self.fleet_agent = vd_spares.VDNodeAgent.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False),
                name="vd_node_agent_fleet0", namespace=HANDLES_NAMESPACE,
                lifetime="detached", get_if_exists=True,
            ).remote(role="fleet")
            agent_ip = ray.get(self.fleet_agent.node_ip.remote())
            if agent_ip == node_ip:
                break
            print(f"[mech] stale fleet agent on {agent_ip} != replica0 node {node_ip}; "
                  f"killing and recreating (attempt {attempt + 1})", flush=True)
            ray.kill(self.fleet_agent)
            time.sleep(3)
        else:
            raise RuntimeError(f"could not place fleet agent on {node_ip}")
        self.fleet_http_url = ray.get(self.fleet_agent.start_http_server.remote())
        self._fleet_dump_server = server
        self._fleet_dump_addr = addr
        print(f"[mech] fleet wcache source: replica0 {addr}, http {self.fleet_http_url}", flush=True)

    def refresh_wcache_async(self, version: int, on_done):
        """Dump (fleet) + prefetch (both trainer nodes) in a thread; off-window."""
        def work():
            try:
                t0 = time.time()
                import vd_spares
                dump_dir = os.path.join(vd_spares.WCACHE_DIR, f"v{version}")
                ray.get(self._fleet_dump_server.collective_rpc.remote(
                    method=vd_spares.wcache_dump_shard, args=(dump_dir, int(version))))
                agents = self.ctx.get("trainer_node_agents")
                fetches = [a.prefetch.remote(self.fleet_http_url, int(version))
                           for a in agents.values()]
                results = ray.get(fetches)
                dt = time.time() - t0
                self.wcache_ready_version = int(version)
                on_done({"version": version, "seconds": round(dt, 1), "fetches": results})
            except Exception as e:
                on_done({"version": version, "error": f"{type(e).__name__}: {e}"})
            finally:
                with self._wcache_lock:
                    self._wcache_busy = False
        with self._wcache_lock:
            if self._wcache_busy:
                return False
            self._wcache_busy = True
        threading.Thread(target=work, daemon=True).start()
        return True

    # ---------- window invariant ----------

    def log_invariant(self, version: int, extra: dict):
        agents = self.ctx.get("trainer_node_agents")
        receipts = []
        for ip, a in agents.items():
            try:
                receipts.extend({"node": ip, **r} for r in ray.get(a.read_receipts.remote(int(version))))
            except Exception as e:
                receipts.append({"node": ip, "error": str(e)})
        loads = [r for r in receipts if r.get("kind") == "load"]
        dumps = [r for r in receipts if r.get("kind") == "dump"]
        if extra.get("wcache_loaded_this_window", True):
            uniq_dump_ranks = {r.get("tp_rank") for r in dumps}
            ok = (
                len(loads) >= 16  # 4 spares x 4 TP ranks
                and all(int(r["version"]) == int(version) for r in loads)
                and all(r.get("model_fingerprint") == r.get("dump_fingerprint") for r in loads)
                and uniq_dump_ranks == {0, 1, 2, 3}
                and all(int(r["version"]) == int(version) for r in dumps)
            )
        else:
            # paused-resident window without reload: weights unchanged since the
            # last load; verify every spare server stamps this exact version.
            try:
                stamps = ray.get([s.__ray_call__.remote(lambda self: self.global_steps)
                                  for s in self.ctx.spare_servers])
            except Exception as e:
                stamps = [f"err:{e}"]
            extra = {**extra, "spare_global_steps": stamps}
            ok = all(st == int(version) for st in stamps)
        rec = {"ts": now_iso(), "window": self.window_id, "version": int(version),
               "invariant_ok": bool(ok), "n_load_receipts": len(loads),
               "n_dump_receipts": len(dumps), **extra}
        with open(INVARIANTS_FILE, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if not ok:
            with open(INVARIANTS_FILE, "a") as f:
                f.write(json.dumps({"ts": now_iso(), "window": self.window_id,
                                    "receipts": receipts}) + "\n")
        print(f"[mech] window {self.window_id} invariant_ok={ok} v{version} "
              f"(loads={len(loads)}, dumps={len(dumps)})", flush=True)
        return ok

    # ---------- switch ops ----------

    def switch_to_spares(self, version: int) -> dict:
        """Window open. Two park modes (spares_park_mode handle, set by driver):
          paused (default): spares are memory-resident and PAUSED; open =
            wcache-load-if-stale + resume + LB add + concurrency (~2-5s).
          sleep: big-model path — trainer FSDP offload + spare wake_up +
            mandatory wcache load (post-L2 buffers are garbage)."""
        self.window_id += 1
        park_mode = self.ctx.get("spares_park_mode", required=False) or "paused"
        timer = PhaseTimer("switch-to-spares")
        ctx = self.ctx
        try:
            with timer.phase("trainer_hold"):
                # VD-6 guard: batch-ready during the window must stall the
                # trainer (in _fit_generate) until spares are parked again.
                ray.get(ctx.trainer.elastic_hold.remote())
            if park_mode == "sleep":
                with timer.phase("trainer_offload"):
                    ray.get(ctx.trainer.elastic_offload.remote())
                with timer.phase("spares_wake"):
                    ray.get([s.spare_wake.remote() for s in ctx.spare_servers])
            elif park_mode == "sleepL1":
                with timer.phase("trainer_release_cache"):
                    ray.get(ctx.trainer.elastic_release_cache.remote())
                with timer.phase("spares_wake"):
                    ray.get([s.spare_wake.remote() for s in ctx.spare_servers])
            need_load = (park_mode == "sleep") or (self.spare_loaded_version != int(version))
            if need_load:
                with timer.phase("spares_load_wcache"):
                    import vd_spares
                    load_dir = os.path.join(vd_spares.WCACHE_DIR, f"v{int(version)}")
                    ray.get([s.spare_load_wcache.remote(load_dir, int(version))
                             for s in ctx.spare_servers])
                    self.spare_loaded_version = int(version)
            else:
                with timer.phase("spares_stamp_version"):
                    # weights already at this version; refresh the stamp only
                    ray.get([s.set_global_steps.remote(int(version)) for s in ctx.spare_servers])
            with timer.phase("spares_resume_generation"):
                ray.get([s.resume_generation.remote() for s in ctx.spare_servers])
            with timer.phase("lb_add_spares"):
                servers = dict(zip(ctx.spare_addresses, ctx.spare_servers))
                ray.get(ctx.lb.add_servers.remote(servers=servers))
                ray.get(ctx.lb.clear_sticky_cache.remote())
            with timer.phase("raise_concurrency"):
                base = ctx.get("max_concurrent_base") or 16
                ray.get(ctx.rollouter.elastic_set_max_concurrent_samples.remote(int(base) * 2))
        except Exception as first:
            print(f"[mech] switch-in FAILED ({first}); emergency spare-park", flush=True)
            try:
                for s in ctx.spare_servers:
                    try:
                        ray.get(s.abort_all_requests.remote())
                        ray.get(s.wait_for_requests_to_drain.remote())
                        if park_mode == "sleep":
                            ray.get(s.spare_sleep.remote(2))
                        elif park_mode == "sleepL1":
                            ray.get(s.spare_sleep.remote(1))
                    except Exception as e2:
                        print(f"[mech] emergency spare park error: {e2}", flush=True)
                if park_mode == "sleep":
                    ray.get(ctx.trainer.elastic_onload.remote())
                else:
                    ray.get(ctx.trainer.elastic_release.remote())
            except Exception as second:
                raise RuntimeError(
                    f"FATAL_TRAINER_YIELDED: switch-in failed ({first}) and emergency onload failed ({second})"
                ) from second
            raise
        ctx.put("spares_state", "serving")
        rec = timer.finish(extra={"window": self.window_id, "version": int(version),
                                  "park_mode": park_mode, "loaded": need_load})
        self.log_invariant(version, {"phase": "post-switch-in", "total_seconds": rec["total_seconds"],
                                     "wcache_loaded_this_window": need_load})
        return rec

    def switch_to_trainer(self) -> dict:
        park_mode = self.ctx.get("spares_park_mode", required=False) or "paused"
        timer = PhaseTimer("switch-to-trainer")
        ctx = self.ctx
        with timer.phase("lb_remove_spares"):
            ray.get(ctx.lb.remove_servers.remote(server_ids=list(ctx.spare_addresses)))
        with timer.phase("restore_concurrency"):
            base = ctx.get("max_concurrent_base") or 16
            ray.get(ctx.rollouter.elastic_set_max_concurrent_samples.remote(int(base)))
        with timer.phase("abort_spares_partial_rollout"):
            aborted = ray.get([s.abort_all_requests.remote() for s in ctx.spare_servers])
            n = sum(a.get("aborted_count", 0) for a in aborted)
            print(f"[mech] aborted {n} in-flight spare requests (partial rollout resumes on fleet)", flush=True)
        with timer.phase("spares_drain"):
            ray.get([s.wait_for_requests_to_drain.remote() for s in ctx.spare_servers])
        if park_mode == "sleep":
            with timer.phase("spares_sleep"):
                ray.get([s.spare_sleep.remote(2) for s in ctx.spare_servers])
            with timer.phase("trainer_onload"):
                ray.get(ctx.trainer.elastic_onload.remote())
        elif park_mode == "sleepL1":
            with timer.phase("spares_sleep_L1"):
                ray.get([s.spare_sleep.remote(1) for s in ctx.spare_servers])
        with timer.phase("trainer_release"):
            ray.get(ctx.trainer.elastic_release.remote())
        ctx.put("spares_state", "parked")
        return timer.finish(extra={"window": self.window_id, "park_mode": park_mode})


# ============================================================================
# Controller (M2 policy core, adapted)
# ============================================================================


class Controller:
    def __init__(self, args):
        self.args = args
        t0 = time.time()
        while True:
            try:
                self.ctx = Ctx(ray_address=args.ray_address)
                if self.ctx.get("init_complete", required=False):
                    break
                raise RuntimeError("driver init not complete yet")
            except Exception as e:
                if time.time() - t0 > GATE_TIMEOUT_S:
                    print(f"[policy] GATE-TIMEOUT waiting for handles ({e})", flush=True)
                    sys.exit(7)
                print(f"[policy] waiting for driver init ({e}); retry in 20s", flush=True)
                time.sleep(20)
        self.mech = Mechanics(self.ctx, args)
        self.log = TrainLogParser(args.train_log)
        self.mode = "dry"
        self.state = "TRAINER_ACTIVE"
        self.sim_state = "TRAINER_ACTIVE"
        self.fill_ema = Ema(args.fill_tau)
        self.fill_updates = 0
        self.prev_produced = None
        self.prev_produced_t = None
        self.rt_in = float(os.environ.get('VD_RT_IN_SEED', '8'))
        self.rt_out = float(os.environ.get('VD_RT_OUT_SEED', '8'))
        self.last_op_end = 0.0
        self.last_switch_block = -1
        self.switch_in_t = 0.0
        self.consec_fail = 0
        self.pending_verify_deadline = None
        self.pending_verify_marker = None
        self.prev_dropped_stale = None
        self.last_stale_incr_t = 0.0
        self.tick = 0
        self.cycles_completed = 0
        self.dry_blocks_baseline = None
        self.would_in = 0
        self.would_back = 0
        self.live_in = 0
        self.live_back = 0
        self.last_action = "none-yet"
        self.last_progress_t = time.time()
        self.wcache_events = []

    def record(self, rec):
        rec = {"ts": now_iso(), "tick": self.tick, "mode": self.mode, **rec}
        with open(self.args.decisions_file, "a") as f:
            f.write(json.dumps(rec) + "\n")
        a = rec.get("action") or rec.get("event")
        if a and a != "none":
            self.last_action = f"{a}@{rec['ts']}"
            if a == "would_switch_to_rollout":
                self.would_in += 1
            elif a == "would_switch_to_trainer":
                self.would_back += 1
            elif a == "switch_to_rollout":
                self.live_in += 1
            elif a == "switch_to_trainer":
                self.live_back += 1
        return rec

    def say(self, msg):
        print(f"[{now_iso()}] [policy] {msg}", flush=True)

    def progress(self, force=False):
        t = time.time()
        if not force and t - self.last_progress_t < 600:
            return
        self.last_progress_t = t
        self.say(
            f"PROGRESS mode={self.mode} state={self.state} sim={self.sim_state} tick={self.tick} "
            f"would={self.would_in}/{self.would_back} live={self.live_in}/{self.live_back} "
            f"cycles={self.cycles_completed} psyncs={self.log.param_syncs} "
            f"blocks={self.log.blocks_closed} wcache_v={self.mech.wcache_ready_version} "
            f"last={self.last_action}"
        )

    # ---------------- signals ----------------

    def sample_signals(self):
        sig = {}
        t = time.time()
        try:
            stats = ray.get(self.ctx.mq.get_statistics.remote(), timeout=15)
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
            rs = ray.get(self.ctx.rollouter.get_statistics.remote(), timeout=15)
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
            blocked=L.block_open, collected=L.collected, samples_needed=L.samples_needed,
            mq_len_log=L.mq_len_log, block_id=L.requests, blocks_closed=L.blocks_closed,
            param_syncs=L.param_syncs, param_version=L.param_version,
            wcache_version=self.mech.wcache_ready_version,
            fill_ema=round(self.fill_ema.value, 5) if self.fill_ema.value is not None else None,
        )
        return sig

    def eta_seconds(self, sig):
        need = sig.get("samples_needed", 32)
        progress = min(sig.get("collected", 0) + sig.get("queue_size", 0), need)
        remaining = need - progress
        fill = self.fill_ema.value
        if remaining <= 0:
            return 0.0
        if not fill or fill < self.args.fill_floor:
            return float("inf")
        return remaining / fill

    # ---------------- gates ----------------

    def eval_switch_in(self, sig, t):
        eta = self.eta_seconds(sig)
        threshold = self.args.c * (self.rt_in + self.rt_out)
        gate = {
            "eta_s": None if math.isinf(eta) else round(eta, 1),
            "eta_inf": math.isinf(eta),
            "threshold_s": round(threshold, 1),
            "rt_in": round(self.rt_in, 1), "rt_out": round(self.rt_out, 1),
        }
        if "mq_error" in sig:
            return False, "signal_error_mq", gate
        warmup_needed = (self.args.fill_warmup if (self.live_in + self.would_in) == 0
                         else self.args.fill_warmup_after_reset)
        if self.fill_updates < warmup_needed:
            gate["fill_updates"] = self.fill_updates
            return False, "fill_ema_warmup", gate
        if not sig.get("blocked"):
            return False, "not_blocked", gate
        if sig.get("collected", 0) < 1:
            return False, "block_burst_not_absorbed", gate
        if sig.get("block_id", -1) == self.last_switch_block:
            return False, "already_switched_this_block", gate
        if t - self.last_op_end < self.args.min_dwell:
            return False, "min_dwell", gate
        if sig.get("dropped_samples", 0) > 0:
            return False, "mq_dropped_samples", gate
        if t - self.last_stale_incr_t < self.args.stale_holdoff and self.last_stale_incr_t > 0:
            return False, "staleness_guard", gate
        v = sig.get("param_version", -1)
        if self.mech.wcache_ready_version is None or int(self.mech.wcache_ready_version) != int(v):
            gate["wcache_version"] = self.mech.wcache_ready_version
            gate["param_version"] = v
            return False, "wcache_not_fresh", gate
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
            "window_s": round(t - active_since, 1),
        }
        if not sig.get("blocked") and sig.get("collected", 0) >= sig.get("samples_needed", 32):
            return True, "batch_completed_late", gate
        if sig.get("collected", 0) >= self.args.hard_collect:
            return True, "hard_collect_failsafe", gate
        if t - active_since >= self.args.window_cap:
            return True, "window_cap", gate
        if eta <= trigger_at:
            return True, "predictive_eta", gate
        return False, "eta_above_trigger", gate

    # ---------------- actions ----------------

    def _run_op(self, name, fn):
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
            self.say(f"ABORT: {name} hung > {self.args.op_timeout}s")
            sys.exit(3)
        if "err" in result:
            return False, result["err"]
        return True, result.get("rec")

    def maybe_refresh_wcache(self):
        """Consume freshly observed param_sync versions -> dump+prefetch."""
        while self.log.new_psync_versions:
            v = self.log.new_psync_versions.pop(0)
            def on_done(res, v=v):
                self.wcache_events.append(res)
                self.record({"event": "wcache_refresh", **res})
                print(f"[policy] wcache refresh: {res}", flush=True)
            started = self.mech.refresh_wcache_async(v, on_done)
            if not started:
                # busy: re-queue the LATEST version only
                self.log.new_psync_versions = [v]
                break

    def maybe_go_live(self):
        if self.mode != "dry" or self.dry_blocks_baseline is None:
            return
        done = self.log.blocks_closed - self.dry_blocks_baseline
        if done < self.args.dry_run_blocks:
            return
        flag = os.path.exists(self.args.live_flag)
        aligned = self.would_in >= 1
        overdue = done >= self.args.dry_run_blocks + 4
        if not (flag or self.args.auto_live):
            return
        if self.args.auto_live and not flag and not aligned and not overdue:
            return
        self.mode = "live"
        via = "flag" if flag else ("auto" if aligned else "auto_failsafe_no_would_in")
        self.record({"event": "mode_change", "to": "live", "dry_blocks": done,
                     "would_in": self.would_in, "would_back": self.would_back, "via": via})
        self.say(f"LIVE mode after {done} dry blocks (would={self.would_in}/{self.would_back}, via={via})")

    # ---------------- main loop ----------------

    def wait_steady(self):
        self.say(f"waiting for steady state (>= {self.args.steady_syncs} param syncs "
                 f"AND >= {self.args.steady_blocks} closed blocks)...")
        t0 = time.time()
        while True:
            self.log.poll()
            if self.log.param_syncs >= self.args.steady_syncs and self.log.blocks_closed >= self.args.steady_blocks:
                break
            waited = time.time() - t0
            if waited > self.args.steady_timeout:
                self.say(f"GATE-TIMEOUT: psyncs={self.log.param_syncs} blocks={self.log.blocks_closed} "
                         f"after {waited:.0f}s")
                sys.exit(7)
            time.sleep(15)
        self.say(f"steady: psyncs={self.log.param_syncs} blocks={self.log.blocks_closed} "
                 f"version={self.log.param_version}")

    def run(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        self.record({"event": "controller_start",
                     "config": {k: getattr(self.args, k) for k in vars(self.args)},
                     "seeds": {"rt_in": self.rt_in, "rt_out": self.rt_out}})
        self.mech.setup_fleet_agent()
        self.wait_steady()
        # initial wcache: current version (covers windows before the next sync)
        self.log.new_psync_versions.append(self.log.param_version)
        self.maybe_refresh_wcache()
        if self.args.throttle_concurrency > 0:
            base = self.ctx.get("max_concurrent_base")
            ray.get(self.ctx.rollouter.elastic_set_max_concurrent_samples.remote(
                int(self.args.throttle_concurrency)))
            # switch ops must restore to the THROTTLED value, not the true base,
            # while shakedown is active: override the cached base.
            self.ctx._cache["max_concurrent_base"] = int(self.args.throttle_concurrency)
            self.record({"event": "shakedown_throttle", "true_base": base,
                         "throttled_to": self.args.throttle_concurrency})
            self.say(f"SHAKEDOWN: rollouter concurrency throttled {base} -> "
                     f"{self.args.throttle_concurrency}")
        self.dry_blocks_baseline = self.log.blocks_closed
        self.record({"event": "dry_run_start", "blocks": self.log.blocks_closed})
        self.say(f"poll loop (dry for {self.args.dry_run_blocks} blocks, auto_live={self.args.auto_live})")
        self.progress(force=True)

        while True:
            self.tick += 1
            t = time.time()
            self.progress()
            if os.path.exists(STOP_FLAG):
                self.record({"event": "controller_stop", "reason": "stop_flag"})
                if self.state == "R2_ACTIVE":
                    self.say("stop flag while spares active: switching back first")
                    self._run_op("switch-to-trainer", self.mech.switch_to_trainer)
                self.say("stop flag; exiting")
                return 0
            sig = self.sample_signals()
            self.maybe_refresh_wcache()
            if self.log.ray_actor_error:
                self.record({"event": "abort", "reason": "RayActorError", "signals": sig})
                self.say("ABORT: RayActorError in train.log")
                return 6

            if self.pending_verify_deadline is not None:
                progressed = (self.log.requests > self.pending_verify_marker[0]
                              or self.log.param_syncs > self.pending_verify_marker[1])
                if progressed:
                    self.cycles_completed += 1
                    self.record({"event": "cycle_verified", "cycle": self.cycles_completed,
                                 "param_version": self.log.param_version, "signals": sig})
                    self.say(f"cycle {self.cycles_completed} verified")
                    self.pending_verify_deadline = None
                    if self.args.max_cycles and self.cycles_completed >= self.args.max_cycles:
                        if self.args.throttle_concurrency > 0:
                            # restore the true base before exiting
                            true_base = ray.get(
                                self.ctx.handles.get.remote("max_concurrent_base"))
                            ray.get(self.ctx.rollouter.elastic_set_max_concurrent_samples.remote(
                                int(true_base)))
                            self.say(f"SHAKEDOWN: concurrency restored to {true_base}")
                        self.record({"event": "controller_stop",
                                     "reason": f"max_cycles={self.args.max_cycles} reached"})
                        self.say(f"max cycles reached ({self.cycles_completed}); exiting 0")
                        return 0
                elif t > self.pending_verify_deadline:
                    self.record({"event": "abort", "reason": "no_progress_after_switch_back", "signals": sig})
                    self.say("ABORT: no trainer progress within 900s of onload")
                    return 5

            self.maybe_go_live()

            if self.mode == "live" and self.state == "TRAINER_ACTIVE":
                ok, reason, gate = self.eval_switch_in(sig, t)
                if ok:
                    self.log.poll()
                    if not self.log.block_open:
                        self.record({"action": "skip_switch_in", "reason": "block_closed_at_fire",
                                     "signals": sig, "gate": gate})
                    else:
                        self.record({"action": "switch_to_rollout", "reason": reason,
                                     "signals": sig, "gate": gate})
                        self.say(f"SWITCH-IN (block {sig['block_id']}, {sig['collected']}/{sig['samples_needed']}, "
                                 f"ETA {gate['eta_s']}s > {gate['threshold_s']}s)")
                        v = sig.get("param_version", -1)
                        ok2, rec = self._run_op("switch-to-spares",
                                                lambda: self.mech.switch_to_spares(v))
                        self.last_op_end = time.time()
                        if ok2:
                            self.rt_in = 0.5 * rec["total_seconds"] + 0.5 * self.rt_in
                            self.state = "R2_ACTIVE"
                            self.last_switch_block = sig["block_id"]
                            self.switch_in_t = self.last_op_end
                            self.consec_fail = 0
                            self.fill_ema.reset()
                            self.fill_updates = 0
                            self.record({"event": "switch_in_done", "total_seconds": rec["total_seconds"]})
                        else:
                            if "FATAL_TRAINER_YIELDED" in str(rec):
                                self.record({"event": "abort", "reason": "trainer_yielded_unrecovered",
                                             "error": str(rec)})
                                return 4
                            self.consec_fail += 1
                            self.record({"event": "switch_in_failed", "error": str(rec),
                                         "consec_fail": self.consec_fail})
                            if self.consec_fail >= 2:
                                self.say("ABORT: two consecutive switch-in failures")
                                return 2
                else:
                    self.record({"action": "none", "state": self.state, "reason": reason,
                                 "signals": sig, "gate": gate})

            elif self.mode == "live" and self.state == "R2_ACTIVE":
                ok, reason, gate = self.eval_switch_back(sig, t, self.switch_in_t)
                if ok:
                    self.record({"action": "switch_to_trainer", "reason": reason,
                                 "signals": sig, "gate": gate})
                    self.say(f"SWITCH-BACK ({reason}: {sig['collected']}/{sig['samples_needed']}, "
                             f"ETA {gate['eta_s']}s)")
                    ok2, rec = self._run_op("switch-to-trainer", self.mech.switch_to_trainer)
                    self.last_op_end = time.time()
                    if not ok2:
                        self.record({"event": "abort", "reason": "switch_back_failed", "error": str(rec)})
                        return 4
                    self.rt_out = 0.5 * rec["total_seconds"] + 0.5 * self.rt_out
                    self.state = "TRAINER_ACTIVE"
                    self.fill_ema.reset()
                    self.fill_updates = 0
                    self.pending_verify_marker = (self.log.requests, self.log.param_syncs)
                    self.pending_verify_deadline = self.last_op_end + 900
                    self.record({"event": "switch_back_done", "total_seconds": rec["total_seconds"]})
                else:
                    self.record({"action": "none", "state": self.state, "reason": reason,
                                 "signals": sig, "gate": gate})

            else:  # dry-run shadow
                if self.sim_state == "TRAINER_ACTIVE":
                    ok, reason, gate = self.eval_switch_in(sig, t)
                    if ok:
                        self.sim_state = "R2_ACTIVE"
                        self.last_switch_block = sig["block_id"]
                        self.switch_in_t = t
                        self.last_op_end = t
                        self.record({"action": "would_switch_to_rollout", "reason": reason,
                                     "signals": sig, "gate": gate})
                        self.say(f"[dry] would SWITCH-IN (block {sig['block_id']}, ETA {gate['eta_s']}s)")
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
                        self.say(f"[dry] would SWITCH-BACK ({reason})")
                    else:
                        self.record({"action": "none", "state": "SIM_" + self.sim_state,
                                     "reason": reason, "signals": sig, "gate": gate})

            time.sleep(self.args.poll)


def main(argv=None):
    p = argparse.ArgumentParser(description="elastic-vd policy controller")
    p.add_argument("--ray-address", default="auto")
    p.add_argument("--poll", type=float, default=2.0)
    p.add_argument("--c", type=float, default=1.5)
    p.add_argument("--min-dwell", type=float, default=60.0)
    p.add_argument("--wake-margin", type=float, default=15.0)
    p.add_argument("--hard-collect", type=int, default=28, help="of 32 required samples")
    p.add_argument("--window-cap", type=float, default=900.0)
    p.add_argument("--fill-tau", type=float, default=60.0)
    p.add_argument("--fill-floor", type=float, default=0.002)
    p.add_argument("--fill-warmup", type=int, default=15)
    p.add_argument("--fill-warmup-after-reset", type=int, default=5,
                   help="reduced fill-EMA warmup after regime changes (post first fire)")
    p.add_argument("--stale-holdoff", type=float, default=120.0)
    p.add_argument("--dry-run-blocks", type=int, default=4)
    p.add_argument("--auto-live", action="store_true")
    p.add_argument("--live-flag", default=LIVE_FLAG)
    p.add_argument("--steady-syncs", type=int, default=1, help="init sync counts")
    p.add_argument("--steady-blocks", type=int, default=3)
    p.add_argument("--steady-timeout", type=float, default=7200.0)
    p.add_argument("--op-timeout", type=float, default=600.0)
    p.add_argument("--train-log", default=TRAIN_LOG)
    p.add_argument("--decisions-file", default=DECISIONS_FILE)
    p.add_argument("--throttle-concurrency", type=int, default=0,
                   help="SHAKEDOWN ONLY: cap rollouter max_concurrent_samples at this value "
                        "while the trainer is active, making the fleet slow enough to open real "
                        "gen-wait windows on the real stack (never used in scored runs)")
    p.add_argument("--max-cycles", type=int, default=0,
                   help="exit 0 after this many verified cycles (0 = run forever)")
    args = p.parse_args(argv)

    ctl = Controller(args)
    try:
        return ctl.run()
    except KeyboardInterrupt:
        ctl.record({"event": "controller_stop", "reason": "SIGINT"})
        return 0


if __name__ == "__main__":
    sys.exit(main())
