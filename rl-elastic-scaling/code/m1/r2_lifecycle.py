#!/usr/bin/env python3
# elastic-rl-poc M1 — manual switch controller for trainer <-> R2 GPU time-slicing.
#
# Runs INSIDE the training pod (needs: ray, psutil; the timeslice python client
# for any snapshot-agent operation). Attaches to the running job through the
# named, detached Ray actor "elastic_controller_handles" (namespace "elastic")
# registered by fully_async_main_elastic.py.
#
# Commands (per-phase wall-clock timestamps are printed AND appended as JSONL —
# this produces the C2 latency breakdown):
#   status              one-shot dump of LB / rollouter / MQ / agent state
#   watch-mq            poll MessageQueue.get_statistics(): queue depth + fill rate
#   park-r2             initial park: drain (no-op pre-traffic) + cuda-checkpoint R2,
#                       then set the r2_parked flag the driver init blocks on
#   suspend-r2          LB.remove -> abort -> drain -> cuda-checkpoint R2
#   resume-r2           cuda restore -> set_global_steps(k) -> clear_kv_cache ->
#                       resume_generation -> LB.add + clear_sticky_cache
#   suspend-trainer     SIGRTMIN+1 (ncclCommSuspend via shim) to trainer AND
#                       R1 CheckpointEngineWorker PIDs (two-sided; the CE comm
#                       is a 2-rank group), confirm 'suspend done' from every
#                       pid's stderr, then cuda-checkpoint ONLY the FSDP
#                       worker process(es)
#   resume-trainer      cuda restore of the FSDP worker process(es), then
#                       SIGRTMIN+2 (ncclCommResume via shim) to both sides +
#                       'resume done' confirmation
#   switch-to-rollout   suspend-trainer then resume-r2   (trainer gen-wait window)
#   switch-to-trainer   suspend-r2 then resume-trainer
#
# verl seams used (pinned 983cb0f2, all verified against source):
#   GlobalRequestLoadBalancer.remove_servers(server_ids) / add_servers(servers)
#     / clear_sticky_cache() / get_total_inflight() / get_status()
#     (llm_server.py:120-191); server_id == "ip:port" address string
#     (llm_server.py:556-558).
#   vLLMHttpServer.abort_all_requests() -> vLLM pause_generation(abort+drain),
#     engine stays paused (vllm_async_server.py:874-940);
#     resume_generation() (:942-951); wait_for_requests_to_drain() (:871-872);
#     set_global_steps() (:867-869); clear_kv_cache() (:822-833);
#     collective_rpc(method, timeout, args, kwargs) (:228-240).
#   MessageQueue.get_statistics(): queue_size / total_produced / total_consumed
#     / dropped_samples (message_queue.py:110-119).
#   FullyAsyncRollouter.get_statistics() (fully_async_rollouter.py:1166-1186).
#
# Snapshot-agent path: timeslice.snapshot_agent SnapshotAgentClient with the
# cuda backend and explicit PIDs (works against both standalone and DaemonSet
# agents; see guides/snapshot-agent/README.md "CUDA Checkpoint").

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone

HANDLES_ACTOR_NAME = "elastic_controller_handles"
HANDLES_NAMESPACE = "elastic"

JOB_R2 = "elastic-r2"
JOB_TRAINER = "elastic-trainer"

DEFAULT_AGENT = os.environ.get("AGENT_ENDPOINT", "127.0.0.1:9001")

# NCCL C/R shim (universal_cr_shim.c, LD_PRELOADed into the Ray tree): the
# trainer workers hold the dormant trainer+R1 checkpoint-engine NCCL comm.
# Protocol (shim header + GPU-CR/multi-gpu-cr/REPORT.md):
#   SIGRTMIN+1 (35) -> ncclCommSuspend on all tracked comms, BEFORE freeze
#   SIGRTMIN+2 (36) -> ncclCommResume,                       AFTER restore
#
# TWO-SIDED SUSPEND (run3 fix): the checkpoint-engine comm is a 2-rank NCCL
# group — trainer worker rank 0 <-> R1's CheckpointEngineWorker rank 1 (run3
# evidence: '[cr-shim] PID 3717206: tracked comm ... rank=0/2' and
# '[cr-shim] PID 3708401: tracked comm ... rank=1/2'). ncclCommSuspend on a
# multi-rank comm is collective: signaling only the trainer side blocks the
# handler forever (run3: SUSPEND printed rc for the rank-0/1 FSDP comm, never
# for the 0/2 comm, no 'suspend done'; the 2s settle expired, cuda-checkpoint
# froze the process mid-suspend, and post-restore resume returned rc=5
# 'Not in suspended state'). Therefore BOTH sides get SIGRTMIN+1 before the
# trainer freeze and SIGRTMIN+2 after restore. CE workers are SIGNAL-ONLY:
# they are never cuda-checkpointed (R1's serving engine is a different
# process tree and keeps generating throughout).
#
# CONFIRMATION-BASED SETTLE (run3 fix): instead of a fixed sleep, poll each
# signaled process's stderr (via /proc/<pid>/fd/2, same file Ray redirects
# worker stderr to) for the shim's '[cr-shim] PID <pid>: suspend done' /
# 'resume done' completion markers and check every per-comm 'rc=' line.
NCCL_CONFIRM_TIMEOUT = float(os.environ.get("ELASTIC_NCCL_CONFIRM_TIMEOUT", "30.0"))


class ShimLogWatcher:
    """Polls per-PID stderr for [cr-shim] completion markers.

    The v1 shim prints, per handler invocation, one line per tracked comm
    ('... comm 0x... suspend rc=N') and a final '[cr-shim] PID <pid>:
    suspend done <ms>' marker (universal_cr_shim.c:112-142). Ray gives each
    worker its own stderr file; /proc/<pid>/fd/2 reaches it regardless of
    path or mount namespace. Offsets are recorded BEFORE the signal is sent
    so only post-signal output is considered.
    """

    def __init__(self, pids):
        self.watch = {}
        for pid in pids:
            pid = int(pid)
            path = f"/proc/{pid}/fd/2"
            try:
                offset = os.stat(path).st_size
            except OSError:
                offset = 0
            self.watch[pid] = (path, offset)

    def _read_new(self, pid):
        path, offset = self.watch[pid]
        try:
            with open(path, "rb") as f:
                f.seek(offset)
                return f.read().decode(errors="replace")
        except OSError:
            return ""

    def wait_done(self, verb: str, timeout: float):
        """Block until every watched pid printed '<verb> done'.

        Returns (timed_out_pids, bad_rc_lines). bad_rc_lines are any
        '<verb> rc=' lines with rc != 0 seen from confirmed pids.
        """
        deadline = time.time() + timeout
        pending = set(self.watch)
        bad = []
        while pending:
            for pid in sorted(pending):
                data = self._read_new(pid)
                if f"[cr-shim] PID {pid}: {verb} done" in data:
                    for line in data.splitlines():
                        if f"[cr-shim] PID {pid}:" in line and f"{verb} rc=" in line:
                            try:
                                rc = int(line.rsplit("rc=", 1)[1].strip())
                            except ValueError:
                                rc = -1
                            if rc != 0:
                                bad.append(line.strip())
                    pending.discard(pid)
            if not pending or time.time() >= deadline:
                break
            time.sleep(0.2)
        return sorted(pending), bad


def _shim_signal(offset: int):
    """SIGRTMIN+offset, resolved at runtime (Linux glibc SIGRTMIN=34)."""
    import signal as _signal

    return int(getattr(_signal, "SIGRTMIN", 34)) + offset


def send_shim_signal(pids, offset: int, label: str):
    """Send SIGRTMIN+offset to every pid (hostPID pod: os.kill sees them)."""
    signum = _shim_signal(offset)
    for pid in pids:
        os.kill(int(pid), signum)
    print(f"sent {label} (signal {signum}) to pids={[int(p) for p in pids]}", flush=True)
DEFAULT_TIMINGS_FILE = os.environ.get(
    "ELASTIC_TIMINGS_FILE",
    "/workspace/results/switch_timings.jsonl"
    if os.path.isdir("/workspace/results")
    else os.path.join(os.getcwd(), "switch_timings.jsonl"),
)


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ----------------------------------------------------------------------------
# Phase timing (C2 latency breakdown)
# ----------------------------------------------------------------------------


class PhaseTimer:
    def __init__(self, operation: str, timings_file: str = None):
        self.operation = operation
        self.timings_file = timings_file
        self.phases = []
        self.t_start = time.time()

    @contextmanager
    def phase(self, name: str):
        t0 = time.time()
        print(f"[{_now_iso()}] [{self.operation}] PHASE START {name}", flush=True)
        try:
            yield
        finally:
            t1 = time.time()
            self.phases.append({"phase": name, "start": t0, "end": t1, "seconds": round(t1 - t0, 4)})
            print(
                f"[{_now_iso()}] [{self.operation}] PHASE END   {name}  ({t1 - t0:.3f}s)",
                flush=True,
            )

    def finish(self, extra: dict = None):
        total = time.time() - self.t_start
        record = {
            "ts": _now_iso(),
            "operation": self.operation,
            "total_seconds": round(total, 4),
            "phases": self.phases,
        }
        if extra:
            record.update(extra)
        print(f"\n=== {self.operation} timing breakdown (total {total:.3f}s) ===")
        for p in self.phases:
            print(f"  {p['phase']:<38s} {p['seconds']:>9.3f}s")
        if self.timings_file:
            try:
                os.makedirs(os.path.dirname(self.timings_file), exist_ok=True)
                with open(self.timings_file, "a") as f:
                    f.write(json.dumps(record) + "\n")
                print(f"appended timing record -> {self.timings_file}")
            except OSError as e:
                print(f"WARNING: could not write timings file: {e}")
        return record


# ----------------------------------------------------------------------------
# Snapshot-agent helpers (lazy imports: keep this module importable by the
# driver even before the timeslice client is installed)
# ----------------------------------------------------------------------------


def _snapshot_client(agent: str):
    from timeslice.snapshot_agent import SnapshotAgentClient

    return SnapshotAgentClient(agent)


def _cuda_backend_config(pids):
    from timeslice.snapshot_agent import snapshot_agent_pb2 as pb

    return pb.BackendConfig(
        cuda=pb.CudaBackendConfig(explicit_target=pb.ProcessTarget(pids=[int(p) for p in pids]))
    )


def cuda_snapshot(agent: str, job_id: str, pids) -> dict:
    """cuda-checkpoint suspend of `pids`; blocks until the operation completes."""
    with _snapshot_client(agent) as client:
        result = client.snapshot_and_wait(job_id=job_id, backend_config=_cuda_backend_config(pids))
    print(f"snapshot job={job_id} pids={list(pids)} -> {result.status} in {result.elapsed_ms}ms")
    if result.status != "OPERATION_STATUS_COMPLETE":
        raise RuntimeError(f"snapshot of {job_id} failed: {result.status} error={result.error}")
    return {"elapsed_ms": result.elapsed_ms, "storage_bytes": result.storage_bytes}


def cuda_restore(agent: str, job_id: str, pids) -> dict:
    """cuda-checkpoint restore of `pids`; blocks until the operation completes."""
    with _snapshot_client(agent) as client:
        result = client.restore_and_wait(job_id=job_id, backend_config=_cuda_backend_config(pids))
    print(f"restore job={job_id} pids={list(pids)} -> {result.status} in {result.elapsed_ms}ms")
    if result.status != "OPERATION_STATUS_COMPLETE":
        raise RuntimeError(f"restore of {job_id} failed: {result.status} error={result.error}")
    return {"elapsed_ms": result.elapsed_ms}


def discover_gpu_pids(root_pid: int, include_root: bool = True) -> list:
    """PIDs in root_pid's process tree that hold NVIDIA device handles.

    For the out-of-tree R2 the GPU state lives in the vLLM mp-executor
    subprocesses (EngineCore / worker procs) spawned as children of the
    vLLMHttpServer Ray actor process (AsyncLLM with
    distributed_executor_backend="mp", vllm_async_server.py:310). Discovery,
    not hardcoding: walk the tree and keep processes with /dev/nvidia* fds.
    Requires the pod to share the host PID namespace (hostPID: true) so these
    PIDs are valid for the node-local snapshot-agent.
    """
    import psutil  # ray dependency, always present in the pod

    root = psutil.Process(int(root_pid))
    candidates = ([root] if include_root else []) + root.children(recursive=True)
    gpu_pids = []
    for proc in candidates:
        try:
            fd_dir = f"/proc/{proc.pid}/fd"
            for fd in os.listdir(fd_dir):
                try:
                    target = os.readlink(os.path.join(fd_dir, fd))
                except OSError:
                    continue
                if target.startswith("/dev/nvidia"):
                    gpu_pids.append(proc.pid)
                    break
        except (psutil.NoSuchProcess, PermissionError, FileNotFoundError):
            continue
    print(f"discover_gpu_pids(root={root_pid}): tree={[p.pid for p in candidates]} gpu={gpu_pids}")
    if not gpu_pids:
        raise RuntimeError(
            f"no NVIDIA-holding processes found under pid {root_pid}; "
            "is the engine up (and are we in the host PID namespace)?"
        )
    return gpu_pids


def discover_ce_worker_pids(ctx, trainer_pids) -> list:
    """OS PIDs of R1's CheckpointEngineWorker actor processes.

    These are the rank-1 peers of the trainer's dormant checkpoint-engine NCCL
    comm (2-rank group, see header). They are SEPARATE processes from R1's
    serving engine (vLLMHttpServer + mp-executor subprocs) — run3 process
    evidence: CheckpointEngineWorker pid=3708401 vs vLLMHttpServer pid=3708955.
    They receive the shim suspend/resume signals ONLY; they are NEVER
    cuda-checkpointed.

    Primary discovery: rollouter.get_replicas() -> replica.workers are exactly
    the CheckpointEngineWorker handles verl put in the weight-sync group
    (fully_async_trainer.py:217-224); __ray_call__ fetches each PID (same
    escape hatch as elastic_trainer.get_trainer_worker_pids). Fallback: Ray
    state API by actor class name.
    """
    ray = ctx.ray
    pids = []
    try:
        replicas = ray.get(ctx.rollouter.get_replicas.remote())
        refs = []
        for rep in replicas:
            for w in getattr(rep, "workers", []) or []:
                refs.append(w.__ray_call__.remote(lambda self: __import__("os").getpid()))
        pids = [int(p) for p in ray.get(refs)]
        print(f"discover_ce_worker_pids: replicas={len(replicas)} ce_pids={pids}")
    except Exception as e:
        print(f"discover_ce_worker_pids: replica path failed ({e}); trying ray state API")
    if not pids:
        from ray.util.state import list_actors

        for a in list_actors(filters=[("state", "=", "ALIVE")], detail=True, limit=10000):
            cls = getattr(a, "class_name", "") or ""
            if "CheckpointEngineWorker" in cls and getattr(a, "pid", None):
                pids.append(int(a.pid))
        print(f"discover_ce_worker_pids: state-API ce_pids={pids}")
    trainer_set = {int(p) for p in (trainer_pids or [])}
    pids = sorted({p for p in pids if p not in trainer_set})
    if not pids:
        raise RuntimeError(
            "no CheckpointEngineWorker PIDs found — cannot two-sided-suspend the "
            "checkpoint-engine NCCL comm (one-sided suspend blocks; see run3)"
        )
    return pids


# ----------------------------------------------------------------------------
# Ray-side context
# ----------------------------------------------------------------------------


class ElasticContext:
    """Lazily fetches handles from the named elastic_controller_handles actor."""

    def __init__(self, ray_address: str = "auto"):
        import ray

        self.ray = ray
        if not ray.is_initialized():
            # Attach to the running cluster in the pod. The handles actor is
            # detached + namespaced, so it is visible across Ray jobs.
            ray.init(address=ray_address, namespace=HANDLES_NAMESPACE, ignore_reinit_error=True)
        self.handles = ray.get_actor(HANDLES_ACTOR_NAME, namespace=HANDLES_NAMESPACE)
        self._cache = {}

    def get(self, key, required=True):
        if key not in self._cache:
            value = self.ray.get(self.handles.get.remote(key))
            if value is None and required:
                raise RuntimeError(
                    f"handles actor has no '{key}' yet — is the driver far enough through init?"
                )
            self._cache[key] = value
        return self._cache[key]

    def put(self, key, value):
        self.ray.get(self.handles.put.remote(key, value))
        self._cache[key] = value

    # convenience accessors -------------------------------------------------
    @property
    def lb(self):
        return self.get("load_balancer")

    @property
    def rollouter(self):
        return self.get("rollouter")

    @property
    def trainer(self):
        return self.get("trainer")

    @property
    def mq(self):
        return self.get("message_queue")

    @property
    def r2_server(self):
        return self.get("r2_server")

    @property
    def r2_address(self):
        return self.get("r2_address")


class MQMonitor:
    """queue depth + fill rate from MessageQueue.get_statistics() deltas."""

    def __init__(self, ctx: ElasticContext):
        self.ctx = ctx
        self.prev = None  # (t, total_produced)

    def sample(self, label=""):
        try:
            stats = self.ctx.ray.get(self.ctx.mq.get_statistics.remote())
        except Exception as e:  # MQ may not exist yet during initial park
            print(f"[mq] unavailable ({e})")
            return None
        now = time.time()
        fill = None
        if self.prev is not None and now > self.prev[0]:
            fill = (stats["total_produced"] - self.prev[1]) / (now - self.prev[0])
        self.prev = (now, stats["total_produced"])
        print(
            f"[mq]{(' ' + label) if label else ''} depth={stats['queue_size']} "
            f"produced={stats['total_produced']} consumed={stats['total_consumed']} "
            f"dropped={stats['dropped_samples']} "
            f"fill_rate={f'{fill:.3f} samples/s' if fill is not None else 'n/a'}",
            flush=True,
        )
        return {"queue_size": stats["queue_size"], "fill_rate": fill, **stats}


# ----------------------------------------------------------------------------
# Sequences
# ----------------------------------------------------------------------------


def _drain_r2(ctx: ElasticContext, timer: PhaseTimer, mq: MQMonitor, drain_timeout: float):
    """LB removal + abort + engine drain of R2 (shared by park/suspend)."""
    ray = ctx.ray
    with timer.phase("lb_remove_servers"):
        # server_id == server address string (llm_server.py:556-558).
        ray.get(ctx.lb.remove_servers.remote(server_ids=[ctx.r2_address]))

    with timer.phase("restore_base_concurrency"):
        base = ctx.get("max_concurrent_base", required=False)
        if base:
            ray.get(ctx.rollouter.elastic_set_max_concurrent_samples.remote(int(base)))

    with timer.phase("abort_all_requests"):
        # vLLM >=0.12 path: pause_generation(wait_for_inflight_requests=False,
        # clear_cache=True) — aborts, drains engine-side, clears prefix cache,
        # and leaves the engine PAUSED (vllm_async_server.py:890-907). The
        # partial-rollout client transparently resumes aborted requests on R1
        # with accumulated tokens (llm_server.py:365-418).
        abort_result = ray.get(ctx.r2_server.abort_all_requests.remote())
        print(f"abort_all_requests -> {abort_result.get('aborted_count', '?')} aborted")

    with timer.phase("wait_for_requests_to_drain"):
        ray.get(ctx.r2_server.wait_for_requests_to_drain.remote())

    with timer.phase("lb_inflight_settle"):
        # release_server() from aborted requests is fire-and-forget
        # (llm_server.py:223-226); wait for LB counters to settle. Note
        # get_total_inflight() also counts R1 traffic, so this is a bounded
        # diagnostic wait, NOT a hard ==0 gate (R2's own counter was popped by
        # remove_servers).
        deadline = time.time() + drain_timeout
        while time.time() < deadline:
            status = ray.get(ctx.lb.get_status.remote())
            mq.sample("drain-poll")
            print(f"[lb] {status}")
            if ctx.r2_address not in status["servers"]:
                break
            time.sleep(0.5)
        time.sleep(1.0)  # kernel settle margin before freezing the CUDA context


def op_park_r2(ctx: ElasticContext, agent: str, timer: PhaseTimer, drain_timeout: float):
    """Initial park during driver init: R2 is warmed but carries no traffic."""
    mq = MQMonitor(ctx)
    # Before the trainer exists the LB/rollouter handles are already stored
    # (driver registers them pre-barrier); the MQ is not — MQMonitor degrades.
    _drain_r2(ctx, timer, mq, drain_timeout)
    with timer.phase("discover_gpu_pids"):
        pids = discover_gpu_pids(ctx.get("r2_server_pid"))
        ctx.put("r2_gpu_pids", pids)
    with timer.phase("cuda_snapshot_r2"):
        cuda_snapshot(agent, JOB_R2, pids)
    ctx.put("r2_state", "suspended")
    ctx.put("r2_parked", True)
    print("R2 parked; driver init barrier released.")


def op_suspend_r2(ctx: ElasticContext, agent: str, timer: PhaseTimer, drain_timeout: float):
    mq = MQMonitor(ctx)
    mq.sample("pre-suspend")
    _drain_r2(ctx, timer, mq, drain_timeout)
    with timer.phase("discover_gpu_pids"):
        pids = ctx.get("r2_gpu_pids", required=False) or discover_gpu_pids(ctx.get("r2_server_pid"))
        ctx.put("r2_gpu_pids", pids)
    with timer.phase("cuda_snapshot_r2"):
        cuda_snapshot(agent, JOB_R2, pids)
    ctx.put("r2_state", "suspended")
    mq.sample("post-suspend")


def op_resume_r2(ctx: ElasticContext, agent: str, timer: PhaseTimer, reload_weights: bool):
    ray = ctx.ray
    mq = MQMonitor(ctx)
    mq.sample("pre-resume")

    with timer.phase("cuda_restore_r2"):
        pids = ctx.get("r2_gpu_pids")
        cuda_restore(agent, JOB_R2, pids)

    if reload_weights:
        with timer.phase("reload_weights"):
            # M1 stub for weight catch-up: reload the HF checkpoint from disk.
            # Only useful once trainer.save_freq > 0 has produced a newer disk
            # checkpoint; with save_freq=-1 this reloads the version-0 weights.
            # collective_rpc signature verified (vllm_async_server.py:228-240).
            # UNVERIFIED: zero-arg GPUWorker.reload_weights() disk-reload
            # semantics on the image's vLLM build — verify on the live pod
            # before relying on it.
            ray.get(ctx.r2_server.collective_rpc.remote(method="reload_weights"))

    with timer.phase("set_global_steps"):
        # Landmine #4: a server whose global_steps is None poisons trainer
        # metrics (generate() stamps extra_fields["global_steps"],
        # vllm_async_server.py:650). R2 is outside the weight-sync group, so
        # nothing else ever sets it. NOTE (documented M1 shortcut): this tags
        # R2 samples with the CURRENT version even though R2's weights are
        # older — staleness accounting is metrics-only here; correctness rests
        # on use_rollout_log_probs=True + rollout_correction.bypass_mode=True
        # (both config defaults at this pin).
        version = ray.get(ctx.trainer.get_current_param_version.remote())
        ray.get(ctx.r2_server.set_global_steps.remote(version))
        print(f"R2 global_steps set to current_param_version={version}")

    with timer.phase("clear_kv_cache"):
        # KV hygiene (ServiceNow trap): flush prefix cache before serving.
        ray.get(ctx.r2_server.clear_kv_cache.remote())

    with timer.phase("resume_generation"):
        # abort_all_requests left the engine paused (vllm_async_server.py:884-886).
        ray.get(ctx.r2_server.resume_generation.remote())

    with timer.phase("lb_add_servers"):
        ray.get(ctx.lb.add_servers.remote(servers={ctx.r2_address: ctx.r2_server}))
        ray.get(ctx.lb.clear_sticky_cache.remote())

    with timer.phase("raise_concurrency"):
        # Out-of-tree R2 bypasses rollouter concurrency accounting
        # (fully_async_rollouter.py:503-506 / 1274-1294): without this the
        # rollouter would keep submitting only base(=16) concurrent samples and
        # R2's marginal fill rate would measure ~0.
        base = ctx.get("max_concurrent_base", required=False) or 16
        ray.get(ctx.rollouter.elastic_set_max_concurrent_samples.remote(int(base) * 2))

    ctx.put("r2_state", "active")
    mq.sample("post-resume")


def op_suspend_trainer(ctx: ElasticContext, agent: str, timer: PhaseTimer):
    ray = ctx.ray
    print(
        "REMINDER (M1 manual gate): trigger this only while the trainer log shows it is\n"
        "blocked in gen-wait, i.e. after '[FullyAsyncTrainer] Requesting N samples from queue'\n"
        "and while 'sample collected i/N' lines are still trickling in.\n"
        "Safety property if mistimed: the step stalls inside the first CUDA call of the\n"
        "frozen worker and completes after resume — param_sync cannot start while frozen\n"
        "(fit_step ordering, fully_async_trainer.py:584-594)."
    )
    with timer.phase("get_trainer_pids"):
        pids = ctx.get("trainer_worker_pids", required=False)
        if not pids:
            pids = ray.get(ctx.trainer.get_trainer_worker_pids.remote())
            ctx.put("trainer_worker_pids", pids)
    with timer.phase("get_ce_worker_pids"):
        ce_pids = ctx.get("ce_worker_pids", required=False)
        if not ce_pids:
            ce_pids = discover_ce_worker_pids(ctx, pids)
            ctx.put("ce_worker_pids", ce_pids)
    all_pids = sorted({int(p) for p in list(pids) + list(ce_pids)})
    with timer.phase("nccl_suspend_signal"):
        # TWO-SIDED: ncclCommSuspend is collective on the 2-rank
        # checkpoint-engine comm — both the trainer worker (rank 0) and R1's
        # CheckpointEngineWorker (rank 1) must run their handlers together
        # BEFORE the trainer freeze (one-sided suspend blocks; run3 abort).
        watcher = ShimLogWatcher(all_pids)
        send_shim_signal(all_pids, 1, "SIGRTMIN+1 ncclCommSuspend (trainer+CE)")
    with timer.phase("nccl_suspend_confirm"):
        # Confirmation-based settle: poll each signaled process's stderr for
        # the shim's 'suspend done' marker + per-comm rc lines (replaces the
        # fixed 2s sleep that let run3 freeze mid-suspend).
        timed_out, bad = watcher.wait_done("suspend", NCCL_CONFIRM_TIMEOUT)
        if timed_out or bad:
            # Undo best-effort so the run stays alive for diagnosis: resume
            # whatever did suspend, then fail the switch WITHOUT freezing.
            print(f"suspend confirm FAILED: timed_out={timed_out} bad_rc={bad}")
            try:
                send_shim_signal(all_pids, 2, "SIGRTMIN+2 ncclCommResume (rollback)")
            except OSError as e:
                print(f"rollback resume signal failed: {e}")
            raise RuntimeError(
                f"ncclCommSuspend confirmation failed: no 'suspend done' from "
                f"pids={timed_out} within {NCCL_CONFIRM_TIMEOUT}s, bad rc lines={bad}"
            )
        print(f"suspend confirmed on all pids={all_pids}")
    with timer.phase("cuda_snapshot_trainer"):
        # Freeze the TRAINER worker processes only — CE workers are
        # signal-only and R1 must keep serving throughout.
        cuda_snapshot(agent, JOB_TRAINER, pids)
    ctx.put("trainer_state", "suspended")


def op_resume_trainer(ctx: ElasticContext, agent: str, timer: PhaseTimer):
    with timer.phase("cuda_restore_trainer"):
        pids = ctx.get("trainer_worker_pids")
        cuda_restore(agent, JOB_TRAINER, pids)
    ce_pids = ctx.get("ce_worker_pids", required=False) or []
    all_pids = sorted({int(p) for p in list(pids) + list(ce_pids)})
    with timer.phase("nccl_resume_signal"):
        # ncclCommResume AFTER restore completes, on BOTH sides of the
        # checkpoint-engine comm (shim protocol, universal_cr_shim.c).
        watcher = ShimLogWatcher(all_pids)
        send_shim_signal(all_pids, 2, "SIGRTMIN+2 ncclCommResume (trainer+CE)")
    with timer.phase("nccl_resume_confirm"):
        timed_out, bad = watcher.wait_done("resume", NCCL_CONFIRM_TIMEOUT)
        if timed_out or bad:
            raise RuntimeError(
                f"ncclCommResume confirmation failed: no 'resume done' from "
                f"pids={timed_out} within {NCCL_CONFIRM_TIMEOUT}s, bad rc lines={bad} "
                f"(rc=5 == 'Not in suspended state', the run3 signature)"
            )
        print(f"resume confirmed on all pids={all_pids}")
    ctx.put("trainer_state", "active")


def op_status(ctx: ElasticContext, agent: str):
    ray = ctx.ray
    print(f"--- elastic status @ {_now_iso()} ---")
    for key in ("r2_state", "trainer_state", "r2_address", "r2_gpu", "r2_server_pid",
                "trainer_worker_pids", "ce_worker_pids", "max_concurrent_base", "r2_parked"):
        try:
            print(f"  {key}: {ctx.get(key, required=False)}")
        except Exception as e:
            print(f"  {key}: <error {e}>")
    try:
        print(f"  lb.get_status: {ray.get(ctx.lb.get_status.remote())}")
    except Exception as e:
        print(f"  lb: <error {e}>")
    try:
        stats = ray.get(ctx.rollouter.get_statistics.remote())
        print("  rollouter.get_statistics:")
        for k, v in stats.items():
            print(f"    {k}: {v}")
    except Exception as e:
        print(f"  rollouter: <error {e}>")
    MQMonitor(ctx).sample("status")
    try:
        version = ray.get(ctx.trainer.get_current_param_version.remote())
        print(f"  trainer.current_param_version: {version}")
    except Exception as e:
        print(f"  trainer: <error {e}>")
    try:
        with _snapshot_client(agent) as client:
            agent_status = client.status()
        for job in agent_status.job_statuses:
            print(f"  agent job {job.job_id}: {job.state}")
        for acc in agent_status.accelerator_statuses:
            used = acc.memory_used_bytes / (1 << 30)
            total = acc.memory_total_bytes / (1 << 30)
            print(f"  agent GPU {acc.id}: {used:.1f}/{total:.1f} GiB used")
    except Exception as e:
        print(f"  snapshot-agent: <error {e}>")


def op_watch_mq(ctx: ElasticContext, interval: float, count: int):
    mq = MQMonitor(ctx)
    i = 0
    while count <= 0 or i < count:
        mq.sample()
        try:
            stats = ctx.ray.get(ctx.rollouter.get_statistics.remote())
            print(
                f"[rollouter] active_tasks={stats['monitor/active_tasks_size']} "
                f"total_generated={stats['count/total_generated_samples']} "
                f"staleness={stats['count/staleness_samples']} "
                f"dropped_stale={stats['count/dropped_stale_samples']} "
                f"max_concurrent={stats['static/max_concurrent_samples']}"
            )
        except Exception as e:
            print(f"[rollouter] <error {e}>")
        i += 1
        if count <= 0 or i < count:
            time.sleep(interval)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description="elastic-rl-poc M1 manual switch controller")
    parser.add_argument("--agent", default=DEFAULT_AGENT, help="snapshot-agent endpoint host:port")
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--timings-file", default=DEFAULT_TIMINGS_FILE)
    parser.add_argument("--drain-timeout", type=float, default=30.0,
                        help="bounded wait for LB inflight settle before freezing R2")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status")
    p = sub.add_parser("watch-mq")
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--count", type=int, default=0, help="0 = forever")
    sub.add_parser("park-r2")
    sub.add_parser("suspend-r2")
    p = sub.add_parser("resume-r2")
    p.add_argument("--reload-weights", action="store_true")
    sub.add_parser("suspend-trainer")
    sub.add_parser("resume-trainer")
    p = sub.add_parser("switch-to-rollout")
    p.add_argument("--reload-weights", action="store_true")
    sub.add_parser("switch-to-trainer")

    args = parser.parse_args(argv)
    ctx = ElasticContext(ray_address=args.ray_address)

    if args.command == "status":
        op_status(ctx, args.agent)
        return 0
    if args.command == "watch-mq":
        op_watch_mq(ctx, args.interval, args.count)
        return 0

    timer = PhaseTimer(args.command, timings_file=args.timings_file)
    if args.command == "park-r2":
        op_park_r2(ctx, args.agent, timer, args.drain_timeout)
    elif args.command == "suspend-r2":
        op_suspend_r2(ctx, args.agent, timer, args.drain_timeout)
    elif args.command == "resume-r2":
        op_resume_r2(ctx, args.agent, timer, reload_weights=args.reload_weights)
    elif args.command == "suspend-trainer":
        op_suspend_trainer(ctx, args.agent, timer)
    elif args.command == "resume-trainer":
        op_resume_trainer(ctx, args.agent, timer)
    elif args.command == "switch-to-rollout":
        # Order matters for GPU memory: free the trainer's ~GPU0 allocation
        # first, then restore R2's image into the freed space.
        op_suspend_trainer(ctx, args.agent, timer)
        op_resume_r2(ctx, args.agent, timer, reload_weights=args.reload_weights)
    elif args.command == "switch-to-trainer":
        op_suspend_r2(ctx, args.agent, timer, args.drain_timeout)
        op_resume_trainer(ctx, args.agent, timer)
    timer.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
