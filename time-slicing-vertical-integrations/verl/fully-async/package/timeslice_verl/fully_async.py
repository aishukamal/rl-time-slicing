"""Time-slicing wrapper for verl's experimental fully_async_policy trainer.

Verified against verl 983cb0f24443f87b3d161fad318445130a620b07:

  * `verl/experimental/fully_async_policy/fully_async_trainer.py:53-54` —
    `@ray.remote(num_cpus=10) class FullyAsyncTrainer(SeparateRayPPOTrainer)`.
    The FullyAsyncTrainer is a CPU-ONLY driver actor; all trainer-GPU work is
    RPC'd to a separate FSDP worker process (the cuda-checkpoint target). Every
    lock call below therefore runs in a process that is never checkpointed, so
    blocking gRPC is safe — it just must not stall the actor's asyncio event
    loop (Ray actors run coroutine methods on one loop; max_concurrency defaults
    to 1000), hence all lock RPCs go through `asyncio.to_thread`.

  * Patched methods (signatures preserved exactly):
      - `async def init_workers(self)`                      (L487)
        ACQUIRE before delegating (worker-group creation + model load touch the
        shared GPU), YIELD after it completes: the FSDP worker is checkpointed
        off until steady state restores it. Serializes two jobs' inits.
      - `async def _get_samples_from_queue(self)
             -> tuple[None, None] | tuple[int, Any]`        (L375)
        Delegate first (the queue wait is CPU-only: Ray RPC to the MessageQueue
        actor; zero CUDA in the path), then ACQUIRE after the batch is
        assembled, right before returning — the per-step resume point.
      - `async def _fit_update_weights(self) -> dict | None` (L690)
        Delegate; if it returned non-None a real param sync happened
        (update_actor + NCCL broadcast + rollouter.reset_staleness all done) —
        YIELD the lock. Non-None <=> weights actually updated, per the
        method's own docstring.
        DELIBERATE ADDITION (flagged in the PoC notes, not silent): an
        idempotent ensure-held BEFORE delegating. fully_async_main's
        `_initialize_components` calls `trainer._fit_update_weights()` for the
        initial param sync AFTER init_workers (where we yielded) and BEFORE
        fit() starts; without re-acquiring here that sync would hit a
        checkpointed-off FSDP worker without the lock and deadlock the startup.
        In steady state the lock is already held (acquired at queue-get), so
        this is a no-op.
      - `def _fit_save_checkpoint(self, force=False)`       (L873)
        Guard: if config left trainer.save_freq > 0, warn once and skip —
        saving RPCs the FSDP worker, which may be checkpointed off at this
        point in the step. Our runs use save_freq=-1 (original is then already
        a no-op; we still delegate for exactness).

  * Attach: verl auto-loads the `verl.plugins` entry points at `import verl`
    (verl/__init__.py:49-65) in EVERY process, and swallows loader exceptions —
    so nothing verl-side may be imported at plugin load time (entry points fire
    mid-verl-import). `install()` only registers a `sys.meta_path` finder that
    patches FullyAsyncTrainer when
    `verl.experimental.fully_async_policy.fully_async_trainer` is first
    imported (or immediately, if it already was). Inert unless
    TIMESLICE_FULLY_ASYNC=1.

  * Ray pickling: `FullyAsyncTrainer` in the target module is a Ray ActorClass;
    the real class is `ActorClass.__ray_metadata__.modified_class`, which Ray
    cloudpickles BY VALUE into the actor process at creation time. The patch
    therefore (a) targets modified_class, (b) stores the originals as class
    attributes (`_ts_orig_*`) so they travel with the pickled class, and
    (c) uses module-level wrapper functions (pickled by reference; all mutable
    lock/metrics state is process-local module state resolved at call time in
    whichever process executes — the FullyAsyncTrainer driver actor).

Locks: a single orchestrator group lock (the trainers pool), same
acquire/yield RPC semantics as the sync mode, via a metered PhaseLocks
subclass. Env: TIMESLICE_JOB_ID / TIMESLICE_ORCH_ADDR / TIMESLICE_GROUP
(missing => no-op, same contract as PhaseLocks).

Metrics: every real acquire/yield appends a JSON line to
$TIMESLICE_METRICS_PATH (default /workspace/results/rl_metrics.jsonl) for the
replay/dashboards. Metrics IO can never break training.
"""

import asyncio
import importlib.abc
import importlib.machinery
import json
import os
import sys
import threading
import time
from typing import Any, Optional

from timeslice_verl.locks import ENV_GROUP, ENV_JOB_ID, ENV_ORCH_ADDR, PhaseLocks, _log

ENV_ENABLE = "TIMESLICE_FULLY_ASYNC"
ENV_METRICS_PATH = "TIMESLICE_METRICS_PATH"
DEFAULT_METRICS_PATH = "/workspace/results/rl_metrics.jsonl"
# EXPERIMENTAL (2026-07-26, not wired into any manifest yet): when "1", call
# torch.cuda.empty_cache() immediately before each yield's drop_all, to probe
# whether releasing cached allocator blocks shrinks the cuda-checkpoint
# snapshot (small-run finding: snapshot time grows with allocated training
# memory, ~13s cold -> ~22-27s at 49GB steady state). Inert by default.
ENV_EMPTY_CACHE = "TIMESLICE_EMPTY_CACHE_BEFORE_YIELD"

_TARGET_MODULE = "verl.experimental.fully_async_policy.fully_async_trainer"
_TARGET_CLASS = "FullyAsyncTrainer"
_PATCH_MARKER = "_timeslice_fully_async_patched"

# batch.meta_info key with the rollouter's mq snapshot (embedded by
# assemble_batch_from_rollout_samples, detach_utils.py) — free to read.
_MQ_SIZE_META_KEY = "fully_async/monitor/queue/mq_queue_size"
# trainer.metrics key snapshotted at the end of the previous step
# (fully_async_trainer.py:908) — fallback, also free.
_MQ_SIZE_METRICS_KEY = "dynamic_resource/mq_size"


def enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "").strip().lower() in ("1", "true", "yes")


# ======================================================================
# Lock state (process-local; lives in the FullyAsyncTrainer driver actor)
# ======================================================================

# Test injection point: callable (target, job_id, group_id) -> client.
_client_factory = None

_state_lock = threading.Lock()
_locks: Optional["_MeteredPhaseLocks"] = None
_acquired_monotonic: Optional[float] = None
_warned: set = set()


class _MeteredPhaseLocks(PhaseLocks):
    """PhaseLocks (single group) that surfaces the acquire RPC's result fields
    for rl_metrics.jsonl and accepts an injectable client factory for grpc-free
    tests (mirrors the RoleLocks pattern in locks.py). Semantics — idempotent
    blocking acquire, idempotent swallow-errors release, atexit safety net —
    are inherited from PhaseLocks unchanged."""

    def __init__(self, job_id, orch_addr, group, client_factory=None):
        # Mirrors PhaseLocks.__init__ but routes client construction through
        # the factory. State attrs must match PhaseLocks' exactly so the
        # inherited ensure/drop_all/close/_atexit_cleanup keep working.
        self.job_id = job_id
        self.orch_addr = orch_addr
        self.group = group
        self.enabled = bool(job_id and orch_addr and group)
        self._held = set()
        self._client = None
        self._closed = False

        if not self.enabled:
            _log(
                "WARNING: fully-async time-slicing lock disabled (missing one of "
                f"{ENV_JOB_ID}/{ENV_ORCH_ADDR}/{ENV_GROUP}); lock calls are no-ops."
            )
            return

        if client_factory is None:
            try:
                from timeslice import OrchestratorClient  # lazy: keep import-safe
            except ImportError:  # renamed upstream to TimeSliceOrchestratorClient (small-run finding 2026-07-26; API identical)
                from timeslice import TimeSliceOrchestratorClient as OrchestratorClient

            def client_factory(target, job_id, group_id):
                return OrchestratorClient(target=target, job_id=job_id, group_id=group_id)

        self._client = client_factory(self.orch_addr, self.job_id, self.group)
        _log(
            f"job={self.job_id} connected orchestrator={self.orch_addr} "
            f"group={self.group} (fully_async)"
        )
        import atexit

        atexit.register(self._atexit_cleanup)

    @property
    def held(self) -> bool:
        return self.group in self._held

    def acquire_info(self) -> dict:
        """Blocking-acquire the single group; returns metric fields.

        {"acquired": False} when disabled or already held (idempotent path);
        otherwise {"acquired": True, "wait_ms": ..., "restore_ms": ...,
        "context_restored": ...}. Acquire RPC errors propagate (a job must not
        run unlocked because the orchestrator is unreachable)."""
        if not self.enabled or self.group in self._held:
            return {"acquired": False}
        t0 = time.monotonic()
        result = self._client.acquire(group_id=self.group)  # blocks until granted
        wait_ms = getattr(result, "waited_ms", None)
        if wait_ms is None:
            wait_ms = int((time.monotonic() - t0) * 1000)
        self._held.add(self.group)
        _log(
            f"job={self.job_id} ACQUIRE group={self.group} waited={wait_ms}ms "
            f"context_restored={getattr(result, 'context_restored', '?')}"
        )
        return {
            "acquired": True,
            "wait_ms": wait_ms,
            # Not in the current AcquireResult (success/waited_ms/
            # context_restored only) -> None; kept for schema stability.
            "restore_ms": getattr(result, "restore_ms", None),
            "context_restored": getattr(result, "context_restored", None),
        }


def _get_locks() -> _MeteredPhaseLocks:
    """Lazily build the per-process lock singleton (may open a gRPC channel:
    only call off the event loop)."""
    global _locks
    if _locks is None:
        with _state_lock:
            if _locks is None:
                _locks = _MeteredPhaseLocks(
                    job_id=os.environ.get(ENV_JOB_ID),
                    orch_addr=os.environ.get(ENV_ORCH_ADDR),
                    group=os.environ.get(ENV_GROUP),
                    client_factory=_client_factory,
                )
    return _locks


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        _log(f"WARNING: {msg}")


# ======================================================================
# rl_metrics.jsonl emission (never allowed to break training)
# ======================================================================

def _emit(event: dict) -> None:
    try:
        path = os.environ.get(ENV_METRICS_PATH) or DEFAULT_METRICS_PATH
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        line = json.dumps(event)
        # open-append-close per line: line-atomic-ish and flushed; event rate
        # is two lines per training step.
        with open(path, "a") as f:
            f.write(line + "\n")
    except Exception as e:  # noqa: BLE001 - metrics IO must never break training
        _warn_once("metrics_io", f"rl_metrics emission failed ({e}); continuing without metrics")


def _observed_queue_len(trainer: Any, batch: Any = None):
    """Last observed MessageQueue depth, only from data already in hand."""
    try:
        if batch is not None:
            meta = getattr(batch, "meta_info", None)
            if isinstance(meta, dict) and meta.get(_MQ_SIZE_META_KEY) is not None:
                return meta[_MQ_SIZE_META_KEY]
        metrics = getattr(trainer, "metrics", None)
        if isinstance(metrics, dict) and metrics.get(_MQ_SIZE_METRICS_KEY) is not None:
            return metrics[_MQ_SIZE_METRICS_KEY]
    except Exception:  # noqa: BLE001
        pass
    return None


def _current_step(trainer: Any):
    try:
        return getattr(trainer, "current_param_version", None)
    except Exception:  # noqa: BLE001
        return None


def _base_event(etype: str, trainer: Any, batch: Any = None) -> dict:
    locks = _locks
    workload_id = (locks.job_id if locks is not None else None) or os.environ.get(ENV_JOB_ID)
    pool = (locks.group if locks is not None else None) or os.environ.get(ENV_GROUP) or "trainers"
    return {
        "ts": time.time(),
        "type": etype,
        "workload_id": workload_id,
        "pool": pool,
        "queue_len": _observed_queue_len(trainer, batch),
        "step": _current_step(trainer),
    }


# ======================================================================
# Async lock transitions (called from the patched coroutine methods)
# ======================================================================

async def _ensure_lock(trainer: Any, point: str, batch: Any = None) -> None:
    """Idempotent blocking acquire, run in a worker thread so the actor's
    event loop stays free (the wait can be minutes while the other job holds
    the group). Emits an "acquire" metrics line only on a real acquire."""
    global _acquired_monotonic
    locks = await asyncio.to_thread(_get_locks)
    info = await asyncio.to_thread(locks.acquire_info)
    if not info.get("acquired"):
        return
    _acquired_monotonic = time.monotonic()
    try:
        event = _base_event("acquire", trainer, batch)
        event["wait_ms"] = info.get("wait_ms")
        event["restore_ms"] = info.get("restore_ms")
        event["context_restored"] = info.get("context_restored")
        event["point"] = point
        _emit(event)
    except Exception:  # noqa: BLE001
        pass


def _empty_cache_enabled() -> bool:
    return os.environ.get(ENV_EMPTY_CACHE, "").strip().lower() in ("1", "true", "yes")


def _maybe_empty_cache(point: str) -> None:
    """EXPERIMENTAL, env-gated (TIMESLICE_EMPTY_CACHE_BEFORE_YIELD=1): best-effort
    torch.cuda.empty_cache() right before a yield's drop_all, so the allocator's
    cached-but-free blocks are returned to the driver before the orchestrator
    snapshots the job. Never raises; no-op when disabled, when torch is not
    importable, or when empty_cache itself fails (e.g. no CUDA context in this
    process — note the lock RPCs run in the CPU-only driver actor)."""
    if not _empty_cache_enabled():
        return
    try:
        import torch  # noqa: PLC0415 - only touch torch when the experiment is on
    except Exception:  # noqa: BLE001
        _warn_once("empty_cache_no_torch", f"{ENV_EMPTY_CACHE}=1 but torch is not importable; skipping")
        return
    try:
        torch.cuda.empty_cache()
        _log(f"empty_cache before yield point={point} (experimental {ENV_EMPTY_CACHE}=1)")
    except Exception as e:  # noqa: BLE001 - the experiment must never break training
        _warn_once("empty_cache_failed", f"torch.cuda.empty_cache() failed ({e!r}); continuing")


async def _yield_lock(trainer: Any, point: str) -> None:
    """Idempotent release (PhaseLocks.drop_all: errors logged, never raised),
    run in a worker thread. Emits a "yield" metrics line when the lock was
    actually held."""
    global _acquired_monotonic
    locks = _locks
    if locks is None or not locks.enabled or not locks.held:
        return
    held_ms = None
    if _acquired_monotonic is not None:
        held_ms = int((time.monotonic() - _acquired_monotonic) * 1000)
    _acquired_monotonic = None
    await asyncio.to_thread(_maybe_empty_cache, point)  # experimental, inert by default
    await asyncio.to_thread(locks.drop_all)
    try:
        event = _base_event("yield", trainer)
        event["held_ms"] = held_ms
        event["point"] = point
        _emit(event)
    except Exception:  # noqa: BLE001
        pass


async def _crash_release(trainer: Any, point: str, exc: BaseException) -> None:
    """Best-effort lock release on a crash path (FIX 2026-07-26, small-run
    finding 5: job-a died on NcclError inside _fit_update_weights holding the
    group lock; the orchestrator stayed pinned to the dead job). Any exception
    propagating out of a patched method lands here BEFORE re-raising:
    idempotent drop_all + a final {"type":"crash-release",...} metrics line.
    Never raises — crash hygiene must not mask the real error."""
    global _acquired_monotonic
    try:
        locks = _locks
        if locks is None or not locks.enabled:
            return
        was_held = locks.held
        _acquired_monotonic = None
        await asyncio.to_thread(locks.drop_all)  # idempotent; RPC errors swallowed
        event = _base_event("crash-release", trainer)
        event["point"] = point
        event["was_held"] = was_held
        event["error"] = repr(exc)[:500]
        _emit(event)
    except Exception:  # noqa: BLE001
        pass


# ======================================================================
# Patched methods (module-level so cloudpickle serializes them by reference)
# ======================================================================

async def _patched_init_workers(self):
    """ACQUIRE -> delegate (worker groups + model load on the shared GPU) ->
    YIELD (the FSDP worker gets checkpointed off; steady state restores it)."""
    await _ensure_lock(self, point="init_workers")
    try:
        result = await self._ts_orig_init_workers()
    except BaseException as e:
        # Failed init: release so the peer job is not blocked forever.
        await _crash_release(self, "init_workers", e)
        raise
    await _yield_lock(self, point="init_workers")
    return result


async def _patched_get_samples_from_queue(self):
    """Delegate first (CPU-only queue wait), then ACQUIRE once the batch is
    assembled — the per-step resume point. (None, None) = shutdown: no acquire."""
    try:
        result = await self._ts_orig_get_samples_from_queue()
        batch = None
        try:
            if isinstance(result, tuple) and len(result) == 2:
                batch = result[1]
        except Exception:  # noqa: BLE001
            batch = None
        if batch is not None:
            await _ensure_lock(self, point="samples_ready", batch=batch)
        return result
    except BaseException as e:
        await _crash_release(self, "samples_ready", e)
        raise


async def _patched_fit_update_weights(self):
    """Ensure-held (idempotent; real acquire only for the pre-fit initial param
    sync — see module docstring) -> delegate -> YIELD iff a real sync happened
    (non-None return)."""
    await _ensure_lock(self, point="update_weights")
    try:
        result = await self._ts_orig_fit_update_weights()
    except BaseException as e:
        # THE crash path of the aborted small run (NcclError out of the NCCL
        # checkpoint-engine sync): never die holding the group lock.
        await _crash_release(self, "update_weights", e)
        raise
    if result is not None:
        await _yield_lock(self, point="update_weights")
    else:
        _warn_once(
            "trigger_gt_1",
            "_fit_update_weights returned None (trigger_parameter_sync_step > 1?): "
            "no yield this step — the trainer lock stays held across the next "
            "queue wait. Time-slicing is designed for trigger_parameter_sync_step=1.",
        )
    return result


def _patched_fit_save_checkpoint(self, force=False):
    """No-op-with-warning when save_freq was left > 0: saving RPCs the FSDP
    worker, which may be checkpointed off here. With save_freq<=0 (-1 in our
    runs) delegate — the original is a no-op then anyway."""
    save_freq = None
    try:
        save_freq = self.config.trainer.save_freq
    except Exception:  # noqa: BLE001
        save_freq = None
    if save_freq is not None and save_freq > 0:
        _warn_once(
            "save_checkpoint",
            f"skipping _fit_save_checkpoint (save_freq={save_freq}, force={force}): "
            "checkpoint saving is disabled under fully-async time-slicing "
            "(the trainer worker may be checkpointed off). Set trainer.save_freq=-1.",
        )
        return None
    return self._ts_orig_fit_save_checkpoint(force=force)


# ======================================================================
# Patching + lazy import hook
# ======================================================================

def _patch_class(cls: type) -> bool:
    """Install the wrappers on the (plain) trainer class. Idempotent."""
    if getattr(cls, _PATCH_MARKER, False):
        return False
    cls._ts_orig_init_workers = cls.init_workers
    cls._ts_orig_get_samples_from_queue = cls._get_samples_from_queue
    cls._ts_orig_fit_update_weights = cls._fit_update_weights
    cls._ts_orig_fit_save_checkpoint = cls._fit_save_checkpoint
    cls.init_workers = _patched_init_workers
    cls._get_samples_from_queue = _patched_get_samples_from_queue
    cls._fit_update_weights = _patched_fit_update_weights
    cls._fit_save_checkpoint = _patched_fit_save_checkpoint
    setattr(cls, _PATCH_MARKER, True)
    _log(f"fully_async: patched {cls.__module__}.{cls.__qualname__} for time-slicing")
    return True


def _resolve_trainer_class(obj: Any) -> Optional[type]:
    """FullyAsyncTrainer is @ray.remote-decorated: the module attribute is a
    Ray ActorClass whose real class is __ray_metadata__.modified_class (a
    dynamically created subclass, cloudpickled by value into the actor
    process — patches on it, and the _ts_orig_* attrs, travel with it).
    Duck-typed so a plain class (tests, future verl) also works."""
    meta = getattr(obj, "__ray_metadata__", None)
    if meta is not None:
        cls = getattr(meta, "modified_class", None)
        if isinstance(cls, type):
            return cls
    if isinstance(obj, type):
        return obj
    return None


def _patch_module(module) -> bool:
    try:
        obj = getattr(module, _TARGET_CLASS, None)
        if obj is None:
            _log(f"fully_async: {_TARGET_MODULE} has no {_TARGET_CLASS}; not patching")
            return False
        cls = _resolve_trainer_class(obj)
        if cls is None:
            _log(f"fully_async: cannot resolve trainer class from {type(obj)!r}; not patching")
            return False
        return _patch_class(cls)
    except Exception as e:  # noqa: BLE001 - never break the verl import
        _log(f"fully_async: patch failed: {e!r}")
        return False


class _PatchingLoader(importlib.abc.Loader):
    """Wraps the real loader; runs the patch right after module exec."""

    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        _patch_module(module)  # never raises

    def __getattr__(self, name):  # delegate everything else (get_filename, ...)
        return getattr(self._inner, name)


class _FullyAsyncPatchFinder(importlib.abc.MetaPathFinder):
    """sys.meta_path finder: intercepts only the target verl module.

    Uses PathFinder directly (not importlib.util.find_spec) so it cannot
    recurse into itself, and returns the real spec with a wrapped loader."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchingLoader(spec.loader)
        return spec


def install() -> bool:
    """Entry point hook, called from timeslice_verl/__init__ at plugin load.

    MUST NOT import anything verl-side: the `verl.plugins` entry points fire
    in the middle of `import verl`, and verl swallows loader exceptions.
    Returns True iff fully-async time-slicing is active in this process."""
    try:
        if not enabled():
            return False
        # Post-import case (e.g. VERL_USE_EXTERNAL_MODULES ordering): patch now.
        if _TARGET_MODULE in sys.modules:
            _patch_module(sys.modules[_TARGET_MODULE])
        if not any(isinstance(f, _FullyAsyncPatchFinder) for f in sys.meta_path):
            sys.meta_path.insert(0, _FullyAsyncPatchFinder())
        return True
    except Exception as e:  # noqa: BLE001 - a broken hook must not kill the plugin load
        _log(f"fully_async: install failed: {e!r}")
        return False


def _reset_for_tests() -> None:
    """Test-only: drop the finder and all process-local state."""
    global _locks, _acquired_monotonic, _client_factory
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _FullyAsyncPatchFinder)]
    with _state_lock:
        _locks = None
    _acquired_monotonic = None
    _client_factory = None
    _warned.clear()
