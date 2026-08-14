"""Unit tests for timeslice_verl.fully_async (pure python: no GPU, no verl,
no grpc, no ray).

Run:  python3 -m pytest tests/test_fully_async.py -v

What is covered:
  * inert-without-env: install() does nothing unless TIMESLICE_FULLY_ASYNC=1
  * lazy import hook: patching happens when the (faked, on-disk) verl module
    `verl.experimental.fully_async_policy.fully_async_trainer` is imported —
    and also post-import if the module was already loaded
  * Ray ActorClass unwrap: patch lands on __ray_metadata__.modified_class
  * patch order invariants on a fake FullyAsyncTrainer with the real method
    signatures:
      - init_workers: acquire BEFORE delegation, yield AFTER
      - _get_samples_from_queue: delegate first, acquire only after a real
        batch is returned (no acquire on the (None, None) shutdown path)
      - _fit_update_weights: yield only when it returns non-None; the
        ensure-held before delegation acquires when the lock is not held
        (initial-param-sync path) and is a no-op when it is (steady state)
      - _fit_save_checkpoint: skipped with a warning when save_freq > 0,
        delegated when save_freq <= 0
  * rl_metrics.jsonl lines are well-formed and carry the required fields;
    metrics IO failure never breaks a training call

The timeslice_verl package __init__ imports verl (absent here), so the tests
load the package namespace by path without executing __init__.
"""

import asyncio
import json
import os
import sys
import textwrap
import types
from types import SimpleNamespace

import pytest

# ----------------------------------------------------------------------
# Load timeslice_verl.{locks,fully_async} without running the package
# __init__ (which imports verl): pre-seed a namespace-package stand-in.
# ----------------------------------------------------------------------
_PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "timeslice_verl"))
if "timeslice_verl" not in sys.modules:
    _pkg = types.ModuleType("timeslice_verl")
    _pkg.__path__ = [_PKG_DIR]
    sys.modules["timeslice_verl"] = _pkg

import timeslice_verl.fully_async as fa  # noqa: E402
import timeslice_verl.locks as locks_mod  # noqa: E402

JOB = "job-a"
ADDR = "orch:50051"
GROUP = "trainers"
MQ_META_KEY = "fully_async/monitor/queue/mq_queue_size"


class FakeAcquireResult:
    def __init__(self, waited_ms=7, context_restored=True):
        self.success = True
        self.waited_ms = waited_ms
        self.context_restored = context_restored


class FakeYieldResult:
    success = True
    pending_waiters = 1
    snapshot_deferred = False


class FakeClient:
    """Records acquire/release into a shared event log (same list the fake
    trainer methods append to, so cross-ordering is assertable)."""

    def __init__(self, events, job_id, group_id):
        self.events = events
        self.job_id = job_id
        self.group_id = group_id
        self.closed = False

    def acquire(self, group_id):
        assert group_id == self.group_id
        self.events.append(("acquire", group_id))
        return FakeAcquireResult()

    def release(self, group_id):
        assert group_id == self.group_id
        self.events.append(("release", group_id))
        return FakeYieldResult()

    def close(self):
        self.closed = True


def make_batch(mq_len=5):
    return SimpleNamespace(meta_info={MQ_META_KEY: mq_len})


class FakeFullyAsyncTrainerBase:
    """Mimics FullyAsyncTrainer's patched-method signatures at verl@983cb0f2:
    async init_workers(self); async _get_samples_from_queue(self) ->
    tuple[None,None]|tuple[int,Any]; async _fit_update_weights(self) ->
    dict|None; _fit_save_checkpoint(self, force=False)."""

    def __init__(self, events, save_freq=-1):
        self.events = events
        self.current_param_version = 3
        self.metrics = {}
        self.config = SimpleNamespace(trainer=SimpleNamespace(save_freq=save_freq))
        self.queue_result = (0, make_batch())
        self.update_result = {"timing": 1.0}

    async def init_workers(self):
        self.events.append("orig_init_workers")

    async def _get_samples_from_queue(self):
        self.events.append("orig_get_samples")
        return self.queue_result

    async def _fit_update_weights(self):
        self.events.append("orig_update_weights")
        return self.update_result

    def _fit_save_checkpoint(self, force=False):
        self.events.append(("orig_save_checkpoint", force))
        return "saved"


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_state():
    fa._reset_for_tests()
    yield
    fa._reset_for_tests()
    for name in [m for m in sys.modules if m == "verl" or m.startswith("verl.")]:
        del sys.modules[name]


@pytest.fixture
def events():
    return []


@pytest.fixture
def ts_env(monkeypatch, tmp_path, events):
    """Full fully-async env + injected grpc-free client factory."""
    metrics_path = tmp_path / "results" / "rl_metrics.jsonl"
    monkeypatch.setenv("TIMESLICE_FULLY_ASYNC", "1")
    monkeypatch.setenv("TIMESLICE_JOB_ID", JOB)
    monkeypatch.setenv("TIMESLICE_ORCH_ADDR", ADDR)
    monkeypatch.setenv("TIMESLICE_GROUP", GROUP)
    monkeypatch.setenv("TIMESLICE_METRICS_PATH", str(metrics_path))
    fa._client_factory = lambda target, job_id, group_id: FakeClient(events, job_id, group_id)
    return SimpleNamespace(metrics_path=metrics_path, events=events)


def make_patched(events, **kw):
    """Fresh subclass per test so class-level patching never leaks."""
    cls = type("FakeTrainer", (FakeFullyAsyncTrainerBase,), {})
    assert fa._patch_class(cls) is True
    assert fa._patch_class(cls) is False  # idempotent
    return cls(events, **kw)


def read_metrics(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ----------------------------------------------------------------------
# fake on-disk verl tree (for the import-hook tests)
# ----------------------------------------------------------------------

FAKE_TRAINER_PY = textwrap.dedent(
    """
    CALLS = []

    class FullyAsyncTrainer:
        def __init__(self):
            self.current_param_version = 0
            self.metrics = {}

        async def init_workers(self):
            CALLS.append("orig_init_workers")

        async def _get_samples_from_queue(self):
            CALLS.append("orig_get_samples")
            return (None, None)

        async def _fit_update_weights(self):
            CALLS.append("orig_update_weights")
            return None

        def _fit_save_checkpoint(self, force=False):
            CALLS.append(("orig_save_checkpoint", force))
    """
)


@pytest.fixture
def fake_verl_tree(tmp_path, monkeypatch):
    root = tmp_path / "fake_site"
    pkg = root / "verl" / "experimental" / "fully_async_policy"
    pkg.mkdir(parents=True)
    (root / "verl" / "__init__.py").write_text("")
    (root / "verl" / "experimental" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "fully_async_trainer.py").write_text(FAKE_TRAINER_PY)
    monkeypatch.syspath_prepend(str(root))
    return root


# ======================================================================
# attach / inertness
# ======================================================================

class TestInstall:
    def test_inert_without_env(self, monkeypatch, fake_verl_tree):
        monkeypatch.delenv("TIMESLICE_FULLY_ASYNC", raising=False)
        meta_before = list(sys.meta_path)
        assert fa.install() is False
        assert sys.meta_path == meta_before
        import verl.experimental.fully_async_policy.fully_async_trainer as m

        assert not getattr(m.FullyAsyncTrainer, fa._PATCH_MARKER, False)

    def test_lazy_hook_patches_on_first_import(self, ts_env, fake_verl_tree):
        assert "verl" not in sys.modules
        assert fa.install() is True
        # nothing verl-side imported by install itself
        assert "verl" not in sys.modules
        import verl.experimental.fully_async_policy.fully_async_trainer as m

        cls = m.FullyAsyncTrainer
        assert getattr(cls, fa._PATCH_MARKER, False)
        assert cls.init_workers is fa._patched_init_workers
        assert cls._get_samples_from_queue is fa._patched_get_samples_from_queue
        assert cls._fit_update_weights is fa._patched_fit_update_weights
        assert cls._fit_save_checkpoint is fa._patched_fit_save_checkpoint
        # originals stashed on the class (they must travel with Ray's
        # by-value pickle of the class)
        assert cls._ts_orig_init_workers is not None

    def test_install_idempotent(self, ts_env, fake_verl_tree):
        assert fa.install() is True
        assert fa.install() is True
        finders = [f for f in sys.meta_path if isinstance(f, fa._FullyAsyncPatchFinder)]
        assert len(finders) == 1

    def test_patch_post_import(self, ts_env, fake_verl_tree):
        import verl.experimental.fully_async_policy.fully_async_trainer as m

        assert not getattr(m.FullyAsyncTrainer, fa._PATCH_MARKER, False)
        assert fa.install() is True  # module already in sys.modules -> patch now
        assert getattr(m.FullyAsyncTrainer, fa._PATCH_MARKER, False)

    def test_ray_actorclass_unwrap(self):
        class Plain(FakeFullyAsyncTrainerBase):
            pass

        modified = type("Modified", (Plain,), {})  # ray's derived actor class
        actor_cls = SimpleNamespace(__ray_metadata__=SimpleNamespace(modified_class=modified))
        module = SimpleNamespace(FullyAsyncTrainer=actor_cls)
        assert fa._patch_module(module) is True
        assert getattr(modified, fa._PATCH_MARKER, False)
        assert modified.init_workers is fa._patched_init_workers
        # the plain base class is untouched (patch is on the pickled subclass)
        assert Plain.init_workers is not fa._patched_init_workers

    def test_patch_module_never_raises(self):
        assert fa._patch_module(SimpleNamespace()) is False
        assert fa._patch_module(SimpleNamespace(FullyAsyncTrainer=42)) is False


# ======================================================================
# lock transition order on the fake trainer
# ======================================================================

class TestPatchOrder:
    def test_init_workers_acquire_then_delegate_then_yield(self, ts_env, events):
        t = make_patched(events)
        asyncio.run(t.init_workers())
        assert events == [("acquire", GROUP), "orig_init_workers", ("release", GROUP)]

    def test_init_workers_releases_on_failure(self, ts_env, events):
        t = make_patched(events)

        async def boom():
            events.append("orig_init_workers")
            raise RuntimeError("init failed")

        t._ts_orig_init_workers = boom
        with pytest.raises(RuntimeError):
            asyncio.run(t.init_workers())
        assert events == [("acquire", GROUP), "orig_init_workers", ("release", GROUP)]

    def test_queue_get_delegates_first_then_acquires(self, ts_env, events):
        t = make_patched(events)
        result = asyncio.run(t._get_samples_from_queue())
        assert result == t.queue_result  # return value preserved exactly
        assert events == ["orig_get_samples", ("acquire", GROUP)]

    def test_queue_get_shutdown_no_acquire(self, ts_env, events):
        t = make_patched(events)
        t.queue_result = (None, None)
        result = asyncio.run(t._get_samples_from_queue())
        assert result == (None, None)
        assert events == ["orig_get_samples"]

    def test_update_weights_yields_only_on_real_sync(self, ts_env, events):
        t = make_patched(events)
        # steady state: lock already held from the queue-get acquire
        asyncio.run(t._get_samples_from_queue())
        del events[:]
        result = asyncio.run(t._fit_update_weights())
        assert result == t.update_result
        # no re-acquire (ensure-held is a no-op), yield after non-None return
        assert events == ["orig_update_weights", ("release", GROUP)]

    def test_update_weights_none_keeps_lock(self, ts_env, events):
        t = make_patched(events)
        asyncio.run(t._get_samples_from_queue())
        del events[:]
        t.update_result = None
        assert asyncio.run(t._fit_update_weights()) is None
        assert events == ["orig_update_weights"]  # no release

    def test_initial_param_sync_acquires_when_not_held(self, ts_env, events):
        # fully_async_main calls _fit_update_weights between init_workers
        # (after which we yielded) and fit(): ensure-held must acquire.
        t = make_patched(events)
        asyncio.run(t.init_workers())
        del events[:]
        result = asyncio.run(t._fit_update_weights())
        assert result == t.update_result
        assert events == [("acquire", GROUP), "orig_update_weights", ("release", GROUP)]

    def test_full_step_sequence_alternates(self, ts_env, events):
        t = make_patched(events)
        asyncio.run(t.init_workers())
        asyncio.run(t._fit_update_weights())  # initial param sync
        for _ in range(2):
            asyncio.run(t._get_samples_from_queue())
            asyncio.run(t._fit_update_weights())
        lock_events = [e for e in events if isinstance(e, tuple)]
        # strict alternation, starting with acquire, ending released
        held = False
        for op, group in lock_events:
            assert group == GROUP
            if op == "acquire":
                assert not held, f"double acquire: {lock_events}"
                held = True
            else:
                assert held, f"release while not held: {lock_events}"
                held = False
        assert not held, f"lock leaked at end: {lock_events}"
        assert len(lock_events) == 2 * 4  # init, initial sync, 2 steps

    def test_save_checkpoint_skipped_when_save_freq_positive(self, ts_env, events):
        t = make_patched(events, save_freq=5)
        assert t._fit_save_checkpoint(force=True) is None
        assert events == []  # original never called, no lock traffic

    def test_save_checkpoint_delegates_when_disabled(self, ts_env, events):
        t = make_patched(events, save_freq=-1)
        assert t._fit_save_checkpoint(force=True) == "saved"
        assert events == [("orig_save_checkpoint", True)]


# ======================================================================
# rl_metrics.jsonl
# ======================================================================

class TestMetrics:
    def test_lines_well_formed(self, ts_env, events):
        t = make_patched(events)
        asyncio.run(t.init_workers())
        asyncio.run(t._get_samples_from_queue())
        asyncio.run(t._fit_update_weights())
        lines = read_metrics(ts_env.metrics_path)
        assert [ln["type"] for ln in lines] == ["acquire", "yield", "acquire", "yield"]
        for ln in lines:
            assert ln["workload_id"] == JOB
            assert ln["pool"] == GROUP
            assert isinstance(ln["ts"], float)
            assert "queue_len" in ln and "step" in ln
            if ln["type"] == "acquire":
                assert ln["wait_ms"] == 7  # from FakeAcquireResult.waited_ms
                assert "restore_ms" in ln  # null: not in current AcquireResult
                assert ln["context_restored"] is True
            else:
                assert isinstance(ln["held_ms"], int) and ln["held_ms"] >= 0

    def test_queue_len_and_step_sources(self, ts_env, events):
        t = make_patched(events)
        t.queue_result = (0, make_batch(mq_len=11))
        t.current_param_version = 42
        asyncio.run(t._get_samples_from_queue())
        (acq,) = read_metrics(ts_env.metrics_path)
        assert acq["queue_len"] == 11  # from batch.meta_info mq snapshot
        assert acq["step"] == 42  # current_param_version

    def test_no_duplicate_acquire_line_when_already_held(self, ts_env, events):
        t = make_patched(events)
        asyncio.run(t._get_samples_from_queue())  # real acquire
        asyncio.run(t._fit_update_weights())  # ensure-held no-op + yield
        lines = read_metrics(ts_env.metrics_path)
        assert [ln["type"] for ln in lines] == ["acquire", "yield"]

    def test_metrics_io_failure_never_breaks_training(self, ts_env, events, monkeypatch, tmp_path):
        # a directory as the metrics path -> every open("a") fails
        monkeypatch.setenv("TIMESLICE_METRICS_PATH", str(tmp_path))
        t = make_patched(events)
        asyncio.run(t.init_workers())
        result = asyncio.run(t._get_samples_from_queue())
        assert result == t.queue_result
        # lock protocol unaffected
        assert events == [
            ("acquire", GROUP),
            "orig_init_workers",
            ("release", GROUP),
            "orig_get_samples",
            ("acquire", GROUP),
        ]

    def test_no_lock_env_means_noop_and_no_metrics(self, ts_env, events, monkeypatch):
        # TIMESLICE_FULLY_ASYNC=1 but no orchestrator env: patches installed,
        # lock layer is a PhaseLocks-style no-op, nothing emitted.
        monkeypatch.delenv("TIMESLICE_JOB_ID")
        t = make_patched(events)
        asyncio.run(t.init_workers())
        asyncio.run(t._get_samples_from_queue())
        asyncio.run(t._fit_update_weights())
        assert [e for e in events if isinstance(e, tuple)] == []
        assert read_metrics(ts_env.metrics_path) == []


# ======================================================================
# end-to-end through the import hook
# ======================================================================

class TestEndToEnd:
    def test_imported_fake_trainer_takes_turns(self, ts_env, events, fake_verl_tree):
        assert fa.install() is True
        import verl.experimental.fully_async_policy.fully_async_trainer as m

        t = m.FullyAsyncTrainer()
        asyncio.run(t.init_workers())
        assert m.CALLS == ["orig_init_workers"]
        assert events == [("acquire", GROUP), ("release", GROUP)]
        # shutdown path from the fake: (None, None) -> no acquire
        assert asyncio.run(t._get_samples_from_queue()) == (None, None)
        assert events == [("acquire", GROUP), ("release", GROUP)]


# ======================================================================
# client-rename compat import (small-run finding 2026-07-26: upstream renamed
# OrchestratorClient -> TimeSliceOrchestratorClient, API identical)
# ======================================================================

def _fake_timeslice_module(class_name):
    """A fake `timeslice` module exposing ONLY `class_name` as the client."""
    mod = types.ModuleType("timeslice")

    class Client:
        def __init__(self, target, job_id, group_id):
            self.target = target
            self.job_id = job_id
            self.group_id = group_id

        def acquire(self, group_id):
            return FakeAcquireResult()

        def release(self, group_id):
            return FakeYieldResult()

        def close(self):
            pass

    Client.__name__ = class_name
    setattr(mod, class_name, Client)
    return mod, Client


class TestClientRenameCompat:
    def test_phaselocks_new_client_name_only(self, monkeypatch):
        mod, cls = _fake_timeslice_module("TimeSliceOrchestratorClient")
        assert not hasattr(mod, "OrchestratorClient")
        monkeypatch.setitem(sys.modules, "timeslice", mod)
        pl = locks_mod.PhaseLocks(job_id=JOB, orch_addr=ADDR, group=GROUP)
        assert pl.enabled and isinstance(pl._client, cls)
        assert (pl._client.target, pl._client.job_id, pl._client.group_id) == (ADDR, JOB, GROUP)

    def test_phaselocks_old_client_name_still_works(self, monkeypatch):
        mod, cls = _fake_timeslice_module("OrchestratorClient")
        monkeypatch.setitem(sys.modules, "timeslice", mod)
        pl = locks_mod.PhaseLocks(job_id=JOB, orch_addr=ADDR, group=GROUP)
        assert pl.enabled and isinstance(pl._client, cls)

    def test_metered_phaselocks_new_client_name_only(self, monkeypatch):
        # client_factory=None -> _MeteredPhaseLocks does its own lazy import
        mod, cls = _fake_timeslice_module("TimeSliceOrchestratorClient")
        monkeypatch.setitem(sys.modules, "timeslice", mod)
        ml = fa._MeteredPhaseLocks(job_id=JOB, orch_addr=ADDR, group=GROUP, client_factory=None)
        assert ml.enabled and isinstance(ml._client, cls)
        assert ml.acquire_info()["acquired"] is True


# ======================================================================
# experimental TIMESLICE_EMPTY_CACHE_BEFORE_YIELD (inert by default)
# ======================================================================

def _fake_torch(events, raise_on_call=False):
    mod = types.ModuleType("torch")

    def empty_cache():
        if raise_on_call:
            raise RuntimeError("no CUDA context")
        events.append("empty_cache")

    mod.cuda = SimpleNamespace(empty_cache=empty_cache)
    return mod


class TestEmptyCacheBeforeYield:
    def test_inert_by_default(self, ts_env, events, monkeypatch):
        monkeypatch.delenv(fa.ENV_EMPTY_CACHE, raising=False)
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(events))
        t = make_patched(events)
        asyncio.run(t.init_workers())
        asyncio.run(t._get_samples_from_queue())
        asyncio.run(t._fit_update_weights())
        assert "empty_cache" not in events

    def test_enabled_calls_before_each_yields_drop_all(self, ts_env, events, monkeypatch):
        monkeypatch.setenv(fa.ENV_EMPTY_CACHE, "1")
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(events))
        t = make_patched(events)
        asyncio.run(t.init_workers())
        assert events == [("acquire", GROUP), "orig_init_workers", "empty_cache", ("release", GROUP)]
        del events[:]
        asyncio.run(t._get_samples_from_queue())
        asyncio.run(t._fit_update_weights())
        # steady state: empty_cache immediately before the yield's release only
        assert events == [
            "orig_get_samples",
            ("acquire", GROUP),
            "orig_update_weights",
            "empty_cache",
            ("release", GROUP),
        ]

    def test_enabled_without_torch_is_safe(self, ts_env, events, monkeypatch):
        monkeypatch.setenv(fa.ENV_EMPTY_CACHE, "1")
        monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
        t = make_patched(events)
        asyncio.run(t.init_workers())
        # lock protocol unaffected, no empty_cache event, no exception
        assert events == [("acquire", GROUP), "orig_init_workers", ("release", GROUP)]

    def test_empty_cache_failure_never_breaks_yield(self, ts_env, events, monkeypatch):
        monkeypatch.setenv(fa.ENV_EMPTY_CACHE, "1")
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(events, raise_on_call=True))
        t = make_patched(events)
        asyncio.run(t.init_workers())
        assert events == [("acquire", GROUP), "orig_init_workers", ("release", GROUP)]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
