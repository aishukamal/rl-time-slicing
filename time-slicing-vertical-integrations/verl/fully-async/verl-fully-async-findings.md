# verl fully_async_policy analysis for time-slicing integration

Commit: `983cb0f24443f87b3d161fad318445130a620b07` ("[vllm] fix: guard legacy FusedMoE loader patch (#7147)", Sat Jul 25 2026) — exact requested commit, checked out clean.
Clone: `/Users/aishuk/workspaces/verl-research` (durable, not /tmp).
All paths below relative to the clone root unless absolute.

Status: COMPLETE (all Q1-Q7 answered).

---

## Q1. Process/actor map and trainer step sequence

### Actors created by `fully_async_main.py`

`verl/experimental/fully_async_policy/fully_async_main.py`:

- L35-36: `@ray.remote(num_cpus=1) class FullyAsyncTaskRunner` — driver actor, launched by `run_ppo(config, task_runner_class=FullyAsyncTaskRunner)` (L238; `run_ppo` is `verl/trainer/main_ppo.py:34`, launches it at L93-94 `runner = task_runner_class.remote(); ray.get(runner.run.remote(config))`).
- L146-153: `trainer = FullyAsyncTrainer.remote(...)` — a `@ray.remote(num_cpus=10)` actor (`fully_async_trainer.py:53`). This is a **CPU-only driver actor** (no `num_gpus`); it drives GPU work via RPC to a worker group.
- L119-124: `rollouter = FullyAsyncRollouter.remote(...)` — same pattern for the rollout side (`fully_async_rollouter.py`, `@ray.remote(num_cpus=10)` — verify line).
- L97: `message_queue = MessageQueue.remote(config, max_queue_size)` — `@ray.remote(num_cpus=2, max_concurrency=20)` (`message_queue.py:26`). Pure-CPU deque holder.
- Inside `trainer.init_workers()` (`fully_async_trainer.py:487-496` → `SeparateRayPPOTrainer._init_worker_groups`, `verl/experimental/separation/ray_trainer.py:204-237`): the **FSDP training worker group** — separate Ray actor processes, one per trainer GPU (1 in our config). These are the processes that own trainer CUDA state.
- Inside `rollouter.init_workers()`: vLLM rollout replica server + its CheckpointEngineWorker colocated workers (one per rollout GPU).

Startup order (`_initialize_components`, fully_async_main.py:51-115): trainer created first (L78), then rollouter (L84), then `trainer.set_rollouter` (L87), MessageQueue (L97), `load_checkpoint` both (L106-107), **initial param sync** `trainer._fit_update_weights` (L110), optional val (L112-113). Then `_run_training_loop` (L186-219) starts both loops concurrently:

```python
rollouter_future = self.components["rollouter"].fit.remote()   # L190
trainer_future = self.components["trainer"].fit.remote()       # L191
```

### Trainer loop and step sequence

Class: `FullyAsyncTrainer` (`fully_async_trainer.py:54`), subclass of `SeparateRayPPOTrainer` (`verl/experimental/separation/ray_trainer.py:52`), subclass of `RayPPOTrainer`.

`fit()` (fully_async_trainer.py:498-534): `while True: await self.fit_step()` (L521-526), stops on `TrainingStopException` (raised when queue returns None).

`fit_step()` (fully_async_trainer.py:536-604) — exact sequence:

```python
with marked_timer("step", self.timing_raw):                     # L559
    ... dynamic-schedule deactivation branch (disabled for us)  # L560-582
    batch = await self._fit_generate(None)                      # L585  <- COLLECT SAMPLES (waits on MessageQueue)
    batch = self._fit_compute_reward(batch)                     # L586
    batch = self._fit_compute_log_prob(batch)                   # L587  <- first trainer-GPU work of the step
    batch = self._fit_compute_ref_log_prob(batch)               # L588
    batch = self._fit_compute_critic(batch)                     # L589  (no-op, use_critic False for GRPO)
    batch = self._fit_compute_advantage(batch)                  # L590  (driver-side CPU)
    batch = self._fit_update_critic(batch)                      # L591  (no-op)
    batch = self._fit_update_actor(batch)                       # L592  <- UPDATE_ACTOR (GPU, via actor_wg RPC)
    self._fit_update_local_step()                                # L593  (bumps current_param_version)
    rollout_reset_timing_raw = await self._fit_update_weights() # L594  <- PARAM SYNC (NCCL broadcast)
    self._fit_dump_data(batch)                                   # L595
    self._record_train_resource_utilization(...)                 # L596
await self._fit_validate()                                       # L598
self._fit_save_checkpoint()                                      # L599  <- GPU RPC if save triggered!
self._fit_stop_profile(...)                                      # L600
self._fit_collect_metrics(batch)                                 # L601  (driver CPU)
if rollout_reset_timing_raw is not None:
    self._fit_log_aggregated_training_metrics(...)                # L602-603 (console log)
self._fit_postprocess_step()                                      # L604  <- STEP ENDS HERE
```

A "step" ends at `_fit_postprocess_step()` (fully_async_trainer.py:900-915): `global_steps += 1` (L901), snapshots `dynamic_resource/mq_size` (L908), feeds `metrics_aggregator` (L910-912), updates progress bar. Then the loop re-enters `fit_step()` and blocks in `_fit_generate → _get_samples_from_queue`.

With `trigger_parameter_sync_step=1`: `_fit_update_local_step` (L676-688) increments `current_param_version` every step and keeps `local_trigger_step=1`, so `_fit_update_weights` does a real sync **every step** (it early-returns None at L700-701 only when `local_trigger_step != 1`).

### Update flow inside `_fit_update_actor`
`SeparateRayPPOTrainer._fit_update_actor` (separation/ray_trainer.py:635-646): `with marked_timer("update_actor", timing_raw): actor_output = self._update_actor(batch)` — RPC into the FSDP worker group (GPU work happens in worker process, not the FullyAsyncTrainer actor).

---

## Q2. The natural YIELD point (update_actor + param sync complete)

`FullyAsyncTrainer._fit_update_weights` (`fully_async_trainer.py:690-788`):

- L700-701: `if self.local_trigger_step != 1: return None` (never taken with trigger=1).
- L729-736: the actual sync:
```python
with marked_timer("timing_s/param_sync", self.timing_raw):
    if not self.only_hybrid:
        await self.checkpoint_manager.update_weights(
            global_steps=self.current_param_version,
        )
```
- L756: `timing_raw = await asyncio.wrap_future(self.rollouter.reset_staleness.remote().future())` — tells rollouter the new param version is live.
- L758-762: prints `"[FullyAsyncTrainer] _fit_update_weights, timing_s/param_sync: {…:.4f} seconds self.current_param_version: {N}"`.
- L783-786: `self.logger.log(data=timing_raw, step=self.current_param_version)`.
- L788: `return timing_raw`.

**There is no hook/callback mechanism at this point.** No callback registry, no event emit — nothing pluggable in `fit_step` or `_fit_update_weights`. The clean external wrap is: monkeypatch `FullyAsyncTrainer._fit_update_weights` — after `ret = await orig(self)` and `ret is not None`, update_actor AND param sync AND rollouter reset_staleness are all complete → release the GPU lock there.

Caveats after the yield point but still inside the step (all driver-CPU except one):
- `_fit_validate` (L812-830): with `test_freq>0` and version%test_freq==0 does rollouter-side validate (rollout GPU, not trainer GPU) unless `use_trainer_do_validate` (keep False).
- `_fit_save_checkpoint` (L873-898): **if `trainer.save_freq > 0` and version % save_freq == 0 it RPCs `actor_rollout_wg.save_checkpoint` → trainer-GPU worker must be resident.** Set `trainer.save_freq=-1` for the PoC, or make the wrapper re-acquire the lock around saves.
- `_fit_postprocess_step` L908 `get_queue_size_sync()` — Ray RPC, CPU only.

---

## Q3. Where the trainer waits for samples; CUDA audit of the wait path

Wait location: `FullyAsyncTrainer._get_samples_from_queue` (`fully_async_trainer.py:375-452`), called from `_fit_generate` (L643-652) under `marked_timer("gen", ...)`.

```python
while len(queue_samples) < self.required_samples:                   # L402
    sample, queue_len = await self.message_queue_client.get_sample() # L404
    ...
```

- `required_samples = ppo_mini_batch_size * async_training.require_batches` (L159).
- `MessageQueueClient.get_sample` (`message_queue.py:198-201`) = `await asyncio.wrap_future(self.queue_actor.get_sample.remote().future())` — pure Ray RPC to the MessageQueue actor, which blocks on an `asyncio.Condition` (`message_queue.py:92-103`) until a sample arrives.
- Returns `(sample_bytes, queue_len)`; `sample is None` signals shutdown → `TrainingStopException`.
- **Queue depth observable**: every `get_sample` returns `len(self.queue)` after pop (`message_queue.py:103`), printed by trainer at L414-418 (`"sample collected k/N. mq_len: X"`); plus `get_queue_size()/get_queue_size_sync()` RPCs (`message_queue.py:105-108`, client 203-206, 236-242).

After enough samples (still inside `_get_samples_from_queue`):
- L439: `queue_samples = [ray.cloudpickle.loads(x) for x in queue_samples]` — deserializes `RolloutSample` (detach_utils.py:27-39). Tensors inside came from the rollouter's agent-loop output shipped through the MessageQueue actor (a CPU-only actor), so they are CPU tensors; `loads` does not touch CUDA. (Labeled: high-confidence inference — the MessageQueue actor has no GPU, so any CUDA tensor would have failed deserialization there already; rollouter pickles them before `put_sample`.)
- L441-444: `assemble_batch_from_rollout_samples` (detach_utils.py:84-176) — `DataProto.concat`, `compute_response_mask`, optional `self._balance_batch`, `torch.sum(attention_mask)` — all CPU tensor ops in the driver actor.

**CUDA calls in the wait path: none.** The `FullyAsyncTrainer` actor is scheduled with `num_cpus=10` and no GPUs (`fully_async_trainer.py:53`); it never runs CUDA kernels itself — all GPU work is RPC'd to the FSDP worker group. While the driver sits in `await get_sample()`, the FSDP worker process (the cuda-checkpoint target) is completely idle, blocked in Ray's actor event loop waiting for the next RPC. First trainer-GPU RPC of the next step is `save_model_to_cpu`/`_compute_old_log_prob` inside `_fit_compute_log_prob` (fully_async_trainer.py:665-673 → separation/ray_trainer.py:520-521); `_fit_compute_reward` (separation/ray_trainer.py:488-500) is driver-CPU (`extract_reward`) when no reward model worker is used.

**Implication for the wrapper**: the process being cuda-checkpointed is the FSDP *worker* (separate pid from the FullyAsyncTrainer driver). Safe window = from after `_fit_update_weights` returns (and no pending save_checkpoint) until the wrapper re-acquires the lock before `_fit_compute_log_prob` — simplest: re-acquire at the end of `_get_samples_from_queue` (batch assembled, right before return at L452).

---

## Q4. NCCL param-sync communicator lifecycle (backend=nccl, trigger=1)

Sync path: `FullyAsyncTrainer._fit_update_weights` → `CheckpointEngineManager.update_weights` (`verl/checkpoint_engine/base.py:486-538`). **Every invocation** does:

```python
# base.py
await self.abort_replicas()                       # L499
rollout = RayWorkerGroup(worker_handles=workers, ...)  # L505 temp WG over replica CE workers
await self.release_kv_cache_replicas()            # L509
self.build_process_group(rollout)                 # L512  <- every sync
results = ray.get(actor_wg.update_weights(...) + rollout.update_weights(...))  # L515-518
ray.get(actor_wg.execute_checkpoint_engine(["finalize"]*...) + rollout....)    # L527-530
await self.resume_kv_cache_replicas()             # L533
await self.resume_generation_replicas()           # L536
```

`build_process_group` (base.py:403-428) calls `prepare()` on every worker then `init_process_group(...)`.

`NCCLCheckpointEngine` (`verl/checkpoint_engine/nccl_checkpoint_engine.py`):

- Constructor L116-127: `rebuild_group: bool = False` — **default False**.
- `init_process_group` L201-228 — the deciding code:
```python
if self.rebuild_group or not collective.is_group_initialized(self.group_name):
    collective.init_collective_group(world_size, rank, "nccl", self.group_name)   # L214-215
    self.rank = rank
    self.world_size = world_size
else:
    assert self.rank == rank, ...                                                  # L219-222
...
collective.barrier(self.group_name)                                                # L226
```
- `finalize` L147-158:
```python
def finalize(self):
    """Destroy the NCCL process group if rebuild_group is True."""
    if self.rebuild_group:
        if self.rank >= 0:
            collective.destroy_collective_group(self.group_name)                   # L150-151
        self.rank = None
        self.world_size = None
    self.send_buf = None
    self.recv_buf = None
    torch.cuda.empty_cache()                                                        # L158
```
- `prepare` L135-145 re-allocates the send/recv CUDA buckets each sync (freed in finalize regardless of rebuild_group).

**Answer: with the default `rebuild_group=False`, the `ray.util.collective` NCCL group ("default") is created on the FIRST sync and PERSISTS in the trainer FSDP worker process (rank 0) and every rollout CheckpointEngineWorker across steps.** Only the buckets are freed per sync; the NCCL communicator (and its CUDA/network state) stays live between syncs → cuda-checkpointing the trainer FSDP worker between syncs would freeze a process holding a live cross-process NCCL communicator (corruption risk on restore, same class of problem we solved with ncclCommSuspend).

**Mitigation without source changes**: the flag is plumbed from config — `CheckpointEngineConfig.engine_kwargs` (`verl/workers/config/rollout.py:125-141`, `engine_kwargs: dict = field(default_factory=dict)`) is passed as `engine_kwargs = ...checkpoint_engine.engine_kwargs.get(backend, {})` then `CheckpointEngineRegistry.new(backend, ..., **engine_kwargs)` on both sides:
- trainer FSDP worker: `verl/workers/engine_workers.py:672-682` (`is_master=(torch.distributed.get_rank()==0)`)
- rollout CE worker: `verl/checkpoint_engine/base.py:313-320`

So a hydra override `+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.rebuild_group=true` makes every sync do init_collective_group → broadcast → destroy_collective_group, leaving **no live sync communicator between syncs**. Cost: NCCL group setup per step (~1-3 s for 2 ranks). Residual caveat (labeled speculation): torch.distributed's own process group inside the 1-rank FSDP worker may still own NCCL/CUDA state; with world_size=1 FSDP typically has no cross-process NCCL communicator, which is what makes single-trainer-GPU cuda-checkpoint viable.

The default config used by our run: `verl/experimental/fully_async_policy/config/fully_async_ppo_trainer.yaml:82-83` sets `checkpoint_engine: backend: "nccl"`; `engine_kwargs` defaults to `{}` (`verl/trainer/config/rollout/rollout.yaml:295`), i.e. persistent group unless overridden.

---

## Q5. Plugin/attach mechanics — zero-source-change wrapper

**Both hooks exist at this commit**, in `verl/__init__.py` — executed at `import verl` in EVERY process that imports verl (TaskRunner, FullyAsyncTrainer actor, FullyAsyncRollouter actor, FSDP worker actors, CheckpointEngineWorkers, vLLM server actors):

```python
# verl/__init__.py:38-41
modules = os.getenv("VERL_USE_EXTERNAL_MODULES", "")
if modules:
    modules = modules.split(",")
    import_external_libs(modules)          # importlib.import_module each (verl/utils/import_utils.py:81-88)

# verl/__init__.py:49-65  — entry-point autodiscovery, DEFAULT ON
_plugins_policy = os.getenv("VERL_USE_EXTERNAL_PLUGINS", "auto").strip().lower()
if _plugins_policy != "none":
    _discovered = _entry_points(group="verl.plugins")
    ...
    for _ep in _discovered: _ep.load()     # exceptions swallowed to debug log (L62-65)
```

So: **install a pip package that declares `[project.entry-points."verl.plugins"] gpulock = "our_pkg.patch:install"` and it auto-loads in every verl process with no env var and no source change.** `fully_async_main` itself does `from verl.experimental...` (imports verl) at module import; every Ray actor re-imports verl in its own process, so the plugin runs inside the FullyAsyncTrainer actor process and inside the FSDP worker process alike.

Env-var alternative: `VERL_USE_EXTERNAL_MODULES=our_pkg.patch`. To ensure Ray actors see it on a pre-started cluster, inject via config `+ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=our_pkg.patch` (merged at `verl/trainer/main_ppo.py:62-75`). When `ray.init()` is local (our single-pod case), actors inherit the driver shell env anyway. Note `get_ppo_ray_runtime_env` (`verl/trainer/constants_ppo.py:93-121`) does not forward arbitrary env vars, only a fixed list — hence use runtime_env injection or rely on entry points.

**What to patch (executes inside the FullyAsyncTrainer driver actor):**
1. `verl.experimental.fully_async_policy.fully_async_trainer.FullyAsyncTrainer._fit_update_weights` (async): `ret = await orig(self); if ret is not None: release_gpu_lock()` — update_actor, NCCL sync, and rollouter reset_staleness are all complete here (Q2).
2. `FullyAsyncTrainer._get_samples_from_queue` (async): `epoch, batch = await orig(self); if batch is not None: acquire_gpu_lock()  # blocks until agent restores FSDP worker; return`. The wait inside orig is CPU-only (Q3).
3. Guard: also wrap `_fit_save_checkpoint` / `_save_checkpoint` to acquire the lock first if `trainer.save_freq > 0` (it RPCs the checkpointed FSDP worker; simplest is save_freq=-1).

Plugin-load-time caveat: entry points load mid-`verl/__init__` — do NOT import `verl.experimental...` at plugin import time (circular-ish, and `_ep.load()` swallows the failure silently). Instead register a lazy patcher: a `sys.meta_path` import hook (or `importlib` post-import hook) that patches `FullyAsyncTrainer` when `verl.experimental.fully_async_policy.fully_async_trainer` is first imported. A `.pth` file works too but is unnecessary given the entry-point group.

Identifying the cuda-checkpoint target pid: the FSDP worker actor is named `"{prefix}WorkerDict_{pg_idx}:{local_rank}"` (`verl/single_controller/ray/base.py:653-656`); the wrapper (running in the driver actor) can resolve it via `ray.util.list_named_actors` / `self.actor_wg.workers[0]` → `worker.get_node_id.remote()` / a custom `execute` RPC returning `os.getpid()` — `Worker` already exposes `get_cuda_visible_devices()` (used at base.py:392-394). Publish `{pid, cuda_visible_devices}` to the external agent at lock-release time.

**No pre-existing callback/hook** in fit_step/_fit_update_weights (Q2) — monkeypatch is the only attach mechanism; the entry-point loader is the sanctioned way to get the monkeypatch into the processes.

---

## Q6. Trainer per-step metrics (console logger)

Console backend: `Tracking` (`verl/utils/tracking.py:178,190-193`) → `LocalLogger.log` → `print(concat_dict_to_str(data, step), flush=True)` with format `step:N - key1:v1 - key2:v2 - ...` (`verl/utils/logger/aggregate_logger.py:26-51`). Trainer logs with `step=current_param_version`.

Three `logger.log` call sites per sync cycle in the trainer:
1. `_fit_update_weights` L783-786: logs the **rollouter's reset_staleness timing_raw** (`fully_async_rollouter.py:594-638`): keys `fully_async/rollouter/active_time`, `fully_async/rollouter/version_time`, `fully_async/rollouter/idle_ratio`, `fully_async/rollouter/step_generated_samples`, `dynamic_resource/rollout_resource_utilization`.
2. `_fit_log_aggregated_training_metrics` L790-810: logs `metrics_aggregator.get_aggregated_metrics(...)` — the union of per-micro-step `self.metrics` keys aggregated per rules in `MetricsAggregator` (`detach_utils.py:179-437`).
3. `_fit_validate` L830 / L856: val metrics.

Keys inside the aggregated dict (sources):
- `training/global_step`, `training/epoch` (fit_step L548).
- From batch meta_info via `_collect_metrics_from_samples` (fully_async_trainer.py:1033-1049): `fully_async/count/stale_trajectory_processed`, `fully_async/count/current_param_version`, plus every meta_info key starting with `fully_async` or `timing_s` (L1047-1049). Via `assemble_batch_from_rollout_samples` (detach_utils.py:108-172) that includes:
  - rollout status snapshot (prefixed `fully_async/`, from `FullyAsyncRollouter.get_statistics`, fully_async_rollouter.py:1166-1186): `fully_async/monitor/active_tasks_size`, `fully_async/monitor/queue/pending_queue_size`, **`fully_async/monitor/queue/mq_queue_size`**, `fully_async/count/total_generated_samples`, `fully_async/count/staleness_samples`, **`fully_async/count/dropped_stale_samples`**, `fully_async/static/max_required_samples`, `fully_async/static/required_samples`, `fully_async/static/staleness_threshold`, `fully_async/static/max_queue_size`, `fully_async/static/max_concurrent_samples`.
  - `fully_async/processing_time/{avg,max,min,tp50,tp95,tp99}` (detach_utils.py:134-149)
  - `fully_async/partial/{total_partial_num,partial_ratio,max_partial_span}` (detach_utils.py:155-159)
  - `fully_async/total_wait_time` (set at fully_async_trainer.py:446)
- Timing: `_fit_collect_metrics` (separation/ray_trainer.py:727-739) → `compute_timing_metrics` (`verl/trainer/ppo/metric_utils.py:645`): `timing_s/{name}` for every `timing_raw` key → **`timing_s/step`, `timing_s/gen` (queue-wait time), `timing_s/reward`, `timing_s/old_log_prob`, `timing_s/adv`, `timing_s/update_actor`**, plus `timing_per_token_ms/{gen,update_actor,...}`; param sync appears as **`timing_s/timing_s/param_sync`** (the marked_timer key at fully_async_trainer.py:729 is already `"timing_s/param_sync"`, then compute_timing_metrics re-prefixes). Aggregation rule: any `timing_s/` key is summed over the cycle's micro-steps (detach_utils.py:254-256).
- `perf/total_num_tokens`, `perf/time_per_step`, `perf/throughput` (metric_utils.py:680-689, throughput recomputed cycle-wide at detach_utils.py:350-354).
- `dynamic_resource/mq_size` — end-of-step MessageQueue depth snapshot (fully_async_trainer.py:908, "last" rule detach_utils.py:221).
- `dynamic_resource/train_compute_time_s`, `dynamic_resource/train_allocated_time_s` (fully_async_trainer.py:639-641), ratio `dynamic_resource/train_resource_utilization` (detach_utils.py:363-368).
- `fully_async/trainer/idle_ratio` = timing_s/gen / timing_s/step (detach_utils.py:357-358) — direct measure of the trainer-GPU idle fraction, i.e. our time-slicing win.
- Reward/actor keys from `compute_data_metrics` (standard verl): `critic/rewards/mean`, `critic/score/mean`, `actor/entropy`, `actor/grad_norm`, etc.
- `MessageQueue` actor prints `MessageQueue stats: produced=N, queue_size=K` every 100 produced (message_queue.py:79-80); per-sample trainer prints `... mq_len: X` (fully_async_trainer.py:414-418).

Note `staleness_threshold` semantics: rollouter caps generation at `max_required_samples = required_samples * trigger_parameter_sync_step * (staleness_threshold + 1)` per param version (fully_async_rollouter.py:491-496); when reached it pauses (`_should_pause_generation`, L1155-1163) — with threshold=8 the rollout GPU can run far ahead, which is what keeps it busy while the trainer is checkpointed.

---

## Q7. Ray actor OS-process/GPU layout (1 trainer GPU + 1 rollout GPU, one 2-GPU pod)

Every Ray actor is its own OS process. Distinct pids in the pod:
1. `FullyAsyncTaskRunner` (num_cpus=1) — driver.
2. `FullyAsyncTrainer` (num_cpus=10) — trainer driver, **no GPU**.
3. `FullyAsyncRollouter` (num_cpus=10, max_concurrency=100, fully_async_rollouter.py:329) — rollout driver, no GPU.
4. `MessageQueue` (num_cpus=2) — no GPU.
5. **FSDP trainer worker** ×1 — the cuda-checkpoint target. Created via `ResourcePoolManager` pool `"trainer_pool" = [1]` (`verl/experimental/separation/utils.py:36-47`), placement-group bundle `{"CPU": max_colocate_count, "GPU": 1}` (`verl/single_controller/ray/base.py:146-148`), actor created with `num_gpus = 1/max_colocate_count` (base.py:629, 675-681). Role string `"actor"` (trainer strips Role.Rollout, fully_async_main.py:140-144; train_role=Role.Actor when no hybrid, fully_async_trainer.py:155). Actor name: `{prefix}WorkerDict_0:0` (base.py:653-656).
6. **Rollout CheckpointEngineWorker** ×1 — standalone replica creates its own pool: `resource_pool_spec = {"rollout_pool_0...": [1]}` with `max_colocate_count=2` (`verl/workers/rollout/replica.py:189-226` init_standalone → num_gpus=0.5).
7. **vLLM server actor(s)** — launched by `launch_servers()`, colocated on the CE worker's GPU via `sharing_with` (base.py:391-394: reads `get_cuda_visible_devices()` from the CE worker and schedules with NodeAffinity + explicit `cuda_visible_devices`). For vLLM async mode the engine may spawn further child processes (`VLLM_USE_V1` EngineCore) — all pinned to the rollout GPU.

GPU→actor mapping mechanism: Ray placement groups reserve `{"GPU": 1}` bundles; Ray sets `CUDA_VISIBLE_DEVICES` in each actor to the bundle's assigned physical GPU. So the FSDP worker sees exactly one GPU as `cuda:0` internally, and the rollout stack sees the other as its `cuda:0`.

**Which physical ordinal goes to whom**: NOT contractually deterministic — Ray assigns whichever GPU ids are free when each placement group is created. Ordering fact: the trainer pool PG is created first (`_create_trainer` at fully_async_main.py:78 runs `init_workers` before `_create_rollouter` at L84 → `init_standalone` creates the rollout PG later). In practice Ray hands out ascending free ids, so trainer→GPU 0, rollout→GPU 1 is the typical outcome (labeled: observed Ray behavior, not a documented guarantee). For the external agent, don't assume: resolve at runtime by reading `/proc/<fsdp_worker_pid>/environ` `CUDA_VISIBLE_DEVICES`, or RPC `worker.get_cuda_visible_devices()` (verl `Worker` API, used at base.py:392-394), or `ray list actors` + `nvidia-smi` pid→GPU mapping.

Trainer worker env set by verl (base.py:675-683): `WORLD_SIZE=1, RANK=0, WG_PREFIX, WG_BACKEND=ray, RAY_LOCAL_WORLD_SIZE=1, MASTER_ADDR/PORT` — pid discovery can also match on these.

---

## Summary design implications

1. **Wrap points (monkeypatch inside FullyAsyncTrainer driver actor)**: release GPU lock after `_fit_update_weights` returns non-None; re-acquire at end of `_get_samples_from_queue` before it returns a batch. Both are async methods on `FullyAsyncTrainer` (fully_async_trainer.py:690, 375).
2. **Checkpoint target** = the single FSDP worker process (separate pid from the driver); it is idle (blocked on Ray RPC dispatch) during the whole queue-wait window; the driver's wait path has zero CUDA calls.
3. **Must set** `+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.rebuild_group=true` so no live trainer↔rollout NCCL communicator exists between syncs; and `trainer.save_freq=-1` (or lock-guard `_save_checkpoint`) so no GPU RPC hits the worker while checkpointed; keep `use_trainer_do_validate=false` and `use_dynamic_resource_scheduling=false`.
4. **Attach** via pip package with a `verl.plugins` entry point (auto-loaded, `verl/__init__.py:49-65`) that installs a lazy import hook patching `FullyAsyncTrainer`; or `VERL_USE_EXTERNAL_MODULES` env / runtime_env injection.
5. **Metrics to parse from console**: `step:N - ... fully_async/monitor/queue/mq_queue_size:… - fully_async/count/dropped_stale_samples:… - timing_s/update_actor:… - timing_s/timing_s/param_sync:… - timing_s/gen:… - fully_async/trainer/idle_ratio:… - critic/score/mean:…` plus rollouter line-1 keys (`fully_async/rollouter/idle_ratio` etc.).
