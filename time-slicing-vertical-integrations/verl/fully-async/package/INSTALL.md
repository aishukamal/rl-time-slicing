# timeslice_verl fully-async wrapper — PoC pod install

Extends the `timeslice-verl` package (aishukamal/rl-time-slicing,
`time-slicing-vertical-integrations/verl/package/timeslice_verl/`) to
time-slice verl's experimental `fully_async_policy` trainer
(verl @ `983cb0f24443f87b3d161fad318445130a620b07`).

## What is in this tree

Overlay onto the upstream package tree:

| file | status |
|---|---|
| `timeslice_verl/fully_async.py` | NEW — lazy import hook + FullyAsyncTrainer patches + rl_metrics.jsonl emission + crash-release hygiene + client-rename compat import + experimental empty-cache hook |
| `timeslice_verl/__init__.py` | CHANGED — installs the fully-async hook first; v1 trainer imports tolerated missing in fully-async mode |
| `timeslice_verl/locks.py` | CHANGED — client-rename compat import (`OrchestratorClient` → `TimeSliceOrchestratorClient` fallback, small-run finding 2026-07-26); otherwise upstream |
| `pyproject.toml` | NEW — canonical package metadata + `verl.plugins` entry point (the job manifests still heredoc-generate an identical copy; keep in sync) |
| `tests/test_fully_async.py` | NEW — 29 tests, pure python (no verl/ray/grpc/GPU) |

Not included (unchanged upstream): `trainer.py`, `trainer_disagg.py`,
`tests/test_transitions.py`. The `pyproject.toml` entry point that makes verl
load the wrapper in every process:

```toml
[project.entry-points."verl.plugins"]
timeslice_verl = "timeslice_verl"
```

verl auto-loads every `verl.plugins` entry point at `import verl`
(verl/__init__.py:49-65, default `VERL_USE_EXTERNAL_PLUGINS=auto`) in every
process — driver, FullyAsyncTrainer actor, FSDP worker. The hook is inert
unless `TIMESLICE_FULLY_ASYNC=1`, so the same image/package serves the
sync/disagg modes unchanged.

## Pod install (ConfigMap-mounted source + pip install -e)

1. **timeslice python client** (gRPC client for the accelerator-orchestrator),
   from `pkg/client/python/{pyproject.toml, timeslice/…}` on GitHub main:

   ```bash
   pip install "git+https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git#subdirectory=pkg/client/python"
   # REQUIRED (small-run finding 2026-07-26): the client's generated stubs need
   # grpcio>=1.81 and protobuf gencode 7.35; the verl image ships grpcio 1.80.0
   # / protobuf 6.33.6 and fails at import time without this upgrade.
   pip install "grpcio>=1.81.0" "protobuf>=7.35.0"
   ```

   Client class name: upstream renamed `OrchestratorClient` →
   `TimeSliceOrchestratorClient` (API identical). `locks.py` and
   `fully_async.py` import with a try/except compat fallback, so both old and
   new client versions work.

2. **Package source via ConfigMaps.** ConfigMap keys cannot contain `/`, so
   mount two ConfigMaps to rebuild the tree:

   ```bash
   kubectl create configmap timeslice-verl-root --from-file=pyproject.toml
   kubectl create configmap timeslice-verl-src  --from-file=timeslice_verl/
   ```

   ```yaml
   volumeMounts:
     - {name: tsv-root, mountPath: /mnt/timeslice-verl}
     - {name: tsv-src,  mountPath: /mnt/timeslice-verl/timeslice_verl}
   volumes:
     - {name: tsv-root, configMap: {name: timeslice-verl-root}}
     - {name: tsv-src,  configMap: {name: timeslice-verl-src}}
   ```

3. **Editable install.** ConfigMap mounts are read-only and `pip install -e`
   writes build metadata into the source tree — copy to a writable dir first
   (pod command / initContainer):

   ```bash
   cp -rL /mnt/timeslice-verl /workspace/timeslice-verl-pkg
   pip install -e /workspace/timeslice-verl-pkg
   ```

   The editable install registers the `verl.plugins` entry point; no verl
   source changes and no `VERL_USE_EXTERNAL_MODULES` needed (entry-point
   discovery reaches every Ray actor process because each one re-imports verl).

## Platform setup

Deploy the time-slicing platform (Accelerator Orchestrator + Snapshot Agent + DRA driver) on your GKE cluster following the [deployment guide](https://github.com/llm-d-incubation/llm-d-rl-time-slicing/tree/main/deploy). Label and taint GPU nodes per the [orchestrator guide](https://github.com/llm-d-incubation/llm-d-rl-time-slicing/tree/main/guides/accelerator-orchestrator).

## CUDA_VISIBLE_DEVICES orientation

The shared trainer GPU must be first in each job's CUDA device mask so that
`torch.cuda.device(0)` maps to it:

```yaml
- {name: CUDA_VISIBLE_DEVICES, value: "2,0"}  # GPU 2 = shared trainer (logical 0), GPU 0 = dedicated rollout
```

Both time-sliced jobs must use the same physical GPU as their logical device 0.
A watchdog in the manifest verifies placement at startup.

## Swapping the workload

To use a different model or dataset, change these verl config overrides in the
job manifest:

- `actor_rollout_ref.model.model_path` — HuggingFace model name or local path
- `actor_rollout_ref.rollout.prompt_data` — dataset path
- `actor_rollout_ref.rollout.reward_fn` — reward function

The manifests in `manifests/` use DAPO-Math-17k (Job A) and Eurus-Code (Job B)
with DeepSeek-R1-Distill-Qwen-1.5B.

## Required environment (trainer job container)

```yaml
- {name: TIMESLICE_FULLY_ASYNC, value: "1"}          # activation gate (inert otherwise)
- {name: TIMESLICE_JOB_ID,      value: "job-a"}
- {name: TIMESLICE_ORCH_ADDR,   value: "127.0.0.1:50051"}
- {name: TIMESLICE_GROUP,       value: "trainers"}   # the trainers-pool group
- {name: TIMESLICE_METRICS_PATH, value: "/workspace/results/rl_metrics.jsonl"}  # optional (this is the default)
# REQUIRED for time-sliced jobs (small-run finding 2026-07-26): NVLS/cumem NCCL
# transports don't survive cuda-checkpoint restore (NCCL 2.28.9 dies in
# transport/nvls.cc, cuda error 101, on the first post-restore comm creation).
- {name: NCCL_CUMEM_ENABLE, value: "0"}
- {name: NCCL_NVLS_ENABLE,  value: "0"}
```

Single-pod `ray.init()` (our case): Ray actors inherit the driver shell env.
On a pre-started cluster, additionally inject via
`+ray_kwargs.ray_init.runtime_env.env_vars.TIMESLICE_FULLY_ASYNC=1` (etc.).

Missing `TIMESLICE_JOB_ID/ORCH_ADDR/GROUP` with the gate set → lock layer runs
in PhaseLocks-style warn-once no-op mode (patches installed, no lock traffic,
no metrics), so the wrapper is safe to bake into images.

## Required verl config for the run (hydra overrides)

```bash
+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.rebuild_group=true \  # REQUIRED: no live trainer<->rollout NCCL communicator between syncs
trainer.save_freq=-1 \                                # REQUIRED: saving RPCs the (possibly checkpointed-off) FSDP worker; wrapper warn-skips if left >0
async_training.trigger_parameter_sync_step=1 \        # yield-per-step; wrapper warns once if >1 (lock then held across queue waits)
async_training.use_dynamic_resource_scheduling=false \
async_training.use_trainer_do_validate=false          # keep validation off the trainer GPU
```

NOTE on key paths: the last three live under `async_training.*` — an earlier
revision of this doc wrote `trainer.use_trainer_do_validate`, which is wrong
(hydra rejects it; the manifests use `async_training.use_trainer_do_validate`).

## Lock protocol (single trainers-pool group lock)

| point | action |
|---|---|
| `init_workers` | ACQUIRE → delegate (worker groups + model load) → YIELD |
| initial `_fit_update_weights` (pre-fit, fully_async_main) | ensure-held ACQUIRE → sync → YIELD |
| `_get_samples_from_queue` returns a batch | ACQUIRE (resume point; the queue wait itself is CPU-only) |
| `_fit_update_weights` returns non-None | YIELD (update_actor + NCCL param sync + reset_staleness complete) |
| `_fit_save_checkpoint` with save_freq>0 | warn-once no-op |

All lock RPCs run in the CPU-only FullyAsyncTrainer driver actor (never
checkpointed) via `asyncio.to_thread`, so a minutes-long acquire never stalls
the actor's event loop. Acquire failures propagate (a job must not run
unlocked); release failures and ALL metrics IO failures are swallowed.

## rl_metrics.jsonl

Appended (one JSON object per line, flushed per line) to
`$TIMESLICE_METRICS_PATH`:

```json
{"ts": 1769480000.1, "type": "acquire", "workload_id": "job-a", "pool": "trainers", "queue_len": 5, "step": 12, "wait_ms": 3100, "restore_ms": null, "context_restored": true, "point": "samples_ready"}
{"ts": 1769480031.9, "type": "yield",   "workload_id": "job-a", "pool": "trainers", "queue_len": 5, "step": 13, "held_ms": 31800, "point": "update_weights"}
```

`restore_ms` is always null today (the client's AcquireResult exposes
`success/waited_ms/context_restored` only); `context_restored` and `point` are
additive fields. `queue_len` comes from the batch's embedded rollouter mq
snapshot (`fully_async/monitor/queue/mq_queue_size`), falling back to the
trainer's end-of-previous-step `dynamic_resource/mq_size`, else null — never
an extra RPC. `step` is `current_param_version`.

## Experimental: TIMESLICE_EMPTY_CACHE_BEFORE_YIELD

`TIMESLICE_EMPTY_CACHE_BEFORE_YIELD=1` makes the wrapper call
`torch.cuda.empty_cache()` immediately before each yield's `drop_all`, to
probe whether returning the allocator's cached-but-free blocks shrinks the
cuda-checkpoint snapshot (small-run finding: snapshot time grows with
allocated training memory, ~13 s cold → ~22-27 s at 49 GB steady state).

- Inert by default; NOT wired into any manifest.
- Best-effort: skipped (warn-once) if torch is not importable or
  `empty_cache()` raises; can never break training or the lock protocol.
- Caveat: the call runs in the CPU-only FullyAsyncTrainer driver actor, which
  is not the FSDP worker holding the training memory — a real experiment
  likely needs the call RPC'd into the worker; this env gate is the plumbing
  starting point.

## Tests

```bash
python3 -m pytest tests/test_fully_async.py -v   # 29 passed; needs only pytest
```
