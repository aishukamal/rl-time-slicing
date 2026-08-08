# verl fully-async @983cb0f2 — R2 integration findings (agent investigation, 2026-08-04)

Full details with file:line refs in session transcript. Repo: GPU-CR/code-rlvr/verl (read-only reference clone).

## The seam exists: verl already has dynamic replica membership
- Engine list lives in GlobalRequestLoadBalancer Ray actor (llm_server.py:75-77) with atomic
  add_servers/remove_servers/clear_sticky_cache (llm_server.py:120-178). No health checks; least-inflight +
  sticky-session routing; stale sticky entries self-invalidate.
- FullyAsyncLLMServerManager.hybrid_replicas + rollouter RPCs add_replicas/remove_replicas/rebalance_requests
  (fully_async_rollouter.py:54-259, 1199-1272).
- DynamicResourceController.activate/deactivate_hybrid_replicas (dynamic_schedule/dynamic_resource_controller.py:117-160)
  documents the required ordering: remove-from-LB FIRST, then abort, then sleep.
  NB: this is verl's own nascent elastic mechanism — vLLM-sleep-based, rollout-side only, never touches the
  trainer. Include in honest-comparison arm; our differentiator = trainer-side suspend via C/R.

## R2 lifecycle (zero-verl-patch path)
- Init: size rollout for 2 standalone replicas (rollout.nnodes × n_gpus_per_node). Named actors:
  server `vllm_server_{replica}_{node}`, engine workers `rollout_standalone_{rank}_*` → cuda-checkpoint targets.
- SUSPEND: LB.remove_servers([r2]) → vllm_server_1_0.abort_all_requests() → wait get_total_inflight()==0 →
  cuda-checkpoint. Partial-rollout client (llm_server.py:322-423) transparently resumes aborted requests on R1
  with accumulated tokens — graceful abort is invisible to the agent loop.
- RESUME: restore → weight catch-up (below) → set_global_steps(k) → clear_kv_cache → resume_generation →
  LB.add_servers (+clear_sticky_cache).

## Weight path for R2
- R1 sync: NCCL broadcast via CheckpointEngineManager; group membership effectively FROZEN after first sync
  (rebuild_group=False default, nccl_checkpoint_engine.py:214-222).
- **Landmine #1:** if R2 is in checkpoint_manager.replicas while suspended, next _fit_update_weights deadlocks
  (abort RPC + NCCL to frozen process). MUST exclude R2 from the manager → the ONE out-of-tree patch needed:
  ~5-line subclass of FullyAsyncTrainer._setup_checkpoint_manager (fully_async_trainer.py:217-224) filtering replicas.
- Out-of-band load: no vLLM update-weights HTTP route at this pin. Use named server actor
  collective_rpc("reload_weights") (disk HF checkpoint) — or accept version-k weights until next regular sync:
  correctness-safe because recipe runs use_rollout_log_probs=True + rollout_correction.bypass_mode=True;
  staleness accounting is metrics-only and engine-agnostic.

## Landmines
1. NCCL group deadlock (above).
2. Freeze-without-drain hangs the WHOLE run: no timeouts anywhere in generate path; weight-sync fan-out RPCs
   (abort/release_kv_cache/drain) hang on a frozen engine. Always drain before suspend.
3. Engine hard-death poisons the run (RayActorError propagates to TaskRunner, no per-sample retry except aborts).
   MessageQueue samples survive (separate actor); only overflow drops.
4. global_steps=None on fresh spawn before first sync → TypeError in trainer metrics. Post-restore
   set_global_steps is cheap insurance (state survives C/R restore, so only bites fresh spawns).
5. LB/rollouter/MQ actors are UNNAMED → controller must own handles: fork the ~240-line
   fully_async_main.py driver out-of-tree (register named controller actor holding rollouter+LB+MQ handles).

## Controller metrics (1s-poll friendly, Ray actor calls)
- MessageQueue.get_statistics(): queue_size (=buffer depth), total_produced (Δ/Δt = fill rate), dropped_samples.
  Cheapest single poll target.
- Rollouter.get_statistics(): active_tasks, staleness_samples, dropped_stale_samples, max_concurrent_samples.
- LB.get_status()/get_inflight_count(server): per-engine load. Per-engine vLLM /metrics over HTTP if enabled.
- timing_s/gen, update_actor, param_sync are LOG-ONLY (per sync cycle) — controller uses actor calls, report uses logs.

## M1 implications
- Out-of-tree deliverables: forked driver (fully_async_main.py) + 5-line trainer subclass + controller script
  driving snapshot-agent (VLLMAdapter for R2 drain hooks) — zero verl patches.
- max_concurrent_samples note: bypassing rollouter.add_replicas means concurrency won't grow when R2 joins;
  either call rollouter RPCs or accept split concurrency (fully_async_rollouter.py:503-506).
