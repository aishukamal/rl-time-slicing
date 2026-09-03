# elastic-vd — out-of-tree FullyAsyncTrainer subclass for the 32-GPU DAPO
# showcase (successor of m1/elastic_trainer.py; verl pin 983cb0f2).
#
# Differences vs the M1 subclass:
#   - The yield mechanism is verl's OWN manual offload lever
#     (TrainingWorker.to(device, model, optimizer, grad),
#     engine_workers.py:159-167 -> FSDPEngine.to -> offload_fsdp_* utils),
#     NOT cuda-checkpoint. Comms stay alive; no shim, no snapshot-agent.
#   - Spares are workers=[] STANDALONE vLLM replicas, invisible to
#     rollouter.get_replicas() by construction, so no checkpoint-manager
#     replica filter is needed (kept as a belt-and-braces env option).
#   - NEW SAFETY GUARD: FSDP compute while offloaded is a CRASH (offloaded
#     flat params live on CPU; the first dispatched kernel/allgather dies),
#     unlike the M1 freeze which merely stalled. _fit_generate therefore
#     awaits an "elastic resident" event AFTER batch collection and BEFORE
#     returning the batch to fit_step's compute chain. The controller clears
#     the event at offload and sets it at onload. A self-heal timeout
#     (ELASTIC_STALL_TIMEOUT_S, default 600s) onloads autonomously if the
#     controller dies while the trainer is yielded — overnight insurance.

import asyncio
import os
import time

import ray

from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

_TRAINER_BASE = FullyAsyncTrainer.__ray_actor_class__

ELASTIC_STALL_TIMEOUT_S = float(os.environ.get("ELASTIC_STALL_TIMEOUT_S", "600"))


@ray.remote(num_cpus=10)  # same actor options as FullyAsyncTrainer (fully_async_trainer.py:53)
class ElasticVDTrainer(_TRAINER_BASE):
    """FullyAsyncTrainer with a manual FSDP offload/onload lever + resident guard."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._elastic_resident_event = asyncio.Event()
        self._elastic_resident_event.set()  # trainer starts resident
        self._elastic_state = "resident"    # resident | offloaded
        self._elastic_ops = []              # (ts, op, seconds) history

    # ------------------------------------------------------------------
    # Yield lever (called by the external controller via the handles actor)
    # ------------------------------------------------------------------

    def elastic_offload(self) -> dict:
        """Offload FSDP params+grads+optimizer of ALL trainer ranks to CPU.

        Blocking dispatch: TrainingWorker.to is @register(ONE_TO_ALL) with the
        default blocking dispatch, so this returns only after every rank has
        finished its H2D->D2H copies and emptied its cache. Runs on the
        trainer actor's event loop while fit() is awaiting the MessageQueue
        (async actor, max_concurrency default) — blocking that loop for the
        offload wall is harmless during gen-wait.
        """
        assert self._elastic_state == "resident", f"offload from state {self._elastic_state}"
        t0 = time.time()
        self._elastic_resident_event.clear()
        self._elastic_state = "offloading"
        try:
            self.actor_wg.to("cpu", model=True, optimizer=True, grad=True)
        except Exception:
            # Half-offloaded is NOT resident: leave the event cleared so the
            # guard still protects compute; controller must onload/abort.
            self._elastic_state = "offload_failed"
            raise
        dt = time.time() - t0
        self._elastic_state = "offloaded"
        self._elastic_ops.append((t0, "offload", round(dt, 3)))
        print(f"[ElasticVDTrainer] OFFLOAD complete in {dt:.2f}s (16-rank FSDP -> CPU)", flush=True)
        return {"seconds": dt}

    def elastic_onload(self) -> dict:
        """Load FSDP params+grads+optimizer back to the GPUs and release the guard."""
        t0 = time.time()
        try:
            self.actor_wg.to("device", model=True, optimizer=True, grad=True)
        finally:
            # Even a partially-failed onload should unblock the guard: the
            # subsequent compute will surface the real error where it is
            # visible instead of deadlocking silently.
            self._elastic_resident_event.set()
        dt = time.time() - t0
        self._elastic_state = "resident"
        self._elastic_ops.append((t0, "onload", round(dt, 3)))
        print(f"[ElasticVDTrainer] ONLOAD complete in {dt:.2f}s", flush=True)
        return {"seconds": dt}

    def elastic_hold(self) -> dict:
        """Clear the resident-guard event WITHOUT moving any state (sleepL1
        window safety, defect VD-6): if the batch completes mid-window, the
        trainer stalls at _fit_generate until elastic_release() instead of
        entering update_actor against wake'd spare memory."""
        self._elastic_resident_event.clear()
        self._elastic_state = "held"
        return {"held": True}

    def elastic_release(self) -> dict:
        self._elastic_resident_event.set()
        if self._elastic_state == "held":
            self._elastic_state = "resident"
        return {"held": False}

    async def elastic_release_cache(self) -> dict:
        """torch.cuda.empty_cache() on every trainer rank: frees the caching
        allocator's reserved-but-unused blocks (near-peak after an update)
        WITHOUT moving any state. Only ever called while the trainer is
        blocked in gen-wait (controller gate). ~0.5-2s."""
        import asyncio as _aio
        t0 = time.time()
        def _empty(self):
            import torch
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            return round((total - free) / (1 << 30), 1)
        refs = [w.__ray_call__.remote(_empty) for w in self.actor_wg.workers]
        used_gb = list(await _aio.gather(*refs))
        dt = time.time() - t0
        print(f"[ElasticVDTrainer] release_cache in {dt:.2f}s; per-rank used GiB: "
              f"min={min(used_gb)} max={max(used_gb)}", flush=True)
        return {"seconds": dt, "used_gb_max": max(used_gb)}

    def elastic_state(self) -> dict:
        return {
            "state": self._elastic_state,
            "resident_event": self._elastic_resident_event.is_set(),
            "ops_tail": self._elastic_ops[-6:],
        }

    # ------------------------------------------------------------------
    # Resident guard: no GPU compute may be dispatched while offloaded.
    # ------------------------------------------------------------------

    async def _fit_generate(self, batch=None):
        batch = await super()._fit_generate(batch)
        if not self._elastic_resident_event.is_set():
            t0 = time.time()
            print("[ElasticVDTrainer] batch ready but trainer is YIELDED; awaiting onload...", flush=True)
            while True:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(self._elastic_resident_event.wait()), timeout=30.0
                    )
                    break
                except asyncio.TimeoutError:
                    waited = time.time() - t0
                    print(f"[ElasticVDTrainer] still yielded {waited:.0f}s after batch-ready", flush=True)
                    if waited > ELASTIC_STALL_TIMEOUT_S:
                        print(
                            f"[ElasticVDTrainer] SELF-HEAL: controller did not onload within "
                            f"{ELASTIC_STALL_TIMEOUT_S}s of batch-ready; onloading autonomously",
                            flush=True,
                        )
                        self.elastic_onload()
                        break
            print(f"[ElasticVDTrainer] resident again after {time.time() - t0:.1f}s wait", flush=True)
        return batch

    # ------------------------------------------------------------------
    # Introspection RPCs for the external controller
    # ------------------------------------------------------------------

    def get_current_param_version(self) -> int:
        return self.current_param_version

    def get_local_trigger_step(self) -> int:
        return self.local_trigger_step

    async def get_trainer_worker_info(self) -> list:
        """(pid, ray_node_id, node_ip, cuda_visible_devices) per FSDP rank.

        Used once at init to discover WHICH physical nodes/GPUs the trainer
        landed on, so the spares can be placed 1:1 on the same GPUs."""
        def probe(self):
            import os as _os
            import ray as _ray
            ctx = _ray.get_runtime_context()
            try:
                accel = ctx.get_accelerator_ids().get("GPU", [])
            except Exception:
                accel = []
            return (
                _os.getpid(),
                ctx.get_node_id(),
                _ray.util.get_node_ip_address(),
                _os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                list(accel),
            )
        refs = [w.__ray_call__.remote(probe) for w in self.actor_wg.workers]
        return list(await asyncio.gather(*refs))
