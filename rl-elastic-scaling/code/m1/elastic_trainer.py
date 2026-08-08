# elastic-rl-poc M1 — out-of-tree FullyAsyncTrainer subclass.
#
# Pinned against verl 983cb0f24443f87b3d161fad318445130a620b07 (read-only
# reference clone: GPU-CR/code-rlvr/verl). Zero verl patches: everything here
# subclasses the unwrapped actor class, the same pattern verl itself uses in
# verl/single_controller/ray/base.py:968-971 (_unwrap_ray_remote:
# `cls.__ray_actor_class__`).
#
# Why this subclass exists (landmine #1 from verl-integration-notes.md):
# FullyAsyncTrainer._setup_checkpoint_manager (fully_async_trainer.py:217-224)
# builds the CheckpointEngineManager over EVERY replica returned by
# rollouter.get_replicas(). The NCCL weight-sync group membership is frozen
# after the first sync (NCCLCheckpointEngine.rebuild_group=False default,
# nccl_checkpoint_engine.py:146 + init_process_group at :214-222), so if a
# suspended (cuda-checkpointed) replica is ever in that manager, the next
# _fit_update_weights deadlocks: the abort/release_kv_cache RPC fan-out
# (checkpoint_engine/base.py:499-509) and the NCCL broadcast both hang on the
# frozen process.
#
# In the M1 topology R2 is launched out-of-tree (see fully_async_main_elastic.py)
# and never appears in rollouter.get_replicas() at all, so the filter below is
# defense-in-depth. It becomes load-bearing the moment anyone flips back to the
# in-tree "size rollout for two standalone replicas" layout.

import asyncio
import os

import ray

from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.utils.config import omega_conf_to_dataclass

# Unwrap the @ray.remote-decorated class to get a subclassable base
# (verl pattern, single_controller/ray/base.py:968-971).
_TRAINER_BASE = FullyAsyncTrainer.__ray_actor_class__


def _excluded_replica_ranks() -> set:
    """Replica ranks to exclude from the NCCL weight-sync group.

    Default "1": the R2 rank in the in-tree two-replica layout. Harmless when
    R2 is out-of-tree (rank 1 simply doesn't exist in get_replicas()).
    """
    raw = os.environ.get("ELASTIC_SYNC_EXCLUDE_RANKS", "1")
    raw = raw.replace(",", " ").strip()
    if not raw:
        return set()
    return {int(tok) for tok in raw.split()}


@ray.remote(num_cpus=10)  # same actor options as FullyAsyncTrainer (fully_async_trainer.py:53)
class ElasticFullyAsyncTrainer(_TRAINER_BASE):
    """FullyAsyncTrainer whose weight-sync group excludes the time-sliced R2."""

    async def _setup_checkpoint_manager(self):
        """Copy of fully_async_trainer.py:217-224 with a replica-rank filter.

        Base body (verbatim from the pin):
            replicas = await self.rollouter.get_replicas.remote()
            checkpoint_engine_config = omega_conf_to_dataclass(
                self.config.actor_rollout_ref.rollout.checkpoint_engine)
            self.checkpoint_manager = CheckpointEngineManager(
                config=checkpoint_engine_config, actor_wg=self.actor_wg, replicas=replicas)
        """
        replicas = await self.rollouter.get_replicas.remote()
        excluded = _excluded_replica_ranks()
        kept, dropped = [], []
        for replica in replicas:
            if getattr(replica, "replica_rank", None) in excluded:
                dropped.append(replica.replica_rank)
            else:
                kept.append(replica)

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config, actor_wg=self.actor_wg, replicas=kept
        )
        print(
            f"[ElasticFullyAsyncTrainer] Checkpoint manager initialized "
            f"(backend={checkpoint_engine_config.backend}) with "
            f"{len(kept)}/{len(replicas)} replicas; excluded replica_ranks={sorted(dropped)} "
            f"(R2 must NEVER join the frozen NCCL param_sync group)"
        )

    # ------------------------------------------------------------------
    # Small introspection RPCs for the external switch controller
    # (Ray cannot read actor attributes remotely).
    # ------------------------------------------------------------------

    def get_current_param_version(self) -> int:
        """Weight version k for R2's post-restore set_global_steps(k)."""
        return self.current_param_version

    async def get_trainer_worker_pids(self) -> list:
        """OS PIDs of the FSDP actor worker processes (cuda-checkpoint targets).

        Uses the generic `actor.__ray_call__.remote(fn)` escape hatch, the same
        mechanism verl uses on CheckpointEngineWorker handles in
        vllm_async_server.py:1184-1190. Lambdas are pickled by value, so the
        target worker needs no extra imports.
        """
        refs = [
            worker.__ray_call__.remote(lambda self: __import__("os").getpid())
            for worker in self.actor_wg.workers
        ]
        return list(await asyncio.gather(*refs))
