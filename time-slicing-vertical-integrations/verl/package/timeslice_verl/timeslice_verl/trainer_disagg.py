"""Time-sliced disaggregated (two-pool) async PPO/GRPO trainer for verl v1.

Registers trainer mode "separate_async_timesliced": verl's "separate_async"
mode (PPOTrainerSeparateAsync) with role-based orchestrator locks so TWO jobs
can share a trainers pool and a samplers pool and cross-pipeline (A trains
while B samples).

Verified against verl 6a6242f3d8ec7d9f8b4936f4905144707d91fe3b (v0.9.0.dev):

  * Placement is config-driven and disjoint: the trainer worker group uses
    trainer.nnodes x trainer.n_gpus_per_node (Ray "global_pool"); the
    standalone rollout uses actor_rollout_ref.rollout.nnodes x
    actor_rollout_ref.rollout.n_gpus_per_node (its own Ray resource pool,
    replica.py init_standalone). 1 GPU each => 1 FSDP trainer + 1 vLLM server.
    NOTE: Ray does not pin which physical node each 1-GPU pool lands on; the
    run script enforces it by startup order (sampler node joins Ray only after
    the trainer pool has claimed the trainer node's GPU).

  * The trainer blocks for samples inside PPOTrainer._step_once:
    on_sample_begin -> replay_buffer.sample(...) -> on_sample_end, where
    ReplayBuffer.sample polls TransferQueue metadata in a
    time.sleep(poll_interval) loop (replay_buffer.py:250-271). Pure CPU: no
    trainer-GPU work happens while blocked => safe to yield the trainer pool.

  * on_step_end -> standalone_checkpoint_manager.update_weights is a
    cross-pool broadcast (checkpoint_engine backend nccl/nixl/mooncake;
    "naive" is asserted out because it requires colocation). BOTH pools must
    be resident for the whole call => dual-lock span. With
    engine_kwargs.nccl.rebuild_group=true the NCCL group is created and
    destroyed inside each sync, so no cross-pool NCCL communicator exists
    outside the dual-lock span (required for checkpoint/restore safety).

  * switch_to_rollout()/switch_to_trainer() manage only the HYBRID replicas
    colocated on the trainer GPUs (wake+add / abort+sleep+remove from the load
    balancer). should_switch_to_rollout() is a stub returning False at this
    commit, so the hybrid engine stays in TRAINER mode during training; only
    on_validate_begin wakes it. They complement (do not conflict with)
    external lock gating: all their GPU work is on the trainer pool and every
    call site runs while we hold TRAINER.

  * trainer_base branches on the literal string `self.trainer_mode != "sync"`
    (gen_batch_size forcing, TransferQueue checkpointing, prompt persistence,
    in-flight re-issue). Any non-"sync" name inherits async semantics, so no
    trainer_mode reset is needed here (unlike the sync_timesliced trick).
    Mode-keyed config lookups (trainer.v1.<mode> -> parameter_sync_step,
    ReplayBuffer trainer_config) are satisfied by mirroring
    trainer.v1.separate_async into trainer.v1.separate_async_timesliced before
    super().__init__; PPOTrainerSeparateAsync.__init__ itself reads the
    trainer.v1.separate_async section via hardcoded keys, so both stay
    consistent.

  * On the natural last step fit() returns early WITHOUT calling on_train_end
    (trainer_base.py fit: `if is_last_step: ... return`). The TRAINER lock is
    still held at that point; RoleLocks' atexit hook releases it when the
    python process exits (the stay-alive wrapper keeps the POD alive, not the
    python process).

Sampler-pool lock ownership: the DRIVER process (TaskRunnerV1 ray actor, CPU)
holds the samplers-pool lock on behalf of its job for the span in which its
generation executes: [on_sample_begin, on_sample_end], i.e. exactly the
replay_buffer.sample wait. When sample() returns, this job's generation for
the step is complete and the lock is released. In-flight partial rollouts that
continue past that point are frozen/restored by the platform when the other
job takes the pool; partial rollout tolerates this (trajectory staleness
metrics capture the extra spans).
"""

from omegaconf import OmegaConf, open_dict

from timeslice_verl.locks import PhaseTransitions, RoleLocks

try:
    from verl.trainer.ppo.v1.trainer_base import register_trainer
    from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "timeslice_verl requires verl with the v1 separate_async trainer "
        "(verl.trainer.ppo.v1, >= v0.9.0.dev / commit 6a6242f3)."
    ) from e

MODE_NAME = "separate_async_timesliced"


def _log(msg: str) -> None:
    print(f"[timeslice] {msg}", flush=True)


@register_trainer(MODE_NAME)
class PPOTrainerSeparateAsyncTimesliced(PPOTrainerSeparateAsync):
    """PPOTrainerSeparateAsync with role-based two-pool lock gating.

    Per-step lock timeline (see PhaseTransitions for the full phase model):

      feed                        TRAINER held
      on_sample_begin (starved)   -TRAINER +SAMPLER   <- other job can train
      ... replay_buffer.sample poll-wait; this job's generation executes ...
      on_sample_end               -SAMPLER +TRAINER   <- other job can sample
      reward/old_log_prob/adv/update_actor            TRAINER held
      on_step_end (weight sync)   +SAMPLER ... -SAMPLER (dual-lock span)
    """

    def __init__(self, config):
        # Acquire BOTH pools before any verl construction: trainer.init()
        # creates worker groups and vLLM servers on both pools and on_init_end
        # broadcasts weights to hybrid AND standalone replicas.
        self.role_locks = RoleLocks.from_env()
        self.phases = PhaseTransitions(self.role_locks)
        self.phases.init_begin()

        # Mirror the separate_async mode section under our mode name so
        # trainer.v1.<trainer_mode> lookups in trainer_base resolve
        # identically (parameter_sync_step, ReplayBuffer trainer_config).
        v1_cfg = config.trainer.v1
        with open_dict(v1_cfg):
            if not v1_cfg.get(MODE_NAME):
                v1_cfg[MODE_NAME] = OmegaConf.to_container(v1_cfg.separate_async, resolve=True)

        super().__init__(config)

    # ------------------------------------------------------------ turn-taking
    def on_train_begin(self):
        # Warmup prompt submission is CPU-side (TransferQueue put + async
        # dispatch to the agent loop actors); then yield the sampler pool
        # until this job actually waits on generation.
        super().on_train_begin()
        self.phases.train_begin()

    def on_sample_begin(self):
        self.phases.sample_begin(starved=self._sample_wait_expected())
        super().on_sample_begin()

    def on_sample_end(self):
        # Re-acquire TRAINER before super(): separate_async's on_sample_end
        # may switch the hybrid engine to trainer mode (abort+sleep of the
        # hybrid replicas = GPU work on the trainer pool).
        self.phases.sample_end()
        super().on_sample_end()

    def on_step_end(self):
        # Dual-lock span around the cross-pool NCCL weight broadcast.
        self.phases.weight_sync_begin()
        super().on_step_end()
        self.phases.weight_sync_end()

    # ------------------------------------------------------------ validation
    def on_validate_begin(self):
        # separate_async validation wakes the hybrid replicas (trainer pool)
        # and serves from the standalone pool as well: hold both.
        self.phases.validate_begin()
        super().on_validate_begin()

    def on_validate_end(self):
        super().on_validate_end()
        self.phases.validate_end()

    # -------------------------------------------------------------- shutdown
    def on_train_end(self):
        # NOTE: not reached on the natural last-step exit of fit() at the
        # pinned verl commit (early return); RoleLocks' atexit hook covers
        # that path. This handles the epoch-exhaustion exit.
        super().on_train_end()
        self.phases.train_end()

    # ------------------------------------------------------------- internals
    def _sample_wait_expected(self) -> bool:
        """Conditional-release probe: True if replay_buffer.sample will block.

        Uses the replay buffer's own gating (metadata sync + _has_enough_samples)
        so the answer matches sample()'s first loop iteration exactly. On any
        probe failure, assume starvation (conservative: yield the trainer pool).
        """
        try:
            batch_size = self.config.data.train_batch_size // self.parameter_sync_step
            rb = self.replay_buffer
            rb._sync_metadata_from_transfer_queue()
            ready = rb._has_enough_samples(self.global_steps, "train", batch_size)
            if ready:
                _log(
                    f"job={self.role_locks.job_id} sample_begin: batch already "
                    f"buffered (>= {batch_size}); keeping TRAINER, skipping SAMPLER"
                )
            return not ready
        except Exception as e:  # noqa: BLE001 - probe is best-effort
            _log(f"replay-buffer probe failed ({e}); assuming sample will block")
            return True
