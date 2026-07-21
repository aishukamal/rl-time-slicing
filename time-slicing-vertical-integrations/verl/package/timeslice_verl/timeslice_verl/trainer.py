"""Time-sliced synchronous PPO/GRPO trainer for verl v1.

Registers trainer mode "sync_timesliced": identical to verl's "sync" mode
(PPOTrainerSync) but wraps every GPU-touching phase in an orchestrator group
lock so two colocated jobs can take whole-step turns on the same GPU(s).

Verified against verl 6a6242f3d8ec7d9f8b4936f4905144707d91fe3b (v0.9.0.dev):
  * PPOTrainer.__init__(config) is CPU-only; GPU work starts in trainer.init()
    (worker-group creation, model load, first update_weights via on_init_end).
    We still acquire before super().__init__ so the lock is held for the whole
    init sequence, whichever verl version moves work around.
  * fit() hook order per step:
      on_step_begin -> [sample: on_sample_begin/on_sample_end -> train] ->
      optional save_checkpoint -> on_step_end -> optional (on_validate_begin ->
      _validate -> on_validate_end)
  * NOTE: on the natural last step, fit() returns early WITHOUT calling
    on_train_end (trainer_base.py: `if is_last_step: ... return` precedes the
    loop-exit on_train_end call). The lock is already released by the last
    on_step_end / on_validate_end; PhaseLocks' atexit hook covers crashes.
  * trainer_base branches on the literal string `self.trainer_mode != "sync"`
    (gen_batch_size forcing, TransferQueue checkpointing, in-flight prompt
    re-issue, prompt-field persistence). We reset self.trainer_mode = "sync"
    right after super().__init__ so "sync_timesliced" inherits exact sync
    semantics in those branches. Config lookups keyed by the mode name
    (trainer.v1.<mode>) happen inside super().__init__ via .get(mode, {}) and
    fall back to {} => parameter_sync_step=1, same as "sync".
"""

from timeslice_verl.locks import PhaseLocks

try:
    from verl.trainer.ppo.v1.trainer_base import register_trainer
    from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "timeslice_verl requires verl with the v1 trainer "
        "(verl.trainer.ppo.v1, >= v0.9.0.dev / commit 6a6242f3)."
    ) from e


@register_trainer("sync_timesliced")
class PPOTrainerSyncTimesliced(PPOTrainerSync):
    """PPOTrainerSync with whole-step turn-taking on a single shared group lock.

    Timeline per training step (colocated single-group PoC):
      on_step_begin   -> ensure lock held (blocks while the other job runs)
      ... sample + train + weight sync (on_step_end -> super()) ...
      on_step_end     -> release lock => this job's GPU state is checkpointed
                         out by the platform, the other job gets the GPU.
      on_validate_*   -> same bracket around validation.
    """

    def __init__(self, config):
        # Acquire BEFORE any verl construction so every GPU touch downstream
        # (trainer.init(): worker groups, model load, rollout engine, the
        # on_init_end weight sync) happens while we own the group.
        self.phase_locks = PhaseLocks.from_env()
        self.phase_locks.ensure()
        super().__init__(config)
        # Inherit exact "sync" semantics in trainer_base's literal
        # `trainer_mode != "sync"` branches (see module docstring).
        self.trainer_mode = "sync"

    # ------------------------------------------------------------ step turns
    def on_step_begin(self):
        self.phase_locks.ensure()
        super().on_step_begin()

    def on_step_end(self):
        # Let the sync trainer finish the step first (checkpoint_manager.
        # update_weights -> rollout replicas hold the new weights), THEN yield
        # the GPU. The job re-queues for the lock at the next on_step_begin.
        super().on_step_end()
        self.phase_locks.drop_all()

    # ------------------------------------------------------------ validation
    def on_validate_begin(self):
        self.phase_locks.ensure()
        super().on_validate_begin()

    def on_validate_end(self):
        super().on_validate_end()
        self.phase_locks.drop_all()

    # -------------------------------------------------------------- shutdown
    def on_train_end(self):
        # NOTE: not reached on the natural last-step exit of fit() at the
        # pinned verl commit (early return). Locks are already released by the
        # final on_step_end; this is for the epoch-exhaustion exit path.
        super().on_train_end()
        self.phase_locks.drop_all()
        self.phase_locks.close()
