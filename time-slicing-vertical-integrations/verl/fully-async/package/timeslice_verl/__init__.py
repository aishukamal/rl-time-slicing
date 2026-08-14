"""timeslice_verl: verl <-> llm-d-rl-time-slicing integration.

Importing this package registers two trainer modes with verl's v1 registry:
  * "sync_timesliced"            colocated whole-step turn-taking (v1 PoC)
  * "separate_async_timesliced"  disaggregated two-pool cross-pipelining (v2)
and, when TIMESLICE_FULLY_ASYNC=1, installs a lazy import hook that patches
verl's experimental fully_async_policy FullyAsyncTrainer for single-lock
turn-taking on the trainers pool (v3, see timeslice_verl/fully_async.py).

It is loaded either via:
  * VERL_USE_EXTERNAL_MODULES=timeslice_verl  (verl imports it at `import verl` time), or
  * the `verl.plugins` entry point (auto-discovered by verl in every process).

Select v1/v2 with: trainer.use_v1=True trainer.v1.trainer_mode=<mode>
Select v3 with:    TIMESLICE_FULLY_ASYNC=1 (verl fully_async_main entrypoint)
"""

from timeslice_verl.locks import PhaseLocks, PhaseTransitions, RoleLocks

# Fully-async hook FIRST, before anything that imports verl: the `verl.plugins`
# entry point fires mid-`import verl`, exceptions are swallowed by verl's
# loader, and fully_async imports nothing verl-side (it only registers a
# sys.meta_path finder, and only when TIMESLICE_FULLY_ASYNC=1 — inert otherwise).
from timeslice_verl import fully_async as _fully_async

_FULLY_ASYNC_ACTIVE = _fully_async.install()

# Importing the trainer modules performs the @register_trainer registrations.
# They require verl's v1 trainer registry (verl.trainer.ppo.v1); the locks and
# fully_async modules alone do not. In fully-async mode a missing v1 registry
# must not abort the plugin load (the fully_async hook is already installed);
# otherwise preserve the original loud ImportError.
try:
    from timeslice_verl.trainer import PPOTrainerSyncTimesliced
    from timeslice_verl.trainer_disagg import PPOTrainerSeparateAsyncTimesliced
except ImportError:
    if not _FULLY_ASYNC_ACTIVE:
        raise
    PPOTrainerSyncTimesliced = None
    PPOTrainerSeparateAsyncTimesliced = None

__all__ = [
    "PhaseLocks",
    "PhaseTransitions",
    "RoleLocks",
    "PPOTrainerSyncTimesliced",
    "PPOTrainerSeparateAsyncTimesliced",
]
