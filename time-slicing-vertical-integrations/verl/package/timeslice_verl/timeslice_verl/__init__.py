"""timeslice_verl: verl <-> llm-d-rl-time-slicing integration.

Importing this package registers two trainer modes with verl's v1 registry:
  * "sync_timesliced"            colocated whole-step turn-taking (v1 PoC)
  * "separate_async_timesliced"  disaggregated two-pool cross-pipelining (v2)

It is loaded either via:
  * VERL_USE_EXTERNAL_MODULES=timeslice_verl  (verl imports it at `import verl` time), or
  * the `verl.plugins` entry point (auto-discovered by verl in every process).

Select with: trainer.use_v1=True trainer.v1.trainer_mode=<mode>
"""

from timeslice_verl.locks import PhaseLocks, PhaseTransitions, RoleLocks

# Importing the trainer modules performs the @register_trainer registrations.
# They require verl to be importable; the locks module alone does not.
from timeslice_verl.trainer import PPOTrainerSyncTimesliced
from timeslice_verl.trainer_disagg import PPOTrainerSeparateAsyncTimesliced

__all__ = [
    "PhaseLocks",
    "PhaseTransitions",
    "RoleLocks",
    "PPOTrainerSyncTimesliced",
    "PPOTrainerSeparateAsyncTimesliced",
]
