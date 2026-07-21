"""timeslice_verl: verl <-> llm-d-rl-time-slicing integration.

Importing this package registers the "sync_timesliced" trainer mode with verl's
v1 trainer registry. It is loaded either via:
  * VERL_USE_EXTERNAL_MODULES=timeslice_verl  (verl imports it at `import verl` time), or
  * the `verl.plugins` entry point (auto-discovered by verl in every process).

Select it with: trainer.use_v1=True trainer.v1.trainer_mode=sync_timesliced
"""

from timeslice_verl.locks import PhaseLocks

# Importing the trainer module performs the @register_trainer("sync_timesliced")
# registration. It requires verl to be importable; PhaseLocks alone does not.
from timeslice_verl.trainer import PPOTrainerSyncTimesliced

__all__ = ["PhaseLocks", "PPOTrainerSyncTimesliced"]
