"""
main_ppo_timeslice_sync.py
===========================
Sync disaggregated timeslice entry point — deep-research port.

Seeded from rl-time-slicing/verl/main_ppo_timeslice_sync.py
(with the latent import bug fixed: verl_timeslice_sync_modular_trainer ->
verl_timeslice_sync_trainer).

Uses verl's one_step_off_policy recipe to get separate actor_wg and rollout_wg
on separate GPUs with NCCL weight sync. Replaces the async training loop with
a clean sync loop in SyncTimesliceTrainer.

Deep-research additions (activated when +env.env_name=deepresearch is set):
  - builds the CMU Ray env actors (make_envs) and TrajectoryCollector and
    passes them into SyncTimesliceTrainer -> multi-turn rollout in the
    sampler phase
  - disables the streaming reward loop inside the agent loop workers (the
    per-turn batches carry no reward_model/data_source keys; rewards come
    from the env + EpisodeRewardManager instead)
  - TIMESLICE_* env vars are optional (TIMESLICE_DISABLED=1 runs the pure
    disagg benchmark with no orchestrator)

Architecture:
  GPU 0: rollout worker (vLLM, server mode) — sampler pool
  GPU 1: actor worker (FSDP)                — trainer pool

Usage (deep-research):
  python3 main_ppo_timeslice_sync.py \
    actor_rollout_ref.hybrid_engine=False \
    trainer.n_gpus_per_node=1 trainer.nnodes=1 \
    rollout.n_gpus_per_node=1 rollout.nnodes=1 \
    +env.env_name=deepresearch +env.dataset=deepresearch_mhqa \
    +env.max_steps=6 +env.rollout.n=4 ...
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import ray
import hydra
from omegaconf import DictConfig, OmegaConf

from verl_timeslice_sync_trainer import SyncTimesliceTrainer
from gpu_client import GpuClient


def _is_deepresearch(config) -> bool:
  try:
    env_cfg = config.get("env", None)
    return bool(env_cfg) and "deepresearch" in str(env_cfg.get("env_name", "")).lower()
  except Exception:
    return False


@ray.remote(num_cpus=1)
class SyncTimesliceTaskRunner:
  """TaskRunner for sync disaggregated timeslice training.

  Uses one_step_off_policy's worker setup (separate actor_wg + rollout_wg)
  but replaces the async training loop with SyncTimesliceTrainer.fit().
  """

  def run(self, config):
    import socket

    print(f'[SyncTimesliceTaskRunner] hostname={socket.gethostname()}')

    from verl.utils.fs import copy_to_local
    from verl.utils import hf_tokenizer, hf_processor
    from verl.utils.dataset.rl_dataset import collate_fn
    # moved out of verl.trainer.main_ppo on verl main (pinned a35908ca)
    from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler
    from torchdata.stateful_dataloader import StatefulDataLoader

    # Timeslice env vars are optional: with TIMESLICE_DISABLED=1 the GpuClient
    # calls are no-ops and no orchestrator/daemon stack is needed.
    job_id = os.environ.get('TIMESLICE_JOB_ID', 'job-disagg')
    pool_sampler = os.environ.get('TIMESLICE_POOL_SAMPLER', 'sampler')
    pool_trainer = os.environ.get('TIMESLICE_POOL_TRAINER', 'trainer')

    deepresearch_mode = _is_deepresearch(config)

    # ── Model and tokenizer ───────────────────────────────────────────
    local_path = copy_to_local(
        config.actor_rollout_ref.model.path,
        use_shm=config.actor_rollout_ref.model.get('use_shm', False),
    )
    trust_remote_code = config.data.get('trust_remote_code', False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(
        local_path, trust_remote_code=trust_remote_code, use_fast=True
    )

    # ── Datasets ──────────────────────────────────────────────────────
    # For deep-research the parquet is only a sizing + raw_prompt/data_source
    # vehicle: the actual questions come from the env's own JSON dataset.
    train_dataset = create_rl_dataset(
        config.data.train_files,
        config.data,
        tokenizer,
        processor,
        is_train=True,
    )
    val_dataset = create_rl_dataset(
        config.data.val_files, config.data, tokenizer, processor, is_train=False
    )
    train_sampler = create_rl_sampler(config.data, train_dataset)

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=config.data.train_batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=config.data.val_batch_size or len(val_dataset),
        collate_fn=collate_fn,
        drop_last=False,
    )

    # ── Worker groups (one_step_off_policy setup) ─────────────────────
    from verl.experimental.separation.utils import create_resource_pool_manager, create_role_worker_mapping
    from verl.experimental.one_step_off_policy.ray_trainer import OneStepOffRayTrainer
    from verl.experimental.separation.engine_workers import DetachActorWorker
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

    original_detach_init = DetachActorWorker.__init__
    def timeslice_detach_init(self, *args, **kwargs):
      original_detach_init(self, *args, **kwargs)
      from gpu_client import GpuClient
      job_id = os.environ.get('TIMESLICE_JOB_ID', 'job-disagg')
      pool = os.environ.get('TIMESLICE_POOL_TRAINER', 'trainer')
      GpuClient(workload_id=f'{job_id}-{pool}', pool=pool)

    DetachActorWorker.__init__ = timeslice_detach_init

    original_agent_init = AgentLoopWorker.__init__
    def timeslice_agent_init(self, *args, **kwargs):
      original_agent_init(self, *args, **kwargs)
      from gpu_client import GpuClient
      job_id = os.environ.get('TIMESLICE_JOB_ID', 'job-disagg')
      pool = os.environ.get('TIMESLICE_POOL_SAMPLER', 'sampler')
      GpuClient(workload_id=f'{job_id}-{pool}', pool=pool)

    AgentLoopWorker.__init__ = timeslice_agent_init

    if deepresearch_mode:
      # The streaming reward loop scores every generated sample inside the
      # agent loop worker using reward_model/data_source keys that the CMU
      # per-turn batches do not carry (rewards come from the env instead),
      # and — worse — when it is enabled the agent loop does NOT propagate
      # input non-tensor fields back into the output batch. Force the
      # AgentLoopManager to be built without reward loop handles.
      original_init_mgr = OneStepOffRayTrainer._init_async_rollout_manager

      def patched_init_mgr(trainer_self):
        class _NoRewardLoop:
          reward_loop_workers = None

        real_manager = trainer_self.reward_loop_manager
        trainer_self.reward_loop_manager = _NoRewardLoop()
        try:
          original_init_mgr(trainer_self)
        finally:
          trainer_self.reward_loop_manager = real_manager

      OneStepOffRayTrainer._init_async_rollout_manager = patched_init_mgr

    config.actor_rollout_ref.rollout.nnodes = 1
    config.actor_rollout_ref.rollout.n_gpus_per_node = 1

    role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
    resource_pool_manager = create_resource_pool_manager(config, role_worker_mapping.keys())

    # Register the workload shell from the Head Actor before workers boot up
    GpuClient(workload_id=f'{job_id}-trainer', pool=pool_trainer).register(pids=[])
    GpuClient(workload_id=f'{job_id}-sampler', pool=pool_sampler).register(pids=[])

    one_step_trainer = OneStepOffRayTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        train_sampler=train_sampler,
        device_name=config.trainer.device,
    )

    # Initialize workers under BOTH locks to prevent arbitrary VRAM collisions
    # during Ray Actor startup (since vLLM allocates VRAM eagerly).
    gpu_trainer = GpuClient(workload_id=f'{job_id}-trainer', pool=pool_trainer)
    gpu_sampler = GpuClient(workload_id=f'{job_id}-sampler', pool=pool_sampler)
    gpu_trainer.acquire_gpu()
    gpu_sampler.acquire_gpu()
    try:
      one_step_trainer.init_workers()
      # SHORTCUT FOR VERL POC: Now that vLLM and FSDP have spawned on the GPUs,
      # let the node daemon crawl the host to resolve PIDs (no-op when disabled).
      gpu_trainer.update_pids(pids=['bypass_pod_logic'])
      gpu_sampler.update_pids(pids=['bypass_pod_logic'])
    except Exception as e:
      # Only yield if initialization spectacularly crashes
      gpu_trainer.yield_gpu()
      gpu_sampler.yield_gpu()
      raise e

    # ── Deep-research env actors + trajectory collector ───────────────
    traj_collector = None
    envs = None
    val_envs = None
    if deepresearch_mode:
      from agent_system.environments import make_envs
      from agent_system.multi_turn_rollout import TrajectoryCollector

      print('[SyncTimesliceTaskRunner] building deepresearch env actors '
            f'(env_num={config.data.train_batch_size}, '
            f'group_n={config.env.rollout.n})', flush=True)
      envs, val_envs = make_envs(config)
      traj_collector = TrajectoryCollector(
          config=config, tokenizer=tokenizer, processor=processor
      )

    # ── Timeslice trainer ─────────────────────────────────────────────
    trainer = SyncTimesliceTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        one_step_trainer=one_step_trainer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        job_id=job_id,
        pool_sampler=pool_sampler,
        pool_trainer=pool_trainer,
        traj_collector=traj_collector,
        envs=envs,
        val_envs=val_envs,
    )
    import asyncio
    asyncio.run(trainer.fit())


# Root of the verl REPO clone (the dir that contains the verl/ package).
# Default preserves the original cluster layout (/data/verl/timeslice/ next to
# /data/verl/verl/); the K8s job sets VERL_ROOT=/opt/verl explicitly.
# NOTE: the one_step_off yaml has a hydra searchpath of file://verl/trainer/config
# which resolves against CWD — run this script with CWD at $VERL_ROOT.
_VERL_ROOT = os.environ.get(
    'VERL_ROOT',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'),
)


@hydra.main(
    config_path=os.path.join(
        _VERL_ROOT,
        'verl',
        'experimental',
        'one_step_off_policy',
        'config',
    ),
    config_name='one_step_off_ppo_trainer',
    version_base=None,
)
def main(config: DictConfig):
  from verl.experimental.reward_loop import migrate_legacy_reward_impl
  from verl.utils.device import auto_set_device

  auto_set_device(config)
  config = migrate_legacy_reward_impl(config)

  if not ray.is_initialized():
    ray_init_kwargs = config.ray_kwargs.get('ray_init', {})
    ray.init(**OmegaConf.to_container(ray_init_kwargs))

  runner = SyncTimesliceTaskRunner.remote()
  ray.get(runner.run.remote(config))
  print("\n[RL_JOB_COMPLETED]\n", flush=True)


if __name__ == '__main__':
  main()
