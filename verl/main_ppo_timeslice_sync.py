"""
main_ppo_timeslice_sync.py
===========================
Sync disaggregated timeslice entry point.

Uses verl's one_step_off_policy recipe to get separate actor_wg and rollout_wg
on separate GPUs with NCCL weight sync. Replaces the async training loop with
a clean sync loop in SyncTimesliceTrainer.

This is the cleanest customer integration story:
  - No async machinery
  - No hooks into verl internals
  - The sync loop is self-contained and readable
  - Timeslice integration is 4 acquire/yield calls in the loop

Architecture:
  GPU 0: rollout worker (vLLM) — sampler pool
  GPU 1: actor worker (FSDP)  — trainer pool

Differences from main_ppo_timeslice_disagg.py (async version):
  - Uses one_step_off_policy recipe instead of fully_async_policy
  - Training loop is sync (no queue, no staleness threshold)
  - Strictly on-policy (generate then train, no overlap)
  - Simpler to understand and debug

Usage:
  python3 main_ppo_timeslice_sync.py \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.strategy=fsdp \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    +rollout.n_gpus_per_node=1 \
    +rollout.nnodes=1 \
    +timeslice.job_id=job-a \
    +timeslice.pool_sampler=sampler \
    +timeslice.pool_trainer=trainer \
    ...
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import ray
import hydra
from omegaconf import DictConfig, OmegaConf

from verl_timeslice_sync_modular_trainer import SyncTimesliceTrainer
from gpu_client import GpuClient


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
    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
    from verl.trainer.ppo.utils import need_reference_policy, need_critic
    from verl.utils.config import validate_config
    from verl.experimental.reward_loop import migrate_legacy_reward_impl
    from torchdata.stateful_dataloader import StatefulDataLoader

    # Set env vars for downstream code that reads them
    job_id = os.environ['TIMESLICE_JOB_ID']
    pool_sampler = os.environ['TIMESLICE_POOL_SAMPLER']
    pool_trainer = os.environ['TIMESLICE_POOL_TRAINER']
    sampler_ip = os.environ['TIMESLICE_SAMPLER_IP']
    trainer_ip = os.environ['TIMESLICE_TRAINER_IP']

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
    from verl.single_controller.base.decorator import register as ray_register, Dispatch

    original_detach_init = DetachActorWorker.__init__
    def timeslice_detach_init(self, *args, **kwargs):
      original_detach_init(self, *args, **kwargs)
      from gpu_client import GpuClient
      job_id = os.environ['TIMESLICE_JOB_ID']
      pool   = os.environ['TIMESLICE_POOL_TRAINER']
      import psutil
      all_pids = []
      import sys
      from datetime import datetime
      for p in psutil.pids():
        try:
          proc = psutil.Process(p)
          cmd = " ".join(proc.cmdline())[:80]
          all_pids.append(str(p))
          ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
          # SHORTCUT FOR VERL POC: Commented out noisy process logs
          # print(f"[{ts}] [pool={pool}] Captured PID {p}: {cmd}", file=sys.stderr)
        except Exception:
          pass
      # Monkey patched functions won't read runtime env vars, so job_id/pool are not available here
      gpu_step = GpuClient(workload_id=f'{job_id}-{pool}', pool=pool)
      # SHORTCUT FOR VERL POC: Node daemon will ignore this and natively crawl the bare-metal host using NVIDIA-SMI to resolve vLLM / WorkerDict PIDs directly.
      # gpu_step.update_pids(pids=all_pids)

    DetachActorWorker.__init__ = timeslice_detach_init

    original_agent_init = AgentLoopWorker.__init__
    def timeslice_agent_init(self, *args, **kwargs):
      original_agent_init(self, *args, **kwargs)
      from gpu_client import GpuClient
      job_id = os.environ['TIMESLICE_JOB_ID']
      pool   = os.environ['TIMESLICE_POOL_SAMPLER']
      import psutil
      all_pids = []
      import sys
      from datetime import datetime
      for p in psutil.pids():
        try:
          proc = psutil.Process(p)
          cmd = " ".join(proc.cmdline())[:80]
          all_pids.append(str(p))
          ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
          # SHORTCUT FOR VERL POC: Commented out noisy process logs
          # print(f"[{ts}] [pool={pool}] Captured PID {p}: {cmd}", file=sys.stderr)
        except Exception:
          pass
      # Monkey patched functions won't read runtime env vars, so job_id/pool are not available here
      gpu_step = GpuClient(workload_id=f'{job_id}-{pool}', pool=pool)
      # SHORTCUT FOR VERL POC: Node daemon will ignore this and natively crawl the bare-metal host using NVIDIA-SMI to resolve vLLM / WorkerDict PIDs directly.
      # gpu_step.update_pids(pids=all_pids)

    AgentLoopWorker.__init__ = timeslice_agent_init

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
      # send the bypass string to update_pids to trigger the Daemon to globally scan the host!
      gpu_trainer.update_pids(pids=['bypass_pod_logic'])
      gpu_sampler.update_pids(pids=["bypass_pod_logic"])
    except Exception as e:
      # Only yield if initialization spectacularly crashes
      gpu_trainer.yield_gpu()
      gpu_sampler.yield_gpu()
      raise e
    actor_wg = one_step_trainer.actor_wg
    rollout_wg = one_step_trainer.actor_rollout_wg

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
    )
    import asyncio
    asyncio.run(trainer.fit())


@hydra.main(
    config_path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..',
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
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    from verl.utils.device import is_cuda_available

    ray_init_kwargs = config.ray_kwargs.get('ray_init', {})
    ray.init(**OmegaConf.to_container(ray_init_kwargs))

  runner = SyncTimesliceTaskRunner.remote()
  ray.get(runner.run.remote(config))
  print("\n[RL_JOB_COMPLETED]\n", flush=True)


if __name__ == '__main__':
  main()
