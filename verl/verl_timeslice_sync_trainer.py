"""verl_timeslice_sync_modular_trainer.py

================================
Sync disaggregated RL loop with timeslice integration.

This is the cleanest possible integration — a self-contained synchronous
training loop (no async, no queues, no hooks into internals) with our
4 acquire/yield calls at the natural phase boundaries.

The loop mirrors what ray_trainer.py does in sync mode with hybrid_engine=False:
  1. sync weights actor → rollout
  2. rollout generates sequences        (GPU 0)
  3. compute advantages                 (CPU)
  4. actor trains                       (GPU 1)
  5. repeat

Timeslice integration is just 4 lines:
  sampler_gpu.acquire() before generate
  sampler_gpu.yield()   after generate
  trainer_gpu.acquire() before train + weight sync (needs both for NCCL)
  sampler_gpu.acquire()   both needed for NCCL
  trainer_gpu.yield()  after sync  after sync
  sampler_gpu.yield()  after sync

Used by: main_ppo_timeslice_sync.py
"""

import datetime
import json
import os
import sys
import time


def ts_print(*args, **kwargs):
  ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  print(f"[{ts}]", *args, flush=True, **kwargs)


sys.path.insert(0, os.path.dirname(__file__))

import ray
from gpu_client import GpuClient


class SyncTimesliceTrainer:
  """Self-contained sync disaggregated trainer with timeslice integration.

  Uses verl's one_step_off_policy worker classes (which already have
  separate actor_wg and rollout_wg with NCCL weight sync).

  The fit() loop here is a clean sync loop — no async, no queues.
  Compare to FullyAsyncTrainer.fit() which is ~300 lines of async machinery.
  """

  def __init__(
      self,
      config,
      tokenizer,
      processor,
      one_step_trainer,
      train_dataloader,
      val_dataloader,
      job_id: str,
      pool_sampler: str = "sampler",
      pool_trainer: str = "trainer",
  ):

    self.config = config
    self.tokenizer = tokenizer
    self.one_step_trainer = one_step_trainer
    self.train_dataloader = train_dataloader
    self.val_dataloader = val_dataloader

    self.sampler_gpu = GpuClient(
        workload_id=f"{job_id}-sampler", pool=pool_sampler
    )
    self.trainer_gpu = GpuClient(
        workload_id=f"{job_id}-trainer", pool=pool_trainer
    )

  async def _sync_weights(self):
    import time

    t0_sync = time.perf_counter()
    await self.one_step_trainer.checkpoint_manager.update_weights(
        self.one_step_trainer.global_steps
    )
    await self.one_step_trainer.async_rollout_manager.clear_kv_cache()
    return round((time.perf_counter() - t0_sync) * 1000)

  async def _generate(self, batch):
    import time
    import uuid
    import numpy as np
    from verl.trainer.ppo.ray_trainer import compute_response_mask

    t0_gen = time.perf_counter()
    batch.non_tensor_batch["uid"] = np.array(
        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
    )
    gen_batch = self.one_step_trainer._get_gen_batch(batch)
    gen_batch.meta_info["global_steps"] = self.one_step_trainer.global_steps

    batch_repeat = gen_batch.repeat(
        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
    )
    gen_batch_output = (
        await self.one_step_trainer.async_rollout_manager.generate_sequences(
            batch_repeat
        )
    )
    gen_ms = round((time.perf_counter() - t0_gen) * 1000)

    # Concatenate prompt + generated output
    batch_repeat = batch.repeat(
        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
    )
    batch = batch_repeat.union(gen_batch_output)

    if "response_mask" not in batch.batch:
      batch.batch["response_mask"] = compute_response_mask(batch)

    return batch, gen_ms

  def _compute_rewards_and_advantages(self, batch):
    from verl.trainer.ppo import core_algos
    from verl.trainer.ppo.reward import extract_reward

    if self.one_step_trainer.use_rm and "rm_scores" not in batch.batch.keys():
      batch_reward = self.one_step_trainer._compute_reward_colocate(batch)
      batch = batch.union(batch_reward)

    reward_tensor, reward_extra_infos_dict = extract_reward(batch)
    batch.batch["token_level_rewards"] = reward_tensor

    adv, returns = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=batch.batch["token_level_rewards"],
        response_mask=batch.batch["response_mask"],
        index=batch.non_tensor_batch.get("uid"),
    )
    batch.batch["advantages"] = adv
    batch.batch["returns"] = returns
    return batch, reward_tensor, reward_extra_infos_dict

  def _compute_old_log_probs(self, batch):
    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
    bypass_recomputing_logprobs = (
        rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
    )

    if bypass_recomputing_logprobs:
      from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

      apply_bypass_mode(
          batch=batch,
          rollout_corr_config=rollout_corr_config,
          policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
      )
    else:
      old_log_prob, _ = self.one_step_trainer._compute_old_log_prob(batch)
      batch = batch.union(old_log_prob)
    return batch

  def _train(self, batch):
    import time

    t0_train = time.perf_counter()
    actor_output = self.one_step_trainer._update_actor(batch)
    train_ms = round((time.perf_counter() - t0_train) * 1000)
    return actor_output, train_ms

  def _record_metrics(
      self,
      metrics_file,
      curr_step,
      train_ms,
      gen_ms,
      sync_ms,
      actor_output,
      reward_tensor,
      reward_extra_infos_dict,
  ):
    import json
    import numpy as np

    loss, kl, clipfrac, grad_norm, lr = 0.0, 0.0, 0.0, 0.0, 0.0

    if actor_output and hasattr(actor_output, "meta_info"):
      metrics = actor_output.meta_info.get("metrics", {})
      loss = float(np.mean(metrics.get("actor/loss", [0.0])))
      kl = float(np.mean(metrics.get("actor/ppo_kl", [0.0])))
      clipfrac = float(np.mean(metrics.get("actor/pg_clipfrac", [0.0])))
      grad_norm = float(np.mean(metrics.get("actor/grad_norm", [0.0])))
      lr = float(metrics.get("actor/lr", [0.0])[0])

    mean_reward = 0.0
    if reward_tensor is not None:
      mean_reward = float(reward_tensor.sum(-1).mean().item())

    acc = (
        float(np.mean(reward_extra_infos_dict.get("acc", [0.0])))
        if "acc" in reward_extra_infos_dict
        else 0.0
    )

    metric_record = {
        "step": curr_step,
        "train_ms": train_ms,
        "gen_ms": gen_ms,
        "sync_ms": sync_ms,
        "mean_reward": mean_reward,
        "acc": acc,
        "loss": loss,
        "kl": kl,
        "grad_norm": grad_norm,
        "clipfrac": clipfrac,
        "lr": lr,
    }

    with open(metrics_file, "a") as f:
      f.write(json.dumps(metric_record) + "\n")

  async def fit(self):
    from verl import DataProto
    import os, time

    pause_file = "/tmp/timeslice_pause"

    self.one_step_trainer.metrics = {
        "training/global_step": self.one_step_trainer.global_steps,
        "training/epoch": self.one_step_trainer.epoch,
    }
    self.one_step_trainer.timing_raw = {}

    log_dir = os.environ.get("LOG_DIR", "/data/rl_logs")
    os.makedirs(log_dir, exist_ok=True)
    job_id = self.trainer_gpu.workload_id.split("-trainer")[0]
    metrics_file = os.path.join(log_dir, f"metrics_{job_id}.jsonl")

    if os.path.exists(metrics_file):
      os.remove(metrics_file)

    for epoch in range(self.config.trainer.total_epochs):
      for batch_dict in self.train_dataloader:
        ts_print(
            "[DEBUG] Fetched batch for step"
            f" {self.one_step_trainer.global_steps + 1}"
        )
        batch = DataProto.from_single_dict(batch_dict)

        # ── Step 1: Sync weights ──────────────────
        self.trainer_gpu.acquire_gpu()
        self.sampler_gpu.acquire_gpu()
        sync_ms = await self._sync_weights()
        self.trainer_gpu.yield_gpu()

        # ── Step 2: Generate ──────────────────────
        batch, gen_ms = await self._generate(batch)
        self.sampler_gpu.yield_gpu()

        # ── Step 3: Rewards & Advantages (CPU) ─
        batch, reward_tensor, reward_extra_infos_dict = (
            self._compute_rewards_and_advantages(batch)
        )

        # ── Step 4 & 5: Old Log Probs & Train ─
        self.trainer_gpu.acquire_gpu()
        batch = self._compute_old_log_probs(batch)
        actor_output, train_ms = self._train(batch)

        # NOTE: We don't yield trainer here!
        # It rolls seamlessly into Step 1 of the next loop.

        self.one_step_trainer.global_steps += 1
        curr_step = self.one_step_trainer.global_steps
        ts_print(f"[SyncTimesliceTrainer] step {curr_step} done")

        self._record_metrics(
            metrics_file,
            curr_step,
            train_ms,
            gen_ms,
            sync_ms,
            actor_output,
            reward_tensor,
            reward_extra_infos_dict,
        )

        if (
            self.one_step_trainer.total_training_steps > 0
            and self.one_step_trainer.global_steps
            >= self.one_step_trainer.total_training_steps
        ):
          ts_print(
              "[SyncTimesliceTrainer] reached max steps"
              f" {self.one_step_trainer.global_steps}"
          )
          self.trainer_gpu.yield_gpu()
          return

    self.trainer_gpu.yield_gpu()
