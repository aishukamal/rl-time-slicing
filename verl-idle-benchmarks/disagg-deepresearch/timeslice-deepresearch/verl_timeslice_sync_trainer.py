"""verl_timeslice_sync_trainer.py

================================
Sync disaggregated RL loop with timeslice integration — deep-research port.

Seeded from rl-time-slicing/verl/verl_timeslice_sync_trainer.py. In addition
to the original single-turn GSM8K path, this copy supports the CMU
verl-agent-deepresearch multi-turn rollout loop in the sampler phase:

  per step:
    1. weight sync   : checkpoint_manager.update_weights (NCCL)      [both GPUs]
    2. generate      : TrajectoryCollector.multi_turn_loop           [sampler GPU]
                       (per turn: AgentLoopManager.generate_sequences
                        for the WHOLE batch + Ray env actors step,
                        search I/O against the local Wikipedia server)
    3. rewards/adv   : EpisodeRewardManager + GRPO outcome advantage [CPU]
    4. train         : old_log_prob + update_actor                   [trainer GPU]

The timeslice boundaries are unchanged: the sampler lease covers the whole
multi-turn loop (generation + search I/O), the trainer lease covers
old_log_prob + update + the next weight sync. With TIMESLICE_DISABLED=1 the
GpuClient calls are no-ops (pure disagg benchmark, no orchestrator).

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


class AgentLoopRolloutAdapter:
  """Sync facade over verl main's AgentLoopManager for CMU's per-turn calls.

  The CMU multi-turn loop calls `actor_rollout_wg.generate_sequences(batch)`
  synchronously once per turn. verl main's AgentLoopManager.generate_sequences
  is an @auto_await coroutine: when invoked from a plain sync frame it runs
  the coroutine to completion itself (fresh event loop in a worker thread if
  one is already running), so this adapter just forwards the call.

  The per-turn batch built by TrajectoryCollector.preprocess_batch already
  contains the `raw_prompt` chat that verl main's SingleTurnAgentLoop
  re-tokenizes (same tokenizer + template => same ids CMU used for the
  training tensors).
  """

  def __init__(self, async_rollout_manager, get_global_steps):
    self._mgr = async_rollout_manager
    self._get_global_steps = get_global_steps

  def generate_sequences(self, batch):
    batch.meta_info["global_steps"] = self._get_global_steps()
    return self._mgr.generate_sequences(batch)


class SyncTimesliceTrainer:
  """Self-contained sync disaggregated trainer with timeslice integration.

  Uses verl's one_step_off_policy worker classes (which already have
  separate actor_wg and rollout_wg with NCCL weight sync).

  When constructed with a `traj_collector` + `envs`, the sampler phase runs
  the CMU deep-research multi-turn loop instead of a single generate call;
  the resulting per-(env, turn) training batch REPLACES the dataloader batch
  (CMU semantics), and rewards come from EpisodeRewardManager (episode reward
  broadcast to the last response token of every row).
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
      traj_collector=None,
      envs=None,
      val_envs=None,
  ):

    self.config = config
    self.tokenizer = tokenizer
    self.one_step_trainer = one_step_trainer
    self.train_dataloader = train_dataloader
    self.val_dataloader = val_dataloader

    # Multi-turn (deep-research) extras — all optional; None => original
    # single-turn behavior.
    self.traj_collector = traj_collector
    self.envs = envs
    self.val_envs = val_envs
    self.multi_turn = traj_collector is not None and envs is not None
    self.episode_reward_manager = None
    if self.multi_turn:
      from agent_system.reward_manager import EpisodeRewardManager
      self.episode_reward_manager = EpisodeRewardManager(
          tokenizer=tokenizer, num_examine=1, normalize_by_length=False
      )
    self.rollout_adapter = AgentLoopRolloutAdapter(
        one_step_trainer.async_rollout_manager,
        lambda: self.one_step_trainer.global_steps,
    )

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
    # AgentLoopManager.clear_kv_cache() was removed upstream — the pinned
    # CheckpointEngineManager.update_weights already releases/rebuilds the
    # kv cache around the NCCL sync. Keep a guarded call for older trees.
    clear_kv = getattr(
        self.one_step_trainer.async_rollout_manager, "clear_kv_cache", None
    )
    if clear_kv is not None:
      await clear_kv()
    return round((time.perf_counter() - t0_sync) * 1000)

  async def _generate(self, batch):
    if self.multi_turn:
      return await self._generate_multi_turn(batch)
    return await self._generate_single_turn(batch)

  async def _generate_single_turn(self, batch):
    """Original PoC path: one generate call for the repeated batch (GSM8K)."""
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

  async def _generate_multi_turn(self, batch):
    """Deep-research path: CMU multi-turn env loop on the sampler GPU.

    The dataloader batch only provides `raw_prompt`/`data_source` and batch
    sizing — the questions come from the env's own JSON dataset. The returned
    per-(env, turn) batch REPLACES the dataloader batch wholesale (CMU
    semantics), then gets padded to a divisor of the train batch layout.
    """
    import time
    import asyncio
    from verl.trainer.ppo.ray_trainer import compute_response_mask
    from agent_system.multi_turn_rollout import adjust_batch

    t0_gen = time.perf_counter()

    gen_batch = batch  # needs: input_ids (sizing), raw_prompt, data_source
    gen_batch.meta_info["global_steps"] = self.one_step_trainer.global_steps

    # The loop is synchronous (per-turn generate + ray.get on env actors);
    # run it off the event loop thread so heartbeats/logging stay alive.
    loop = asyncio.get_running_loop()
    final_batch = await loop.run_in_executor(
        None,
        lambda: self.traj_collector.multi_turn_loop(
            gen_batch=gen_batch,
            actor_rollout_wg=self.rollout_adapter,
            envs=self.envs,
            is_train=True,
        ),
    )
    gen_ms = round((time.perf_counter() - t0_gen) * 1000)

    ts_print(
        f"[SyncTimesliceTrainer] multi-turn rollout done: {len(final_batch)}"
        " training rows"
    )

    # Variable batch size (Σ episode lengths) → pad to a divisor of the
    # mini/micro batch layout (mode="copy" duplicates random rows).
    final_batch = adjust_batch(self.config, final_batch, mode="copy")

    # CMU recomputes the response mask from the attention mask (per-turn rows
    # are single-turn generations, so this matches the agent-loop mask).
    final_batch.batch["response_mask"] = compute_response_mask(final_batch)

    return final_batch, gen_ms

  def _compute_rewards_and_advantages(self, batch):
    from verl.trainer.ppo import core_algos

    if self.multi_turn:
      # Episode reward (broadcast onto every row of the trajectory by
      # gather_rollout_data) → last valid response token of each row.
      reward_tensor = self.episode_reward_manager(batch)
      batch.batch["token_level_scores"] = reward_tensor
      batch.batch["token_level_rewards"] = reward_tensor
      reward_extra_infos_dict = {}
    else:
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
      extra=None,
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
    if extra:
      metric_record.update(extra)

    with open(metrics_file, "a") as f:
      f.write(json.dumps(metric_record) + "\n")

  def _multi_turn_batch_stats(self, batch):
    """Episode stats for the metrics JSONL (multi-turn only)."""
    import numpy as np

    stats = {"rows": len(batch)}
    ntb = batch.non_tensor_batch
    if "episode_lengths" in ntb:
      stats["mean_episode_len"] = float(np.mean(ntb["episode_lengths"].astype(np.float64)))
    if "episode_rewards" in ntb:
      stats["mean_episode_reward"] = float(np.mean(ntb["episode_rewards"].astype(np.float64)))
    if "is_action_valid" in ntb:
      stats["valid_action_ratio"] = float(np.mean(ntb["is_action_valid"].astype(np.float64)))
    return stats

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

        # ── Step 2: Generate (multi-turn env loop when enabled) ──
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
            extra=self._multi_turn_batch_stats(batch) if self.multi_turn else None,
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
