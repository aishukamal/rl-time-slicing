#!/usr/bin/env bash
# Two-job GPU time-slicing PoC: minimal single-GPU colocated GRPO run.
#
# Time-slicing knobs (set per job by the platform / pod spec):
#   TIMESLICE_JOB_ID    e.g. "job-a" | "job-b"
#   TIMESLICE_ORCH_ADDR e.g. "127.0.0.1:50051"
#   TIMESLICE_GROUP     e.g. "gpu0"   (single shared group, whole-step turn-taking)
# If unset, the job runs standalone (timeslice_verl degrades to no-op).
#
# Optional overrides: MODEL_ID, DATA_DIR, MODEL_DIR, TOTAL_TRAIN_STEPS,
# TRAIN_BATCH_SIZE, PPO_MINI_BATCH_SIZE, ROLLOUT_GPU_MEM_UTIL, EXP_NAME.
# Extra hydra overrides can be appended as arguments.
set -xeuo pipefail

# Load the timeslice_verl plugin in every verl-importing process (belt) —
# the package also registers a `verl.plugins` entry point (braces).
export VERL_USE_EXTERNAL_MODULES=timeslice_verl

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
DATA_DIR=${DATA_DIR:-/tmp/data/gsm8k}
MODEL_DIR=${MODEL_DIR:-/tmp/models/$(basename "${MODEL_ID}")}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-12}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
# Leave VRAM headroom: the whole job state is checkpointed to host RAM on every handoff.
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.25}
EXP_NAME=${EXP_NAME:-${TIMESLICE_JOB_ID:-standalone}}

# CONFIG_CHECK=1: skip data/model prep and only hydra-compose the config
# (--cfg job). Used as a build-time gate that every override key below exists
# in the pinned verl's config schema. No GPU, no training.
CONFIG_CHECK=${CONFIG_CHECK:-0}
EXTRA_ARGS=()
if [ "${CONFIG_CHECK}" = "1" ]; then
    EXTRA_ARGS+=(--cfg job)
fi

# ---- 1. GSM8K parquet (verl's own preprocess script, baked at the pinned SHA)
if [ "${CONFIG_CHECK}" != "1" ] && { [ ! -f "${DATA_DIR}/train.parquet" ] || [ ! -f "${DATA_DIR}/test.parquet" ]; }; then
    python3 /opt/poc/gsm8k_preprocess.py --local_save_dir "${DATA_DIR}"
fi

# ---- 2. Model download
if [ "${CONFIG_CHECK}" != "1" ] && [ ! -f "${MODEL_DIR}/config.json" ]; then
    python3 - "$MODEL_ID" "$MODEL_DIR" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
EOF
fi

# ---- 3. Launch: minimal single-GPU colocated GRPO, v1 sync_timesliced trainer.
# Config keys verified against verl 6a6242f3 (tests/special_e2e/ppo_trainer/
# run_function_reward.sh and examples/grpo_trainer/run_qwen3_4b_fsdp.sh).
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_DIR}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.skip_tokenizer_init=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=sync_timesliced \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=timeslice-poc \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_epochs=100 \
    trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" \
    trainer.device=cuda \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    "$@"
