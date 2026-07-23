#!/usr/bin/env bash
# Disaggregated two-pool verl time-slicing PoC: one verl job split across a
# shared SAMPLERS pool (1 GPU, standalone vLLM) and a shared TRAINERS pool
# (1 GPU, FSDP actor + sleeping hybrid replica), mode separate_async_timesliced.
#
# Each verl job is its own 3-pod Ray cluster:
#   ROLE=head-driver     CPU pod: ray head + verl driver (TaskRunnerV1) + TransferQueue/agent-loop actors
#   ROLE=trainer-worker  GPU pod on the TRAINERS node: ray worker (1 GPU)
#   ROLE=sampler-worker  GPU pod on the SAMPLERS node: ray worker (1 GPU);
#                        joins Ray ONLY after the trainer pool has claimed the
#                        trainer GPU (placement-by-startup-order: Ray cannot
#                        pin a 1-GPU bundle to a specific node by config)
#
# Time-slicing knobs (per job, set by the experiment track):
#   TIMESLICE_JOB_ID        e.g. "job-a" | "job-b"  (fresh id per attempt!)
#   TIMESLICE_ORCH_ADDR     cluster-reachable orchestrator addr (K8s Service,
#                           NOT 127.0.0.1: the lock holder is the driver pod)
#   TIMESLICE_TRAINER_GROUP e.g. "trainers"
#   TIMESLICE_SAMPLER_GROUP e.g. "samplers"
# If unset, the job runs standalone (RoleLocks degrades to no-op).
#
# Topology knobs:
#   RAY_HEAD_ADDR   host[:port] of this job's ray head (required for workers;
#                   default port 6379). Per-job GCS port must differ if two
#                   heads share a pod network namespace (they don't here).
#
# Baked-in platform workarounds (validated in the sync PoC):
#   * CUDA keepalive process per GPU pod (first-acquire deadlock)
#   * stay-alive wrapper after main command exits (pod must outlive python so
#     the platform can finish its C/R bookkeeping; python exit still runs the
#     RoleLocks atexit release)
#   * NCCL_CUMEM_ENABLE=0 everywhere
#   * checkpoint_engine nccl rebuild_group=true: the cross-pool NCCL group
#     exists only inside the dual-lock weight-sync span
set -xeuo pipefail

export NCCL_CUMEM_ENABLE=0
export VERL_USE_EXTERNAL_MODULES=timeslice_verl

ROLE=${ROLE:-head-driver}
RAY_HEAD_ADDR=${RAY_HEAD_ADDR:-127.0.0.1:6379}
[[ "${RAY_HEAD_ADDR}" == *:* ]] || RAY_HEAD_ADDR="${RAY_HEAD_ADDR}:6379"

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
DATA_DIR=${DATA_DIR:-/tmp/data/gsm8k}
MODEL_DIR=${MODEL_DIR:-/tmp/models/$(basename "${MODEL_ID}")}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-12}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PARAM_SYNC_STEP=${PARAM_SYNC_STEP:-2}   # must satisfy TRAIN = SYNC_STEP * MINI
# Low VRAM: job state is checkpointed to host RAM on every pool handoff.
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.25}
EXP_NAME=${EXP_NAME:-${TIMESLICE_JOB_ID:-standalone}-disagg}

CONFIG_CHECK=${CONFIG_CHECK:-0}

# ---------------------------------------------------------------- helpers
stay_alive() {
    local rc=$1
    echo "[run_job_disagg] main command exited rc=${rc}; staying alive for platform C/R bookkeeping"
    sleep infinity
}

start_cuda_keepalive() {
    # Tiny CUDA-context holder: works around the first-acquire deadlock (the
    # snapshot agent needs at least one live CUDA PID in the pod cgroup).
    python3 - <<'EOF' &
import time
import torch
torch.cuda.init()
x = torch.zeros(1, device="cuda")
print("[keepalive] CUDA context held, pid ready for snapshot agent", flush=True)
while True:
    time.sleep(60)
EOF
    KEEPALIVE_PID=$!
    echo "[run_job_disagg] cuda keepalive pid=${KEEPALIVE_PID}"
}

wait_for_ray_head() {
    until ray status --address "${RAY_HEAD_ADDR}" >/dev/null 2>&1; do
        echo "[run_job_disagg] waiting for ray head at ${RAY_HEAD_ADDR}..."
        sleep 5
    done
}

wait_for_trainer_gpu_claimed() {
    # Placement-by-startup-order gate: block until the cluster shows exactly
    # one GPU and it is fully claimed (the trainer worker group took it).
    # Only then may the sampler node join, so the standalone rollout pool is
    # forced onto the sampler node's GPU.
    while true; do
        local line
        line=$(ray status --address "${RAY_HEAD_ADDR}" 2>/dev/null | grep -E '^[[:space:]]*[0-9.]+/[0-9.]+ GPU' | head -1) || true
        # expected form: " 1.0/1.0 GPU"
        if [[ "${line}" =~ ([0-9.]+)/([0-9.]+)\ GPU ]]; then
            local used=${BASH_REMATCH[1]} total=${BASH_REMATCH[2]}
            echo "[run_job_disagg] ray GPU usage: ${used}/${total}"
            if [[ "${total}" == "1.0" && "${used}" == "1.0" ]]; then
                echo "[run_job_disagg] trainer pool has claimed the trainer GPU; joining as sampler node"
                return 0
            fi
        else
            echo "[run_job_disagg] waiting for ray head / GPU resources..."
        fi
        sleep 10
    done
}

# ---------------------------------------------------------------- workers
if [ "${ROLE}" = "trainer-worker" ]; then
    start_cuda_keepalive
    wait_for_ray_head
    ray start --address="${RAY_HEAD_ADDR}" --num-gpus=1 --block && rc=0 || rc=$?
    stay_alive ${rc}
fi

if [ "${ROLE}" = "sampler-worker" ]; then
    start_cuda_keepalive
    wait_for_ray_head
    wait_for_trainer_gpu_claimed
    ray start --address="${RAY_HEAD_ADDR}" --num-gpus=1 --block && rc=0 || rc=$?
    stay_alive ${rc}
fi

if [ "${ROLE}" != "head-driver" ]; then
    echo "[run_job_disagg] unknown ROLE=${ROLE}" >&2
    exit 2
fi

# ---------------------------------------------------------------- head-driver
EXTRA_ARGS=()
if [ "${CONFIG_CHECK}" = "1" ]; then
    EXTRA_ARGS+=(--cfg job)
else
    ray start --head --port="${RAY_HEAD_ADDR##*:}" --num-gpus=0 \
        --dashboard-host=0.0.0.0
    export RAY_ADDRESS="${RAY_HEAD_ADDR}"
fi

# ---- 1. GSM8K parquet (verl's own preprocess script, baked at the pinned SHA)
if [ "${CONFIG_CHECK}" != "1" ] && { [ ! -f "${DATA_DIR}/train.parquet" ] || [ ! -f "${DATA_DIR}/test.parquet" ]; }; then
    python3 /opt/poc/gsm8k_preprocess.py --local_save_dir "${DATA_DIR}"
fi

# ---- 2. Model download (driver side; workers get weights via ray/NCCL)
if [ "${CONFIG_CHECK}" != "1" ] && [ ! -f "${MODEL_DIR}/config.json" ]; then
    python3 - "$MODEL_ID" "$MODEL_DIR" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
EOF
fi

# ---- 3. Env propagation to every Ray actor (TaskRunnerV1 = the lock holder,
# agent-loop actors, GPU workers): merged into verl's default runtime_env.
RUNTIME_ENV_ARGS=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_CUMEM_ENABLE=\"0\""
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=\"timeslice_verl\""
)
if [ -n "${TIMESLICE_JOB_ID:-}" ]; then
    RUNTIME_ENV_ARGS+=(
        "+ray_kwargs.ray_init.runtime_env.env_vars.TIMESLICE_JOB_ID=\"${TIMESLICE_JOB_ID}\""
        "+ray_kwargs.ray_init.runtime_env.env_vars.TIMESLICE_ORCH_ADDR=\"${TIMESLICE_ORCH_ADDR:?}\""
        "+ray_kwargs.ray_init.runtime_env.env_vars.TIMESLICE_TRAINER_GROUP=\"${TIMESLICE_TRAINER_GROUP:?}\""
        "+ray_kwargs.ray_init.runtime_env.env_vars.TIMESLICE_SAMPLER_GROUP=\"${TIMESLICE_SAMPLER_GROUP:?}\""
    )
fi

# ---- 4. Launch: 1 trainer GPU + 1 standalone rollout GPU, disjoint pools.
# Config keys verified against verl 6a6242f3 (trainer_separate_async.py,
# ppo_trainer.yaml trainer.v1.*, rollout.yaml nnodes/n_gpus_per_node/
# checkpoint_engine, base.py CheckpointEngineManager.update_weights).
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
    actor_rollout_ref.rollout.nnodes=1 \
    actor_rollout_ref.rollout.n_gpus_per_node=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.skip_tokenizer_init=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512 \
    '+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.rebuild_group=True' \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=separate_async_timesliced \
    trainer.v1.separate_async.parameter_sync_step="${PARAM_SYNC_STEP}" \
    trainer.v1.separate_async.num_warmup_batches=1 \
    trainer.v1.sampler.max_off_policy_threshold=100 \
    trainer.v1.sampler.max_off_policy_strategy=drop \
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
    "${RUNTIME_ENV_ARGS[@]}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    "$@" && RC=0 || RC=$?
if [ "${CONFIG_CHECK}" = "1" ]; then
    exit ${RC}
fi
stay_alive ${RC}
