#!/usr/bin/env bash
# Phase 1: dapo_7b_math_fsdp2_{4_4,8_8,4_12} at recipe defaults (Mode 4), ~30 steps each.
# Requires 5 free mega nodes. Pods stay alive after DONE for data extraction.
set -euo pipefail

CTX="gke_aishuk-test_us-west1-c_verl-research-cluster-west"
KUBECTL="/opt/homebrew/bin/kubectl --context ${CTX}"
IMAGE="verlai/verl:vllm020.dev2"
ROLLOUT_STEPS=15360
: "${MODE_NAME:?set MODE_NAME}" ; : "${STALENESS:?}" ; : "${SYNC_STEP:?}" ; : "${PARTIAL:?}"

# Pods are scheduled by pool label + anti-affinity; workers find heads via Services.

$KUBECTL delete configmap phase1-scripts --ignore-not-found
cat <<'CMEOF' | $KUBECTL apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: phase1-scripts
data:
  gpu_monitor.sh: |
    #!/usr/bin/env bash
    set -euo pipefail
    OUTPUT="${1:?Usage}"
    echo "timestamp_ms,gpu_index,gpu_util_pct,mem_util_pct,mem_used_mib,power_w" > "$OUTPUT"
    trap 'exit 0' SIGINT SIGTERM
    while true; do
        ts=$(date +%s%3N)
        nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw \
                   --format=csv,noheader,nounits 2>/dev/null | \
        while IFS=', ' read -r idx gpu mem memused pwr; do
            echo "${ts},${idx},${gpu},${mem},${memused},${pwr}"
        done >> "$OUTPUT"
        sleep 0.1
    done
  setup.sh: |
    #!/usr/bin/env bash
    set -euo pipefail
    pip install datasets 2>&1 | tail -1
    pip install "verl @ git+https://github.com/verl-project/verl.git@main" --no-deps 2>&1 | tail -2
    pip install cupy-cuda12x 2>&1 | tail -1
    python3 -c "import verl; print('verl ok')"
    hf download Qwen/Qwen2.5-Math-7B --local-dir /workspace/models/Qwen2.5-Math-7B 2>&1 | tail -1
    python3 -c "
    import json
    p = '/workspace/models/Qwen2.5-Math-7B/config.json'
    c = json.load(open(p)); c['max_position_embeddings'] = 32768
    json.dump(c, open(p,'w'), indent=2); print('config patched')
    "
    mkdir -p /workspace/data
    hf download BytedTsinghua-SIA/DAPO-Math-17k --repo-type dataset --local-dir /workspace/data/dapo 2>&1 | tail -1
    echo "Setup complete"
CMEOF

# ---- shared pod fragments -------------------------------------------------
common_train_args() {
  # $1=train_nnodes $2=train_gpus_pn $3=rollout_nnodes $4=rollout_gpus_pn $5=exp_name
  cat <<ARGS
        data.train_files=/workspace/data/dapo/data/dapo-math-17k.parquet \\
        data.val_files=/workspace/data/dapo/data/dapo-math-17k.parquet \\
        data.prompt_key=prompt data.truncation=left \\
        data.max_prompt_length=2048 data.max_response_length=8192 \\
        data.train_batch_size=0 data.gen_batch_size=1 data.return_raw_chat=True \\
        actor_rollout_ref.rollout.n=16 actor_rollout_ref.rollout.calculate_log_probs=True \\
        algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False algorithm.kl_ctrl.kl_coef=0.0 \\
        actor_rollout_ref.hybrid_engine=False \\
        actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0.0 \\
        actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 actor_rollout_ref.actor.clip_ratio_c=10.0 \\
        actor_rollout_ref.model.path=/workspace/models/Qwen2.5-Math-7B \\
        actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True \\
        +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \\
        actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.lr_warmup_steps=10 actor_rollout_ref.actor.optim.weight_decay=0.1 \\
        actor_rollout_ref.actor.ppo_mini_batch_size=32 actor_rollout_ref.actor.entropy_coeff=0 \\
        actor_rollout_ref.actor.grad_clip=1.0 actor_rollout_ref.actor.loss_agg_mode=token-mean \\
        actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \\
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \\
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \\
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=30720 \\
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=30720 \\
        actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \\
        actor_rollout_ref.actor.fsdp_config.fsdp_size=4 \\
        actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \\
        actor_rollout_ref.ref.fsdp_config.param_offload=True \\
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \\
        critic.strategy=fsdp2 \\
        actor_rollout_ref.rollout.gpu_memory_utilization=0.80 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\
        actor_rollout_ref.rollout.enable_chunked_prefill=True \\
        actor_rollout_ref.rollout.max_num_batched_tokens=10240 \\
        actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 \\
        actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \\
        actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \\
        actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024 \\
        reward.reward_manager.name=dapo \\
        +reward.reward_kwargs.overlong_buffer_cfg.enable=True +reward.reward_kwargs.overlong_buffer_cfg.len=4096 \\
        +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 +reward.reward_kwargs.overlong_buffer_cfg.log=False \\
        +reward.reward_kwargs.max_resp_len=8192 \\
        trainer.logger="['console']" trainer.val_before_train=False trainer.save_freq=-1 trainer.resume_mode=disable \\
        trainer.nnodes=$1 trainer.n_gpus_per_node=$2 trainer.total_epochs=10 trainer.test_freq=-1 \\
        rollout.nnodes=$3 rollout.n_gpus_per_node=$4 rollout.total_rollout_steps=${ROLLOUT_STEPS} \\
        trainer.project_name=phase1 trainer.experiment_name=$5 \\
        trainer.default_local_dir=/workspace/results/ckpts \\
        async_training.staleness_threshold=${STALENESS} async_training.trigger_parameter_sync_step=${SYNC_STEP} \\
        async_training.partial_rollout=${PARTIAL} async_training.require_batches=4
ARGS
}

make_head_pod() {
  # $1=pod_name $2=role_label $3=num_gpus $4=expected_total_gpus $5=train_args
  $KUBECTL apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: $1
  labels: {experiment: phase1, role: "$2"}
spec:
  restartPolicy: Never
  hostPID: true
  hostNetwork: true
  dnsPolicy: ClusterFirstWithHostNet
  nodeSelector: {cloud.google.com/gke-nodepool: h100-mega-8gpu-spot-a}
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector: {matchLabels: {experiment: phase1}}
        topologyKey: kubernetes.io/hostname
  tolerations:
  - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
  containers:
  - name: run
    image: ${IMAGE}
    securityContext:
      capabilities: {add: ["SYS_PTRACE"]}
      seccompProfile: {type: Unconfined}
    env:
    - {name: VLLM_USE_V1, value: "1"}
    - {name: PYTORCH_CUDA_ALLOC_CONF, value: "expandable_segments:True"}
    - {name: RAY_DEDUP_LOGS, value: "0"}
    command: ["/bin/bash", "-c"]
    args:
    - |
      set -euo pipefail
      bash /workspace/scripts/setup.sh
      ray start --head --port=6379 --num-gpus=$3
      for i in \$(seq 1 180); do
        GT=\$(ray status 2>/dev/null | grep -oP '[\d.]+/\K[\d.]+(?=\s+GPU)' || echo 0)
        python3 -c "exit(0 if float('\$GT')>=$4 else 1)" 2>/dev/null && { echo "cluster ready: \$GT GPUs"; break; }
        sleep 5
      done
      mkdir -p /workspace/results
      bash /workspace/scripts/gpu_monitor.sh /workspace/results/gpu_util_head.csv &
      date +%s%3N > /workspace/results/start_ts
      python3 -m verl.experimental.fully_async_policy.fully_async_main \\
$5 \\
        2>&1 | tee /workspace/results/train.log || true
      date +%s%3N > /workspace/results/end_ts
      echo "=== DONE ==="
      sleep 172800
    resources:
      limits: {nvidia.com/gpu: "$3"}
      requests: {nvidia.com/gpu: "$3", cpu: "16", memory: "96Gi"}
    volumeMounts:
    - {name: scripts, mountPath: /workspace/scripts}
    - {name: results, mountPath: /workspace/results}
    - {name: dshm, mountPath: /dev/shm}
  volumes:
  - {name: scripts, configMap: {name: phase1-scripts, defaultMode: 493}}
  - {name: results, emptyDir: {}}
  - {name: dshm, emptyDir: {medium: Memory, sizeLimit: 64Gi}}
EOF
}

make_worker_pod() {
  # $1=pod_name $2=head_service_dns
  $KUBECTL apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: $1
  labels: {experiment: phase1}
spec:
  restartPolicy: Never
  hostPID: true
  hostNetwork: true
  dnsPolicy: ClusterFirstWithHostNet
  nodeSelector: {cloud.google.com/gke-nodepool: h100-mega-8gpu-spot-a}
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector: {matchLabels: {experiment: phase1}}
        topologyKey: kubernetes.io/hostname
  tolerations:
  - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
  containers:
  - name: run
    image: ${IMAGE}
    securityContext:
      capabilities: {add: ["SYS_PTRACE"]}
      seccompProfile: {type: Unconfined}
    env:
    - {name: VLLM_USE_V1, value: "1"}
    - {name: PYTORCH_CUDA_ALLOC_CONF, value: "expandable_segments:True"}
    command: ["/bin/bash", "-c"]
    args:
    - |
      set -euo pipefail
      bash /workspace/scripts/setup.sh
      for i in \$(seq 1 240); do
        HEAD_IP=\$(getent hosts "$2" | awk '{print \$1}' | head -1)
        [ -n "\$HEAD_IP" ] && ray start --address="\$HEAD_IP:6379" --num-gpus=8 2>/dev/null && break
        sleep 5
      done
      mkdir -p /workspace/results
      bash /workspace/scripts/gpu_monitor.sh /workspace/results/gpu_util_worker.csv &
      while ray status >/dev/null 2>&1; do sleep 15; done
      echo "=== WORKER DONE ==="
      sleep 172800
    resources:
      limits: {nvidia.com/gpu: "8"}
      requests: {nvidia.com/gpu: "8", cpu: "16", memory: "96Gi"}
    volumeMounts:
    - {name: scripts, mountPath: /workspace/scripts}
    - {name: results, mountPath: /workspace/results}
    - {name: dshm, mountPath: /dev/shm}
  volumes:
  - {name: scripts, configMap: {name: phase1-scripts, defaultMode: 493}}
  - {name: results, emptyDir: {}}
  - {name: dshm, emptyDir: {medium: Memory, sizeLimit: 64Gi}}
EOF
}

# ---- services for head discovery -------------------------------------------
for svc in p1-88-head p1-412-head; do
$KUBECTL apply -f - <<EOF
apiVersion: v1
kind: Service
metadata: {name: ${svc}-svc}
spec:
  clusterIP: None
  selector: {role: ${svc}}
  ports: [{port: 6379, targetPort: 6379}]
EOF
done

# ---- launch the three runs (pods first; nodes may still be scaling up) ------
make_head_pod p1-44 p1-44 8 8 "$(common_train_args 1 4 1 4 dapo44_${MODE_NAME})"
make_head_pod p1-88-head p1-88-head 8 16 "$(common_train_args 1 8 1 8 dapo88_${MODE_NAME})"
make_worker_pod p1-88-worker "p1-88-head-svc.default.svc.cluster.local"
make_head_pod p1-412-head p1-412-head 8 16 "$(common_train_args 2 6 2 2 dapo412_${MODE_NAME} | sed 's/ppo_mini_batch_size=32/ppo_mini_batch_size=24/')"
make_worker_pod p1-412-worker "p1-412-head-svc.default.svc.cluster.local"

echo ""
$KUBECTL get pods -l experiment=phase1 -o wide
