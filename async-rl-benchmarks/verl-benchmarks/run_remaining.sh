#!/usr/bin/env bash
# Re-run multi-node recipes with proper worker CSV extraction via kubectl cp.
set -euo pipefail

KUBECTL="/opt/homebrew/bin/kubectl"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_BASE="${SCRIPT_DIR}/full_results"
IMAGE="verlai/verl:vllm020.dev2"

NODES=($($KUBECTL get nodes -l "cloud.google.com/gke-nodepool=h100-mega-8gpu-spot-a" -o jsonpath='{.items[*].metadata.name}'))
NODE1="${NODES[0]}"
NODE2="${NODES[1]}"
NODE1_IP=$($KUBECTL get node "$NODE1" -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}')
NODE2_IP=$($KUBECTL get node "$NODE2" -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}')
echo "Node1: $NODE1 ($NODE1_IP)"
echo "Node2: $NODE2 ($NODE2_IP)"

# Recipes needing worker data: recipe|train_gpus_pn|rollout_gpus_pn|nnodes|fsdp_size|gen_tp|sp_size
RECIPES=(
    "dapo_7b_8_8|8|8|2|2|1|1"
    "dapo_7b_4_12|6|2|2|2|1|1"
)

MODES=(
    "mode1_on_policy|0|4|False|4"
    "mode2_stream_offpolicy|0|16|False|4"
    "mode3_async_stale|0.3|16|False|4"
    "mode4_async_partial|0.3|16|True|4"
)

cleanup() {
    $KUBECTL delete pod ray-head ray-worker --force --ignore-not-found 2>/dev/null
    $KUBECTL delete svc ray-head-svc --ignore-not-found 2>/dev/null
    $KUBECTL delete configmap async-exp-scripts --ignore-not-found 2>/dev/null
}

for recipe_spec in "${RECIPES[@]}"; do
    IFS='|' read -r rname train_gpus_pn rollout_gpus_pn nnodes fsdp_size gen_tp sp_size <<< "$recipe_spec"

    for mode_spec in "${MODES[@]}"; do
        IFS='|' read -r mname staleness sync_step partial req_batches <<< "$mode_spec"
        run_name="${rname}_${mname}"
        run_dir="${RESULTS_BASE}/${run_name}"
        mkdir -p "$run_dir"

        echo ""
        echo "============================================================"
        echo " $run_name (${nnodes}x${train_gpus_pn}T + ${nnodes}x${rollout_gpus_pn}R)"
        echo "============================================================"

        cleanup

        # ConfigMap
        cat <<'CMEOF' | $KUBECTL apply -f - 2>/dev/null
apiVersion: v1
kind: ConfigMap
metadata:
  name: async-exp-scripts
data:
  gpu_monitor.sh: |
    #!/usr/bin/env bash
    set -euo pipefail
    OUTPUT="${1:?Usage}"
    INTERVAL_MS="${2:-100}"
    echo "timestamp_ms,gpu_index,gpu_util_pct,mem_util_pct,power_w" > "$OUTPUT"
    trap 'exit 0' SIGINT SIGTERM
    SLEEP_S=$(python3 -c "print($INTERVAL_MS/1000)")
    while true; do
        ts=$(date +%s%3N)
        nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,power.draw \
                   --format=csv,noheader,nounits 2>/dev/null | \
        while IFS=', ' read -r idx gpu mem pwr; do
            echo "${ts},${idx},${gpu},${mem},${pwr}"
        done >> "$OUTPUT"
        sleep "$SLEEP_S"
    done
CMEOF

        # Head pod
        cat <<HEOF | $KUBECTL apply -f - 2>/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: ray-head
spec:
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: ${NODE1}
  tolerations:
  - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
  hostNetwork: true
  dnsPolicy: ClusterFirstWithHostNet
  containers:
  - name: head
    image: ${IMAGE}
    env:
    - {name: VLLM_USE_V1, value: "1"}
    - {name: RAY_DEDUP_LOGS, value: "0"}
    command: ["/bin/bash", "-c"]
    args:
    - |
      set -euo pipefail
      pip install datasets 2>&1 | tail -1
      pip install "verl @ git+https://github.com/verl-project/verl.git@main" --no-deps 2>&1 | tail -1
      pip install cupy-cuda12x 2>&1 | tail -1

      MODEL_PATH="/workspace/models/Qwen2.5-7B-Instruct"
      [ -d "\$MODEL_PATH" ] || hf download Qwen/Qwen2.5-7B-Instruct --local-dir "\$MODEL_PATH" 2>&1
      mkdir -p /workspace/data/gsm8k
      [ -f /workspace/data/gsm8k/train.parquet ] || python3 -c "
      import json; from datasets import load_dataset
      d = load_dataset('openai/gsm8k', 'main')
      for s,p in [('train','/workspace/data/gsm8k/train.parquet'),('test','/workspace/data/gsm8k/test.parquet')]:
        ds = d[s].map(lambda x: {'prompt': json.dumps([{'role':'user','content':x['question']}]), 'data_source':'gsm8k'})
        ds.select_columns(['prompt','data_source']).to_parquet(p)
      "

      ray start --head --port=6379 --num-gpus=8
      for i in \$(seq 1 120); do
        GT=\$(ray status 2>/dev/null | grep -oP '[\d.]+/\K[\d.]+(?=\s+GPU)' || echo 0)
        python3 -c "exit(0 if float('\$GT')>=16 else 1)" 2>/dev/null && break
        sleep 5
      done
      ray status

      mkdir -p /workspace/results
      bash /workspace/scripts/gpu_monitor.sh /workspace/results/gpu_util_head.csv 100 &
      MP=\$!
      export VLLM_USE_V1=1

      python3 -m verl.experimental.fully_async_policy.fully_async_main \
        data.train_files=/workspace/data/gsm8k/train.parquet \
        data.val_files=/workspace/data/gsm8k/test.parquet \
        data.prompt_key=prompt data.truncation=left \
        data.max_prompt_length=2048 data.max_response_length=8192 \
        data.train_batch_size=0 data.gen_batch_size=1 data.return_raw_chat=True \
        actor_rollout_ref.rollout.n=16 actor_rollout_ref.rollout.calculate_log_probs=True \
        algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False algorithm.kl_ctrl.kl_coef=0.0 \
        actor_rollout_ref.hybrid_engine=False \
        actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.actor.clip_ratio_c=10.0 \
        actor_rollout_ref.model.path=/workspace/models/Qwen2.5-7B-Instruct \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
        actor_rollout_ref.actor.optim.weight_decay=0.1 \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.actor.loss_agg_mode=token-mean \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
        critic.strategy=fsdp2 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
        reward.reward_manager.name=dapo \
        +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
        +reward.reward_kwargs.overlong_buffer_cfg.len=4096 \
        +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
        +reward.reward_kwargs.overlong_buffer_cfg.log=False \
        +reward.reward_kwargs.max_resp_len=8192 \
        trainer.logger="['console']" trainer.val_before_train=False \
        trainer.save_freq=-1 trainer.resume_mode=disable \
        trainer.nnodes=${nnodes} trainer.n_gpus_per_node=${train_gpus_pn} \
        trainer.total_epochs=2 trainer.test_freq=-1 \
        rollout.nnodes=${nnodes} rollout.n_gpus_per_node=${rollout_gpus_pn} \
        rollout.total_rollout_steps=512 \
        trainer.project_name=timeslice-exp trainer.experiment_name=${run_name} \
        trainer.default_local_dir=/workspace/results/ckpts \
        async_training.staleness_threshold=${staleness} \
        async_training.trigger_parameter_sync_step=${sync_step} \
        async_training.partial_rollout=${partial} \
        async_training.require_batches=${req_batches} \
        2>&1 | tee /workspace/results/train.log || true

      kill \$MP 2>/dev/null || true; wait \$MP 2>/dev/null || true
      echo "=== HEAD DONE ==="
      sleep 600
    resources:
      limits: {nvidia.com/gpu: "8"}
      requests: {nvidia.com/gpu: "8", cpu: "16", memory: "96Gi"}
    volumeMounts:
    - {name: scripts, mountPath: /workspace/scripts}
    - {name: results, mountPath: /workspace/results}
    - {name: dshm, mountPath: /dev/shm}
  volumes:
  - {name: scripts, configMap: {name: async-exp-scripts, defaultMode: 493}}
  - {name: results, emptyDir: {}}
  - {name: dshm, emptyDir: {medium: Memory, sizeLimit: 64Gi}}
HEOF

        # Worker pod
        cat <<WEOF | $KUBECTL apply -f - 2>/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: ray-worker
spec:
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: ${NODE2}
  tolerations:
  - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
  hostNetwork: true
  dnsPolicy: ClusterFirstWithHostNet
  containers:
  - name: worker
    image: ${IMAGE}
    env:
    - {name: VLLM_USE_V1, value: "1"}
    command: ["/bin/bash", "-c"]
    args:
    - |
      set -euo pipefail
      pip install datasets 2>&1 | tail -1
      pip install "verl @ git+https://github.com/verl-project/verl.git@main" --no-deps 2>&1 | tail -1
      pip install cupy-cuda12x 2>&1 | tail -1
      MODEL_PATH="/workspace/models/Qwen2.5-7B-Instruct"
      [ -d "\$MODEL_PATH" ] || hf download Qwen/Qwen2.5-7B-Instruct --local-dir "\$MODEL_PATH" 2>&1
      mkdir -p /workspace/data/gsm8k
      [ -f /workspace/data/gsm8k/train.parquet ] || python3 -c "
      import json; from datasets import load_dataset
      d = load_dataset('openai/gsm8k', 'main')
      for s,p in [('train','/workspace/data/gsm8k/train.parquet'),('test','/workspace/data/gsm8k/test.parquet')]:
        ds = d[s].map(lambda x: {'prompt': json.dumps([{'role':'user','content':x['question']}]), 'data_source':'gsm8k'})
        ds.select_columns(['prompt','data_source']).to_parquet(p)
      "
      for i in \$(seq 1 60); do
        ray start --address="${NODE1_IP}:6379" --num-gpus=8 2>/dev/null && break
        sleep 5
      done
      mkdir -p /workspace/results
      bash /workspace/scripts/gpu_monitor.sh /workspace/results/gpu_util_worker.csv 100 &
      while ray status >/dev/null 2>&1; do sleep 10; done
      echo "WORKER DONE"
      sleep 600
    resources:
      limits: {nvidia.com/gpu: "8"}
      requests: {nvidia.com/gpu: "8", cpu: "16", memory: "96Gi"}
    volumeMounts:
    - {name: scripts, mountPath: /workspace/scripts}
    - {name: results, mountPath: /workspace/results}
    - {name: dshm, mountPath: /dev/shm}
  volumes:
  - {name: scripts, configMap: {name: async-exp-scripts, defaultMode: 493}}
  - {name: results, emptyDir: {}}
  - {name: dshm, emptyDir: {medium: Memory, sizeLimit: 64Gi}}
WEOF

        # Wait for head to finish training (look for HEAD DONE)
        echo "Waiting for run to complete..."
        for i in $(seq 1 120); do
            if $KUBECTL logs ray-head --tail=3 2>/dev/null | grep -q "HEAD DONE"; then
                echo "Training done"
                break
            fi
            phase=$($KUBECTL get pod ray-head -o jsonpath='{.status.phase}' 2>/dev/null)
            if [ "$phase" = "Failed" ] || [ "$phase" = "Error" ]; then
                echo "Head pod failed"
                break
            fi
            sleep 15
        done

        # Extract CSVs via kubectl cp (not Ray remote)
        echo "Extracting head CSV..."
        $KUBECTL cp ray-head:/workspace/results/gpu_util_head.csv "$run_dir/gpu_util_head.csv" 2>/dev/null
        echo "Extracting worker CSV..."
        $KUBECTL cp ray-worker:/workspace/results/gpu_util_worker.csv "$run_dir/gpu_util_worker.csv" 2>/dev/null

        echo "CSVs collected:"
        ls -la "$run_dir"/gpu_util_*.csv 2>/dev/null

        cleanup
        echo "Done: $run_name"
    done
done

echo ""
echo "All remaining runs complete."
