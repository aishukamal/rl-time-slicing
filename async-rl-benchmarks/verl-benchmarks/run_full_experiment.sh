#!/usr/bin/env bash
# Run all async RL recipes × all 4 modes on h100-mega-8gpu-spot-a pool.
# Collects raw nvidia-smi CSVs from both nodes for each run.
#
# Recipes:
#   dapo_7b_4_4:  4 train + 4 rollout = 8 GPUs  (1 node)
#   dapo_7b_8_8:  8 train + 8 rollout = 16 GPUs (2 nodes)
#   dapo_7b_4_12: 12 train + 4 rollout = 16 GPUs (2 nodes)
#
# Each recipe runs 4 modes (on-policy, stream off-policy, async+stale, async+partial)

set -euo pipefail

KUBECTL="/opt/homebrew/bin/kubectl"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_BASE="${SCRIPT_DIR}/full_results"
mkdir -p "$RESULTS_BASE"

NODE_POOL="h100-mega-8gpu-spot-a"
# Discover nodes dynamically from the pool
NODES=($($KUBECTL get nodes -l "cloud.google.com/gke-nodepool=${NODE_POOL}" -o jsonpath='{.items[*].metadata.name}'))
if [ ${#NODES[@]} -lt 1 ]; then
    echo "ERROR: No nodes in pool ${NODE_POOL}"
    exit 1
fi
NODE1="${NODES[0]}"
NODE2="${NODES[1]:-${NODES[0]}}"
NODE1_IP=$($KUBECTL get node "$NODE1" -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}')
NODE2_IP=$($KUBECTL get node "$NODE2" -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}')
echo "Node1: $NODE1 ($NODE1_IP)"
echo "Node2: $NODE2 ($NODE2_IP)"
IMAGE="verlai/verl:vllm020.dev2"

# Mode configs: name|staleness|sync_step|partial|require_batches
MODES=(
    "mode1_on_policy|0|4|False|4"
    "mode2_stream_offpolicy|0|16|False|4"
    "mode3_async_stale|0.3|16|False|4"
    "mode4_async_partial|0.3|16|True|4"
)

# Recipe configs: name|train_gpus_per_node|rollout_gpus_per_node|nnodes|fsdp_size|gen_tp|sp_size
RECIPES=(
    "dapo_7b_4_4|4|4|1|2|1|1"
    "dapo_7b_8_8|8|8|1|2|1|1"
    "dapo_7b_4_12|4|4|2|2|1|1"
)
# Note: dapo_7b_4_12 is 2 nodes, 6 train + 2 rollout per node in the original recipe.
# But veRL async splits trainer and rollouter into separate resource pools.
# With nnodes=2: trainer.nnodes=2 trainer.n_gpus_per_node=4 → 8 train GPUs total (originally 12)
# Adjusting: for 4_12 we use trainer.nnodes=1 + 4gpus and rollout.nnodes=1 + 4gpus on a single 8gpu node,
# but to get 12+4 we need 2 nodes differently. Let's stick to what maps cleanly.

# Actually, re-reading the 4_12 recipe: NNODES=2, n_gpus_training=6, n_gpus_rollout=2 per node
# So: trainer total = 2*6=12, rollout total = 2*2=4
# With our 2 nodes of 8 GPUs each, this maps to: each node has 6 train + 2 rollout GPUs
# veRL handles this via trainer.nnodes=2 trainer.n_gpus_per_node=6 rollout.nnodes=2 rollout.n_gpus_per_node=2

# Updated recipes:
RECIPES=(
    "dapo_7b_4_4|4|4|1|2|1|1"
    "dapo_7b_8_8|8|8|2|2|1|1"
    "dapo_7b_4_12|6|2|2|2|1|1"
)
# dapo_7b_8_8: trainer.nnodes=1 n_gpus=8, rollout.nnodes=1 n_gpus=8 → each on separate node
# dapo_7b_4_12: trainer.nnodes=2 n_gpus=6, rollout.nnodes=2 n_gpus=2 → shared nodes, GPU split

cleanup() {
    echo "Cleaning up..."
    $KUBECTL delete pod ray-head ray-worker --force --ignore-not-found 2>/dev/null
    $KUBECTL delete svc ray-head-svc --ignore-not-found 2>/dev/null
    $KUBECTL delete configmap async-exp-scripts --ignore-not-found 2>/dev/null
}

wait_for_pod() {
    local pod="$1" timeout="${2:-300}"
    for i in $(seq 1 $timeout); do
        phase=$($KUBECTL get pod "$pod" -o jsonpath='{.status.phase}' 2>/dev/null)
        if [ "$phase" = "Running" ]; then return 0; fi
        if [ "$phase" = "Succeeded" ] || [ "$phase" = "Failed" ] || [ "$phase" = "Error" ]; then return 1; fi
        sleep 1
    done
    return 1
}

run_experiment() {
    local recipe="$1" mode_spec="$2"
    IFS='|' read -r rname train_gpus_pn rollout_gpus_pn nnodes fsdp_size gen_tp sp_size <<< "$recipe"
    IFS='|' read -r mname staleness sync_step partial req_batches <<< "$mode_spec"

    local run_name="${rname}_${mname}"
    local run_dir="${RESULTS_BASE}/${run_name}"
    mkdir -p "$run_dir"

    echo ""
    echo "============================================================"
    echo " Recipe: $rname  Mode: $mname"
    echo " Trainer: ${nnodes}×${train_gpus_pn} GPUs  Rollouter: ${nnodes}×${rollout_gpus_pn} GPUs"
    echo " staleness=$staleness sync=$sync_step partial=$partial batches=$req_batches"
    echo "============================================================"

    cleanup

    # Determine node allocation
    local total_gpus=$(( (train_gpus_pn + rollout_gpus_pn) * nnodes ))
    local use_two_nodes=false
    if [ "$total_gpus" -gt 8 ] || [ "$nnodes" -gt 1 ]; then
        use_two_nodes=true
    fi

    # Create ConfigMap with the run script
    cat <<CMEOF | $KUBECTL apply -f - 2>/dev/null
apiVersion: v1
kind: ConfigMap
metadata:
  name: async-exp-scripts
data:
  gpu_monitor.sh: |
    #!/usr/bin/env bash
    set -euo pipefail
    OUTPUT="\${1:?Usage}"
    INTERVAL_MS="\${2:-100}"
    echo "timestamp_ms,gpu_index,gpu_util_pct,mem_util_pct,power_w" > "\$OUTPUT"
    trap 'exit 0' SIGINT SIGTERM
    SLEEP_S=\$(python3 -c "print(\$INTERVAL_MS/1000)")
    while true; do
        ts=\$(date +%s%3N)
        nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,power.draw \
                   --format=csv,noheader,nounits 2>/dev/null | \
        while IFS=', ' read -r idx gpu mem pwr; do
            echo "\${ts},\${idx},\${gpu},\${mem},\${pwr}"
        done >> "\$OUTPUT"
        sleep "\$SLEEP_S"
    done
  collect_worker_gpu.py: |
    #!/usr/bin/env python3
    import ray, sys
    ray.init(address="auto", ignore_reinit_error=True)
    @ray.remote(num_cpus=0.01)
    def read_csv():
        try:
            return open("/workspace/results/gpu_util_worker.csv").read()
        except:
            return ""
    worker_ip = sys.argv[2] if len(sys.argv) > 2 else "10.138.0.48"
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/worker_gpu.csv"
    nodes = [n for n in ray.nodes() if n["Alive"] and n["NodeManagerAddress"] == worker_ip]
    if nodes:
        nid = nodes[0]["NodeID"]
        strat = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(node_id=nid, soft=False)
        data = ray.get(read_csv.options(scheduling_strategy=strat).remote())
        if data:
            with open(out_path, "w") as f:
                f.write(data)
            print("Collected " + str(len(data)) + " bytes -> " + out_path)
        else:
            print("Worker GPU trace empty")
    else:
        print("Worker node not found at " + worker_ip)
CMEOF

    # Create head pod
    local head_node="$NODE1"
    local worker_node="$NODE2"
    local head_gpus=8
    local worker_gpus=8

    if [ "$use_two_nodes" = false ]; then
        head_gpus=8
        worker_gpus=0
    fi

    cat <<HEOF | $KUBECTL apply -f - 2>/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: ray-head
  labels:
    app: async-exp-head
spec:
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: ${head_node}
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  hostNetwork: true
  dnsPolicy: ClusterFirstWithHostNet
  containers:
  - name: head
    image: ${IMAGE}
    env:
    - name: VLLM_USE_V1
      value: "1"
    - name: RAY_DEDUP_LOGS
      value: "0"
    command: ["/bin/bash", "-c"]
    args:
    - |
      set -euo pipefail

      # Install deps
      pip install datasets 2>&1 | tail -2
      pip install "verl @ git+https://github.com/verl-project/verl.git@main" --no-deps 2>&1 | tail -2
      pip install cupy-cuda12x 2>&1 | tail -2
      python3 -c "import verl; print('verl ok')"

      # Download model + data
      MODEL_PATH="/workspace/models/Qwen2.5-7B-Instruct"
      if [ ! -d "\$MODEL_PATH" ]; then
          hf download Qwen/Qwen2.5-7B-Instruct --local-dir "\$MODEL_PATH" 2>&1 || \
            python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='\$MODEL_PATH')"
      fi
      TRAIN_FILE="/workspace/data/gsm8k/train.parquet"
      if [ ! -f "\$TRAIN_FILE" ]; then
          mkdir -p /workspace/data/gsm8k
          python3 -c "
      import json
      from datasets import load_dataset
      d = load_dataset('openai/gsm8k', 'main')
      for split, path in [('train', '/workspace/data/gsm8k/train.parquet'), ('test', '/workspace/data/gsm8k/test.parquet')]:
          ds = d[split].map(lambda ex: {'prompt': json.dumps([{'role': 'user', 'content': ex['question']}]), 'data_source': 'gsm8k'})
          ds = ds.select_columns(['prompt', 'data_source'])
          ds.to_parquet(path)
          print(f'{split}: {len(ds)} samples')
      "
      fi

      # Start Ray head
      ray start --head --port=6379 --num-gpus=${head_gpus}
      echo "Ray head started"

      # Wait for cluster (if multi-node)
      EXPECTED_GPUS=$((${head_gpus} + ${worker_gpus}))
      for i in \$(seq 1 120); do
          GPU_TOTAL=\$(ray status 2>/dev/null | grep -oP '[\d.]+/\K[\d.]+(?=\s+GPU)' || echo "0")
          if python3 -c "exit(0 if float('\${GPU_TOTAL}') >= \${EXPECTED_GPUS} else 1)" 2>/dev/null; then
              echo "Ray cluster ready: \${GPU_TOTAL} GPUs"
              break
          fi
          echo "Waiting... (\$i/120, GPUs: \${GPU_TOTAL}/\${EXPECTED_GPUS})"
          sleep 5
      done

      # Start GPU monitor
      mkdir -p /workspace/results
      bash /workspace/scripts/gpu_monitor.sh /workspace/results/gpu_util_head.csv 100 &
      MONITOR_PID=\$!

      # Run veRL
      export VLLM_USE_V1=1
      date +%s%3N > /workspace/results/start_ts

      python3 -m verl.experimental.fully_async_policy.fully_async_main \
          data.train_files=/workspace/data/gsm8k/train.parquet \
          data.val_files=/workspace/data/gsm8k/test.parquet \
          data.prompt_key=prompt \
          data.truncation=left \
          data.max_prompt_length=2048 \
          data.max_response_length=8192 \
          data.train_batch_size=0 \
          data.gen_batch_size=1 \
          data.return_raw_chat=True \
          actor_rollout_ref.rollout.n=16 \
          actor_rollout_ref.rollout.calculate_log_probs=True \
          algorithm.adv_estimator=grpo \
          algorithm.use_kl_in_reward=False \
          algorithm.kl_ctrl.kl_coef=0.0 \
          actor_rollout_ref.hybrid_engine=False \
          actor_rollout_ref.actor.use_kl_loss=False \
          actor_rollout_ref.actor.kl_loss_coef=0.0 \
          actor_rollout_ref.actor.clip_ratio_low=0.2 \
          actor_rollout_ref.actor.clip_ratio_high=0.28 \
          actor_rollout_ref.actor.clip_ratio_c=10.0 \
          actor_rollout_ref.model.path=/workspace/models/Qwen2.5-7B-Instruct \
          actor_rollout_ref.model.use_remove_padding=True \
          actor_rollout_ref.model.enable_gradient_checkpointing=True \
          actor_rollout_ref.actor.optim.lr=1e-6 \
          actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
          actor_rollout_ref.actor.optim.weight_decay=0.1 \
          actor_rollout_ref.actor.ppo_mini_batch_size=32 \
          actor_rollout_ref.actor.entropy_coeff=0 \
          actor_rollout_ref.actor.grad_clip=1.0 \
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
          actor_rollout_ref.rollout.temperature=1.0 \
          actor_rollout_ref.rollout.top_p=1.0 \
          actor_rollout_ref.rollout.top_k=-1 \
          actor_rollout_ref.rollout.name=vllm \
          actor_rollout_ref.rollout.mode=async \
          actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
          reward.reward_manager.name=dapo \
          +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
          +reward.reward_kwargs.overlong_buffer_cfg.len=4096 \
          +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
          +reward.reward_kwargs.overlong_buffer_cfg.log=False \
          +reward.reward_kwargs.max_resp_len=8192 \
          trainer.logger="['console']" \
          trainer.val_before_train=False \
          trainer.save_freq=-1 \
          trainer.resume_mode=disable \
          trainer.nnodes=${nnodes} \
          trainer.n_gpus_per_node=${train_gpus_pn} \
          trainer.total_epochs=2 \
          trainer.test_freq=-1 \
          rollout.nnodes=${nnodes} \
          rollout.n_gpus_per_node=${rollout_gpus_pn} \
          rollout.total_rollout_steps=512 \
          trainer.project_name=timeslice-exp \
          trainer.experiment_name=${run_name} \
          trainer.default_local_dir=/workspace/results/ckpts \
          async_training.staleness_threshold=${staleness} \
          async_training.trigger_parameter_sync_step=${sync_step} \
          async_training.partial_rollout=${partial} \
          async_training.require_batches=${req_batches} \
          2>&1 | tee /workspace/results/train.log || echo "Training completed (or failed)"

      date +%s%3N > /workspace/results/end_ts

      # Stop monitor
      kill \$MONITOR_PID 2>/dev/null || true
      wait \$MONITOR_PID 2>/dev/null || true

      # Collect worker GPU trace if multi-node
      if [ "${use_two_nodes}" = "true" ]; then
          python3 /workspace/scripts/collect_worker_gpu.py /workspace/results/gpu_util_worker.csv ${worker_node_ip:-$NODE2_IP} 2>&1 || echo "Worker collection failed"
      fi

      echo "=== Run complete: ${run_name} ==="
      ls -la /workspace/results/gpu_util_*.csv 2>/dev/null
      # Keep alive for extraction
      sleep 300
    resources:
      limits:
        nvidia.com/gpu: ${head_gpus}
      requests:
        nvidia.com/gpu: ${head_gpus}
        cpu: "16"
        memory: "96Gi"
    volumeMounts:
    - name: scripts
      mountPath: /workspace/scripts
    - name: results
      mountPath: /workspace/results
    - name: dshm
      mountPath: /dev/shm
  volumes:
  - name: scripts
    configMap:
      name: async-exp-scripts
      defaultMode: 0755
  - name: results
    emptyDir: {}
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 64Gi
HEOF

    # Create worker pod if multi-node
    if [ "$use_two_nodes" = true ]; then
        cat <<WEOF | $KUBECTL apply -f - 2>/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: ray-worker
  labels:
    app: async-exp-worker
spec:
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: ${worker_node}
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  hostNetwork: true
  dnsPolicy: ClusterFirstWithHostNet
  containers:
  - name: worker
    image: ${IMAGE}
    env:
    - name: VLLM_USE_V1
      value: "1"
    command: ["/bin/bash", "-c"]
    args:
    - |
      set -euo pipefail
      pip install datasets 2>&1 | tail -2
      pip install "verl @ git+https://github.com/verl-project/verl.git@main" --no-deps 2>&1 | tail -2
      pip install cupy-cuda12x 2>&1 | tail -2

      # Download model (needed on worker too for FSDP)
      MODEL_PATH="/workspace/models/Qwen2.5-7B-Instruct"
      if [ ! -d "\$MODEL_PATH" ]; then
          hf download Qwen/Qwen2.5-7B-Instruct --local-dir "\$MODEL_PATH" 2>&1 || \
            python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='\$MODEL_PATH')"
      fi

      # Data
      mkdir -p /workspace/data/gsm8k
      TRAIN_FILE="/workspace/data/gsm8k/train.parquet"
      if [ ! -f "\$TRAIN_FILE" ]; then
          python3 -c "
      import json
      from datasets import load_dataset
      d = load_dataset('openai/gsm8k', 'main')
      for split, path in [('train', '/workspace/data/gsm8k/train.parquet'), ('test', '/workspace/data/gsm8k/test.parquet')]:
          ds = d[split].map(lambda ex: {'prompt': json.dumps([{'role': 'user', 'content': ex['question']}]), 'data_source': 'gsm8k'})
          ds = ds.select_columns(['prompt', 'data_source'])
          ds.to_parquet(path)
      "
      fi

      # Connect to Ray head
      for i in \$(seq 1 60); do
          if ray start --address="${NODE1_IP}:6379" --num-gpus=${worker_gpus} 2>/dev/null; then
              echo "Connected to Ray head"
              break
          fi
          sleep 5
      done

      # Start GPU monitor
      mkdir -p /workspace/results
      bash /workspace/scripts/gpu_monitor.sh /workspace/results/gpu_util_worker.csv 100 &

      # Block until head finishes
      while ray status >/dev/null 2>&1; do sleep 10; done
      echo "Worker done"
      sleep 30
    resources:
      limits:
        nvidia.com/gpu: ${worker_gpus}
      requests:
        nvidia.com/gpu: ${worker_gpus}
        cpu: "16"
        memory: "96Gi"
    volumeMounts:
    - name: scripts
      mountPath: /workspace/scripts
    - name: results
      mountPath: /workspace/results
    - name: dshm
      mountPath: /dev/shm
  volumes:
  - name: scripts
    configMap:
      name: async-exp-scripts
      defaultMode: 0755
  - name: results
    emptyDir: {}
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 64Gi
WEOF
    fi

    # Wait for head to start
    echo "Waiting for head pod..."
    if ! wait_for_pod ray-head 600; then
        echo "ERROR: Head pod failed to start"
        $KUBECTL logs ray-head --tail=20 2>/dev/null
        return 1
    fi

    # Wait for completion (check every 30s, timeout 30min)
    echo "Experiment running..."
    for i in $(seq 1 60); do
        phase=$($KUBECTL get pod ray-head -o jsonpath='{.status.phase}' 2>/dev/null)
        if [ "$phase" != "Running" ]; then
            echo "Head pod finished: $phase"
            break
        fi
        # Check for completion marker
        if $KUBECTL logs ray-head --tail=5 2>/dev/null | grep -q "Run complete"; then
            echo "Run complete detected"
            break
        fi
        sleep 30
    done

    # Extract results
    echo "Extracting results..."
    $KUBECTL cp ray-head:/workspace/results "$run_dir" 2>/dev/null
    echo "Results saved to $run_dir/"
    ls -la "$run_dir"/gpu_util_*.csv 2>/dev/null

    cleanup
    echo "Done: $run_name"
}

# Main loop
echo "Starting full async RL experiment"
echo "Recipes: ${#RECIPES[@]}, Modes: ${#MODES[@]}, Total runs: $(( ${#RECIPES[@]} * ${#MODES[@]} ))"
echo ""

for recipe in "${RECIPES[@]}"; do
    for mode in "${MODES[@]}"; do
        run_experiment "$recipe" "$mode"
    done
done

echo ""
echo "============================================================"
echo " All experiments complete. Results in $RESULTS_BASE"
echo "============================================================"
ls -d "$RESULTS_BASE"/*/
