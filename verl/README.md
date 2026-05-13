# Synchronous Time-Slicing RL Orchestrator PoC

## Concept Overview
Verl does not natively support synchronous Reinforcement Learning (RL) in a disaggregated architecture out of the box. 

This PoC orchestrates a custom **Synchronous RL Loop in Disaggregated Mode** tightly coupled with an active Time-Slicing orchestrator mapping concurrently to a shared GPU cluster:
1. **Sync RL Engine**: Modifies and overrides Verl's intrinsically async "off-by-one" deployment framework, replacing it with a custom synchronous execution loop that drives strictly interleaved Rollout and Training gracefully.
2. **Orchestrator Locks**: A dedicated FastAPI endpoint (`gpu-orchestrator`) enforces global structural Total Ordering across cluster semaphores (`[Trainer] -> [Sampler]`) rigidly securing true mutually-exclusive bounds gracefully across competing synchronous jobs targeting identical hardware.
3. **Daemon Eviction**: A privileged daemon observer (`node_daemon.py`) leverages explicit hooks tied to the orchestrator utilizing NVIDIA `cuda-checkpoint` to natively freeze and aggressively flush competing inactive Job VRAM pages temporarily to disk executing rigid hardware Time-Slicing robustly.
4. **Baseline Mode**: Alternatively, the pipeline supports `--mode=baseline`, collapsing the context-switching cleanly and defaulting purely to a fundamental synchronous runtime barrier.

### Issues Encountered & Bypassed
* **Ray Resource Deadlocks**: Ray inherently blocks multiple heavy GPU actors due to logical fractional ledger limitations (`PENDING` starvation). Within this PoC, Ray's internal logical GPU calculus was explicitly neutered to `0.01` internally inside `verl/single_controller/ray/base.py`, functionally allowing Ray to blindly map an unlimited volume of independent Job architectures concurrently, fully delegating actual hardware arbitration safely directly to the Orchestrator.

## How to Deploy

The unified deployment script dynamically sets up KubeRay, the Orchestrator, the Daemon, builds out physical GCSFuse mapped directories securely across pods, and automatically triggers natively configured baseline/timesliced workers identically. 

```bash
# Clean state deployment (installs cluster, mounts storage, establishes jobs natively)
bash deploy_verl.sh --mode=timeslice

# Incremental update (redeploys the orchestrator logic without nuking Ray resources)
bash deploy_verl.sh --mode=timeslice --skip-data --skip-ray --skip-gcs

# Run securely without time-slicing eviction routines (simple HTTP lock barrier only)
bash deploy_verl.sh --mode=baseline
```

## Kicking off Jobs

To manually submit the training jobs to the Ray cluster, you can use the following commands. These configurations intentionally restrict `actor_rollout_ref.rollout.max_num_seqs=8` to safely throttle generation throughput, dramatically extending the Phase wall-clock time without blowing up VRAM.

```bash
# Kick off Job A
kubectl exec -it -n $NAMESPACE verl-grpo-cluster-head-mj488 -- bash -c "
ray job submit \
  --address http://localhost:8265 \
  --runtime-env-json '{\"env_vars\": {
    \"PYTHONPATH\": \"/ray-deps:/data/verl:/data/verl/timeslice\",
    \"LD_LIBRARY_PATH\": \"/usr/local/nvidia/lib64:/usr/local/cuda/lib64\",
    \"TIMESLICE_JOB_ID\": \"job-a\",
    \"TIMESLICE_POOL_SAMPLER\": \"sampler\",
    \"TIMESLICE_POOL_TRAINER\": \"trainer\",
    \"TIMESLICE_SAMPLER_IP\": \"10.96.4.59\",
    \"TIMESLICE_TRAINER_IP\": \"10.96.4.58\"
  }}' \
  -- bash -c \"cd /data/verl && python3 timeslice/main_ppo_timeslice_sync.py \
     data.train_files=/data/data/gsm8k/train.parquet \
     data.val_files=/data/data/gsm8k/test.parquet \
     data.max_prompt_length=256 \
     data.max_response_length=1024 \
     data.train_batch_size=256 \
     data.val_batch_size=32 \
     actor_rollout_ref.model.path=/data/models/Qwen2.5-0.5B-Instruct \
     actor_rollout_ref.hybrid_engine=False \
     actor_rollout_ref.actor.strategy=fsdp \
     actor_rollout_ref.actor.optim.lr=1e-5 \
     actor_rollout_ref.actor.ppo_epochs=2 \
     actor_rollout_ref.actor.ppo_mini_batch_size=256 \
     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
     +actor_rollout_ref.actor.use_rollout_log_probs=False \
     actor_rollout_ref.rollout.name=vllm \
     actor_rollout_ref.rollout.free_cache_engine=False \
     actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
     actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
     actor_rollout_ref.rollout.n=16 \
     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
     algorithm.adv_estimator=grpo \
     algorithm.kl_ctrl.kl_coef=0.001 \
     trainer.n_gpus_per_node=1 \
     trainer.nnodes=1 \
     trainer.total_training_steps=10 \
     trainer.save_freq=-1 \
     trainer.test_freq=-1 \
     trainer.val_before_train=False \
     trainer.logger=console \
     rollout.n_gpus_per_node=1 \
     rollout.nnodes=1\
     actor_rollout_ref.rollout.max_num_seqs=8 \
     actor_rollout_ref.rollout.max_num_batched_tokens=4096 \" \
  2>&1 | tee /data/rl_logs/rl_job-a.log
cat /data/rl_logs/rl_job-a.log
"

# Kick off Job B
kubectl exec -it -n $NAMESPACE verl-grpo-cluster-head-mj488 -- bash -c "
ray job submit \
  --address http://localhost:8265 \
  --runtime-env-json '{\"env_vars\": {
    \"PYTHONPATH\": \"/ray-deps:/data/verl:/data/verl/timeslice\",
    \"LD_LIBRARY_PATH\": \"/usr/local/nvidia/lib64:/usr/local/cuda/lib64\",
    \"TIMESLICE_JOB_ID\": \"job-b\",
    \"TIMESLICE_POOL_SAMPLER\": \"sampler\",
    \"TIMESLICE_POOL_TRAINER\": \"trainer\",
    \"TIMESLICE_SAMPLER_IP\": \"10.96.4.59\",
    \"TIMESLICE_TRAINER_IP\": \"10.96.4.58\"
  }}' \
  -- bash -c \"cd /data/verl && python3 timeslice/main_ppo_timeslice_sync.py \
     data.train_files=/data/data/gsm8k/train.parquet \
     data.val_files=/data/data/gsm8k/test.parquet \
     data.max_prompt_length=256 \
     data.max_response_length=1024 \
     data.train_batch_size=256 \
     data.val_batch_size=32 \
     actor_rollout_ref.model.path=/data/models/Qwen2.5-0.5B-Instruct \
     actor_rollout_ref.hybrid_engine=False \
     actor_rollout_ref.actor.strategy=fsdp \
     actor_rollout_ref.actor.optim.lr=1e-5 \
     actor_rollout_ref.actor.ppo_epochs=2 \
     actor_rollout_ref.actor.ppo_mini_batch_size=256 \
     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
     +actor_rollout_ref.actor.use_rollout_log_probs=False \
     actor_rollout_ref.rollout.name=vllm \
     actor_rollout_ref.rollout.free_cache_engine=False \
     actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
     actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
     actor_rollout_ref.rollout.n=16 \
     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
     algorithm.adv_estimator=grpo \
     algorithm.kl_ctrl.kl_coef=0.001 \
     trainer.n_gpus_per_node=1 \
     trainer.nnodes=1 \
     trainer.total_training_steps=10 \
     trainer.save_freq=-1 \
     trainer.test_freq=-1 \
     trainer.val_before_train=False \
     trainer.logger=console \
     rollout.n_gpus_per_node=1 \
     rollout.nnodes=1\
     actor_rollout_ref.rollout.max_num_seqs=8 \
     actor_rollout_ref.rollout.max_num_batched_tokens=4096 \" \
  2>&1 | tee /data/rl_logs/rl_job-b.log
cat /data/rl_logs/rl_job-b.log
"
```

## How to Monitor & Export Logs

Once the orchestrator sequentially triggers jobs inside the cluster natively, you can explicitly hook the export module mechanically. It elegantly watches the head node seamlessly blocking natively until exactly when the `[RL_JOB_COMPLETED]` tag is printed!

```bash
# Block uniquely and securely natively trigger export once Job A and B sequentially formally finish
./export_logs.sh --watch "/data/rl_logs/rl_job-a.log,/data/rl_logs/rl_job-b.log"

# Run log exports abruptly manually ad-hoc
./export_logs.sh
```
*Note: Any natively parsed log metrics are fundamentally pushed sequentially to organically named payload directories (e.g. `rl_logs_export_timeslice_YYYYMMDD_HHMMSS/`) automatically.*

## Generating Dashboards

Using the automatically captured `/data/rl_logs/metrics_xyz.jsonl` traces explicitly generated during the export procedure organically:

```bash
# Point the dashboard generation script securely straight at the dynamically captured generated export directory
python3 dashboard_generator.py --dir ./rl_logs_export_timeslice_YOUR_TS
```
