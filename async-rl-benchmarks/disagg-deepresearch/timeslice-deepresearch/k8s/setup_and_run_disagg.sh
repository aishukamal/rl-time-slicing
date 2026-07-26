#!/usr/bin/env bash
# Disaggregated deep-research benchmark — main container entrypoint.
# 1 trainer GPU (FSDP) + 1 sampler GPU (server-mode vLLM), CMU multi-turn
# rollout in the sampler phase, local Wikipedia search sidecar on :8877.
set -xeuo pipefail

# On ANY exit, kill background children (gpu monitor). Otherwise they hold
# the stdout pipe to tee open and the container hangs instead of surfacing
# the failure in pod status.
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

RESULTS_DIR="/workspace/results"
MODEL_PATH="/workspace/models/Qwen2.5-3B-Instruct"
SEARCH_PORT=8877
# Pinned commits — the port was written and API-verified against these exact
# trees. Do NOT float to HEAD (AgentLoopManager/CheckpointEngineManager APIs
# drift; see timeslice-deepresearch docstrings).
VERL_COMMIT="a35908ca3c9632859c58d6a2855d858918ae21dc"
CMU_COMMIT="9c311053d607d9c63d3f148f03f712cc6469a52d"

mkdir -p "$RESULTS_DIR" "$RESULTS_DIR/rl_logs"

echo "=== Phase 1: Runtime deps ==="
# torchdata: StatefulDataLoader (same runtime install the PoC's RayCluster
# init-container does). openai: env-side LLM-judge client (fails fast on the
# dummy key -> score 0, matching the colocated baseline). Neither touches torch.
python3 -m pip install --no-cache-dir torchdata openai 2>&1 | tail -2
python3 -c "import torchdata.stateful_dataloader, openai; print('runtime deps OK')"

echo "=== Phase 2: Pinned verl clone + pod patches ==="
if [ ! -d /opt/verl ]; then
    git clone --filter=blob:none https://github.com/volcengine/verl.git /opt/verl
fi
git -C /opt/verl fetch origin "$VERL_COMMIT" || git -C /opt/verl fetch origin
git -C /opt/verl checkout "$VERL_COMMIT"
# Same live patches deploy_verl.sh applies to worker pods:
# layered_summon / peft_merge may be missing on DetachActorWorker.
sed -i 's/layered_summon=self\.layered_summon/layered_summon=getattr(self, "layered_summon", False)/g' \
    /opt/verl/verl/workers/engine_workers.py
sed -i 's/if not self\.peft_merge/if not getattr(self, "peft_merge", False)/g' \
    /opt/verl/verl/workers/engine_workers.py
# Make sure the image's preinstalled verl can't shadow the pinned clone.
python3 -m pip uninstall -y verl 2>/dev/null | tail -1 || true

echo "=== Phase 3: Assemble timeslice-deepresearch code tree ==="
# The code ConfigMap uses flat keys with '--' as the path separator
# (ConfigMap keys cannot contain '/'; '__' would collide with __init__.py).
rm -rf /opt/timeslice && mkdir -p /opt/timeslice
for f in /workspace/code/*; do
    name="$(basename "$f")"
    dest="/opt/timeslice/${name//--//}"
    mkdir -p "$(dirname "$dest")"
    cp "$f" "$dest"
done
find /opt/timeslice -name "*.py" | wc -l

echo "=== Phase 4: MHQA question data from pinned CMU repo ==="
CMU_DIR=/opt/cmu-deepresearch
if [ ! -d "$CMU_DIR/.git" ]; then
    mkdir -p "$CMU_DIR" && cd "$CMU_DIR"
    git init -q
    git remote add origin https://github.com/cxcscmu/verl-agent-deepresearch.git
    git fetch --depth 1 origin "$CMU_COMMIT"
    git checkout -q FETCH_HEAD
fi
DATA_SRC="$CMU_DIR/agent_system/environments/env_package/deepresearch/deepresearch/data/deepresearch_mhqa"
DATA_DST="/opt/timeslice/agent_system/environments/env_package/deepresearch/deepresearch/data/deepresearch_mhqa"
mkdir -p "$DATA_DST"
cp "$DATA_SRC"/*.json "$DATA_DST/"
ls -la "$DATA_DST"

echo "=== Phase 5: Download model ==="
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-3B-Instruct', local_dir='$MODEL_PATH')
print('Model download complete')
"

echo "=== Phase 6: Wait for search sidecar (:$SEARCH_PORT) ==="
python3 - << 'PYWAIT'
import json, time, urllib.request
url = "http://localhost:8877/search"
deadline = time.time() + 2400  # 12GB index download on first run
while time.time() < deadline:
    try:
        req = urllib.request.Request(
            url, data=json.dumps({"q": "capital of France"}).encode(),
            headers={"Content-Type": "application/json"})
        body = urllib.request.urlopen(req, timeout=10).read()
        organic = json.loads(body).get("organic", [])
        print(f"Search server ready, {len(organic)} results for sanity query")
        print(str(body[:300]))
        raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        time.sleep(5)
raise SystemExit("FATAL: search sidecar not ready after 2400s")
PYWAIT

echo "=== Phase 7: Prepare parquet (sizing + raw_prompt vehicle) ==="
python3 /opt/timeslice/prepare_deepresearch_data.py \
    --train_json "$DATA_DST/train.json" \
    --val_json "$DATA_DST/val.json" \
    --out_dir /workspace/data \
    --train_data_size 128 \
    --val_data_size 64

echo "=== Phase 8: Start GPU monitor (100ms) ==="
bash /workspace/scripts/gpu_monitor.sh "$RESULTS_DIR/gpu_util.csv" 100 &
MONITOR_PID=$!
date +%s%3N > "$RESULTS_DIR/start_ts"

echo "=== Phase 9: Training ==="
echo "  Model: Qwen2.5-3B-Instruct | Layout: 1 trainer GPU + 1 sampler GPU (TP=1)"
echo "  Multi-turn: env.max_steps=6, env_num=8, group_n=4 (32 concurrent envs)"
echo "  Data: MHQA, train_data_size=128 -> 16 steps"
nvidia-smi

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TIMESLICE_DISABLED=1
export TIMESLICE_JOB_ID=job-disagg
export VERL_ROOT=/opt/verl
export PYTHONPATH="/opt/verl:/opt/timeslice:${PYTHONPATH:-}"
export SEARCH_URL="http://localhost:$SEARCH_PORT/search"
export DEEPRESEARCH_TOKENIZER_PATH="$MODEL_PATH"
export DEEPRESEARCH_LOG_ROOT="$RESULTS_DIR"
export LOG_DIR="$RESULTS_DIR/rl_logs"
export OPENAI_API_KEY="dummy-not-used"
export GOOGLE_API_KEY="dummy-not-used"
export GEMINI_API_KEY="dummy-not-used"

# hydra searchpath in one_step_off_ppo_trainer.yaml is CWD-relative
cd /opt/verl
python3 /opt/timeslice/main_ppo_timeslice_sync.py \
    data.train_files=/workspace/data/train.parquet \
    data.val_files=/workspace/data/val.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=16 \
    data.max_prompt_length=10000 \
    data.max_response_length=1024 \
    data.return_raw_chat=True \
    data.truncation=left \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=12288 \
    actor_rollout_ref.rollout.max_num_batched_tokens=12288 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.bypass_mode=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    rollout.n_gpus_per_node=1 \
    rollout.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.logger=console \
    trainer.project_name=disagg-deepresearch \
    trainer.experiment_name=qwen3b-mhqa-disagg \
    +env.env_name=deepresearch \
    +env.dataset=deepresearch_mhqa \
    +env.max_steps=6 \
    +env.rollout.n=4 \
    +env.seed=0 \
    +env.use_explicit_thinking=True \
    +env.is_evaluation=False \
    +env.use_critique=False \
    +env.use_rule_reward=False \
    +env.rule_reward_coef=0 \
    +env.use_dense_reward=False \
    +env.rule_number=5 \
    2>&1 | tee "$RESULTS_DIR/train.log" || {
        echo "WARNING: Training failed"
        date +%s%3N > "$RESULTS_DIR/end_ts"
        kill "$MONITOR_PID" 2>/dev/null || true
        echo "=== Partial results ==="
        wc -l "$RESULTS_DIR/gpu_util.csv" || true
        echo "=== GPU trace dump (gzip+base64, prefix GPUCSV:) ==="
        echo "GPUCSV-BEGIN md5=$(md5sum "$RESULTS_DIR/gpu_util.csv" | cut -d' ' -f1)"
        gzip -c "$RESULTS_DIR/gpu_util.csv" | base64 -w 3000 | sed 's/^/GPUCSV:/' || true
        echo "GPUCSV-END"
        sleep 7200
        exit 1
    }

date +%s%3N > "$RESULTS_DIR/end_ts"
kill "$MONITOR_PID" 2>/dev/null || true

echo "=== Per-GPU summary ==="
python3 -c "
import csv
from collections import defaultdict
traces = defaultdict(list)
with open('$RESULTS_DIR/gpu_util.csv') as f:
    for row in csv.DictReader(f):
        traces[int(row['gpu_index'])].append(float(row['gpu_util_pct']))
for gi in sorted(traces.keys()):
    u = traces[gi]
    n = len(u)
    mean_u = sum(u)/n
    idle = sum(1 for v in u if v <= 15)/n
    busy = sum(1 for v in u if v >= 50)/n
    print(f'GPU {gi}: mean={mean_u:.1f}% idle={idle:.1%} busy={busy:.1%} samples={n}')
"

echo "=== Training complete ==="
echo "GPU trace: $(wc -l < $RESULTS_DIR/gpu_util.csv) samples"

# Insurance: dump the raw GPU trace to stdout as gzip+base64 so Cloud
# Logging preserves it even if the live kubectl-cp window is missed.
echo "=== GPU trace dump (gzip+base64, prefix GPUCSV:) ==="
echo "GPUCSV-BEGIN md5=$(md5sum "$RESULTS_DIR/gpu_util.csv" | cut -d' ' -f1)"
gzip -c "$RESULTS_DIR/gpu_util.csv" | base64 -w 3000 | sed 's/^/GPUCSV:/' || true
echo "GPUCSV-END"

echo "=== Metrics JSONL dump ==="
cat "$RESULTS_DIR"/rl_logs/metrics_*.jsonl 2>/dev/null || true

echo "=== Keeping pod alive for result collection ==="
sleep 7200
