#!/bin/bash
set -e

MODEL_DIR="/tmp/Qwen2.5-0.5B-Instruct"
DATA_FILE="/tmp/dapo-math-17k/dapo-math-17k.jsonl"

if [ ! -d "$MODEL_DIR" ]; then
    echo "[setup] Downloading model..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='$MODEL_DIR')"
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "[setup] Downloading dataset..."
    mkdir -p /tmp/dapo-math-17k
    python3 -c "from datasets import load_dataset; ds = load_dataset('zhuzilin/dapo-math-17k'); ds['train'].to_json('$DATA_FILE')"
fi

echo "[setup] Done"
