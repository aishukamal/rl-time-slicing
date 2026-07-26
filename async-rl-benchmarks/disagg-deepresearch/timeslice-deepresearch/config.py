"""config.py — all tuneable parameters, read from env vars"""

import os

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "/data/model")
MODEL_NAME = "qwen"

# ── Ports ────────────────────────────────────────────────────────────────────
RL_ORCH_PORT = int(os.environ.get("ORCH_PORT", "9000"))
RL_DAEMON_PORT = int(os.environ.get("DAEMON_PORT", "9001"))
RL_TRAINER_A_PORT = int(os.environ.get("RL_TRAINER_A_PORT", "8200"))
RL_TRAINER_B_PORT = int(os.environ.get("RL_TRAINER_B_PORT", "8201"))
RL_SAMPLER_VLLM_PORT = int(os.environ.get("SAMPLER_PORT", "8100"))
RL_SAMPLER_A_PORT = int(os.environ.get("RL_SAMPLER_A_PORT", "8300"))
RL_SAMPLER_B_PORT = int(os.environ.get("RL_SAMPLER_B_PORT", "8301"))

# ── GPU pool assignment ───────────────────────────────────────────────────────
# Pool names map to a CUDA device index visible in each pod/process
RL_TRAINER_GPU = int(os.environ.get("TRAINER_GPU", "0"))
RL_SAMPLER_GPU = int(os.environ.get("SAMPLER_GPU", "1"))

# ── RL hyperparameters ───────────────────────────────────────────────────────
N_RL_STEPS = int(os.environ.get("N_RL_STEPS", "10"))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "8"))  # G per prompt
PROMPTS_PER_STEP = int(os.environ.get("PROMPTS_PER_STEP", "4"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "2048"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.8"))
LR = float(os.environ.get("LR", "1e-6"))
KL_COEF = float(os.environ.get("KL_COEF", "0.01"))  # beta in willccbb
MAX_SEQ_LEN = int(
    os.environ.get("MAX_SEQ_LEN", "2560")
)  # 2048 gen + ~512 prompt
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "0.1"))  # willccbb uses 0.1

# ── Swap ─────────────────────────────────────────────────────────────────────
CUDA_CKPT_BIN = os.environ.get(
    "CUDA_CKPT_BIN", "/usr/local/bin/cuda-checkpoint"
)
# MODE: "timeslice" = with context switches | "baseline" = no context switches
MODE = os.environ.get("MODE", "timeslice")

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET_PATH = os.environ.get("DATASET_PATH", "/data/gsm8k_prompts.json")

# ── Weight transfer ───────────────────────────────────────────────────────────
WEIGHT_DIR = os.environ.get("WEIGHT_DIR", "/data/weights_transfer")

# ── vLLM ─────────────────────────────────────────────────────────────────────
VLLM_GPU_UTIL = float(os.environ.get("VLLM_GPU_UTIL", "0.25"))
VLLM_MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096"))

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR = os.environ.get("LOG_DIR", "/data/rl_logs")
METRICS_FILE = os.environ.get("METRICS_FILE", "/data/rl_metrics.jsonl")

# ── Weight sync ─────────────────────────────────────────────────────────────
# Set to "nccl" to enable GPU-to-GPU NCCL weight transfer after each train step.
# Set to "off" to skip weight transfer (sampler uses original model weights).
WEIGHT_SYNC = os.environ.get("WEIGHT_SYNC", "off")  # "nccl" | "off"
NCCL_MASTER_PORT = int(os.environ.get("NCCL_MASTER_PORT", "29600"))


# ── LoRA ────────────────────────────────────────────────────────────────────
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "64"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
