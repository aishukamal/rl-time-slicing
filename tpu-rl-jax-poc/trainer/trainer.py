"""trainer.py — JAX/Flax GRPO trainer for GPT-2 (124M params) on TPU v5e (8 chips)

Full fine-tuning (no LoRA). Loads the model via HuggingFace FlaxGPT2LMHeadModel,
shards across 8 TPU chips, and exposes a FastAPI service for training.

Design notes on JIT:
  - The loss/grad function is built lazily after model load, since it needs the
    model's apply method as a closure.
  - All sequences are right-padded to MAX_SEQ_LEN so that jax.jit traces a single
    static shape and reuses the compiled XLA program across calls.
  - prompt_len is passed as a regular (traced) argument, not a static one — the
    masking logic handles variable prompt lengths within the fixed buffer.

Endpoints:
  GET  /health
  POST /train   {prompts, completions, advantages, prompt_ids}
  GET  /metrics
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [trainer] %(levelname)s %(message)s"
)
log = logging.getLogger("trainer")

# -- Config (from env) --------------------------------------------------------

MODEL_PATH = os.environ.get("MODEL_PATH", "gpt2")
LR = float(os.environ.get("LR", "1e-6"))
KL_COEF = float(os.environ.get("KL_COEF", "0.01"))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "2560"))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "0.1"))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "8"))
TRAINER_PORT = int(os.environ.get("TRAINER_PORT", "8200"))
LOG_DIR = os.environ.get("LOG_DIR", "/data/rl_logs")
METRICS_FILE = os.environ.get("METRICS_FILE", "/data/rl_metrics.jsonl")

# -- Global state -------------------------------------------------------------

g_policy_params = None  # trainable params (pytree of jax arrays)
g_ref_params = None  # frozen reference params
g_opt_state = None  # optax optimizer state
g_model = None  # FlaxGPT2LMHeadModel instance (holds apply logic)
g_tokenizer = None
g_optimizer = None  # optax GradientTransformation
g_mesh = None  # JAX device mesh
g_grad_fn = None  # JIT-compiled gradient function (built after model load)
g_step = 0
g_metrics = []


# -- TPU mesh setup -----------------------------------------------------------


def create_tpu_mesh() -> Mesh:
    """Create a 1D mesh across all available TPU chips for data-parallel sharding."""
    devices = jax.devices()
    log.info(f"JAX devices: {len(devices)} -- {[str(d) for d in devices]}")
    mesh = Mesh(jax.devices(), axis_names=("dp",))
    return mesh


def shard_params(params, mesh: Mesh):
    """Replicate all parameters across the data-parallel axis.

    For a 124M model (~250MB in bf16) with 128GB total HBM across 8 chips,
    full replication is the simplest approach and fits easily.
    """
    replicated = NamedSharding(mesh, P())  # fully replicated

    def _shard(x):
        return jax.device_put(x, replicated)

    return jax.tree.map(_shard, params)


# -- Model loading ------------------------------------------------------------


def build_grad_fn(model):
    """Build a JIT-compiled loss+grad function that closes over the model's apply method.

    Called once after model load. The returned function has signature:
      (policy_params, ref_params, input_ids, attention_mask, prompt_len, advantage)
      -> (grads, (loss, policy_loss, kl_loss))
    """
    apply_fn = model.module.apply
    kl_coef = KL_COEF

    def grpo_loss(policy_params, ref_params, input_ids, attention_mask, prompt_len, advantage):
        """GRPO loss for a single padded sequence. All args are traced (no static)."""
        seq_len = input_ids.shape[1]
        position_ids = jnp.broadcast_to(jnp.arange(seq_len), input_ids.shape)

        # Forward pass through both models
        policy_logits = apply_fn(
            {"params": policy_params}, input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids
        ).logits
        ref_logits = apply_fn(
            {"params": ref_params}, input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids
        ).logits

        # Build a mask for completion tokens only
        # logits[t] predicts token[t+1], so for completion starting at prompt_len:
        #   prediction positions: prompt_len-1 .. seq_len-2
        #   target tokens:        prompt_len   .. seq_len-1
        seq_len = input_ids.shape[1]
        positions = jnp.arange(seq_len - 1)  # (seq_len-1,)
        # mask: 1.0 where position >= prompt_len-1 AND the target token has attention
        comp_mask = ((positions >= (prompt_len - 1)) & (attention_mask[0, 1:] == 1)).astype(jnp.float32)
        n_valid = jnp.maximum(comp_mask.sum(), 1.0)

        # Log-softmax over vocabulary for shifted positions
        policy_lp_all = jax.nn.log_softmax(policy_logits[0, :-1], axis=-1)  # (seq_len-1, vocab)
        ref_lp_all = jax.nn.log_softmax(ref_logits[0, :-1], axis=-1)

        # Gather log-prob of actual next token at each position
        target_ids = input_ids[0, 1:]  # (seq_len-1,)
        policy_lp = policy_lp_all[positions, target_ids]  # (seq_len-1,)
        ref_lp = ref_lp_all[positions, target_ids]

        # TRL-style GRPO loss (only over completion tokens)
        log_ratio = policy_lp - jax.lax.stop_gradient(ref_lp)
        policy_loss = -(advantage * (log_ratio * comp_mask).sum() / n_valid)
        kl_loss = ((jnp.exp(log_ratio) - log_ratio - 1.0) * comp_mask).sum() / n_valid
        total_loss = policy_loss + kl_coef * kl_loss

        return total_loss, (policy_loss, kl_loss)

    # value_and_grad w.r.t. first arg (policy_params), with aux outputs
    @jax.jit
    def grad_fn(policy_params, ref_params, input_ids, attention_mask, prompt_len, advantage):
        (loss, (policy_loss, kl_loss)), grads = jax.value_and_grad(grpo_loss, argnums=0, has_aux=True)(
            policy_params, ref_params, input_ids, attention_mask, prompt_len, advantage
        )
        return grads, loss, policy_loss, kl_loss

    return grad_fn


def load_model():
    """Load GPT-2 (124M) with Flax backend and set up optimizer."""
    global g_policy_params, g_ref_params, g_model, g_tokenizer
    global g_optimizer, g_opt_state, g_mesh, g_grad_fn

    from transformers import AutoTokenizer, FlaxGPT2LMHeadModel

    log.info(f"Loading model from {MODEL_PATH}")
    t0 = time.perf_counter()

    g_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if g_tokenizer.pad_token is None:
        g_tokenizer.pad_token = g_tokenizer.eos_token

    # Load the Flax model
    g_model = FlaxGPT2LMHeadModel.from_pretrained(
        MODEL_PATH,
        dtype=jnp.bfloat16,
    )

    # Set up TPU mesh and shard params
    g_mesh = create_tpu_mesh()
    g_policy_params = shard_params(g_model.params, g_mesh)
    g_ref_params = jax.tree.map(lambda x: x.copy(), g_policy_params)

    # Optimizer: AdamW with gradient clipping
    g_optimizer = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP),
        optax.adamw(learning_rate=LR, weight_decay=0.01, b1=0.9, b2=0.99),
    )
    g_opt_state = g_optimizer.init(g_policy_params)

    # Build JIT-compiled grad function now that model is loaded
    g_grad_fn = build_grad_fn(g_model)

    param_count = sum(x.size for x in jax.tree.leaves(g_policy_params))
    elapsed = time.perf_counter() - t0
    log.info(
        f"Model loaded in {elapsed:.1f}s. "
        f"Params: {param_count:,} ({param_count * 2 / 1e9:.2f} GB in bf16). "
        f"Mesh: {g_mesh}"
    )


# -- Tokenization helper (pad to MAX_SEQ_LEN) ---------------------------------


def tokenize_and_pad(text: str, prompt: str):
    """Tokenize a full sequence (prompt + completion), pad to MAX_SEQ_LEN.

    Returns (input_ids, attention_mask, prompt_len, was_truncated) where
    input_ids and attention_mask are numpy arrays of shape (1, MAX_SEQ_LEN).
    """
    full_enc = g_tokenizer(text, return_tensors="np", truncation=True, max_length=MAX_SEQ_LEN)
    prompt_enc = g_tokenizer(prompt, return_tensors="np", truncation=True, max_length=MAX_SEQ_LEN)

    ids = full_enc["input_ids"][0]  # (actual_len,)
    actual_len = len(ids)
    prompt_len = prompt_enc["input_ids"].shape[1]
    was_truncated = len(g_tokenizer(text)["input_ids"]) > MAX_SEQ_LEN

    # Right-pad to MAX_SEQ_LEN
    pad_token_id = g_tokenizer.pad_token_id or 0
    padded_ids = np.full((1, MAX_SEQ_LEN), pad_token_id, dtype=np.int32)
    padded_mask = np.zeros((1, MAX_SEQ_LEN), dtype=np.int32)
    padded_ids[0, :actual_len] = ids
    padded_mask[0, :actual_len] = 1

    return padded_ids, padded_mask, prompt_len, was_truncated


# -- Training step (batched over completions) ---------------------------------


def train_step(prompts, completions, advantages):
    """Run one GRPO training step over a batch of (prompt, completion, advantage) triples.

    Accumulates gradients across all samples then applies a single optimizer update.
    """
    global g_policy_params, g_opt_state, g_step

    t0 = time.perf_counter()
    total_policy_loss = 0.0
    total_kl = 0.0
    n_items = 0
    n_truncated = 0
    accumulated_grads = None
    batch_size = max(len(completions), 1)

    for prompt, completion, adv in zip(prompts, completions, advantages):
        full_text = prompt + completion
        padded_ids, padded_mask, prompt_len, was_truncated = tokenize_and_pad(full_text, prompt)

        if was_truncated:
            n_truncated += 1

        # Skip if no completion tokens after tokenization
        actual_len = int(padded_mask.sum())
        if actual_len <= prompt_len:
            continue

        # Convert to JAX arrays
        input_ids = jnp.array(padded_ids, dtype=jnp.int32)
        attention_mask = jnp.array(padded_mask, dtype=jnp.int32)
        jax_prompt_len = jnp.int32(prompt_len)
        jax_advantage = jnp.float32(adv)

        # Compute gradients for this sample (hits JIT cache since shape is fixed)
        grads, loss_val, policy_loss, kl_loss = g_grad_fn(
            g_policy_params, g_ref_params,
            input_ids, attention_mask,
            jax_prompt_len, jax_advantage,
        )

        # Accumulate gradients (mean over batch)
        scale = 1.0 / batch_size
        scaled_grads = jax.tree.map(lambda g: g * scale, grads)

        if accumulated_grads is None:
            accumulated_grads = scaled_grads
        else:
            accumulated_grads = jax.tree.map(jnp.add, accumulated_grads, scaled_grads)

        total_policy_loss += float(policy_loss)
        total_kl += float(kl_loss)
        n_items += 1

    if accumulated_grads is None:
        log.warning("No valid samples in batch, skipping update")
        return {"step": g_step, "error": "no valid samples"}

    # Apply accumulated gradients via optax
    updates, g_opt_state = g_optimizer.update(accumulated_grads, g_opt_state, g_policy_params)
    g_policy_params = optax.apply_updates(g_policy_params, updates)

    # Compute grad norm for logging
    grad_norm = float(
        jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(accumulated_grads)))
    )

    g_step += 1
    n = max(n_items, 1)
    ms = (time.perf_counter() - t0) * 1000

    rec = {
        "step": g_step,
        "policy_loss": round(total_policy_loss / n, 6),
        "kl_loss": round(total_kl / n, 6),
        "loss": round((total_policy_loss + KL_COEF * total_kl) / n, 6),
        "grad_norm": round(grad_norm, 4),
        "n_samples": n_items,
        "n_truncated": n_truncated,
        "elapsed_ms": round(ms),
    }
    g_metrics.append(rec)
    log.info(
        f"step={g_step} loss={rec['loss']:.4f} kl={rec['kl_loss']:.4f} "
        f"grad_norm={rec['grad_norm']:.3f} "
        f"truncated={n_truncated}/{n_items} {ms:.0f}ms"
    )

    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "a") as f:
        f.write(json.dumps({"type": "train", **rec, "ts": time.time()}) + "\n")

    return rec


# -- FastAPI app --------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Trainer starting -- model will load on first /train call")
    yield


app = FastAPI(title="JAX GRPO Trainer (TPU)", lifespan=lifespan)


class TrainRequest(BaseModel):
    prompts: List[str]
    completions: List[str]
    advantages: List[float]
    prompt_ids: Optional[List[str]] = None
    group_size: int = GROUP_SIZE


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": g_policy_params is not None,
        "step": g_step,
    }


@app.get("/metrics")
def metrics_endpoint():
    return {"metrics": g_metrics[-30:], "step": g_step}


@app.post("/train")
def train_endpoint(req: TrainRequest):
    """GRPO training step.

    Expects pre-computed advantages (group-normalized rewards).
    The caller is responsible for:
      1. Generating completions (via a sampler)
      2. Computing rewards (via reward.py)
      3. Computing advantages (via reward.compute_advantages)
    """
    if g_policy_params is None:
        log.info("First /train call -- loading model")
        load_model()

    log.info(
        f"train request: prompts={len(req.prompts)} completions={len(req.completions)} "
        f"advantages={len(req.advantages)}"
    )

    # Expand prompts to match completions (each prompt has group_size completions)
    expanded_prompts = [p for p in req.prompts for _ in range(req.group_size)]

    if len(expanded_prompts) != len(req.completions):
        # If prompts are already expanded (1:1 with completions), use as-is
        if len(req.prompts) == len(req.completions):
            expanded_prompts = req.prompts
        else:
            return {
                "error": (
                    f"Mismatch: {len(req.prompts)} prompts * {req.group_size} "
                    f"!= {len(req.completions)} completions"
                )
            }

    return train_step(expanded_prompts, req.completions, req.advantages)


@app.post("/export_weights")
def export_weights_endpoint():
    """Export current policy weights as safetensors for sampler to download."""
    if g_policy_params is None:
        return {"error": "model not loaded"}

    import weight_sync
    t0 = time.perf_counter()
    weight_sync.export_weights(g_model, g_policy_params)
    elapsed = time.perf_counter() - t0
    log.info(f"Weights exported in {elapsed:.1f}s")
    return {"status": "ok", "elapsed_s": round(elapsed, 1), "step": g_step}


@app.get("/weights/{filename}")
def serve_weight_file(filename: str):
    """Serve exported weight files for HTTP-based weight transfer."""
    from fastapi.responses import FileResponse
    filepath = os.path.join("/tmp/exported_weights", filename)
    if not os.path.exists(filepath):
        from fastapi import HTTPException
        raise HTTPException(404, f"File not found: {filename}")
    return FileResponse(filepath)


@app.get("/get_pids")
def get_pids():
    return {"pids": [os.getpid()]}


@app.post("/checkpoint")
def checkpoint_endpoint():
    """Checkpoint this process's TPU state via libtpu-uds gRPC."""
    import grpc
    import glob
    pid = os.getpid()
    sock = f"/run/tpu_hal_{pid}.sock"
    if not os.path.exists(sock):
        socks = glob.glob(f"/run/tpu_hal_*.sock")
        return {"error": f"socket {sock} not found, available: {socks}"}

    t0 = time.perf_counter()
    channel = grpc.insecure_channel(f"unix://{sock}")
    call = channel.unary_unary("/tpu.TpuHalService/Checkpoint",
        request_serializer=lambda x: x, response_deserializer=lambda x: x)
    call(b"", timeout=300)
    channel.close()
    elapsed = round((time.perf_counter() - t0) * 1000)
    log.info(f"Checkpoint complete: {elapsed}ms")
    return {"status": "ok", "elapsed_ms": elapsed}


@app.post("/restore")
def restore_endpoint():
    """Restore this process's TPU state via libtpu-uds gRPC."""
    import grpc
    pid = os.getpid()
    sock = f"/run/tpu_hal_{pid}.sock"

    t0 = time.perf_counter()
    channel = grpc.insecure_channel(f"unix://{sock}")
    call = channel.unary_unary("/tpu.TpuHalService/Restore",
        request_serializer=lambda x: x, response_deserializer=lambda x: x)
    call(b"", timeout=300)
    channel.close()
    elapsed = round((time.perf_counter() - t0) * 1000)
    log.info(f"Restore complete: {elapsed}ms")
    return {"status": "ok", "elapsed_ms": elapsed}


@app.get("/get_tpu_stats")
def get_tpu_stats():
    """Return per-chip TPU memory and duty cycle via tpu-info."""
    try:
        from tpu_info.device import get_local_chips
        from tpu_info.metrics import get_chip_usage
        chip_type, _ = get_local_chips()
        usages = get_chip_usage(chip_type)
        chips = []
        for u in usages:
            chips.append({
                "device_id": u.device_id,
                "mem_used_bytes": u.memory_usage,
                "mem_total_bytes": u.total_memory,
                "mem_used_mib": round(u.memory_usage / (1024 * 1024)),
                "duty_cycle_pct": round(u.duty_cycle_pct, 1),
            })
        return {"chips": chips}
    except Exception as e:
        return {"error": str(e)}


# -- Entrypoint ---------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=TRAINER_PORT, log_level="info")
