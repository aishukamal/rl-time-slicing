# train.py
"""Simple nano-GPT trainer in JAX/Flax designed to verify gVisor checkpoint/restore.

Prints losses and parameter stats at each step and measures restore latency.
"""

import os
import sys
import time
import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax


# Model definition (smaller than Jessica's for fast startup and training)
class TransformerBlock(nn.Module):
  num_heads: int
  emb_dim: int

  @nn.compact
  def __call__(self, x, mask=None):
    norm_x = nn.LayerNorm()(x)
    attn_out = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        qkv_features=self.emb_dim,
        out_features=self.emb_dim,
    )(norm_x, norm_x, mask=mask)
    x = x + attn_out

    norm_x = nn.LayerNorm()(x)
    mlp_out = nn.Dense(4 * self.emb_dim)(norm_x)
    mlp_out = nn.gelu(mlp_out)
    mlp_out = nn.Dense(self.emb_dim)(mlp_out)
    x = x + mlp_out
    return x


class MiniGPT(nn.Module):
  vocab_size: int
  seq_len: int
  emb_dim: int
  num_heads: int
  num_layers: int

  @nn.compact
  def __call__(self, x):
    tok_emb = nn.Embed(num_embeddings=self.vocab_size, features=self.emb_dim)(x)
    positions = jnp.arange(x.shape[1])
    pos_emb = nn.Embed(num_embeddings=self.seq_len, features=self.emb_dim)(
        positions
    )
    x_emb = tok_emb + pos_emb
    mask = nn.make_causal_mask(x)

    for _ in range(self.num_layers):
      x_emb = TransformerBlock(
          num_heads=self.num_heads,
          emb_dim=self.emb_dim,
      )(x_emb, mask=mask)

    x_emb = nn.LayerNorm()(x_emb)
    logits = nn.Dense(self.vocab_size)(x_emb)
    return logits


def cross_entropy_loss(logits, targets):
  one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])
  loss = optax.softmax_cross_entropy(logits, one_hot_targets)
  return jnp.mean(loss)


@jax.jit
def train_step(state, batch):
  inputs, targets = batch

  def loss_fn(params):
    logits = state.apply_fn({"params": params}, inputs)
    loss = cross_entropy_loss(logits, targets)
    return loss

  loss, grads = jax.value_and_grad(loss_fn)(state.params)
  state = state.apply_gradients(grads=grads)
  return state, loss


def get_param_fingerprint(params):
  # Compute check sum / mean of model parameters to verify no corruption
  flat_params, _ = jax.tree_util.tree_flatten(params)
  total_sum = sum(jnp.sum(p) for p in flat_params)
  return float(total_sum)


def main():
  print("=" * 60)
  print("Starting JAX nano-GPT Trainer process inside gVisor")
  print("=" * 60)

  print(f"JAX Devices: {jax.devices()}")

  from jax_smi import initialise_tracking

  initialise_tracking()

  # Configuration
  vocab_size = 50257
  seq_len = 128
  emb_dim = 192
  num_heads = 3
  num_layers = 3
  batch_size = 8
  learning_rate = 3e-4
  num_steps = 10
  checkpoint_step = 5

  # Initial key and model
  rng = jax.random.PRNGKey(42)
  model = MiniGPT(
      vocab_size=vocab_size,
      seq_len=seq_len,
      emb_dim=emb_dim,
      num_heads=num_heads,
      num_layers=num_layers,
  )

  # Prepare mock input dataset
  rng, input_rng = jax.random.split(rng)
  dummy_x = jax.random.randint(input_rng, (batch_size, seq_len), 0, vocab_size)
  dummy_y = jax.random.randint(input_rng, (batch_size, seq_len), 0, vocab_size)

  # Initialize TrainState
  rng, init_rng = jax.random.split(rng)
  params = model.init(init_rng, dummy_x)["params"]
  tx = optax.adamw(learning_rate)
  state = train_state.TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=tx,
  )

  # Run training loop
  for step in range(1, num_steps + 1):
    t_start = time.time()
    state, loss = train_step(state, (dummy_x, dummy_y))
    loss = float(loss)
    fingerprint = get_param_fingerprint(state.params)
    t_elapsed = time.time() - t_start
    print(
        f"Step {step}/{num_steps} | Loss: {loss:.6f} | Param Fingerprint:"
        f" {fingerprint:.6f} | Tool time: {t_elapsed:.4f}s"
    )
    sys.stdout.flush()

    if step == checkpoint_step:
      # Distinct ready-file path per process instance, set via env var.
      # This allows multiple trainer processes to be time-sliced
      # independently without one process's ready signal waking another.
      ready_file = os.environ.get(
          "CHECKPOINT_READY_FILE", "/tmp/checkpoint/ready"
      )
      # Remove ready file if it exists from a previous run
      if os.path.exists(ready_file):
        os.remove(ready_file)
      print(
          f"--- PAUSED AFTER STEP {step}: run checkpoint now, then"
          f" touch {ready_file} to continue ---"
      )
      sys.stdout.flush()
      while not os.path.exists(ready_file):
        time.sleep(1)
      print("--- READY FILE DETECTED, RESUMING TRAINING ---")
      sys.stdout.flush()

  print("=" * 60)
  print("Training session finished successfully!")
  print("=" * 60)


if __name__ == "__main__":
  main()
