"""weight_sync.py — Flax GPT-2 → safetensors export for vLLM reload

Converts FlaxGPT2LMHeadModel params to PyTorch-compatible safetensors format.
Weights are served via HTTP from the trainer; no GCS dependency.

The key mapping follows HuggingFace's FlaxGPT2 ↔ PyTorch GPT2 conversion:
  - Dense weights are transposed (Flax uses (in, out), PyTorch uses (out, in))
  - Key names are mapped from Flax-style to PyTorch-style
"""

import json
import logging
import os

import numpy as np

log = logging.getLogger("weight_sync")

EXPORT_DIR = "/tmp/exported_weights"


def flax_to_pytorch_key(flax_key_tuple):
    parts = list(flax_key_tuple)
    if parts[-1] == "kernel":
        parts[-1] = "weight"
    elif parts[-1] == "scale":
        parts[-1] = "weight"
    elif parts[-1] == "embedding":
        parts[-1] = "weight"
    return ".".join(parts)


def should_transpose(key_tuple):
    return key_tuple[-1] == "kernel" and len(key_tuple) > 1


def flatten_params(params, prefix=()):
    result = {}
    if isinstance(params, dict):
        for k, v in params.items():
            result.update(flatten_params(v, prefix + (k,)))
    else:
        result[prefix] = params
    return result


def export_weights(model, params):
    """Export Flax GPT-2 params as safetensors + config.json to EXPORT_DIR."""
    from safetensors.numpy import save_file

    flat = flatten_params(params)
    tensors = {}
    for key_tuple, array in flat.items():
        pt_key = flax_to_pytorch_key(key_tuple)
        arr = np.array(array)
        if should_transpose(key_tuple) and arr.ndim == 2:
            arr = arr.T
        tensors[pt_key] = arr

    os.makedirs(EXPORT_DIR, exist_ok=True)

    sf_path = os.path.join(EXPORT_DIR, "model.safetensors")
    save_file(tensors, sf_path)
    model.config.save_pretrained(EXPORT_DIR)

    size_mb = os.path.getsize(sf_path) / 1e6
    log.info(f"Exported {len(tensors)} tensors ({size_mb:.1f} MB) to {EXPORT_DIR}")
    return EXPORT_DIR
