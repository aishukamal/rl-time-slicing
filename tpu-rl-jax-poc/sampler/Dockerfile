# Dockerfile -- vLLM TPU sampler for RL rollout generation
#
# Base: official vLLM TPU image from Docker Hub.
# Target: TPU v5e (8 chips), ct5lp-hightpu-8t machine type.
#
# Build:
#   docker build -t tpu-rl-sampler -f Dockerfile .
#
# The vllm/vllm-tpu image already includes:
#   - vLLM with tpu-inference plugin
#   - JAX with TPU support (libtpu, jaxlib-tpu)
#   - PyTorch/XLA
#
# First inference request triggers XLA compilation (~20-30 min cold,
# ~5 min warm with cache). Cache persists across restarts if you mount
# /tmp/xla_cache to a PVC.

FROM vllm/vllm-tpu:latest

WORKDIR /app

# Install minimal additional deps for the FastAPI wrapper
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic \
    requests \
    safetensors \
    tpu-info

# Note: JAX fallback sampler (sampler_jax_fallback.py) needs transformers[flax]<5.0
# but that conflicts with vllm-tpu's jaxlib. The fallback requires a separate image.
# This image is for the primary vLLM path only.

COPY sampler.py .
COPY sampler_jax_fallback.py .

# Default: run the vLLM-based sampler
# Override with CMD ["python", "sampler_jax_fallback.py"] to use JAX fallback
EXPOSE 8300
CMD ["python", "sampler.py"]
