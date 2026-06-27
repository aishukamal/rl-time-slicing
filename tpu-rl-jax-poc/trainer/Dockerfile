FROM python:3.11-slim

WORKDIR /app

# System deps for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install JAX with TPU support
# The jax[tpu] package bundles libtpu and jaxlib for TPU
RUN pip install --no-cache-dir \
    "jax[tpu]>=0.4.30" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install ML and serving deps
# Install transformers WITHOUT [flax] extra to avoid its jaxlib<=0.4.13 pin
# conflicting with jax[tpu]'s jaxlib. Flax is installed separately;
# FlaxGPT2LMHeadModel is available as long as flax is importable.
RUN pip install --no-cache-dir \
    flax>=0.8.0 \
    optax>=0.2.0 \
    "transformers>=4.44.0,<5.0.0" \
    tokenizers \
    fastapi>=0.111.0 \
    "uvicorn[standard]>=0.29.0" \
    pydantic>=2.0 \
    safetensors \
    tpu-info \
    matplotlib

# Copy trainer code
COPY trainer.py reward.py weight_sync.py /app/

EXPOSE 8200

# TPU runtime expects these
ENV JAX_PLATFORMS=tpu
ENV TOKENIZERS_PARALLELISM=false

CMD ["python", "trainer.py"]
