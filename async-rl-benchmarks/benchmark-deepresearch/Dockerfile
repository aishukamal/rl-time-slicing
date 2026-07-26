FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.12 + system deps
RUN apt-get update && apt-get install -y \
    software-properties-common git wget curl openjdk-21-jdk-headless \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3.12-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
    && rm -rf /var/lib/apt/lists/*

# Install core ML deps (pinned to CMU repo's versions)
RUN python3 -m pip install --no-cache-dir \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

RUN python3 -m pip install --no-cache-dir \
    vllm==0.8.5 \
    ray==2.46.0 \
    transformers==4.51.1 \
    accelerate==1.7.0 \
    datasets==3.6.0 \
    peft==0.15.2 \
    tensordict==0.6.2 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0

# Install flash_attn
RUN python3 -m pip install --no-cache-dir flash_attn==2.7.4.post1 --no-build-isolation 2>/dev/null || true

# Remove system blinker that blocks pip installs BEFORE the big dep batch
RUN rm -rf /usr/lib/python3/dist-packages/blinker* 2>/dev/null || true

# Install agent/search deps. NO silent-failure suffix: a previous build used
# `2>/dev/null || true` here (and ran before the blinker removal), which made
# the entire batch silently missing from the image.
RUN python3 -m pip install --no-cache-dir \
    langchain==0.3.27 langchain-core==0.3.74 langchain-openai==0.3.29 \
    langchain-text-splitters==0.3.9 "langchain-community<0.4" \
    diskcache aiohttp httpx fastapi uvicorn python-dotenv \
    tavily-python tenacity timeout-decorator nest-asyncio \
    faiss-cpu cachetools einops google-genai google-generativeai \
    gym gymnasium openai wandb cupy-cuda12x codetiming \
    xformers==0.0.29.post2

RUN python3 -m pip install --no-cache-dir pyserini torchdata

# Re-pin transformers to CMU's version (pyserini may have upgraded it)
RUN python3 -m pip install --no-cache-dir transformers==4.51.1

# Force-reinstall matched torch trio LAST — pyserini/transformers deps break the pairing
RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# torchdata pulled a newer torch whose triton (3.3+) survives the --no-deps
# reinstall above; torch 2.6 inductor needs triton 3.2 (AttrsDescriptor).
RUN python3 -m pip install --no-cache-dir triton==3.2.0

# Clone and install CMU verl-agent-deepresearch
RUN git clone --depth 1 https://github.com/cxcscmu/verl-agent-deepresearch.git /opt/verl-agent-deepresearch \
    && cd /opt/verl-agent-deepresearch \
    && python3 -m pip install -e . --no-deps 2>/dev/null || true

ENV PYTHONPATH="/opt/verl-agent-deepresearch:${PYTHONPATH}"

WORKDIR /workspace
