# timeslice-verl-poc: verl (v1 trainer) + llm-d-rl-time-slicing integration
#
# Base: the exact image verl main's CI runs on at the pinned commit
# (.github/workflows/vllm.yml @ 6a6242f3 -> verlai/verl:vllm023.dev1).
# It ships torch/vllm/flash-attn/TransformerEngine etc. with verl's deps
# preinstalled and verl itself uninstalled.
FROM verlai/verl:vllm023.dev1

ARG VERL_SHA=6a6242f3d8ec7d9f8b4936f4905144707d91fe3b
ARG TIMESLICE_SHA=11a661fd621b6ee447a12c1c72cc7b0f27664da2

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1

# 1. verl at the pinned commit, editable-from-source so hydra config yamls and
#    version files are all present (pip-from-git can drop non-.py package data).
#    --no-deps: the base image already carries the dependency stack; letting pip
#    resolve here could upgrade torch/vllm. Gaps are covered explicitly below.
RUN git clone https://github.com/volcengine/verl.git /opt/verl && \
    cd /opt/verl && \
    git checkout ${VERL_SHA} && \
    pip install --no-deps -e /opt/verl

# 2. Deps verl main needs beyond the base image's frozen set (v1 trainer imports
#    transfer_queue at module import time; reward utils may want math_verify).
RUN pip install "TransferQueue==0.1.8" "math_verify" "latex2sympy2_extended"

# 3. timeslice client (OrchestratorClient) at a pinned commit. --no-deps: needs
#    grpcio+protobuf, both already provided by the base image (ray stack).
#    The client's generated pb2 files carry protobuf gencode 7.35.0; the base
#    image ships runtime 6.33.6, which the gencode rejects (runtime must be >=
#    gencode). Upgrading the runtime is the protobuf-supported direction
#    (old gencode + new runtime); smoke steps re-verify ray/grpc/verl imports.
RUN pip install --no-deps \
    "timeslice @ git+https://github.com/llm-d-incubation/llm-d-rl-time-slicing.git@${TIMESLICE_SHA}#subdirectory=pkg/client/python" && \
    pip install --upgrade "protobuf>=7.35,<8"

# 4. The integration package (registers trainer modes "sync_timesliced"
#    and "separate_async_timesliced") + its unit tests.
COPY timeslice_verl /opt/timeslice_verl
RUN pip install --no-deps /opt/timeslice_verl

# 5. PoC run script + verl's gsm8k preprocess script (baked from the pinned SHA).
COPY poc/run_job.sh poc/run_job_disagg.sh poc/gsm8k_preprocess.py poc/DISAGG_PLAN.md /opt/poc/
RUN chmod +x /opt/poc/run_job.sh /opt/poc/run_job_disagg.sh

ENV VERL_USE_EXTERNAL_MODULES=timeslice_verl

WORKDIR /opt/poc
CMD ["/bin/bash"]
