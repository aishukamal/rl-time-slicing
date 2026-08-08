# elastic-rl-poc M1 image: the exact M0/M1 base (verlai/verl:vllm020.dev2)
# plus the multi-GPU C/R shim bits from GPU-CR/multi-gpu-cr (REPORT.md).
#
# Nothing in the base python environment is touched: the run script still
# pip-installs verl at the 983cb0f2 pin at runtime, exactly as M0 did.
# Everything C/R-related lives under /opt/cr:
#
#   /opt/cr/libcr-shim.so      shim v1 — ncclCommSuspend/Resume on
#                              SIGRTMIN+1/+2 (kill -35 / -36). Requires the
#                              NCCL TCP-transport env (NCCL_P2P_DISABLE=1,
#                              NCCL_SHM_DISABLE=1, NCCL_NVLS_ENABLE=0).
#   /opt/cr/libcr-shim-v2.so   shim v2 — destroy/recreate variant (NVLink
#                              stays on; needs CR_NCCL_LIB + VLLM_NCCL_SO_PATH).
#                              Shipped as a fallback/perf option; M1 defaults
#                              to the v1 recipe per k8s-job-m1.yaml.
#   /opt/cr/nccl/nvidia/nccl/lib/libnccl.so.2
#                              NCCL >= 2.29.7 (ships ncclCommSuspend/Resume,
#                              same delivery as the multi-gpu-cr repro which
#                              pip-installed nvidia-nccl-cu12>=2.29.7).
FROM verlai/verl:vllm020.dev2

ENV PIP_NO_CACHE_DIR=1

# NCCL with ncclCommSuspend/Resume, isolated under /opt/cr (NOT site-packages,
# so nothing changes unless LD_PRELOAD points at it).
RUN pip install --target=/opt/cr/nccl "nvidia-nccl-cu12>=2.29.7" && \
    test -f /opt/cr/nccl/nvidia/nccl/lib/libnccl.so.2

# Build both shims (self-contained C, no nccl.h needed; build lines are the
# ones documented in each source header).
COPY universal_cr_shim.c universal_cr_shim_v2.c /opt/cr/src/
RUN gcc -shared -fPIC -o /opt/cr/libcr-shim.so /opt/cr/src/universal_cr_shim.c -ldl && \
    gcc -shared -fPIC -o /opt/cr/libcr-shim-v2.so /opt/cr/src/universal_cr_shim_v2.c -ldl -lpthread
