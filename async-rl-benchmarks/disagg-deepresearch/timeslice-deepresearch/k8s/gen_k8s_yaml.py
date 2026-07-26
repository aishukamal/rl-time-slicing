#!/usr/bin/env python3
"""Generate k8s-job-disagg.yaml from the timeslice-deepresearch tree.

Embeds:
  - disagg-deepresearch-scripts ConfigMap: gpu_monitor.sh,
    local_search_server.py, setup_and_run_disagg.sh
  - disagg-deepresearch-code ConfigMap: every .py of the timeslice tree,
    flat keys with '--' as the path separator (reassembled by the setup
    script into /opt/timeslice)
  - the Job: main container on the PoC verl-main image (2 GPUs) + search
    sidecar on the deepresearch-benchmark image (pyserini Wikipedia BM25)

Re-run after editing any embedded file:
  python3 timeslice-deepresearch/k8s/gen_k8s_yaml.py
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)  # timeslice-deepresearch/
OUT = os.path.join(os.path.dirname(TREE), "k8s-job-disagg.yaml")

MAIN_IMAGE = "verlai/verl:vllm017.latest"
SEARCH_IMAGE = "us-west1-docker.pkg.dev/aishuk-test/tpu-poc-repo/deepresearch-benchmark:latest"


def block(content: str, indent: int) -> str:
    """Render a YAML literal block scalar at the given indent."""
    pad = " " * indent
    lines = content.rstrip("\n").split("\n")
    body = "\n".join((pad + line).rstrip() if line.strip() == "" else pad + line
                     for line in lines)
    return "|\n" + body + "\n"


def code_files():
    """Yield (configmap_key, relpath) for every embedded code file."""
    top = [
        "main_ppo_timeslice_sync.py",
        "verl_timeslice_sync_trainer.py",
        "gpu_client.py",
        "config.py",
        "prepare_deepresearch_data.py",
    ]
    for f in top:
        yield f, f
    for root, _dirs, files in os.walk(os.path.join(TREE, "agent_system")):
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(root, f), TREE)
            # '--' as path separator: '__' collides with __init__.py, and no
            # path component in this tree contains a hyphen.
            assert "-" not in rel, f"hyphen in path breaks the -- separator: {rel}"
            yield rel.replace(os.sep, "--"), rel


def main():
    parts = []

    # ── scripts ConfigMap ───────────────────────────────────────────────
    parts.append(
        "apiVersion: v1\n"
        "kind: ConfigMap\n"
        "metadata:\n"
        "  name: disagg-deepresearch-scripts\n"
        "  namespace: default\n"
        "data:\n"
    )
    for name in ["gpu_monitor.sh", "local_search_server.py", "setup_and_run_disagg.sh"]:
        content = open(os.path.join(HERE, name)).read()
        parts.append(f"  {name}: " + block(content, 4))
        parts.append("\n")

    # ── code ConfigMap ─────────────────────────────────────────────────
    parts.append(
        "---\n"
        "apiVersion: v1\n"
        "kind: ConfigMap\n"
        "metadata:\n"
        "  name: disagg-deepresearch-code\n"
        "  namespace: default\n"
        "data:\n"
    )
    total = 0
    for key, rel in code_files():
        content = open(os.path.join(TREE, rel)).read()
        if content.strip() == "":
            # ConfigMap values must be non-empty strings; keep a marker line
            content = "# (intentionally empty)\n"
        total += len(content)
        parts.append(f"  {key}: " + block(content, 4))
        parts.append("\n")
    print(f"embedded code bytes: {total}")

    # ── Job ────────────────────────────────────────────────────────────
    parts.append(f"""---
apiVersion: batch/v1
kind: Job
metadata:
  name: disagg-deepresearch
  namespace: default
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 604800
  template:
    metadata:
      labels:
        app: disagg-deepresearch
    spec:
      restartPolicy: Never
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-h100-mega-80gb
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: cloud.google.com/gke-nodepool
                operator: In
                values:
                - h100-mega-8gpu-spot-b
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      containers:
      - name: benchmark
        image: {MAIN_IMAGE}
        env:
        - name: DEBIAN_FRONTEND
          value: "noninteractive"
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -euo pipefail
          echo "=== Disaggregated Deep-Research Multi-Turn Benchmark ==="
          echo "Node: $(hostname)"
          echo "Date: $(date)"
          nvidia-smi
          echo ""
          mkdir -p /workspace/results
          bash /workspace/scripts/setup_and_run_disagg.sh 2>&1 | tee /workspace/results/experiment.log
        resources:
          limits:
            nvidia.com/gpu: 2
          requests:
            nvidia.com/gpu: 2
            cpu: "40"
            memory: "200Gi"
        volumeMounts:
        - name: scripts
          mountPath: /workspace/scripts
        - name: code
          mountPath: /workspace/code
        - name: results
          mountPath: /workspace/results
        - name: dshm
          mountPath: /dev/shm
      # Search sidecar: pyserini BM25 Wikipedia server (Serper-format API).
      # Reuses the colocated benchmark image, which already has pyserini +
      # JDK + the 12GB index bootstrap. Shares localhost with the main
      # container (same pod network namespace).
      - name: search
        image: {SEARCH_IMAGE}
        env:
        - name: SEARCH_PORT
          value: "8877"
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -euo pipefail
          echo "=== Local Wikipedia search server (pyserini BM25) ==="
          python3 /workspace/scripts/local_search_server.py
        resources:
          requests:
            cpu: "8"
            memory: "40Gi"
          limits:
            memory: "60Gi"
        volumeMounts:
        - name: scripts
          mountPath: /workspace/scripts
      volumes:
      - name: scripts
        configMap:
          name: disagg-deepresearch-scripts
          defaultMode: 0755
      - name: code
        configMap:
          name: disagg-deepresearch-code
      - name: results
        emptyDir: {{}}
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: 64Gi
""")

    with open(OUT, "w") as f:
        f.write("".join(parts))
    print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes)")


if __name__ == "__main__":
    main()
