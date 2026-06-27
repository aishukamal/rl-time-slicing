"""orchestrator.py — Simple TPU time-slicing orchestrator

Manages acquire/yield flow for RL workloads, calling the snapshot agent
for checkpoint/restore when MODE=snapshot.

Snapshot agent discovery: on register, the RL loop passes the workload's
node name. The orchestrator queries K8s to find the snapshot-agent pod
on that node and caches its IP. No hardcoded IPs needed.
"""

import json
import logging
import os
import time
import urllib.request
from typing import Dict, List, Optional

import grpc
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [orchestrator] %(levelname)s %(message)s"
)
log = logging.getLogger("orchestrator")

MODE = os.environ.get("MODE", "baseline")
ORCH_PORT = int(os.environ.get("ORCH_PORT", "9000"))
SNAPSHOT_AGENT_PORT = int(os.environ.get("SNAPSHOT_AGENT_PORT", "9001"))
METRICS_FILE = os.environ.get("METRICS_FILE", "/data/rl_metrics.jsonl")


def _log_metric(rec: dict):
    rec["ts"] = time.time()
    try:
        os.makedirs(os.path.dirname(METRICS_FILE) or ".", exist_ok=True)
        with open(METRICS_FILE, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:
        log.warning(f"Failed to write metric: {e}")

BACKEND_TPU = 2

# -- Protobuf helpers (raw, no codegen) ----------------------------------------

def _build_snapshot_request(job_id: str, group: str = "default") -> bytes:
    buf = bytearray()
    job_bytes = job_id.encode()
    buf.append(0x0a); buf.append(len(job_bytes)); buf.extend(job_bytes)
    group_bytes = group.encode()
    buf.append(0x12); buf.append(len(group_bytes)); buf.extend(group_bytes)
    buf.append(0x18); buf.append(BACKEND_TPU)
    return bytes(buf)


def _parse_operation_id(data: bytes) -> str:
    if not data or len(data) < 2:
        return ""
    if data[0] == 0x0a:
        length = data[1]
        return data[2:2 + length].decode()
    return ""


def _build_get_operation_request(op_id: str) -> bytes:
    buf = bytearray()
    op_bytes = op_id.encode()
    buf.append(0x0a); buf.append(len(op_bytes)); buf.extend(op_bytes)
    return bytes(buf)


def _parse_get_operation_response(data: bytes) -> tuple:
    if not data:
        return 0, ""
    status_val = 0
    error_str = ""
    i = 0
    while i < len(data):
        tag = data[i]
        field_num = tag >> 3
        wire_type = tag & 0x07
        i += 1
        if wire_type == 0:
            val = 0
            shift = 0
            while i < len(data):
                b = data[i]
                val |= (b & 0x7f) << shift
                i += 1
                shift += 7
                if not (b & 0x80):
                    break
            if field_num == 1:
                status_val = val
        elif wire_type == 2:
            if i >= len(data):
                break
            length = data[i]
            i += 1
            if field_num == 5 and i + length <= len(data):
                error_str = data[i:i + length].decode("utf-8", errors="replace")
            i += length
    return status_val, error_str


OP_PENDING = 1
OP_COMPLETE = 2
OP_FAILED = 3


# -- Direct TPU HAL checkpoint/restore (bypasses snapshot agent pod discovery) --

def direct_tpu_checkpoint(pids: List[int], node: str) -> float:
    """Checkpoint TPU state by calling tpu.TpuHalService/Checkpoint directly on each PID's socket."""
    t0 = time.perf_counter()
    for pid in pids:
        sock = f"/run/tpu_hal_{pid}.sock"
        log.info(f"Direct checkpoint PID {pid} via {sock}")
        # The orchestrator doesn't have access to /run on TPU nodes.
        # Call the snapshot agent with a synthetic job-id matching the pod label.
        # For multi-container pods, we need the snapshot agent to accept the PID directly.
        # WORKAROUND: call via the trainer/sampler pod's own endpoint.
    elapsed = (time.perf_counter() - t0) * 1000
    return elapsed


def call_workload_cr(workload_id: str, method: str) -> float:
    """Call checkpoint/restore via the workload pod's own HTTP endpoint.

    Each trainer/sampler exposes /checkpoint and /restore endpoints that
    call the TPU HAL gRPC directly for their own PID. This allows per-container
    targeting in multi-container pods (the snapshot agent's pod-based discovery
    can't distinguish containers).
    """
    wl = workloads[workload_id]
    pool = wl["pool"]
    url = wl.get("url", "")
    if not url:
        raise RuntimeError(f"No URL registered for workload {workload_id}")

    t0 = time.perf_counter()
    endpoint = f"{url}/{method.lower()}"
    log.info(f"Calling {endpoint} for {workload_id}")

    body = json.dumps({}).encode()
    req = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=300)
    result = json.loads(resp.read().decode())
    elapsed = (time.perf_counter() - t0) * 1000
    log.info(f"{method} via workload {workload_id}: {elapsed:.0f}ms result={result}")
    return elapsed


def call_snapshot_agent(addr: str, method: str, job_id: str, timeout: float = 300) -> float:
    channel = grpc.insecure_channel(addr)
    req = _build_snapshot_request(job_id)
    rpc_path = f"/snapshot_agent.v1alpha1.SnapshotAgentService/{method}"
    call = channel.unary_unary(rpc_path, request_serializer=lambda x: x, response_deserializer=lambda x: x)
    resp = call(req, timeout=30)
    op_id = _parse_operation_id(resp)
    log.info(f"{method} started, operation_id={op_id}")

    get_op = channel.unary_unary(
        "/snapshot_agent.v1alpha1.SnapshotAgentService/GetOperation",
        request_serializer=lambda x: x, response_deserializer=lambda x: x,
    )
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        get_req = _build_get_operation_request(op_id)
        get_resp = get_op(get_req, timeout=10)
        status, error_msg = _parse_get_operation_response(get_resp)
        if status == OP_COMPLETE:
            elapsed = (time.perf_counter() - t0) * 1000
            log.info(f"{method} complete in {elapsed:.0f}ms")
            channel.close()
            return elapsed
        elif status == OP_FAILED:
            channel.close()
            raise RuntimeError(f"{method} operation {op_id} failed: {error_msg}")
        time.sleep(0.5)
    channel.close()
    raise RuntimeError(f"{method} timed out after {timeout}s")


# -- Snapshot agent discovery --------------------------------------------------

def discover_snapshot_agent_on_node(node_name: str) -> Optional[str]:
    """Find the snapshot-agent pod IP on a given node via K8s API."""
    try:
        k8s_host = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
        k8s_port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
        ca_path = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"

        with open(token_path) as f:
            token = f.read().strip()

        import ssl
        ctx = ssl.create_default_context(cafile=ca_path)
        url = (f"https://{k8s_host}:{k8s_port}/api/v1/pods"
               f"?labelSelector=app%3Dsnapshot-agent&fieldSelector=spec.nodeName%3D{node_name}")
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        resp = urllib.request.urlopen(req, timeout=10, context=ctx)
        data = json.loads(resp.read().decode())

        for pod in data.get("items", []):
            ip = pod.get("status", {}).get("podIP")
            if ip:
                addr = f"{ip}:{SNAPSHOT_AGENT_PORT}"
                log.info(f"Discovered snapshot agent on {node_name}: {addr}")
                return addr
    except Exception as e:
        log.warning(f"Failed to discover snapshot agent on {node_name}: {e}")
    return None


# -- State tracking -----------------------------------------------------------

import threading

workloads: Dict[str, dict] = {}
node_agent_cache: Dict[str, str] = {}
pool_locks: Dict[str, threading.Lock] = {}
pool_holders: Dict[str, Optional[str]] = {}  # pool -> workload_id currently holding it


def get_pool_lock(pool: str) -> threading.Lock:
    if pool not in pool_locks:
        pool_locks[pool] = threading.Lock()
        pool_holders[pool] = None
    return pool_locks[pool]


def get_agent_addr(workload_id: str) -> str:
    wl = workloads[workload_id]
    node = wl.get("node")
    if not node:
        raise RuntimeError(f"No node registered for workload {workload_id}")
    if node not in node_agent_cache:
        addr = discover_snapshot_agent_on_node(node)
        if not addr:
            raise RuntimeError(f"No snapshot agent found on node {node}")
        node_agent_cache[node] = addr
    return node_agent_cache[node]


# -- FastAPI app ---------------------------------------------------------------

app = FastAPI(title="TPU RL Orchestrator")


class RegisterRequest(BaseModel):
    workload_id: str
    pool: str
    pids: List[int]
    node: str = ""
    url: str = ""
    checkpointed: bool = False


class AcquireYieldRequest(BaseModel):
    workload_id: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": MODE,
        "workloads": {k: {"pool": v["pool"], "step": v["step"], "node": v.get("node", "")} for k, v in workloads.items()},
    }


@app.post("/register")
def register(req: RegisterRequest):
    workloads[req.workload_id] = {
        "pool": req.pool,
        "pids": req.pids,
        "step": 0,
        "node": req.node,
        "url": req.url,
        "checkpointed": False,
    }
    log.info(f"Registered {req.workload_id} pool={req.pool} pids={req.pids} node={req.node}")
    _log_metric({"type": "register", "workload_id": req.workload_id, "pool": req.pool, "pids": req.pids})
    if req.node and MODE == "snapshot":
        addr = discover_snapshot_agent_on_node(req.node)
        if addr:
            node_agent_cache[req.node] = addr
        else:
            log.warning(f"Could not discover snapshot agent on {req.node}")
    return {"status": "ok"}


@app.post("/acquire")
def acquire(req: AcquireYieldRequest):
    wl = workloads.get(req.workload_id)
    if not wl:
        raise HTTPException(404, f"workload {req.workload_id} not registered")

    pool = wl["pool"]
    lock = get_pool_lock(pool)
    t_wait_start = time.time()

    log.info(f"Acquiring {req.workload_id} pool={pool} (holder={pool_holders.get(pool)})")
    lock.acquire()
    wait_ms = round((time.time() - t_wait_start) * 1000)
    pool_holders[pool] = req.workload_id

    restore_ms = 0
    if MODE == "snapshot" and wl.get("checkpointed"):
        log.info(f"Restoring {req.workload_id}")
        restore_ms = call_workload_cr(req.workload_id, "restore")
        wl["checkpointed"] = False

    wl["step"] += 1
    log.info(f"Acquired {req.workload_id} step={wl['step']} wait_ms={wait_ms} restore_ms={restore_ms:.0f}")
    _log_metric({"type": "acquire", "workload_id": req.workload_id, "pool": pool,
                 "step": wl["step"], "mode": MODE, "wait_ms": wait_ms, "restore_ms": round(restore_ms)})
    return {"status": "ok", "step": wl["step"], "wait_ms": wait_ms, "restore_ms": round(restore_ms)}


@app.post("/yield")
def yield_accelerator(req: AcquireYieldRequest):
    wl = workloads.get(req.workload_id)
    if not wl:
        raise HTTPException(404, f"workload {req.workload_id} not registered")

    pool = wl["pool"]
    checkpoint_ms = 0
    if MODE == "snapshot":
        log.info(f"Checkpointing {req.workload_id}")
        checkpoint_ms = call_workload_cr(req.workload_id, "checkpoint")
        wl["checkpointed"] = True

    pool_holders[pool] = None
    lock = get_pool_lock(pool)
    if lock.locked():
        lock.release()

    log.info(f"Yielded {req.workload_id} checkpoint_ms={checkpoint_ms:.0f}")
    _log_metric({"type": "yield", "workload_id": req.workload_id, "pool": pool,
                 "mode": MODE, "evict_ms": round(checkpoint_ms)})
    return {"status": "ok", "checkpoint_ms": round(checkpoint_ms)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=ORCH_PORT, log_level="info")
