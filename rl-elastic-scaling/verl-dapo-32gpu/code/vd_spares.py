# elastic-vd — spare sampler machinery for the 32-GPU DAPO showcase.
#
# Spares: 4 x TP4 vLLM engines parked on the 16 trainer GPUs (2 per trainer
# node), launched OUT-OF-TREE exactly like M1's R2 (vLLMReplica STANDALONE,
# workers=[], no Ray GPU accounting, NodeAffinity + NOSET env + explicit
# cuda_visible_devices — the in-tree server-actor mechanism,
# vllm_async_server.py:1197-1235) but with a server SUBCLASS because the
# stock server's sleep/wake are no-ops in STANDALONE mode (:808-820).
#
# Park mode: vLLM sleep level 2 (discard weights+KV; genbound measured
# ~2.9GB/GPU residual, zero step-time cost on the colocated trainer).
# Requires +enable_sleep_mode=True on the SPARE engine config only (the
# fleet stays stock enable_sleep_mode=False per the pinned shell).
#
# Weights: campaign-proven WCACHE pattern (design decision journaled in
# VD-STATE.md). One fleet replica dumps per-TP-rank safetensors to its
# node-local wcache dir right after every param_sync (collective_rpc with a
# CALLABLE — supported at the pin, vllm_async_server.py:228-240); trainer
# nodes prefetch over HTTP during trainer-busy; spares load rank-i -> rank-i
# on wake (TP4 == TP4, the 1:1 shard invariant from GB design decision 1).
# CORRECTNESS: fresh weights each wake (load MANDATORY after L2 sleep — the
# buffers wake_up() allocates are garbage), truthful version stamping
# (set_global_steps(wcache version), which equals the fleet's version by
# construction between syncs), invariant receipts logged per window.

import asyncio
import hashlib
import json
import os
import time
from uuid import uuid4

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.plugin.platform import get_platform
from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica

WCACHE_DIR = os.environ.get("VD_WCACHE_DIR", "/wcache")
WCACHE_HTTP_PORT = int(os.environ.get("VD_WCACHE_HTTP_PORT", "18080"))


# ============================================================================
# collective_rpc callables (run INSIDE vLLM WorkerProc; first arg = worker).
# Everything imported lazily inside; results are surfaced via receipt files +
# stdout (verl's server.collective_rpc discards return values at the pin).
# ============================================================================


def _names_fingerprint(state):
    """Deterministic fingerprint of (name, shape, dtype) — cheap invariant."""
    h = hashlib.sha256()
    for name, t in state:
        h.update(f"{name}|{tuple(t.shape)}|{t.dtype}\n".encode())
    return h.hexdigest()[:16]


def wcache_dump_shard(worker, dump_dir: str, version: int):
    """Dump this TP rank's local weights to {dump_dir}/model-tp{r}.safetensors."""
    import torch  # noqa: F401
    from safetensors.torch import save_file
    from vllm.distributed import get_tensor_model_parallel_rank

    t0 = time.time()
    rank = get_tensor_model_parallel_rank()
    model = worker.model_runner.model
    os.makedirs(dump_dir, exist_ok=True)

    cpu_state, items, total = {}, [], 0
    for name, p in model.named_parameters():
        cpu_state[name] = p.detach().to("cpu", copy=True).contiguous()
        items.append((name, p))
        total += p.numel() * p.element_size()
    fp = _names_fingerprint(sorted(items))  # sorted: must match load-side ordering

    path = os.path.join(dump_dir, f"model-tp{rank}.safetensors")
    save_file(cpu_state, path + ".tmp")
    os.replace(path + ".tmp", path)

    receipt = {
        "kind": "dump",
        "version": int(version),
        "tp_rank": rank,
        "ntensors": len(cpu_state),
        "bytes": total,
        "fingerprint": fp,
        "seconds": round(time.time() - t0, 2),
    }
    with open(os.path.join(dump_dir, f"dump_receipt_tp{rank}.json"), "w") as f:
        json.dump(receipt, f)
    print(f"[vd-wcache] {receipt}", flush=True)


def wcache_load_shard(worker, load_dir: str, version: int):
    """Load rank-i shard into this TP rank's params, in place. Raises on ANY
    mismatch (propagates through collective_rpc -> visible switch failure)."""
    import torch
    from safetensors.torch import safe_open
    from vllm.distributed import get_tensor_model_parallel_rank

    t0 = time.time()
    rank = get_tensor_model_parallel_rank()
    path = os.path.join(load_dir, f"model-tp{rank}.safetensors")
    dump_receipt_path = os.path.join(load_dir, f"dump_receipt_tp{rank}.json")
    with open(dump_receipt_path) as f:
        dump_receipt = json.load(f)
    if int(dump_receipt["version"]) != int(version):
        raise RuntimeError(
            f"wcache version mismatch: dir has v{dump_receipt['version']}, controller wants v{version}"
        )

    model = worker.model_runner.model
    params = dict(model.named_parameters())
    fp = _names_fingerprint(sorted(params.items()))
    loaded, total = 0, 0
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        missing = set(params) - keys
        extra = keys - set(params)
        if missing or extra:
            raise RuntimeError(
                f"wcache shard key mismatch tp{rank}: missing={sorted(missing)[:5]} "
                f"extra={sorted(extra)[:5]} (counts {len(missing)}/{len(extra)})"
            )
        with torch.no_grad():
            for name in f.keys():
                t = f.get_tensor(name)
                p = params[name]
                if tuple(p.shape) != tuple(t.shape):
                    raise RuntimeError(f"shape mismatch {name}: param {tuple(p.shape)} vs shard {tuple(t.shape)}")
                p.copy_(t.to(p.device, p.dtype, non_blocking=True))
                loaded += 1
                total += t.numel() * t.element_size()
    torch.cuda.synchronize()

    receipt = {
        "kind": "load",
        "version": int(version),
        "tp_rank": rank,
        "ntensors": loaded,
        "bytes": total,
        "model_fingerprint": fp,
        "dump_fingerprint": dump_receipt["fingerprint"],
        "seconds": round(time.time() - t0, 2),
    }
    with open(os.path.join(load_dir, f"load_receipt_tp{rank}_{os.getpid()}.json"), "w") as f:
        json.dump(receipt, f)
    print(f"[vd-wcache] {receipt}", flush=True)


# ============================================================================
# Spare server subclass (STANDALONE server methods no-op sleep/wake at pin)
# ============================================================================


class ElasticSpareServer(vLLMHttpServer):
    async def spare_sleep(self, level: int = 1) -> dict:
        """L1: weights -> CPU + KV freed (default; ~1-3s at 7B TP4).
        L2: discard both (big-model mode; wake buffers are garbage)."""
        t0 = time.time()
        await self.engine.reset_prefix_cache()
        await self.engine.sleep(level=int(level))
        return {"seconds": round(time.time() - t0, 2), "level": int(level)}

    async def spare_wake(self) -> dict:
        t0 = time.time()
        await self.engine.wake_up()  # reallocates weights (GARBAGE until wcache load) + KV
        await self.engine.reset_prefix_cache()
        return {"seconds": round(time.time() - t0, 2)}

    async def spare_load_wcache(self, load_dir: str, version: int) -> dict:
        t0 = time.time()
        await self.engine.collective_rpc(method=wcache_load_shard, args=(load_dir, int(version)))
        await self.set_global_steps(int(version))  # truthful stamp: weights ARE version v
        return {"seconds": round(time.time() - t0, 2), "version": int(version)}

    async def spare_probe_generate(self, prompt_ids, max_tokens: int = 8):
        out = await self.generate(
            prompt_ids=prompt_ids,
            sampling_params={"max_tokens": max_tokens},
            request_id=uuid4().hex,
        )
        return len(out.token_ids)


# ============================================================================
# Per-node helper agent: wcache HTTP prefetch + receipt collection.
# Zero-GPU Ray actor pinned by node id; lives next to the spares/fleet dirs.
# ============================================================================


@ray.remote(num_cpus=1)
class VDNodeAgent:
    """Runs on a specific node. Serves (fleet node) or prefetches (trainer
    node) the wcache, and reads receipt files for the invariant log."""

    def __init__(self, role: str):
        self.role = role
        self._http = None
        os.makedirs(WCACHE_DIR, exist_ok=True)

    def node_ip(self):
        return ray.util.get_node_ip_address()

    def start_http_server(self):
        """Fleet node: serve WCACHE_DIR over HTTP. Idempotent: if the port is
        already bound on this node (this agent, or an orphan from a previous
        run's agent), REUSE it — /wcache is the node hostPath, so any server
        on this node serves the same files."""
        import socket
        import subprocess
        url = f"http://{self.node_ip()}:{WCACHE_HTTP_PORT}"
        try:
            with socket.create_connection(("127.0.0.1", WCACHE_HTTP_PORT), timeout=2):
                return url  # something already serving on this node
        except OSError:
            pass
        self._http = subprocess.Popen(
            ["python3", "-m", "http.server", str(WCACHE_HTTP_PORT), "--directory", WCACHE_DIR],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(1.0)
        return url

    def prefetch(self, src_url: str, version: int) -> dict:
        """Trainer node: download v{version} shard set from the fleet node."""
        import urllib.request
        t0 = time.time()
        dst = os.path.join(WCACHE_DIR, f"v{version}")
        tmp = dst + ".tmp"
        os.makedirs(tmp, exist_ok=True)
        files = [f"model-tp{r}.safetensors" for r in range(4)] + [f"dump_receipt_tp{r}.json" for r in range(4)]
        total = 0
        for fname in files:
            with urllib.request.urlopen(f"{src_url}/v{version}/{fname}", timeout=600) as r, \
                 open(os.path.join(tmp, fname), "wb") as w:
                while True:
                    chunk = r.read(1 << 24)
                    if not chunk:
                        break
                    w.write(chunk)
                    total += len(chunk)
        os.replace(tmp, dst)
        # prune older versions (keep current + previous)
        vers = sorted(
            (int(d[1:]) for d in os.listdir(WCACHE_DIR) if d.startswith("v") and d[1:].isdigit()),
            reverse=True,
        )
        for v in vers[2:]:
            import shutil
            shutil.rmtree(os.path.join(WCACHE_DIR, f"v{v}"), ignore_errors=True)
        return {"version": version, "bytes": total, "seconds": round(time.time() - t0, 2), "dst": dst}

    def read_receipts(self, version: int) -> list:
        d = os.path.join(WCACHE_DIR, f"v{version}")
        out = []
        if os.path.isdir(d):
            for fname in sorted(os.listdir(d)):
                if fname.endswith(".json"):
                    try:
                        with open(os.path.join(d, fname)) as f:
                            out.append({"file": fname, **json.load(f)})
                    except Exception as e:
                        out.append({"file": fname, "error": str(e)})
        return out

    def has_version(self, version: int) -> bool:
        d = os.path.join(WCACHE_DIR, f"v{version}")
        return os.path.isdir(d) and all(
            os.path.exists(os.path.join(d, f"model-tp{r}.safetensors")) for r in range(4)
        )


# ============================================================================
# Spare launch (driver-side)
# ============================================================================


def launch_spare(config, model_config_holder, replica_rank: int, node_id: str, cuda_devices: str):
    """Launch one TP4 spare on `node_id` GPUs `cuda_devices` (e.g. '0,1,2,3').

    Mirrors vLLMReplica.launch_servers (vllm_async_server.py:1197-1247) with:
      - server_class = ElasticSpareServer,
      - workers=[]  (never in the weight-sync group; DP=1 never hits the
        self.workers asserts, per M1 deliverable D),
      - no Ray GPU: platform NOSET env + explicit cuda_visible_devices,
      - spare-only engine overrides applied by the caller on the config copy
        (enable_sleep_mode=True, gpu_memory_utilization, load_format auto).
    """
    replica = vLLMReplica(
        replica_rank=replica_rank,
        config=config,
        model_config=model_config_holder,
        gpus_per_node=4,
    )
    replica.rollout_mode = RolloutMode.STANDALONE
    replica.server_class = ray.remote(ElasticSpareServer)

    env_vars = {
        **{var: "1" for var in get_platform().ray_noset_envvars()},
        **get_platform().rollout_env_vars(),
    }
    name = f"vllm_server_{replica_rank}_0"  # in-tree convention; fleet uses ranks 0..3

    server = replica.server_class.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
        runtime_env={"env_vars": env_vars},
        name=name,
        max_concurrency=replica.max_concurrency,
    ).remote(
        config=replica.config,
        model_config=replica.model_config,
        rollout_mode=RolloutMode.STANDALONE,
        workers=[],
        replica_rank=replica_rank,
        node_rank=0,
        gpus_per_node=4,
        nnodes=1,
        cuda_visible_devices=cuda_devices,
    )

    ray.get(server.launch_server.remote())
    server_address, server_port = ray.get(server.get_server_address.remote())

    replica.servers = [server]
    replica._server_handle = server
    replica._server_address = (
        f"[{server_address}]:{server_port}"
        if is_valid_ipv6_address(server_address)
        else f"{server_address}:{server_port}"
    )
    return replica


async def fleet_dump(fleet_server, version: int, dump_root: str = WCACHE_DIR) -> str:
    """Trigger the per-rank dump on ONE fleet replica's server. Returns dir."""
    dump_dir = os.path.join(dump_root, f"v{int(version)}")
    await fleet_server.collective_rpc.remote(
        method=wcache_dump_shard, args=(dump_dir, int(version))
    )
    return dump_dir
