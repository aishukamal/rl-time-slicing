"""sampler.py -- vLLM TPU sampler with dynamic weight loading

Serves GPT-2 on TPU v5e via vLLM's Python API (not subprocess).
Using the Python API directly gives us access to the model internals
for dynamic weight updates without restarting vLLM.

Endpoints:
  GET  /health
  POST /generate     {prompts, group_size, max_new_tokens, temperature}
  POST /reload_weights   downloads new weights from trainer and hot-swaps them
"""

import logging
import os
import threading
import time
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [tpu-sampler] %(levelname)s %(message)s"
)
log = logging.getLogger("tpu-sampler")

MODEL_PATH = os.environ.get("MODEL_PATH", "gpt2")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt2")
SAMPLER_PORT = int(os.environ.get("SAMPLER_PORT", "8300"))
DEFER_VLLM = os.environ.get("DEFER_VLLM", "false").lower() in ("true", "1", "yes")

GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "8"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.8"))
VLLM_MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "1024"))
VLLM_TP_SIZE = int(os.environ.get("VLLM_TP_SIZE", "1"))
TRAINER_URL = os.environ.get("TRAINER_URL", "http://grpo-trainer-svc:8200")

g_llm = None
g_tokenizer = None
g_ready = False
g_loading = False
g_engine_core_pid = None


def _clean_stale_sockets():
    """Remove sockets whose owning process is dead."""
    import glob
    for sock in glob.glob("/run/tpu_hal_*.sock"):
        pid_str = sock.replace("/run/tpu_hal_", "").replace(".sock", "")
        try:
            pid = int(pid_str)
            if not os.path.exists(f"/proc/{pid}"):
                log.info(f"Removing stale socket {sock} (PID {pid} dead)")
                os.remove(sock)
        except (ValueError, OSError) as e:
            log.warning(f"Could not clean socket {sock}: {e}")


def init_vllm():
    """Initialize vLLM engine with GPT-2 on TPU."""
    global g_llm, g_tokenizer, g_ready, g_engine_core_pid
    import glob
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    _clean_stale_sockets()

    before_socks = set(glob.glob("/run/tpu_hal_*.sock"))
    log.info(f"Sockets before init: {before_socks}")

    log.info(f"Initializing vLLM with model={MODEL_PATH} on TPU...")
    g_llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        max_model_len=VLLM_MAX_MODEL_LEN,
        tensor_parallel_size=VLLM_TP_SIZE,
    )
    g_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if g_tokenizer.pad_token is None:
        g_tokenizer.pad_token = g_tokenizer.eos_token

    after_socks = set(glob.glob("/run/tpu_hal_*.sock"))
    new_socks = after_socks - before_socks
    log.info(f"Sockets after init: {after_socks}, new: {new_socks}")

    if new_socks:
        sock = sorted(new_socks)[-1]
        pid_str = sock.replace("/run/tpu_hal_", "").replace(".sock", "")
        g_engine_core_pid = int(pid_str)
        log.info(f"EngineCore PID (from new socket): {g_engine_core_pid}")
    else:
        # Fallback: find the live socket whose PID is a child of our process
        for sock in sorted(after_socks):
            pid_str = sock.replace("/run/tpu_hal_", "").replace(".sock", "")
            try:
                pid = int(pid_str)
                if os.path.exists(f"/proc/{pid}"):
                    g_engine_core_pid = pid
                    log.info(f"EngineCore PID (fallback, live process): {pid}")
                    break
            except (ValueError, OSError):
                pass
        if not g_engine_core_pid:
            log.warning("No TPU HAL socket found after vLLM init")

    g_ready = True
    log.info(f"vLLM initialized and ready (EngineCore PID={g_engine_core_pid})")


app = FastAPI(title="TPU RL Sampler (vLLM)")


@app.on_event("startup")
async def on_startup():
    if DEFER_VLLM:
        log.info("DEFER_VLLM=true — vLLM will start when /start_vllm is called")
        return

    global g_loading
    g_loading = True

    def _boot():
        global g_loading
        try:
            init_vllm()
        except Exception as e:
            log.error(f"Failed to init vLLM: {e}")
        finally:
            g_loading = False

    threading.Thread(target=_boot, daemon=True).start()


@app.post("/start_vllm")
def start_vllm_endpoint():
    """Trigger deferred vLLM initialization."""
    global g_loading
    if g_ready:
        return {"status": "already_started"}
    if g_loading:
        return {"status": "already_loading"}

    g_loading = True

    def _boot():
        global g_loading
        try:
            init_vllm()
        except Exception as e:
            log.error(f"Failed to init vLLM: {e}")
        finally:
            g_loading = False

    threading.Thread(target=_boot, daemon=True).start()
    return {"status": "starting"}


class GenerateReq(BaseModel):
    prompts: List[str]
    group_size: int = GROUP_SIZE
    max_new_tokens: int = MAX_NEW_TOKENS
    temperature: float = TEMPERATURE


@app.get("/health")
def health():
    return {
        "status": "ok",
        "vllm_started": g_ready,
        "loading": g_loading,
        "model": MODEL_PATH,
        "backend": "tpu",
    }


@app.post("/generate")
def generate(req: GenerateReq):
    if not g_ready:
        raise HTTPException(503, "vLLM not ready yet (XLA compilation in progress)")

    from vllm import SamplingParams

    t0 = time.perf_counter()
    completions = []
    prompt_ids = []

    sampling = SamplingParams(
        max_tokens=req.max_new_tokens,
        temperature=req.temperature,
        n=req.group_size,
        stop=["<|endoftext|>"],
        repetition_penalty=1.1,
    )

    outputs = g_llm.generate(req.prompts, sampling)

    for idx, output in enumerate(outputs):
        for completion in output.outputs:
            completions.append(completion.text)
            prompt_ids.append(idx)

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    log.info(f"Generated {len(completions)} completions in {elapsed_ms}ms")

    return {
        "completions": completions,
        "prompt_ids": prompt_ids,
        "elapsed_ms": elapsed_ms,
    }


def _reload_weights_on_worker(worker, weights_path: str):
    """Called inside the EngineCore subprocess via collective_rpc.

    Loads safetensors from disk, converts to JAX arrays, and replaces
    matching entries in the runner's state dict + state_leaves.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from safetensors.numpy import load_file

    runner = worker.model_runner
    tensors = load_file(weights_path)
    state = runner.state

    n_updated = 0
    n_skipped = 0

    if isinstance(state, dict):
        # Build update mapping: pt_name -> jax array with correct shape/dtype/sharding
        updates = {}
        for pt_name, np_arr in tensors.items():
            vllm_key = "vllm_model." + pt_name
            if vllm_key in state:
                old = state[vllm_key]
                new = jnp.array(np_arr)
                if old.shape != new.shape:
                    if new.ndim == 2 and old.shape == new.T.shape:
                        new = new.T
                    else:
                        n_skipped += 1
                        continue
                if old.dtype != new.dtype:
                    new = new.astype(old.dtype)
                sharding = getattr(old, 'sharding', None)
                if sharding is not None:
                    new = jax.device_put(new, sharding)
                updates[vllm_key] = new
                n_updated += 1
            else:
                n_skipped += 1

        # GPT-2 ties lm_head.weight to wte.weight
        wte_key = "vllm_model.transformer.wte.weight"
        lm_head_key = "vllm_model.lm_head.weight"
        if wte_key in updates and lm_head_key in state:
            updates[lm_head_key] = updates[wte_key]

        # Build new state dict preserving keys not in the update
        new_state = {}
        for k, v in state.items():
            new_state[k] = updates.get(k, v)
        runner.state = new_state
        runner.state_leaves = new_state
    else:
        return {"error": f"unsupported state type: {type(state).__name__}"}

    return {"n_updated": n_updated, "n_skipped": n_skipped, "n_tensors": len(tensors)}


@app.post("/reload_weights")
def reload_weights():
    """Download updated weights from trainer and hot-swap into running vLLM."""
    import requests

    if not g_ready:
        raise HTTPException(503, "vLLM not ready")

    t0 = time.perf_counter()

    # 1. Tell trainer to export weights as safetensors
    log.info("Requesting weight export from trainer...")
    r = requests.post(f"{TRAINER_URL}/export_weights", timeout=120)
    if r.status_code != 200:
        raise HTTPException(500, f"Trainer export failed: {r.text[:500]}")

    # 2. Download weight files
    weights_dir = "/tmp/updated_weights"
    os.makedirs(weights_dir, exist_ok=True)
    for fname in ["model.safetensors", "config.json"]:
        log.info(f"Downloading {fname}...")
        r = requests.get(f"{TRAINER_URL}/weights/{fname}", timeout=300, stream=True)
        if r.status_code != 200:
            raise HTTPException(500, f"Failed to download {fname}: {r.status_code}")
        with open(os.path.join(weights_dir, fname), "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)

    download_s = time.perf_counter() - t0
    log.info(f"Weights downloaded in {download_s:.1f}s")

    # 3. Update weights in the EngineCore subprocess via collective_rpc
    log.info("Updating model weights in EngineCore...")
    t_load = time.perf_counter()

    weights_path = os.path.join(weights_dir, "model.safetensors")
    try:
        results = g_llm.llm_engine.collective_rpc(
            _reload_weights_on_worker,
            timeout=120,
            args=(weights_path,),
        )
        load_s = time.perf_counter() - t_load
        total_s = time.perf_counter() - t0
        log.info(f"Weight update complete: download={download_s:.1f}s load={load_s:.1f}s total={total_s:.1f}s results={results}")
        return {
            "status": "ok",
            "download_s": round(download_s, 1),
            "load_s": round(load_s, 1),
            "total_s": round(total_s, 1),
            "worker_results": results,
        }
    except Exception as e:
        load_s = time.perf_counter() - t_load
        log.error(f"Weight update failed after {load_s:.1f}s: {e}")
        raise HTTPException(500, f"Weight update failed: {e}")


@app.get("/debug_weights")
def debug_weights():
    if not g_ready:
        return {"error": "not ready"}
    def _check(worker):
        import jax.numpy as jnp
        state = worker.model_runner.state
        wte = state.get('vllm_model.transformer.wte.weight')
        lm = state.get('vllm_model.lm_head.weight')
        c_attn = state.get('vllm_model.transformer.h.0.attn.c_attn.weight')
        ln = state.get('vllm_model.transformer.h.0.ln_1.weight')
        return {
            'wte_sum': round(float(wte.sum()), 4) if wte is not None else None,
            'lm_sum': round(float(lm.sum()), 4) if lm is not None else None,
            'c_attn_sum': round(float(c_attn.sum()), 4) if c_attn is not None else None,
            'ln_sum': round(float(ln.sum()), 4) if ln is not None else None,
            'wte_shape': list(wte.shape) if wte is not None else None,
            'c_attn_shape': list(c_attn.shape) if c_attn is not None else None,
            'wte_first5': [round(float(x), 6) for x in wte.flatten()[:5]] if wte is not None else None,
            'wte_device': str(wte.devices()) if wte is not None else None,
            'c_attn_device': str(c_attn.devices()) if c_attn is not None else None,
        }
    try:
        return {"result": g_llm.llm_engine.collective_rpc(_check)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/debug_state")
def debug_state():
    """Debug: inspect model state type in the EngineCore."""
    if not g_ready:
        return {"error": "not ready"}
    def _inspect(worker):
        runner = worker.model_runner
        state = runner.state
        sl = runner.state_leaves
        info = {
            "state_type": type(state).__name__,
            "n_leaves": len(sl),
            "has_flat_state": hasattr(state, 'flat_state'),
            "is_dict": isinstance(state, dict),
        }
        if isinstance(state, dict):
            keys = list(state.keys())
            info["state_keys_sample"] = str(keys[:3])
            shapes = {}
            for k in keys:
                if 'c_attn' in k or 'c_proj' in k or 'c_fc' in k or 'lm_head' in k or 'wte' in k:
                    shapes[k] = str(state[k].shape)
            info["key_shapes"] = shapes
        return info
    try:
        result = g_llm.llm_engine.collective_rpc(_inspect)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@app.get("/get_pids")
def get_pids():
    return {"pids": [os.getpid()]}


def _find_my_tpu_socket():
    """Return the TPU HAL socket for this sampler's EngineCore."""
    if g_engine_core_pid:
        sock = f"/run/tpu_hal_{g_engine_core_pid}.sock"
        if os.path.exists(sock):
            return sock
    return None


@app.post("/checkpoint")
def checkpoint_endpoint():
    import grpc
    sock = _find_my_tpu_socket()
    if not sock:
        return {"error": "no TPU HAL socket found"}

    t0 = time.perf_counter()
    channel = grpc.insecure_channel(f"unix://{sock}")
    call = channel.unary_unary("/tpu.TpuHalService/Checkpoint",
        request_serializer=lambda x: x, response_deserializer=lambda x: x)
    call(b"", timeout=300)
    channel.close()
    elapsed = round((time.perf_counter() - t0) * 1000)
    log.info(f"Checkpoint complete via {sock}: {elapsed}ms")
    return {"status": "ok", "elapsed_ms": elapsed, "socket": sock}


@app.post("/restore")
def restore_endpoint():
    import grpc
    sock = _find_my_tpu_socket()
    if not sock:
        return {"error": "no TPU HAL socket found"}

    t0 = time.perf_counter()
    channel = grpc.insecure_channel(f"unix://{sock}")
    call = channel.unary_unary("/tpu.TpuHalService/Restore",
        request_serializer=lambda x: x, response_deserializer=lambda x: x)
    call(b"", timeout=300)
    channel.close()
    elapsed = round((time.perf_counter() - t0) * 1000)
    log.info(f"Restore complete via {sock}: {elapsed}ms")
    return {"status": "ok", "elapsed_ms": elapsed, "socket": sock}


@app.get("/get_tpu_stats")
def get_tpu_stats():
    """Return per-chip TPU memory and duty cycle."""
    try:
        from tpu_info.device import get_local_chips
        from tpu_info.metrics import get_chip_usage
        chip_type, _ = get_local_chips()
        usages = get_chip_usage(chip_type)
        chips = []
        for u in usages:
            chips.append({
                "device_id": u.device_id,
                "mem_used_bytes": u.memory_usage,
                "mem_total_bytes": u.total_memory,
                "mem_used_mib": round(u.memory_usage / (1024 * 1024)),
                "duty_cycle_pct": round(u.duty_cycle_pct, 1),
            })
        return {"chips": chips}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SAMPLER_PORT, log_level="info")
