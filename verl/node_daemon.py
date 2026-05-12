"""node_daemon.py — runs as a DaemonSet on each GPU node

Exposes APIs consumed by the orchestrator:
  POST /register_workload {workload_id, pids}  → resolve pod PIDs to host PIDs
  POST /evict_context     {workload_id, pids}  → dump GPU state to host RAM
  POST /restore_context   {workload_id, pids}  → restore GPU state from host RAM
  GET  /gpu_stats
  GET  /health

Uses cuda-checkpoint --action=dump/restore internally.
Resolves pod-namespace PIDs to host PIDs at registration time.
"""

import glob, logging, os, subprocess, time
from config import *
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [daemon] %(levelname)s %(message)s"
)
log = logging.getLogger("daemon")

NVIDIA_ENV = {
    **os.environ,
    "PATH": os.environ.get("PATH", "") + ":/usr/local/nvidia/bin",
}

app = FastAPI(title="Node Daemon")

# workload_id → [host_pid, ...]  resolved at registration time
_workload_host_pids: dict = {}


class WorkloadPidsRequest(BaseModel):
  workload_id: str
  pool: str = "unknown"
  pids: list[str]  # pod-namespace PIDs from the workload pod


class ContextRequest(BaseModel):
  pool: str  # "trainer" or "sampler"
  pids: list[str]  # pod-namespace PIDs (fallback if workload_id unknown)
  workload_id: str = ""  # preferred — used to look up resolved host PIDs


def _get_gpu_host_pids(pool: str = "unknown") -> set:
  """Return set of host PIDs currently using the pool's target GPU via nvidia-smi."""
  try:
    r = subprocess.run(
        [
            "/usr/local/nvidia/bin/nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        env=NVIDIA_ENV,
    )
    return {p.strip() for p in r.stdout.strip().splitlines() if p.strip()}
  except Exception:
    return set()


def _get_process_name(pid: str) -> str:
  import psutil
  try:
    proc = psutil.Process(int(pid))
    cmd = " ".join(proc.cmdline())
    return cmd[:80] + ("..." if len(cmd) > 80 else "")
  except Exception:
    return "unknown"


def _find_host_pid(pod_pid: str, already_registered: set, pool: str) -> str:
  """Find the host PID for a pod-namespace PID by: 1.

  Scanning /proc for all host PIDs whose innermost NSpid matches pod_pid 2.
  Cross-referencing with nvidia-smi (on the specific pool GPU) to find the one using the GPU 3. Excluding
  already-registered host PIDs
  """
  gpu_pids = _get_gpu_host_pids(pool)
  candidates = []

  for status_file in glob.glob("/proc/[0-9]*/status"):
    try:
      host_pid = status_file.split("/")[2]
      with open(status_file) as f:
        for line in f:
          if line.startswith("NSpid:"):
            parts = line.split()[1:]
            if len(parts) > 1 and parts[-1] == pod_pid:
              if host_pid not in already_registered:
                candidates.append(host_pid)
            break
    except Exception:
      continue

  if not candidates:
    log.warning(f"  No candidates found for pod PID {pod_pid}")
    return None

  # ONLY return the candidate that is using the GPU to avoid crashing CPU-only processes
  for host_pid in candidates:
    if host_pid in gpu_pids:
      name = _get_process_name(host_pid)
      name_lower = name.lower()
      
      # Robustly filter by Actor name since Trainer and Sampler share the same KubeRay Pod
      if "checkpointengineworker" in name_lower:
          log.info(f"  pod PID {pod_pid} → host PID {host_pid} [{name}] matched GPU, but excluded (never evict orchestrators)")
          continue
      
      if pool == "sampler" and "vllm" not in name_lower:
          log.info(f"  pod PID {pod_pid} → host PID {host_pid} [{name}] matched GPU, but excluded (not VLLM)")
          continue
          
      if pool == "trainer" and "workerdict" not in name_lower:
          log.info(f"  pod PID {pod_pid} → host PID {host_pid} [{name}] matched GPU, but excluded (not FSDP Trainer)")
          continue
          
      log.info(f"  pod PID {pod_pid} → host PID {host_pid} [{name}] (via nvidia-smi)")
      return host_pid

  log.warning(f"  pod PID {pod_pid} has no corresponding host PID in nvidia-smi. Not checkpointing.")
  return None


def _get_process_state(host_pid: str) -> str:
  """Get cuda-checkpoint state: 'running', 'checkpointed', or 'unknown'."""
  try:
    r = subprocess.run(
        [CUDA_CKPT_BIN, "--get-state", "--pid", host_pid],
        capture_output=True,
        text=True,
        env=NVIDIA_ENV,
        timeout=10,
    )
    return r.stdout.strip().lower() if r.returncode == 0 else "unknown"
  except Exception:
    return "unknown"


def _checkpoint(
    host_pid: str, action: str, retries: int = 3, retry_delay: float = 2.0
) -> None:
  """Run cuda-checkpoint --toggle with retries and state validation.

  --toggle handles locking internally and alternates dump/restore based on
  current process state.
  """
  expected_state = "running" if action == "evict" else "checkpointed"

  for attempt in range(1, retries + 1):
    state = _get_process_state(host_pid)

    if state == "unknown":
      log.warning(
          f"  pid={host_pid} state=unknown (attempt {attempt}/{retries}),"
          " retrying..."
      )
      time.sleep(retry_delay)
      continue

    # Already in desired state — no-op
    if action == "evict" and state == "checkpointed":
      log.info(f"  pid={host_pid} already checkpointed — skipping")
      return
    if action == "restore" and state == "running":
      log.info(f"  pid={host_pid} already running — skipping")
      return

    if state != expected_state:
      log.warning(
          f"  pid={host_pid} state={state} (expected {expected_state}), "
          f"attempt {attempt}/{retries}"
      )
      time.sleep(retry_delay)
      continue

    log.info(f"host_pid={host_pid} | native_ckpt_state={state} | action={action}")

    r = subprocess.run(
        [CUDA_CKPT_BIN, "--toggle", "--pid", host_pid],
        capture_output=True,
        text=True,
        env=NVIDIA_ENV,
        timeout=120,
    )
    if r.returncode == 0:
      return
    err = r.stderr.strip()
    log.warning(
        f"  cuda-checkpoint toggle failed pid={host_pid} "
        f"(attempt {attempt}/{retries}): {err[:200]}"
    )
    if attempt < retries:
      time.sleep(retry_delay)

  raise RuntimeError(
      f"cuda-checkpoint toggle failed pid={host_pid} after {retries} attempts"
  )


def _resolve_pids(workload_id: str, pod_pids: list[str]) -> list[str]:
  """Return host PIDs for a workload.

  Uses pre-resolved mapping if available, otherwise translates on the fly (less
  reliable when multiple pods share the same pod-namespace PID).
  """
  if workload_id and workload_id in _workload_host_pids:
    return _workload_host_pids[workload_id]
  # Fallback: translate each pod PID individually with no exclusions
  return [_find_host_pid(p, set(), "unknown") for p in pod_pids]


def _get_gpu_mib() -> dict:
  try:
    r = subprocess.run(
        [
            "/usr/local/nvidia/bin/nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        env=NVIDIA_ENV,
    )
    gpus = []
    for line in r.stdout.strip().splitlines():
      idx, used, total, util = [x.strip() for x in line.split(",")]
      gpus.append({
          "gpu": int(idx),
          "used_mib": int(used),
          "total_mib": int(total),
          "util_pct": int(util),
      })
    return {"gpus": gpus}
  except Exception as e:
    return {"error": str(e)}


@app.get("/health")
def health():
  return {"status": "ok"}


@app.get("/gpu_stats")
def gpu_stats():
  return _get_gpu_mib()


@app.post("/register_workload")
def register_workload(req: WorkloadPidsRequest):
  """Resolve pod PIDs to host PIDs.
  When multiple pods share the same pod-namespace PID, skip host PIDs already
  registered to other workloads.
  """
  # VERL POC SHORTCUT: If the user sends the bypass flag, scan the global node for the pids instead!
  if req.pids and req.pids[0] == "bypass_pod_logic":
    shortcut_pids = __verl_poc_shortcut_get_pids(req.pool)
    if shortcut_pids:
      existing = _workload_host_pids.get(req.workload_id, [])
      new_pids = list(set(existing + shortcut_pids))
      _workload_host_pids[req.workload_id] = new_pids
      log.info(f"SHORTCUT registered {req.workload_id}: cached global PIDs -> {shortcut_pids}")
      return {"status": "ok", "workload_id": req.workload_id, "host_pids": shortcut_pids}
    else:
      log.warning(f"SHORTCUT failed: could not find {req.pool} PIDs on gpu")
      return {"status": "ok", "workload_id": req.workload_id, "host_pids": []}

  already_registered = {
      pid
      for wid, pids in _workload_host_pids.items()
      if wid != req.workload_id
      for pid in pids
  }

  host_pids = []
  for pod_pid in req.pids:
    found = _find_host_pid(pod_pid, already_registered, req.pool)
    if found is not None:
      host_pids.append(found)

  existing = _workload_host_pids.get(req.workload_id, [])
  new_pids = list(set(existing + host_pids))
  _workload_host_pids[req.workload_id] = new_pids
  log.info(
      f"registered {req.workload_id}: pod_pids={req.pids} →"
      f" host_pids={host_pids}"
  )
  return {
      "status": "ok",
      "workload_id": req.workload_id,
      "host_pids": host_pids,
  }


def __verl_poc_shortcut_get_pids(pool: str) -> list[str]:
  """SHORTCUT FOR VERL POC: Ignore registered pod PIDs and globally search the host.
  This allows workloads to bypass registering their pod PIDs from the workers.
  """
  try:
    r = subprocess.run(
        ["/usr/local/nvidia/bin/nvidia-smi", "pmon", "-c", "1", "-s", "u"],
        capture_output=True, text=True, env=NVIDIA_ENV, timeout=10
    )
    if r.returncode != 0: return []

    gpu_pids = set()
    for line in r.stdout.splitlines():
      if line.startswith("#") or not line.strip(): continue
      parts = line.split()
      if len(parts) >= 3 and parts[1].isdigit():
        gpu_pids.add(parts[1])

    pids = []
    for host_pid in gpu_pids:
      try:
        with open(f"/proc/{host_pid}/cmdline", "r") as f:
          cmdline = f.read().replace('\x00', ' ').lower()
          # Check if the process is a valid target using our existing native tool
          state = _get_process_state(host_pid)
          log.info(f"SHORTCUT scanning {host_pid} | pool={pool} | native_ckpt_state={state}")

          if pool == "sampler" and "vllm" in cmdline:
            pids.append(host_pid)
          elif pool == "trainer" and "workerdict" in cmdline:
            pids.append(host_pid)
      except Exception:
        pass
    return pids
  except Exception as e:
    log.error(f"Shortcut finding pids error: {e}")
    return []


@app.post("/evict_context")
def evict_context(req: ContextRequest):
  """Checkpoint all PIDs for a workload → GPU memory freed, state in host RAM."""
  host_pids = _resolve_pids(req.workload_id, req.pids)
  gpu_before = _get_gpu_mib()
  log.info(
      f"evict pool={req.pool} workload={req.workload_id} host_pids={host_pids} | PRE-EVICT GPU: {gpu_before}"
  )
  t0 = time.perf_counter()
  errors = []
  for host_pid in host_pids:
    try:
      name = _get_process_name(host_pid)
      log.info(f"  evicting {host_pid} [{name}]")
      _checkpoint(host_pid, action="evict")
    except RuntimeError as e:
      errors.append(str(e))
      log.error(str(e))
  ms = (time.perf_counter() - t0) * 1000
  if errors:
    raise HTTPException(500, {"errors": errors})
  gpu = _get_gpu_mib()
  log.info(f"  evict done {ms:.0f}ms | {gpu}")
  return {"status": "ok", "pool": req.pool, "elapsed_ms": round(ms), "gpu": gpu}


@app.post("/restore_context")
def restore_context(req: ContextRequest):
  """Restore all PIDs for a workload from host RAM back to GPU."""
  host_pids = _resolve_pids(req.workload_id, req.pids)
  log.info(
      f"restore pool={req.pool} workload={req.workload_id}"
      f" host_pids={host_pids}"
  )
  t0 = time.perf_counter()
  errors = []
  for host_pid in reversed(host_pids):
    try:
      name = _get_process_name(host_pid)
      log.info(f"  restoring {host_pid} [{name}]")
      _checkpoint(host_pid, action="restore")
    except RuntimeError as e:
      errors.append(str(e))
      log.error(str(e))
  ms = (time.perf_counter() - t0) * 1000
  if errors:
    raise HTTPException(500, {"errors": errors})
  gpu = _get_gpu_mib()
  log.info(f"  restore done {ms:.0f}ms | {gpu}")
  return {"status": "ok", "pool": req.pool, "elapsed_ms": round(ms), "gpu": gpu}


if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=RL_DAEMON_PORT, log_level="info")
