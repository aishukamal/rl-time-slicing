"""gpu_client.py — orchestrator client library

acquire_gpu() uses a poll-based approach instead of a single long-lived
HTTP connection. This avoids timeout issues when waiting for the lock:
  1. POST /acquire_accelerators — returns immediately with status
     "queued" (lock not yet available) or "ok" (lock acquired)
  2. If "queued": poll GET /acquire_status?workload_id=X every few seconds
  3. Repeat until "ok" or timeout

This is more resilient than holding a single HTTP connection open for
minutes while waiting for another workload to yield.
"""

import functools, logging, os, requests, time
from typing import Callable, List, Optional
from config import RL_ORCH_PORT

log = logging.getLogger("gpu_client")

POLL_INTERVAL = 3  # seconds between status polls
DEFAULT_TIMEOUT = 7200  # 2 hours — more than enough for any cold start


class GpuClient:

  def __init__(
      self,
      workload_id: str,
      pool: str,
      orch_host: str = None,
      timeout: int = DEFAULT_TIMEOUT,
  ):
    host = orch_host or os.environ.get("ORCHESTRATOR_HOST", "localhost")
    port = int(os.environ.get("ORCH_PORT", str(RL_ORCH_PORT)))
    self.base = f"http://{host}:{port}"
    self.workload_id = workload_id
    self.pool = pool
    self.timeout = timeout

  def register(self, pids: List[str]) -> dict:
    """Register with orchestrator, retrying until it's reachable."""
    deadline = time.time() + 120  # wait up to 2 min for orchestrator
    attempt = 0
    while True:
      attempt += 1
      try:
        r = requests.post(
            f"{self.base}/register_workload",
            json={
                "workload_id": self.workload_id,
                "pool": self.pool,
                "pids": pids,
            },
            timeout=(5, 10),
        )  # (connect, read)
        r.raise_for_status()
        result = r.json()
        log.info(f"[{self.workload_id}] registered pool={self.pool}")
        return result
      except Exception as e:
        if time.time() > deadline:
          raise RuntimeError(
              f"[{self.workload_id}] could not reach orchestrator "
              f"after {attempt} attempts: {e}"
          )
        log.warning(
            f"[{self.workload_id}] orchestrator not ready "
            f"(attempt {attempt}), retrying in 3s... ({e})"
        )
        time.sleep(3)

  def update_pids(self, pids: List[str]) -> dict:
    r = requests.post(
        f"{self.base}/update_pids",
        json={"workload_id": self.workload_id, "pids": pids},
        timeout=30,
    )
    r.raise_for_status()
    log.info(f"[{self.workload_id}] updated pids={pids}")
    return r.json()

  def yield_gpu(self) -> dict:
    """Release GPU lock — fast, always returns quickly."""
    r = requests.post(
        f"{self.base}/yield_accelerators",
        json={"workload_id": self.workload_id},
        timeout=120,
    )
    r.raise_for_status()
    result = r.json()
    log.info(
        f"[{self.workload_id}] yielded evict_ms={result.get('evict_ms', 0)}"
    )
    return result

  def acquire_gpu(self) -> dict:
    """Acquire GPU lock — polls until granted.

    Sends a non-blocking request, then polls for status. Avoids long-lived HTTP
    connections that time out.
    """
    t0 = time.time()

    # Non-blocking enqueue — returns immediately
    r = requests.post(
        f"{self.base}/enqueue_acquire",
        json={"workload_id": self.workload_id},
        timeout=30,
    )
    r.raise_for_status()
    result = r.json()

    if result.get("status") == "ok":
      # Got it immediately (lock was free)
      log.info(f"[{self.workload_id}] acquired immediately")
      return result

    # Poll until granted
    log.info(f"[{self.workload_id}] queued for pool={self.pool}, polling...")
    while True:
      elapsed = time.time() - t0
      if elapsed > self.timeout:
        raise TimeoutError(
            f"[{self.workload_id}] timed out waiting for GPU lock after"
            f" {elapsed:.0f}s"
        )

      time.sleep(POLL_INTERVAL)

      r = requests.get(
          f"{self.base}/acquire_status",
          params={"workload_id": self.workload_id},
          timeout=10,
      )
      r.raise_for_status()
      result = r.json()

      if result.get("status") == "ok":
        elapsed = time.time() - t0
        log.info(f"[{self.workload_id}] acquired after {elapsed:.0f}s wait")
        return result

      if result.get("status") == "error":
        raise RuntimeError(f"[{self.workload_id}] acquire failed: {result}")

      # status == "queued" — keep polling
      if int(elapsed) % 30 == 0:
        log.info(
            f"[{self.workload_id}] still waiting for GPU lock"
            f" ({elapsed:.0f}s)..."
        )

  def gpu_step(self, func: Optional[Callable] = None):
    """Decorator: acquire before, yield after."""

    def decorator(f):
      @functools.wraps(f)
      def wrapper(*args, **kwargs):
        self.acquire_gpu()
        try:
          return f(*args, **kwargs)
        finally:
          self.yield_gpu()

      return wrapper

    return decorator(func) if func is not None else decorator
