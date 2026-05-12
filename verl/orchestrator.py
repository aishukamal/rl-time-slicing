"""orchestrator.py — GPU pool scheduling service

Two independent GPU pools, each with its own lock and queue:
  pool "trainer" — GPU0 node, trainerA and trainerB queue for it
  pool "sampler" — GPU1 node, samplerA and samplerB queue for it

These pools are completely independent. A workload holding the trainer
pool lock has no effect on who holds the sampler pool lock, and vice versa.

The RL ordering constraint (train(N) before rollout(N+1)) is enforced by
the rl_loop application logic, not by this service. The orchestrator only
manages "who holds the GPU right now" within each pool.

APIs:
  POST /register_workload   {workload_id, pool, pids}
  POST /update_pids         {workload_id, pids}
  POST /yield_accelerators  {workload_id}      → evict + release lock
  POST /acquire_accelerators {workload_id}     → block until free + restore
  GET  /status
  GET  /metrics

Modes:
  timeslice — yield/acquire trigger real cuda-checkpoint swaps via daemon
  baseline  — yield/acquire are no-ops (workloads keep GPU forever)
"""

import asyncio, json, logging, os, time
from collections import defaultdict
from typing import Dict, List, Optional

from config import *
from fastapi import FastAPI, HTTPException
import httpx
from pydantic import BaseModel
import uvicorn

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [orch] %(levelname)s %(message)s"
)
log = logging.getLogger("orchestrator")

DAEMON_BASE = f"http://gpu-swap-daemon:{RL_DAEMON_PORT}"

app = FastAPI(title="GPU Orchestrator")


# ── State ─────────────────────────────────────────────────────────────────────

# One asyncio.Lock per pool — completely independent
_pool_locks: Dict[str, asyncio.Lock] = {}
_pool_holder: Dict[str, Optional[str]] = {}  # pool → workload_id holding it
_workload_pids: Dict[str, List[str]] = {}  # workload_id → GPU PIDs
_workload_pool: Dict[str, str] = {}  # workload_id → pool name
_step_counter: Dict[str, int] = defaultdict(int)
_metrics: List[dict] = []


def _get_pool_lock(pool: str) -> asyncio.Lock:
  if pool not in _pool_locks:
    _pool_locks[pool] = asyncio.Lock()
    _pool_holder[pool] = None
  return _pool_locks[pool]


def _log_metric(record: dict):
  record["ts"] = time.time()
  _metrics.append(record)
  os.makedirs(LOG_DIR, exist_ok=True)
  with open(METRICS_FILE, "a") as f:
    f.write(json.dumps(record) + "\n")


# ── Request models ────────────────────────────────────────────────────────────


class RegisterRequest(BaseModel):
  workload_id: str
  pool: str
  pids: List[str]


class WorkloadRequest(BaseModel):
  workload_id: str


class UpdatePidsRequest(BaseModel):
  workload_id: str
  pids: List[str]


# ── Startup ───────────────────────────────────────────────────────────────────


@app.on_event("startup")
async def startup():
  for pool in ("trainer", "sampler"):
    _get_pool_lock(pool)
  log.info(
      f"Orchestrator ready | mode={MODE} | pools: trainer, sampler"
      " (independent)"
  )


# ── Health / status ───────────────────────────────────────────────────────────


@app.get("/health")
def health():
  return {"status": "ok", "mode": MODE}


@app.get("/status")
def status():
  return {
      "mode": MODE,
      "holders": _pool_holder,
      "workloads": {
          wid: {"pool": _workload_pool[wid], "pids": _workload_pids[wid]}
          for wid in _workload_pids
      },
  }


@app.get("/metrics")
def metrics():
  return {"metrics": _metrics[-50:]}


# ── Registration ──────────────────────────────────────────────────────────────


@app.post("/register_workload")
async def register_workload(req: RegisterRequest):
  """Register a workload with its pool and PIDs.

  The first workload to register on a pool gets the lock immediately. Subsequent
  workloads queue up behind it.
  """
  _workload_pids[req.workload_id] = req.pids
  _workload_pool[req.workload_id] = req.pool
  log.info(
      f"registered workload={req.workload_id} pool={req.pool} pids={req.pids}"
  )
  _log_metric({
      "type": "register",
      "workload_id": req.workload_id,
      "pool": req.pool,
      "pids": req.pids,
  })

  # Do NOT pre-assign the lock here. The workload calls acquire_gpu()
  # immediately after registering. The first caller acquires freely
  # (lock is unheld), subsequent callers queue behind it.
  # Pre-assigning caused a deadlock: register gave the lock, then
  # lifespan called acquire_gpu() which tried to acquire it again.
  # Reset step counter on re-registration (handles pod restarts).
  # A re-registering workload is treated as fresh — no restore on first acquire.
  if req.workload_id in _step_counter and _step_counter[req.workload_id] > 0:
    log.info(
        f"  {req.workload_id} RE-REGISTERED (was step"
        f" {_step_counter[req.workload_id]}) — resetting state"
    )
    _step_counter[req.workload_id] = 0
  log.info(f"  {req.workload_id} registered, will acquire via acquire_gpu()")
  return {"status": "ok", "initial_holder": False}


@app.post("/update_pids")
async def update_pids(req: UpdatePidsRequest):
  """Update GPU PIDs after vLLM/trainer cold-starts.

  Also registers the workload with the daemon so it can resolve pod PIDs to host
  PIDs unambiguously while this is the only active workload.
  """
  if req.workload_id not in _workload_pids:
    raise HTTPException(404, f"workload {req.workload_id} not registered")
    
  existing = _workload_pids.get(req.workload_id, [])
  new_pids = list(set(existing + req.pids))
  _workload_pids[req.workload_id] = new_pids
  log.info(f"updated pids workload={req.workload_id} pids={req.pids} (total={len(new_pids)})")

  # Register with daemon for unambiguous PID resolution
  if MODE == "timeslice" and req.pids:
    pool = _workload_pool.get(req.workload_id, "unknown")
    try:
      async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{DAEMON_BASE}/register_workload",
            json={"workload_id": req.workload_id, "pool": pool, "pids": req.pids},
        )
      if r.status_code == 200:
        log.info(
            f"  daemon registered {req.workload_id}:"
            f" {r.json().get('host_pids')}"
        )
      else:
        log.warning(f"  daemon register failed: {r.text[:200]}")
    except Exception as e:
      log.warning(f"  daemon register error: {e}")

  _log_metric(
      {"type": "update_pids", "workload_id": req.workload_id, "pids": req.pids}
  )
  return {"status": "ok"}


# ── Yield / Acquire ───────────────────────────────────────────────────────────


@app.post("/yield_accelerators")
async def yield_accelerators(req: WorkloadRequest):
  """Yield GPU: evict this workload (cuda-checkpoint) and release pool lock.

  Only operates on the workload's own pool — does not affect other pools.
  """
  wid = req.workload_id
  pool = _workload_pool.get(wid)
  if pool is None:
    raise HTTPException(404, f"workload {wid} not registered")
  if _pool_holder.get(pool) != wid:
    log.info(f"  {wid} does not hold lock for pool={pool}, idempotent yield ok")
    return {"status": "ok"}

  t0 = time.perf_counter()
  evict_ms = 0

  if MODE == "timeslice":
    pids = _workload_pids.get(wid, [])
    if pids:
      log.info(f"yield workload={wid} pool={pool} — evicting pids={pids}")
      async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{DAEMON_BASE}/evict_context",
            json={"pool": pool, "pids": pids, "workload_id": wid},
        )
        if r.status_code != 200:
          raise HTTPException(500, f"evict_context failed: {r.text}")
      evict_ms = round((time.perf_counter() - t0) * 1000)
      log.info(f"  evicted in {evict_ms}ms")
    else:
      log.info(f"yield workload={wid} pool={pool} — no pids, skip evict")
  else:
    log.info(f"yield workload={wid} pool={pool} [baseline: no eviction]")

  _pool_holder[pool] = None
  _get_pool_lock(pool).release()

  _log_metric({
      "type": "yield",
      "workload_id": wid,
      "pool": pool,
      "mode": MODE,
      "evict_ms": evict_ms,
  })
  return {"status": "ok", "evict_ms": evict_ms}


_acquire_status: Dict[str, str] = {}  # workload_id → "queued"|"ok"|"error"


@app.post("/enqueue_acquire")
async def enqueue_acquire(req: WorkloadRequest):
  """Non-blocking acquire.

  Returns immediately:

    {"status": "ok"}     — lock granted (was free)
    {"status": "queued"} — waiting, poll /acquire_status
  """
  wid = req.workload_id
  pool = _workload_pool.get(wid)
  if pool is None:
    raise HTTPException(404, f"workload {wid} not registered")

  if _pool_holder.get(pool) == wid:
    log.info(f"  {wid} already holds lock for pool={pool}, idempotent ok")
    return {"status": "ok"}

  lock = _get_pool_lock(pool)
  step = _step_counter[wid]

  async def _do_acquire():
    t0 = time.perf_counter()
    log.info(
        f"enqueue_acquire workload={wid} pool={pool} step={step} — waiting..."
    )
    await lock.acquire()
    _pool_holder[pool] = wid
    wait_ms = round((time.perf_counter() - t0) * 1000)
    log.info(f"  {wid} acquired pool={pool} after {wait_ms}ms")

    restore_ms = 0
    if MODE == "timeslice" and step > 0:
      pids = _workload_pids.get(wid, [])
      if pids:
        try:
          async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{DAEMON_BASE}/restore_context",
                json={"pool": pool, "pids": pids, "workload_id": wid},
            )
          if r.status_code != 200:
            log.error(f"  restore failed: {r.text}")
            _acquire_status[wid] = "error"
            return
          restore_ms = round((time.perf_counter() - t0) * 1000) - wait_ms
          log.info(f"  restored in {restore_ms}ms")
        except Exception as e:
          log.error(f"  restore error: {e}")
          _acquire_status[wid] = "error"
          return

    _step_counter[wid] += 1
    _log_metric({
        "type": "acquire",
        "workload_id": wid,
        "pool": pool,
        "step": step,
        "mode": MODE,
        "wait_ms": wait_ms,
        "restore_ms": restore_ms,
    })
    _acquire_status[wid] = "ok"

  _acquire_status[wid] = "queued"
  asyncio.create_task(_do_acquire())

  # If lock was free the task completes almost instantly — check immediately
  await asyncio.sleep(0)
  return {"status": _acquire_status.get(wid, "queued")}


@app.get("/acquire_status")
async def acquire_status(workload_id: str):
  """Poll after enqueue_acquire returns 'queued'."""
  status = _acquire_status.get(workload_id, "unknown")
  return {"status": status, "workload_id": workload_id}


@app.post("/acquire_accelerators")
async def acquire_accelerators(req: WorkloadRequest):
  """Acquire GPU: block until this pool's lock is free, then restore.

  Only operates on the workload's own pool — does not affect other pools.

  Two workloads on DIFFERENT pools can hold their locks simultaneously:
    trainer holds pool="trainer" lock  AND
    sampler holds pool="sampler" lock
  → both fine, independent locks.

  Two workloads on the SAME pool queue up:
    trainerA holds pool="trainer" lock
    trainerB blocks until trainerA yields
  """
  wid = req.workload_id
  pool = _workload_pool.get(wid)
  if pool is None:
    raise HTTPException(404, f"workload {wid} not registered")

  t0 = time.perf_counter()
  step = _step_counter[wid]
  lock = _get_pool_lock(pool)
  log.info(f"acquire workload={wid} pool={pool} step={step} — waiting...")

  await lock.acquire()
  _pool_holder[pool] = wid
  wait_ms = round((time.perf_counter() - t0) * 1000)
  log.info(f"  {wid} acquired pool={pool} after {wait_ms}ms wait")

  restore_ms = 0
  if MODE == "timeslice" and step > 0:
    pids = _workload_pids.get(wid, [])
    if pids:
      log.info(f"  restoring pids={pids}")
      async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{DAEMON_BASE}/restore_context",
            json={"pool": pool, "pids": pids, "workload_id": wid},
        )
        if r.status_code != 200:
          raise HTTPException(500, f"restore_context failed: {r.text}")
      restore_ms = round((time.perf_counter() - t0) * 1000) - wait_ms
      log.info(f"  restored in {restore_ms}ms")
    else:
      log.info(f"  no pids yet — cold start path, skipping restore")

  _step_counter[wid] += 1
  _log_metric({
      "type": "acquire",
      "workload_id": wid,
      "pool": pool,
      "step": step,
      "mode": MODE,
      "wait_ms": wait_ms,
      "restore_ms": restore_ms,
  })
  return {"status": "ok", "wait_ms": wait_ms, "restore_ms": restore_ms}


if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=RL_ORCH_PORT, log_level="info")
