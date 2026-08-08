#!/usr/bin/env bash
# elastic-rl-poc M1 — unattended switch-cycle driver. Runs INSIDE the job pod
# (nohup right after pod start; it self-gates on training progress).
# Performs N_CYCLES full trainer->R2->trainer switch cycles gated on trainer
# gen-wait blocks, and records: per-phase C2 timings (switch_timings.jsonl via
# r2_lifecycle.py), R2-active mq captures, and per-cycle status snapshots.
#
# External dependency (guarded below): agent job "elastic-trainer" must be
# RUNNING before cycle 1 — operator relabels the pod once after auto-park:
#   kubectl label pod <pod> timeslice.io/job-id=elastic-trainer --overwrite
set -uo pipefail
cd /workspace/m1
export PYTHONPATH=/workspace/m1
RES=/workspace/results
LOG=$RES/train.log
N_CYCLES=${N_CYCLES:-4}
# run4: R2 active window spans the full gen-wait block — switch back only when
# the mq nears batch-ready (collected >= SWITCH_BACK_AT) or the cap expires.
# Primary payoff metric is per-step wall time (timing_s/step in train.log) vs
# the M0 baseline (619.0s mean, steps 3-11); fill rate is diagnostics only.
SWITCH_BACK_AT=${SWITCH_BACK_AT:-56}
WINDOW_CAP_S=${WINDOW_CAP_S:-600}

log() { echo "[$(date -u +%FT%TZ)] [cycles] $*"; }

param_syncs() { local v; v=$(grep -c "timing_s/param_sync" "$LOG" 2>/dev/null); echo "${v:-0}"; }
last_collect() { grep -oE "sample collected [0-9]+/64" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+" | head -1; }
requests() { local v; v=$(grep -c "Requesting 64 samples" "$LOG" 2>/dev/null); echo "${v:-0}"; }

log "waiting for steady state (>=3 param syncs)..."
while [ "$(param_syncs)" -lt 3 ]; do sleep 30; done
log "steady state: $(param_syncs) param syncs"

log "waiting for agent job elastic-trainer to be RUNNING (operator relabel)..."
python3 - <<'PYEOF'
import os, time
from timeslice.snapshot_agent import SnapshotAgentClient
from timeslice.snapshot_agent import snapshot_agent_pb2 as pb
ep = os.environ["AGENT_ENDPOINT"]
while True:
    with SnapshotAgentClient(ep) as c:
        st = {}
        for j in c.status().job_statuses:
            try:
                st[j.job_id] = pb.JobState.Name(j.state)
            except Exception:
                st[j.job_id] = str(j.state)
    print("agent jobs:", st, flush=True)
    if st.get("elastic-trainer") == "JOB_STATE_RUNNING":
        break
    time.sleep(15)
PYEOF
log "elastic-trainer RUNNING confirmed"

log "baseline mq capture (120s)"
python3 r2_lifecycle.py watch-mq --interval 2 --count 60 > "$RES/mq_baseline_run4.log" 2>&1

completed=0
consec_fail=0
for c in $(seq 1 "$N_CYCLES"); do
  log "=== cycle $c: waiting for a fresh gen-wait block ==="
  base_req=$(requests)
  while true; do
    r=$(requests); lc=$(last_collect); lc=${lc:-0}
    # The mq backlog (~35) drains in a burst at the start of every block, so
    # collected jumps past 25 within seconds. lc<=45 still leaves several
    # minutes of trickling long-tail collection — and mistiming is safe by
    # design (the stalled step simply completes after resume).
    if [ "$r" -gt "$base_req" ] && [ "$lc" -ge 1 ] && [ "$lc" -le 45 ]; then break; fi
    sleep 3
  done
  log "cycle $c: gen-wait open (collected=$(last_collect)/64); switch-to-rollout"
  python3 r2_lifecycle.py switch-to-rollout > "$RES/cycle${c}_to_rollout.log" 2>&1
  rc1=$?
  log "cycle $c: switch-to-rollout rc=$rc1"
  if [ "$rc1" -ne 0 ]; then
    # suspend confirm failure rolls back (resume signal, no freeze): trainer
    # stays active, R2 stays parked -> safe to try the next cycle once.
    consec_fail=$((consec_fail+1))
    log "cycle $c FAILED at switch-to-rollout (consec_fail=$consec_fail; see cycle${c}_to_rollout.log)"
    if [ "$consec_fail" -ge 2 ]; then log "ABORT: two consecutive failed cycles"; break; fi
    continue
  fi

  # R2 active for the FULL gen-wait block: hold until the batch nears ready,
  # with a continuous mq capture (diagnostics) + a mid-window status snapshot
  # (R1-continuity evidence: LB inflight on R1 while the trainer is frozen).
  log "cycle $c: R2 active; holding until collected>=$SWITCH_BACK_AT/64 or ${WINDOW_CAP_S}s"
  python3 r2_lifecycle.py watch-mq --interval 2 --count 450 > "$RES/mq_active_cycle${c}.log" 2>&1 &
  MQ_PID=$!
  ( sleep 25; python3 r2_lifecycle.py status > "$RES/status_mid_cycle${c}.log" 2>&1 ) &
  w0=$(date +%s)
  while true; do
    lc=$(last_collect); lc=${lc:-0}
    if [ "$lc" -ge "$SWITCH_BACK_AT" ]; then
      log "cycle $c: collected=$lc/64 — batch near-ready, switching back"; break
    fi
    if [ $(( $(date +%s) - w0 )) -ge "$WINDOW_CAP_S" ]; then
      log "cycle $c: window cap ${WINDOW_CAP_S}s reached (collected=$lc/64), switching back"; break
    fi
    sleep 5
  done
  kill "$MQ_PID" 2>/dev/null || true; wait "$MQ_PID" 2>/dev/null

  log "cycle $c: switch-to-trainer"
  python3 r2_lifecycle.py switch-to-trainer > "$RES/cycle${c}_to_trainer.log" 2>&1
  rc2=$?
  log "cycle $c: switch-to-trainer rc=$rc2"
  if [ "$rc2" -ne 0 ]; then
    # state ambiguous (R2 may be frozen / trainer may be half-restored):
    # do NOT continue switching.
    log "ABORT: switch-to-trainer failed (see cycle${c}_to_trainer.log)"; break
  fi

  # Correctness gate (pass/fail line for the NCCL fix): the step must complete
  # and param_sync must succeed within 15 min of resume.
  ps0=$(param_syncs); t0=$(date +%s)
  ok=1
  while [ "$(param_syncs)" -le "$ps0" ]; do
    if [ $(( $(date +%s) - t0 )) -gt 900 ]; then
      log "ABORT: no param_sync within 15 min of trainer resume (dormant-NCCL suspect)"
      ok=0; break
    fi
    sleep 15
  done
  [ "$ok" -eq 0 ] && break
  log "cycle $c: step completed, param_sync ok (total $(param_syncs))"
  python3 r2_lifecycle.py status > "$RES/status_after_cycle${c}.log" 2>&1
  completed=$((completed+1))
  consec_fail=0
done
log "cycle driver done: $completed/$N_CYCLES cycles completed"
