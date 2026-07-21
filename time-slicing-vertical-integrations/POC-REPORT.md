# PoC Report: Time-Slicing Two Real veRL Workloads on One GPU

**Date:** 2026-07-20 · **Cluster:** verl-research-cluster-west (GKE, 1× H100-80GB via spot node) · **Verdict: demonstrated** — mechanism and integration story proven; platform robustness gaps identified.

## Objective

Show that two unmodified, real veRL training workloads can share a single physical GPU under the llm-d-rl-time-slicing platform — orchestrator lock → snapshot agent → cuda-checkpoint on every handoff — with the framework integration delivered exactly as the vertical-integration strategy claims: **a pip package, env vars, one config value, and pod labels. Zero framework source changes. No mocks anywhere.**

This first experiment used veRL's sync **colocated** mode deliberately: it carries no real time-slicing value on its own (a colocated job already keeps its GPU busy), but it is the simplest topology that exercises the proposal's mechanics end to end — DRA oversubscription, lock protocol, whole-VRAM checkpoint/restore under a live training process, and the plug-in integration path. The disaggregated topologies (async- or sync-disagg), where the utilization value actually lives, are the follow-on now in preparation.

## Stack

| Layer | What ran |
|---|---|
| Workloads | 2× veRL GRPO (v0.9.0.dev @ `6a6242f3`), Qwen2.5-0.5B-Instruct, GSM8K, 12 steps each, single-GPU colocated |
| Integration | `timeslice_verl` pip package: `@register_trainer("sync_timesliced")` extending stock `PPOTrainerSync`, whole-step turn-taking (acquire at step begin — including before first model load — release after weight sync); registered via a `verl.plugins` entry point; config via `TIMESLICE_JOB_ID/ORCH_ADDR/GROUP` env |
| Image | `gcr.io/aishuk-test/timeslice-verl-poc:v1` (Cloud Build; build-time gate: trainer-mode registration + config-schema check) |
| Platform | Helm deploy on cluster: accelerator orchestrator + snapshot-agent DaemonSet (k8s mode, cuda-checkpoint @ NVIDIA main, driver 580.126.20) + NVIDIA DRA driver |
| Sharing | Both jobs' pods reference ONE shared `ResourceClaim` (1 GPU) → co-scheduled onto the same physical H100 (DRA oversubscription) |

## Results (run 3 of 3)

- **23 strictly alternating turns; 22 clean automatic snapshot/restore handoffs; zero lock-ordering violations.** Job a3: 12/12 steps. Job b3: 11/12 (final step lost to a platform race — see findings).
- **Real learning across suspensions:** reward_mean a3 0.016→0.141, b3 0.008→0.117, both tracking the solo baseline (0.008→0.117); no NaNs or corruption across ~12 freeze/thaw cycles per job.
- **Timing:** steady-state turn (1 GRPO step) 19.7–20.5 s vs 17.6–19.8 s solo baseline; **switch cost 14.1–15.0 s (mean ~14.6 s)** = snapshot ~10 s + restore ~3 s + ~1.5 s orchestration. Effective contended cycle ~34.6 s.
- **Cooperative blocking:** b3's first `ACQUIRE` blocked 137 s with zero GPU activity while a3 held the group (first turns are long: model + vLLM init under lock, ~124–127 s).
- **Zero-cost-when-alone confirmed:** solo releases logged `snapshot_deferred=True` (no physical snapshot); contended releases `snapshot_deferred=False`. Solo run of the same image: 12/12 steps at ~18.8 s/step.
- **GPU occupancy (2 s nvidia-smi trace):** strict compute alternation — only the active job's processes execute (up to ~18.6 GB during generation). Caveat: eviction is partial; ~4 GB of the suspended job (keepalive + one FSDP worker) stays resident between turns.

## Adoption-story validation

The full user-side delta, as exercised: `pip install` in the image → 3 env vars + `trainer.use_v1=True trainer.v1.trainer_mode=sync_timesliced` in the launch command → labels/claim/tolerations in the Job manifests. veRL source untouched; the trainer mode arrived through veRL's own plugin loader and trainer registry.

Integration findings folded back into the package: verl's `trainer_base` branches on the literal string `trainer_mode != "sync"` (subclass re-declares itself `"sync"` post-construction); `on_train_end` doesn't fire on the natural last-step exit (atexit safety net); entry-point registration is required for Ray actor processes.

## Platform findings (all platform-side; none veRL/training-side)

1. **First-ACQUIRE deadlock** — orchestrator grants only once the agent sees the job's GPU PIDs, but correct clients acquire *before* touching the GPU. Workaround: CUDA keepalive process in each pod. Needs a grant-semantics fix, not a patch.
2. **Snapshotting a completed job faults the whole group** — empty PID discovery fails hard (upstream fix exists, not in the deployed image). Workaround: stay-alive wrapper post-training.
3. **cuda-checkpoint lock flake (~1/20 ops) with no rollback** — one failed snapshot faults job X, then the controller restores Y onto an un-evicted GPU, faulting Y too. Ended run 2; cost b3 its 12th step in run 3. `NCCL_CUMEM_ENABLE=0` did not eliminate it. Needs retry-then-rollback.
4. **Sticky ghost state** — failed acquirers wedge the lock queue until a manual Yield-as-ghost; the agent never forgets FAULTED job IDs (fresh job ids required per attempt).

These align with and extend the pre-existing fault-tolerance backlog: single-job hiccups must never poison the shared pool.

## Conclusions

1. **The core claim holds on real hardware with real workloads**: orchestrated GPU time-slicing of two independent RL jobs, with training integrity preserved through repeated whole-VRAM checkpoint/restore.
2. **The integration cost claim holds**: zero framework changes; the plugin-package path works end to end, including inside Ray actor processes.
3. **Scope this correctly: this run is the mechanism gate, not the value demonstration.** Sync-colocated turn-taking splits one GPU's throughput between tenants (~42% per-step overhead at this toy scale) — an access/fairness story, not a utilization story, since a colocated job already keeps its own GPU busy. The utilization value prop lives in the async topology (dedicated samplers, shared trainer pool, yield-on-starvation): with measured trainer idle of 70–95% in 20–130 s gaps, the same ~14.6 s switch amortizes to single-digit-% overhead and two jobs' trainers should run at near-solo speed on one GPU. That run (`separate_async_timesliced`, needs 3–4 GPUs) is the next milestone and the chart that carries the pitch.
4. **Robustness, not mechanism, is the gap**: the four findings above are the priority platform work before unattended runs or external demos.

## Next steps

File the four platform bugs upstream with this evidence → land fixes (1–2 are small) → drop the keepalive/stay-alive workarounds → unattended overnight two-job run as the robustness gate → then this setup graduates to the public veRL recipe + maintainer conversation.

## Evidence

`poc-evidence/` (this directory): `EVIDENCE-SUMMARY.md` (experiment agent's full writeup), `timeline-run3.txt` (merged per-event timeline), `analyze.py` (turn/switch accounting), `full-evidence.tar.gz` (complete 43 MB set: orchestrator/agent logs with per-op durations, 2 s lock-state + nvidia-smi traces, all pod logs incl. solo baseline, manifests). Platform remains deployed on the west cluster; workload image and package source retained per PLAN.md.
