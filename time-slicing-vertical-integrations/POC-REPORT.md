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

1. **First-ACQUIRE deadlock (pods-before-acquire lifecycle)** — the grant gate (`LoadedJob`) blocks any job whose labeled pods exist but haven't touched the GPU (agent state IDLE). The deploy-pods-*after*-grant lifecycle (guide Pattern A, the e2e tests, autoscaled RayJobs with replicas:0) is unaffected — which is why existing tests pass — but static-pod deployments deadlock on their first acquire. Workaround: CUDA keepalive process. Fix is small: treat cold-start IDLE (no saved context, nothing else loaded) as grantable, like UNSPECIFIED already is.
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

`poc-evidence/` (this directory): `EVIDENCE-SUMMARY.md` (experiment agent's full writeup), `timeline-run3.txt` (merged per-event timeline), `analyze.py` (turn/switch accounting), `full-evidence.tar.gz` (complete 43 MB set: orchestrator/agent logs with per-op durations, 2 s lock-state + nvidia-smi traces, all pod logs incl. solo baseline, manifests). Platform remains deployed on the west cluster; the workload image recipe, package source, and manifests are published at github.com/aishukamal/rl-time-slicing/tree/main/time-slicing-vertical-integrations/verl.

---

## Appendix: Integration design in detail (as built, and what remains)

### Components to build

**1. `timeslice.phases` — framework-agnostic core (in the existing `timeslice` Python client).**
- `PhaseObserver`: consumes `(phase_name, roles, event)` and drives `OrchestratorClient` locks. Rules, encoded once: map role → group lock from config; acquire all roles of a phase in a fixed global order (trainer before sampler); on phase end, release only locks the next phase doesn't need; at a data-wait event, conditionally release the trainer lock if the data isn't ready and re-acquire after (yield-on-starvation).
- `MockOrchestrator`: in-process fake that records the acquire/release trace during a dry run and asserts the invariants (global ordering respected, both locks held across any `roles=[trainer,sampler]` span, no lock leaked at exit, no acquire-while-holding in wrong order). This is the CI gate for every framework binding and for hand-rolled integrations.
- Estimated size: ~300–500 lines + tests.
- *PoC status:* the colocated single-group mode needed only a much simpler `PhaseLocks` helper (idempotent `ensure()`/`drop_all()`, env-driven, atexit safety net, ~100 lines) — the full roles-based `PhaseObserver` is required once the multi-pool modes (`separate_async`) land. `MockOrchestrator` remains to-build; the PoC substituted build-time registration/config gates plus the live run.

**2. `timeslice-verl` — the binding package.**
- Registers trainer modes via veRL's registry, one per stock mode we cover:
  - `sync_timesliced` extends `PPOTrainerSync` (colocated → roles collapse to one group, single lock).
  - `separate_async_timesliced` extends `PPOTrainerSeparateAsync` (disaggregated → two groups; the flagship).
  - `colocate_async_timesliced` extends `PPOTrainerColocateAsync`.
- Each subclass overrides the v1 lifecycle hooks in `verl/trainer/ppo/v1/trainer_base.py`, wraps them with lock calls, and delegates via `super()` so veRL's own logic (engine sleep/wake, weight sync, checkpoint engine) runs unchanged. Three implementation facts learned at verl `6a6242f3` (v0.9.0.dev), now required knowledge:
  1. **Register via a `verl.plugins` entry point**, not just `VERL_USE_EXTERNAL_MODULES` — the v1 trainer runs inside a separate Ray actor process (`TaskRunnerV1`), and the entry point guarantees registration there regardless of env propagation.
  2. **`trainer_base` branches on the literal string `trainer_mode != "sync"`** (gen-batch sizing, TransferQueue paths); a mode named `sync_timesliced` takes the async paths — the subclass must set `self.trainer_mode = "sync"` immediately after `super().__init__()` to inherit exact sync semantics.
  3. **`on_train_end` is not called on the natural last-step exit** (`fit()` returns early) — final release must not depend on it; pair the last `on_step_end` release with an `atexit` safety net.

  | veRL hook | Phase emitted | Roles | Lock action |
  |---|---|---|---|
  | `__init__`/`on_init_end` | `init` | both | acquire all at startup, release per first phase |
  | `on_sample_begin` | `generate` start | sampler | acquire sampler |
  | `on_sample_end` | `generate` end | sampler→trainer | acquire trainer, release sampler (veRL sleeps replicas here — engines quiesce exactly where we release) |
  | *(between)* | `train` (all sub-phases: `old_log_prob`, `ref`, `values`, `adv`, `update_critic`, `update_actor`, `save_checkpoint`) | trainer | held throughout — sub-phases are contiguous, no transitions needed |
  | `on_step_end` | `weight_sync` | **both** | acquire sampler before `super()` (its `update_weights` needs both), release trainer after |
  | `on_validate_begin/end` | `eval` | sampler | acquire/release sampler |
  | `on_train_end` | shutdown | — | release all, close client — *unreliable: skipped on natural last-step exit; atexit net is the real backstop* |

- Config plumbing: env-first (`TIMESLICE_JOB_ID/ORCH_ADDR/GROUP`, set by the manifests — one source of truth with the pod labels); hydra keys optional sugar later. Package is import-safe without the env (no-op with a warning).
- Async data-wait: in the async modes the wait *is* the sample span (`replay_buffer.sample` runs between `on_sample_begin/end`), so yield-on-starvation lands on existing hooks.
- Estimated size: ~200–300 lines — matched reality (the PoC package is ~200 lines, built and validated in under a day via Cloud Build gates: trainer-mode registration resolvable with the env var cleared, plus full hydra config composition at the pinned schema).
- Acquire timing, validated: acquire before `super().__init__()` holds the lock through `trainer.init()` (worker groups, model load, initial weight sync) — note the actual GPU work starts in `init()`, not `__init__` (which is CPU-only).

**3. Recipes + guide**: a `guides/rl-frameworks/verl/` entry mirroring the slime guide structure (main README = the delta spec; `examples/async/` = minimal runnable two-job example) using our already-validated H100 recipes (`dapo_7b_*` from the async idle study).

### Optional fourth mode (experimental, phase 1.5): `sync_disagg_timesliced`

The trainer registry also lets us add a mode veRL itself doesn't have: **synchronous training on disaggregated pools** — the topology our original PoC used (job A trains while job B generates, pipelined across two pools). `separate_async` proves the prerequisites exist (disjoint placement, cross-pool weight transfer); the mode is roughly the sync loop minus the colocation-specific replica sleeping. It reaches beyond the documented hook surface (placement validation, transfer-path selection), so it ships as explicitly experimental with a defined exit criterion: benchmark **colocated turn-taking vs. disaggregated pipelining** for two tenants on the same recipe — promote it if pipelining wins, delete it if not. No veRL approval needed either way.

### Known gaps and workarounds

- **Two platform-side gaps currently require in-pod workarounds** (both are filed platform bugs, not integration design; drop the workarounds when fixed): (a) the orchestrator only grants a lock once the node agent sees the job's GPU PIDs — but a correct integration acquires *before* touching the GPU, which deadlocks; interim: a small CUDA keepalive process per pod. (b) Snapshotting a job whose training already exited faults the whole group; interim: a stay-alive wrapper after training completes. A third platform issue (cuda-checkpoint lock flake ~1/20 ops with no rollback) is a robustness risk for unattended runs but needs no integration-side change.
- **`fully_async_policy` sits outside the v1 registry** (experimental tree, own loop). Interim: a wrapper around its `MessageQueueClient.get_sample()` for the data-wait yield — a readiness probe already exists (`get_queue_size()`, and every `get_sample` returns `queue_len`). This is the one component we must maintain against veRL internals until an upstream emit lands.
- **No sub-phase hooks** (`old_log_prob` … `update_actor` have timer names only). Costs nothing for correctness — all trainer sub-phases occupy one role contiguously — but mid-step yields (finer-grained preemption) wait for the upstream RFC. We deliberately do *not* wrap veRL's private `_compute_*` methods (no stability contract).
- **The single-slot problem**: `trainer.v1.trainer_mode` takes one value, and our integration occupies it. Users who already run their *own* custom trainer mode — often exactly the users who want time-slicing — can't compose the two without hand-merging subclasses, and other hook consumers (metrics, profilers) can't stack either. Related: subclassing scales per-mode — every trainer mode veRL adds needs a matching variant from us. Both are structural limits of the out-of-tree attachment, and both are what the upstream RFC's registered-callback-list design eliminates (any mode emits, any number of consumers listen).

### Validation plan

1. `MockOrchestrator` trace tests per mode (seconds, in CI — no cluster). *(Still to build.)*
2. Two-job e2e on the H100 cluster. *(Done for `sync_timesliced`, 2026-07-20: 22 clean handoffs, rewards tracking solo baseline, ~14.6 s switch — this report. The `separate_async_timesliced` variant with the measured `dapo_7b` recipes remains.)*
3. Clean-room run of the shipped example files only, with real outputs pasted into the guide (the standard we set on the slime PR).
4. Build-time gates proved their worth and are now part of the spec: registration resolvable via entry point (env cleared), full config composition against the pinned verl schema, and dependency smoke-imports (caught a protobuf gencode/runtime clash — image needed protobuf ≥7.35).

### Compatibility and the upstream follow-up

- The package depends only on the *documented* extension surface: the nine hooks, the trainer registry, `trainer.v1.trainer_mode`, and the plugin loader. Pin `verl>=0.8,<0.10` and publish a compat matrix; veRL's own trainer variants are built on the same nine hooks, which makes that surface sticky. Upgrades within the supported range are automatic — the subclass copies no loop code, so everything we don't override comes from the installed veRL at runtime.
- **Drift detection is part of the package, not a hope**: startup assertions verify every overridden hook still exists on the base class (a renamed hook would otherwise fail *silently* — our override just stops firing), and CI runs the trace test against both the pinned range and veRL `main`, so an interface change upstream fails our CI the day it lands, not the day a user's job misbehaves.
- **No veRL approval is needed for any of v1**: the plugin loader, registry, and subclassing are public, user-side mechanisms; nothing changes for users who don't install the package. Maintainer involvement is distribution upside only (a verl-recipe listing, a docs mention) — and the RFC deliberately comes *after* v1, negotiated from "working integration + users + data" rather than "please add hooks."
- Upstream RFC (after the package proves the pattern): convert the `on_*` template methods into a config-registered callback list with fan-out, add role metadata, emit begin/end at the existing `marked_timer` sites (which already carry the right names), and add the data-wait emit to `fully_async`. ~150–250 lines in one file, pitched with metrics/profiling/scheduling as co-equal consumers. What it buys, in order: composability (kills the single-slot problem), any-mode coverage without per-mode subclasses, a versioned contract instead of de-facto stability, mid-step yield points, and default distribution in every veRL install.
