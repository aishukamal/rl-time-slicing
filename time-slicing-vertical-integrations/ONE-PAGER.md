# Making GPU Time-Slicing Easy to Adopt

## The problem

RL post-training runs in a loop with distinct steps: generate samples, train on them, sync weights, evaluate. At any moment, some of the job's GPUs are working and others are waiting. The waste is large: in our measurements of async RL on veRL, the training GPUs sat **idle 70–95% of the time**, in gaps long enough (20–130 seconds) that another job could easily use them — the switch itself costs 2–3 seconds.

Our time-slicing platform recovers that waste: multiple jobs share the same GPUs, and when one job's GPUs go idle, the platform parks its GPU state, lets another job run, and swaps back later. The jobs don't slow each other down meaningfully; the cluster does more work with the same hardware.

## The adoption problem

Connecting an RL framework to the platform today requires changes inside the framework itself — the coordination calls have to live at the right points in its training loop. That leaves users two paths, each with real costs:

- **Use our pre-integrated fork** (what our slime integration provides today). Works out of the box, but it's pinned to the commit we forked from: users don't get mainline fixes and features as the framework evolves, and can't easily combine it with their own modifications.
- **Apply the changes to their own version.** Keeps them on their code and their framework version, but they take on the integration work themselves — including the coordination logic that keeps two jobs from deadlocking, which is easy to get subtly wrong.

We can soften the second path with an agent-driven tool that applies and verifies the changes automatically, and that remains useful for teams on pinned versions or private forks. But whichever path a team picks, the entry barrier and the maintenance tradeoffs stay high — which is what motivates a different approach.

## The idea: frameworks announce, they don't integrate

Every RL framework's loop moves through the same steps. Our proposal: frameworks simply **announce these steps as they happen** — "generation starting," "training finished," "syncing weights" — through a small, generic callback interface, the same way training libraries already report progress to loggers and dashboards. Each announcement also says which GPUs the step uses (the sampler pool, the trainer pool, or both).

The framework carries **zero time-slicing code** and takes no dependency on us. Announcements are useful to everyone — metrics, profilers, spot-instance handlers — and we ship a separate plug-in that listens to them and coordinates the GPU sharing. All the tricky coordination logic lives in our plug-in, written once and tested once, instead of being re-derived inside every framework.

## What we found

We surveyed six frameworks at source level (July 2026: veRL `6a6242f3`, NeMo-RL `ec46ab3f`, TRL `a44ecd07`, Miles/SkyRL/Tunix HEAD). Three facts hold across all of them:

1. **The loops are already segmented and named.** Each framework wraps its loop stages in named timing spans for metrics: veRL's `marked_timer("gen"/"update_actor"/"update_weights"...)`, NeMo-RL's `timer.time("generation")`, SkyRL's `Timer("sync_weights")`, Miles' `timer(...)`, TRL's profiling spans, Tunix's perf spans. These timers measure wall-clock only and also wrap CPU-only stages (reward scoring, data prep) — they don't distinguish GPU from CPU. What they prove is that the segmentation and naming our proposal needs already exist at exactly the right boundaries; the proposal adds the one missing piece of information — which GPU pool (if any) each segment occupies. That classification is assigned once per phase from what the segment calls (generation calls the rollout engines, training calls the trainer workers, reward scoring calls neither), and CPU-only segments are the point: they're where the GPUs are free for another job.
2. **The async data-wait — the moment a trainer should yield its GPUs — is one identifiable call site, usually already timed:** Miles has `inverse_timer("train_wait")`, SkyRL `Timer("wait_for_generation_buffer")`, Tunix `actor_dequeue_time`, NeMo-RL `exposed_generation`, TRL `queue_wait_time_s`.
3. **Which GPUs a step occupies is already declared in config** (colocation flags, placement settings, Tunix's `role_to_mesh`), so phases can announce their resource roles without new bookkeeping.

Per framework:

| Framework | What exists today | Attach without a framework change? | Upstream ask |
|---|---|---|---|
| **veRL** | 9 lifecycle hook methods (`on_sample_begin/end`, `on_step_begin/end`, ...) + trainer registry + plugin loader | **Yes** — our package registers a custom trainer mode extending the stock one; engine sleep/wake already runs inside the hooks we override | Small: let hooks fan out to registered listeners; add role tags |
| **Miles** | ~25 dotted-path hook flags; docs say "if you're patching the trainer, we're missing a hook — open an issue" | Partially (thin, user-owned driver) | Small: one `--phase-callback-path` flag, matching their idiom |
| **SkyRL** | HF-style `TrainingCallback` API (10 events) already exists | Partially (programmatic registration only) | Small: phase/role events, config registration, async-trainer support |
| **NeMo-RL** | No hook API, but the whole loop is one readable file with explicit named call sites | No | Modest: callback protocol + emits (loop duplicated ~3×) |
| **TRL** | Has `TrainerCallback`, **but** generation, weight sync, and engine sleep/wake run inside `training_step` where no callback fires | Via a proxy around its `VLLMGeneration` object (one clean seam) | Small code; design discussion (event set owned by `transformers`) |
| **Tunix** (JAX/TPU) | Perf spans already record which devices each phase uses — role metadata effectively exists | No | Medium: blocking subscribers on the span protocol; JAX's async dispatch needs explicit phase-end sync |

## The plan

1. **veRL first.** veRL has a built-in plugin system: at startup it reads an environment variable, `VERL_USE_EXTERNAL_MODULES`, naming extra Python packages to load, and a loaded package can register itself in veRL's *trainer registry* — the same catalog veRL's own trainer variants live in. So our package (`timeslice-verl`) registers a trainer mode that extends the stock one, adding lock coordination around the nine existing lifecycle hooks; users select it with one config value. **No fork, no upstream approval, and users keep getting mainline veRL updates** — the package only wraps the hooks and delegates everything else to the installed veRL version. We target the async modes (already disaggregated, and where the 70–95% idle was measured); sync veRL jobs are colocated by design, so they get simple whole-pool turn-taking rather than a new placement mode.
2. **Miles and slime next.** Our existing slime integration ports over almost directly (Miles is slime's successor, same loop shape), and Miles' maintainers explicitly invite hook contributions.
3. **SkyRL.** Already has a callback system; we propose a small extension.
4. **The rest as demand appears** — NeMo-RL, TRL, and Tunix. Tunix matters most for TPUs, where no external freeze mechanism exists, so cooperative announcements are the only path — and Tunix already pauses itself between phases (host offload, an internal rollout/sync lock), making it unusually ready.

In parallel, an agent-driven integration tool covers teams running pinned versions or private forks — it inserts the hooks into their code and verifies the result automatically against a mock coordinator that checks the lock protocol.

## What adoption looks like for a user (veRL)

One-time, cluster admin: install the platform via Helm and label the shared GPU node pools (documented separately).

Per job, three steps, **no training code changes**:

1. **Add the package to the container image:**
   ```dockerfile
   RUN pip install timeslice-verl
   ```
2. **Enable it in the existing launch command** — one env var, one trainer-mode selection, and the job/pool names:
   ```bash
   export VERL_USE_EXTERNAL_MODULES=timeslice_verl
   python -m verl.trainer.main_ppo \
       ... existing config ... \
       trainer.use_v1=True \
       trainer.v1.trainer_mode=separate_async_timesliced \
       +timeslice.job_id=my-job-a \
       +timeslice.groups.trainer=trainers +timeslice.groups.sampler=samplers
   ```
3. **Add the standard sharing config to the job's RayJob manifest.** Each workload runs in its own Ray cluster (KubeRay `RayJob`), so jobs never share a Ray cluster — they share only the physical GPU nodes underneath, which is what makes oversubscription via shared DRA claims straightforward: both jobs' worker pods reference the same per-pool `ResourceClaim`, and Kubernetes schedules them onto the same GPUs. Only the GPU worker groups carry the sharing config; the Ray head and submitter are ordinary CPU pods (no changes — the node taints keep them off the shared GPU nodes automatically):
   ```yaml
   apiVersion: ray.io/v1
   kind: RayJob
   metadata:
     name: my-job-a
   spec:
     rayClusterSpec:
       headGroupSpec: { ... unchanged ... }
       workerGroupSpecs:
       - groupName: trainer-group
         template:
           metadata:
             labels:
               timeslice.io/job-id: "my-job-a"     # must match timeslice.job_id above
               timeslice.io/group: "trainers"
           spec:
             nodeSelector:
               group.timeslice.io/trainers: "true"
             tolerations:
             - key: "nvidia.com/gpu"
               operator: "Exists"
             - key: "timeslice.io/shared"
               operator: "Equal"
               value: "true"
               effect: "NoSchedule"
             containers:
             - name: ray-worker
               resources:
                 claims: [{ name: accelerator }]
             resourceClaims:                        # shared claim instead of nvidia.com/gpu limits
             - name: accelerator
               resourceClaimName: shared-trainers-gpu-claim
       # sampler worker group: same shape with the "samplers" group and its claim
       # (async modes: samplers are dedicated — plain GPU workers, no sharing config at all)
   ```

Then submit both jobs (`kubectl apply -f my-job-a.yaml`, `kubectl apply -f my-job-b.yaml`) and watch the sharing live:

```bash
rlts orchestrator status trainers    # shows active job, waiters, per-node GPU state
```

When the job is alone it runs at full speed (the platform skips all switching when no one is waiting); when other jobs share the pool, the platform interleaves them automatically.

Note where the remaining manual work actually sits: steps 1–2 — the framework integration itself — are a one-line image change and a few config lines. Step 3 and the one-time admin setup are Kubernetes deployment and oversubscription plumbing, identical for every framework. That's the next simplification target: an operator/webhook that expands a single job annotation into the scheduling boilerplate (labels, tolerations, claim references) and injects the job id into the container environment, removing both the YAML surface and the one value users must currently keep consistent by hand.

---

## Appendix: the veRL vertical integration in detail

> **Status (2026-07-20): validated end-to-end.** A working `timeslice_verl` package drove two real veRL GRPO jobs through 22 automatic checkpoint/restore handoffs on one H100 with zero veRL source changes — see `POC-REPORT.md`. Corrections learned from the implementation are folded in below.

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
2. Two-job e2e on the H100 cluster. *(Done for `sync_timesliced`, 2026-07-20: 22 clean handoffs, rewards tracking solo baseline, ~14.6 s switch — POC-REPORT.md. The `separate_async_timesliced` variant with the measured `dapo_7b` recipes remains.)*
3. Clean-room run of the shipped example files only, with real outputs pasted into the guide (the standard we set on the slime PR).
4. Build-time gates proved their worth and are now part of the spec: registration resolvable via entry point (env cleared), full config composition against the pinned verl schema, and dependency smoke-imports (caught a protobuf gencode/runtime clash — image needed protobuf ≥7.35).

### Compatibility and the upstream follow-up

- The package depends only on the *documented* extension surface: the nine hooks, the trainer registry, `trainer.v1.trainer_mode`, and the plugin loader. Pin `verl>=0.8,<0.10` and publish a compat matrix; veRL's own trainer variants are built on the same nine hooks, which makes that surface sticky. Upgrades within the supported range are automatic — the subclass copies no loop code, so everything we don't override comes from the installed veRL at runtime.
- **Drift detection is part of the package, not a hope**: startup assertions verify every overridden hook still exists on the base class (a renamed hook would otherwise fail *silently* — our override just stops firing), and CI runs the trace test against both the pinned range and veRL `main`, so an interface change upstream fails our CI the day it lands, not the day a user's job misbehaves.
- **No veRL approval is needed for any of v1**: the plugin loader, registry, and subclassing are public, user-side mechanisms; nothing changes for users who don't install the package. Maintainer involvement is distribution upside only (a verl-recipe listing, a docs mention) — and the RFC deliberately comes *after* v1, negotiated from "working integration + users + data" rather than "please add hooks."
- Upstream RFC (after the package proves the pattern): convert the `on_*` template methods into a config-registered callback list with fan-out, add role metadata, emit begin/end at the existing `marked_timer` sites (which already carry the right names), and add the data-wait emit to `fully_async`. ~150–250 lines in one file, pitched with metrics/profiling/scheduling as co-equal consumers. What it buys, in order: composability (kills the single-slot problem), any-mode coverage without per-mode subclasses, a versioned contract instead of de-facto stability, mid-step yield points, and default distribution in every veRL install.
