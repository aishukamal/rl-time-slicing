# Vertical Integration Plan: Time-Slicing Across RL Frameworks

**Date:** 2026-07-20
**Strategy:** Generic phase callbacks upstream; all time-slicing logic in our external packages. No framework ever carries time-slicing-specific code.

## The primitive

Every accelerator-occupying interval in the RL driver loop is a **named phase** that:
- declares the **resource roles** it occupies (`sampler` / `trainer` / both),
- emits **blocking** `on_phase_start(phase, ctx)` / `on_phase_end(phase, ctx)` callbacks,
- and async loops additionally emit `on_data_wait_start/end` at the trainer's rollout-data wait.

External consumers register via config (metrics, profilers, spot-preemption handlers — and our time-slicing lock client, which maps roles → orchestrator group locks with a fixed global acquire order, making dual-lock rules like weight-sync *derived* rather than special-cased).

## Research result: the primitive applies to all six frameworks

Verified at file level, July 2026 (verl `6a6242f3`, NeMo-RL `ec46ab3f`, TRL `a44ecd07`, Miles/SkyRL/Tunix HEAD):

| Framework | Phases already named? | Callback API today | Out-of-tree possible today? | Upstream ask | Size |
|---|---|---|---|---|---|
| **verl** | Yes — `marked_timer` vocab (`gen/old_log_prob/ref/values/adv/update_actor/update_weights/save_checkpoint/testing`) in v1 `trainer_base.py` | 9 `on_*` template methods + `@register_trainer` registry | **Yes** — custom trainer mode via `VERL_USE_EXTERNAL_MODULES`; sleep/wake lives inside the overridable hooks | Fan-out (callback list via config) + role metadata + sub-phase emits; one file | S–M |
| **Miles** (radixark/miles, slime successor) | Yes — `timer(...)` incl. native `inverse_timer("train_wait")` = data_wait | No; ~25 dotted-path `--*-path` hooks (house idiom: "if you're patching the trainer we're missing a hook") | Partially (thin user-owned driver) | `--phase-callback-path` flag + emits in 2 thin drivers — exactly their idiom | S |
| **SkyRL** | Yes — `Timer(...)` incl. `Timer("wait_for_generation_buffer")` | **Yes** — `TrainingCallback` (HF-style, 10 events, programmatic only) | Partially (custom entrypoint passes callbacks) | Phase/role events + config registration + fully-async support (currently `NotImplementedError` there) | S–M |
| **NeMo-RL** | Yes — `Timer` scopes incl. `exposed_generation` at the async buffer wait | None | No (only proxy-wrapping `policy`/`policy_generation` in user entry script) | Callback protocol + emits, duplicated across 2–3 loop variants | M |
| **TRL** | Partially — `profiling_context` spans (`sync_weights`, `vLLM.generate`) | `TrainerCallback` exists **but generation/sync/sleep-wake run inside `training_step` where no callback fires** | Not as callback pkg; yes via `VLLMGeneration` proxy (clean single seam) | Emit events from generation/sync sites; custom-event dispatch design (transformers owns the event set) | S code, M design |
| **Tunix** (Google, JAX/TPU) | Yes — perf spans **already carry device sets** (= roles), + `StepTraceAnnotation` | SFT-only `TrainingHooks` | No | Blocking-subscriber extension of the span protocol; needs `block_until_ready` semantics for phase-end | M (JAX) |

**Universal facts the research established:**
1. Every framework already names its phases (timers/spans). The RFC pitch everywhere is the same: *"promote your existing timers into a subscriber API and declare what each phase occupies."*
2. Every framework has one identifiable trainer data-wait, usually already timed (`train_wait`, `wait_for_generation_buffer`, `actor_dequeue_time`, `exposed_generation`, `queue_wait_time_s`, verl's `on_sample_begin→replay_buffer.sample→on_sample_end`).
3. Role ground truth exists everywhere (colocate flags, placement configs, `role_to_mesh`, `WorkerDispatch.GPUState`).
4. Blocking `on_phase_start` = the acquire gate works everywhere; the one semantic caveat is Tunix/JAX (async dispatch means phase-end must `block_until_ready` on phase outputs).

## The plan

**Phase 0 — build the framework-agnostic core once (`timeslice.phases`, in our client repo):**
- `PhaseObserver`: maps `(phase, roles)` → lock protocol. Global lock ordering, acquire-per-role on start, release-what-next-phase-doesn't-need on end, conditional yield at data_wait. All protocol correctness lives here, tested once.
- `MockOrchestrator` trace validator: records acquire/release sequences in a dry run, asserts invariants (ordering, dual-lock spans, no leaks). Works for every binding and for hand-rolled integrations.
- Manifest kit templates (labels/tolerations/claims/rayStartParams) shared by all framework guides.

**Phase 1 — verl (no permission needed):** ship `timeslice-verl` pip package: `@register_trainer("sync_timesliced")` (+ colocate_async/separate_async variants) subclassing the stock trainers, overriding the nine `on_*` hooks (super() preserves sleep/wake), loaded via `VERL_USE_EXTERNAL_MODULES`. Validate on our H100 recipes (async idle data already measured). Publish as a verl-recipe. Known gap: `fully_async_policy` is outside the v1 registry — interim wrapper around its message-queue `get_sample` (probe exists: returns `queue_len`); upstream emit later.

**Phase 1.5 (experimental) — `sync_disagg_timesliced`:** a fourth registered mode adding what verl lacks natively: sync training on disaggregated pools (the PoC topology — cross-job phase pipelining). We do NOT propose this upstream; it ships as our experimental mode (registry permits it, no approval). Prereqs proven by `separate_async` (disjoint placement, cross-pool weight transfer); reaches beyond the documented hook surface, so it carries an exit criterion: benchmark colocated turn-taking vs disagg pipelining for 2 tenants, same recipe — promote on a win, delete on a loss. This is also the first same-framework apples-to-apples comparison of the two sharing topologies.

**Phase 2 — Miles + slime (slime-lineage transfer):** port Jessica's sync+async work to Miles (same driver shape, same offload primitives). Upstream `--phase-callback-path` to Miles — their house idiom, >100 merged PRs/month, and their docs invite exactly this ask. slime keeps Jessica's guide; upstream the same flag there after Miles proves it.

**Phase 3 — SkyRL:** PR extending their existing `TrainingCallback` (phase/role events, config-driven registration, fully-async support — their own community flagged the fully-async callback gap in #1922). Binding package `timeslice-skyrl`.

**Phase 4 — RFC standardization + long tail:** publish the phase-callback spec in our repo as the reference all bindings conform to. NeMo-RL: modest single-file PR when bandwidth allows (governance risk: NVIDIA/Run:ai). TRL: `VLLMGeneration` proxy interim, upstream event ask when the audience justifies it. **Tunix: strategic TPU play** — on TPU, app-cooperative suspend is the *only* mechanism (no cuda-checkpoint), and Tunix already self-pauses (host offload to `pinned_host`, `RolloutSyncLock` drain-and-exclusive semantics); pursue alongside our TPU time-slicing maturation.

**Continuous:** the integration skill (agent-driven) covers pinned versions and private forks of all frameworks — hook-capable versions become `pip install` + config; older versions get the hooks inserted by the skill, verified by the MockOrchestrator trace test.

## First target: verl

**Why verl over Miles (the runner-up):**
1. **Zero-permission start.** The v1 trainer registry + `VERL_USE_EXTERNAL_MODULES` means the complete integration ships today as a pip package — no upstream review on the critical path. Every other framework needs either an upstream PR or fragile wrapping.
2. **Our flagship use case is already measured on verl.** The async trainer-sharing story (70–95% trainer idle across 3 recipes × 4 async modes, 20–130s gaps vs 2–3s C/R) is verl data — the recipe validates against known traces.
3. **K8s maturity matches our platform.** Official GKE/ACK/KubeRay tutorials mean verl users already run where our DaemonSet/DRA stack lives. Miles has zero K8s story; SkyRL/Tunix little.
4. **Largest audience + integration-friendly culture** (recipe repo, external-module registry, plugin entry points).
Miles wins on insertion cleanliness and merge velocity, and is the natural Phase 2 because Jessica's slime work transfers almost mechanically — but no releases (rolling main) and no K8s story make it a worse *first* proof point.

## verl user journey (time-slicing an existing verl workload)

**Admin, once per cluster (framework-independent):** helm install platform (orchestrator + snapshot-agent DaemonSet + DRA driver); label/taint group nodes; apply one shared ResourceClaim per group.

**User, per job — three deltas, zero training-code changes:**
1. **Image:** add `timeslice-verl` (pip) to their existing verl image.
2. **Config/env (2–4 lines):**
   ```bash
   export VERL_USE_EXTERNAL_MODULES=timeslice_verl
   # in the run config / CLI overrides:
   trainer.use_v1=True trainer.v1.trainer_mode=sync_timesliced \
   +timeslice.job_id=$JOB_NAME +timeslice.groups.trainer=trainers +timeslice.groups.sampler=samplers
   ```
3. **Manifests:** the standard kit on GPU worker podspecs — `timeslice.io/job-id` + `group` labels, group nodeSelector, tolerations (`nvidia.com/gpu: Exists` + shared taint), shared ResourceClaim reference. Same job-id string in labels and config (the JOB_NAME lesson from PR #92).

**Why zero app-code lines:** verl users launch via `main_ppo` + config — they don't own a driver script. The trainer mode *is* the integration; selecting it via config is the entire app-side change. (Contrast slime: users own `train.py`, hence Jessica's 6-file fork and hand-designed lock protocol.)

**Phase→lock mapping the package implements (verl vocabulary):**
| verl phase | Roles | Lock behavior |
|---|---|---|
| `gen` / `on_sample_*` | sampler | hold sampler; in async modes this span is the data_wait → conditional yield |
| `old_log_prob`, `ref`, `values`, `adv`, `update_critic`, `update_actor` | trainer | hold trainer |
| `update_weights` (`on_step_end`) | **both** | acquire both in global order, release trainer after |
| `save_checkpoint` | trainer | hold trainer |
| `testing` / validate | sampler | hold sampler |

Colocated mode collapses roles to one group/lock automatically (roles → same group id). Uncontended jobs pay ~nothing (`snapshot_deferred` when no waiters).

**Effort comparison:** slime today = fork, 6 files, hand-designed deadlock protocol, postStart hot-patching. verl with `timeslice-verl` = image add + config lines + manifest kit; protocol correctness ships tested in the package and is verified per-integration by the MockOrchestrator trace test.

## PoC execution (started 2026-07-20) — REAL two-job demo, no mocks

Goal: two genuine verl workloads (Qwen2.5-0.5B GRPO on GSM8K, single-GPU colocated, v1 `sync_timesliced` mode) time-slicing ONE physical H100 on verl-research-cluster-west, full stack: orchestrator lock → snapshot agent → cuda-checkpoint on every handoff.

- Cluster state at start: 1 free spot H100 node (2 GPUs), K8s 1.35 (DRA v1 GA), NO platform deployed (existing snapshot-agent DS is TPU-only), no DRA driver.
- Topology: both jobs reference one shared 1-GPU ResourceClaim → same physical GPU; whole-step turn-taking (acquire at step begin, release after weight sync). Plain k8s Jobs (verl single-node runs in-process Ray; no KubeRay). Node labels only, NO taints (shared cluster).
- Track A (background agent): helm deploy orchestrator + agent (+ DEPLOYMENT_MODE=k8s patch, PR#94) + NVIDIA DRA driver; node labels; shared-gpu-claim; smoke pod discovery test.
- Track B (background agent): real `timeslice_verl` package (register_trainer("sync_timesliced") extending PPOTrainerSync; env-driven config; whole-step lock choreography) + workload image via Cloud Build with build-time registration smoke test. Image: gcr.io/aishuk-test/timeslice-verl-poc.
- Track C (pending A+B): two Jobs verl-job-a/b, evidence = orchestrator alternation, agent snapshot/restore logs, both jobs' reward curves progressing, nvidia-smi occupancy trace.
- poc-verl/ dir: protocol simulation files (PhaseObserver + mock orchestrator) — now demoted to future package unit tests; NOT the demo.

### RESULT (2026-07-20): DEMONSTRATED ✅ — see POC-REPORT.md

Sync-colocated mechanism gate passed: 22 clean automatic cuda-checkpoint handoffs between two real verl GRPO jobs on one H100, both learning, zero verl source changes. Four platform bugs found (all platform-side; fault-tolerance memory + report findings). Full numbers, evidence index, and conclusions live in **POC-REPORT.md** (single source of truth for this experiment). Platform left deployed on west; PR #112 (image publishing fix) in draft; code + docs published at github.com/aishukamal/rl-time-slicing/tree/main/time-slicing-vertical-integrations. Next: disagg PoC (Track D prep in flight).

## Risks / open items
- verl v1 API stability: private-method wrapping for sub-phase events has no stability contract — keep the package pinned per verl minor version; upstream RFC removes this fragility.
- verl fully-async is outside the v1 registry (the one wrapper we must maintain until an upstream emit lands).
- Tunix blocking-semantics design (JAX async dispatch) needs a prototype before the RFC.
- All bindings conform to one spec doc so "what is a phase / what are roles" never forks per framework.
