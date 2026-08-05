# Phase Callback Specification for RL Training Frameworks

A universal interface for notifying external systems at GPU phase boundaries in RL training loops.

## 1. Interface

```python
class PhaseCallback:
    def on_phase_begin(self, phase: str, role: str, context: dict | None = None) -> None: ...
    def on_phase_end(self, phase: str, role: str, context: dict | None = None) -> None: ...
    def close(self) -> None: ...
```

### Phases

| Phase | GPU work |
|-------|----------|
| `init` | Model load, weight initialization, NCCL group creation |
| `generate` | Rollout / sample generation (inference) |
| `train` | Forward + backward pass, optimizer step |
| `weight_sync` | Parameter broadcast from trainer to sampler (NCCL/disk/IPC) |
| `save` | Checkpoint save to disk or object store |
| `eval` | Evaluation / validation inference |

### Roles

| Role | Meaning |
|------|---------|
| `trainer` | Trainer GPU pool only (training, old-log-prob, save) |
| `sampler` | Sampler/rollout GPU pool only (generation, evaluation) |
| `both` | Both pools active simultaneously (init, weight sync) |

### Context

Optional dict with phase metadata: `{"rollout_id": int, "step": int, ...}`. Framework-specific keys are allowed.

## 2. Framework Mapping

How each phase maps to concrete code in six RL frameworks.

| Phase | verl v1 | Slime | SkyRL | NeMo-RL | TRL | Tunix |
|-------|---------|-------|-------|---------|-----|-------|
| init | `__init__` → `on_init_end` | `create_*` → `update_weights` | `setup()` | `policy.init()` | `__init__` | `init_components` |
| generate | `on_sample_begin` → `on_sample_end` | `rollout_manager.generate.remote()` | `rollout_manager.generate_batch()` | `policy_generation.generate()` | `_generate_and_score_completions()` | `rollout_step` |
| train | after `on_sample_end` → before `on_step_end` | `actor_model.async_train()` | `train_step()` | `policy.train_step()` | inside `training_step()` | `train_step` |
| weight_sync | inside `on_step_end` | `actor_model.update_weights()` | `sync_weights()` | `policy.update_weights()` | `sync_and_offload()` | `sync_params` |
| save | `save_checkpoint` | `actor_model.save_model()` | `save_checkpoint()` | `policy.save()` | `save_model()` | `save_checkpoint` |
| eval | `on_validate_begin` → `on_validate_end` | `rollout_manager.eval.remote()` | `evaluate()` | `evaluate()` | `evaluate()` | `eval_step` |

### Hook readiness per framework

| Framework | Callback API exists? | Phase boundaries hookable? | Out-of-tree integration path |
|-----------|---------------------|---------------------------|------------------------------|
| **verl v1** | 9 `on_*` template methods + `@register_trainer` | Yes (proven) | Subclass, zero upstream |
| **Slime** | `--phase-callback-path` (our upstream) | Yes (with our commits) | CLI flag, zero code changes |
| **SkyRL** | `TrainingCallback` (10 events, no gen bracket) | Partial | Extend existing callback API |
| **NeMo-RL** | None | No | Fork or upstream callbacks |
| **TRL** | `TrainerCallback` (no gen bracket) | Partial | Subclass `GRPOTrainer` or upstream |
| **Tunix** | `TrainingHooks` (SFT only) | Partial | Extend span protocol |

## 3. GPU Role Semantics

**Colocated mode**: Trainer and sampler share the same physical GPUs. Both roles map to the same lock group. The protocol collapses to single-group whole-step turns — acquire before step, release after.

**Disaggregated mode**: Trainer and sampler run on separate GPU pools with separate lock groups. Enables cross-pipeline concurrency: Job A trains while Job B generates. The dual-lock weight sync span (both pools held) is the only serialization point.

**Fully async mode**: Generation runs continuously on dedicated sampler GPUs. The trainer pulls from a queue. Lock acquire happens when the trainer resumes from the queue (after batch is ready), not when generation starts. The sampler lock may not be needed if sampler GPUs are dedicated.

## 4. Lock Protocol for Time-Slicing

Time-slicing is the primary consumer. Multiple RL jobs cooperatively share GPU hardware by acquiring/releasing orchestrator locks at phase boundaries.

### Requirements

1. **Every GPU phase must hold its role's lock** — no GPU work without the lock
2. **Init**: acquire BOTH before model loads; yield BOTH after
3. **Train**: acquire TRAINER before; yield TRAINER after
4. **Generate**: acquire SAMPLER before; yield SAMPLER after
5. **Weight sync (GPU-GPU)**: acquire BOTH before NCCL broadcast; yield SAMPLER after (keep TRAINER for next step)
6. **Global lock order**: always TRAINER before SAMPLER — prevents deadlock with only two wait shapes

### Invariants

- **Deadlock freedom**: TRAINER is never requested while SAMPLER is held. The only two wait shapes are "want TRAINER holding nothing" and "want SAMPLER holding TRAINER."
- **Idempotent acquire/release**: safe to call redundantly (already-held → no-op, already-released → no-op)
- **Atexit safety**: every lock holder registers an `atexit` handler that releases all held locks on process exit
- **Env-gated no-op**: when `TIMESLICE_JOB_ID` is not set, all lock operations are silent no-ops — the same image runs with and without the time-slicing platform
- **Crash release**: on unhandled exceptions, release all held locks before propagating

### Conditional yield (disagg mode)

During the sample/data-wait phase (trainer idle, waiting for generation), the trainer lock may be released to let another job use the trainer GPU. This is conditional: if the batch is already buffered (no wait expected), skip the release to avoid unnecessary context switches. Probe the framework's replay buffer or queue to decide.

## 5. Other Consumer Patterns

The PhaseCallback interface supports use cases beyond time-slicing:

| Consumer | on_phase_begin | on_phase_end |
|----------|---------------|-------------|
| **Profiling** | Start timer/span | Stop timer, log duration |
| **Spot preemption** | — | Checkpoint at consistent state |
| **Memory management** | Onload tensors to GPU | Offload to CPU |
| **Training dashboards** | — | Log phase metrics |
| **Resource scheduling** | Acquire external GPU broker token | Release token |
| **Fault injection (CI)** | — | Inject failures at phase boundaries |

The callback protocol is minimal by design (two methods + close) to keep framework adoption friction low. Consumers compose: multiple callbacks can be chained by a `CompositePhaseCallback` wrapper if the framework supports only one `--phase-callback-path`.
