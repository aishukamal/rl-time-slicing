# RL Infrastructure Landscape — Pain Points & Universal-Solution Opportunities

*Research date: 2026-07-29. Sources: verl, slime, Miles, NeMo-RL roadmaps/issues/design docs; AReaL, SkyRL, OpenRLHF, ROLL, TRL, prime-rl, PipelineRL, Tinker, Magistral, Kimi K1.5/K2/K3, Seed, MiniMax, Qwen infra publications.*

## 1. Where each framework is right now

| Framework | Current focus | Headline unsolved problems (their own words) |
|---|---|---|
| **verl** (26Q3 roadmap [#6985](https://github.com/verl-project/verl/issues/6985)) | Decomposing into 5 services (rollout / model / weight-transfer engines, agent loop, TransferQueue) per north-star [#3624](https://github.com/volcengine/verl/issues/3624); delta weight sync; KV-aware rollout scheduling; AgentGateway | **Dynamic trainer↔rollout GPU switch in disaggregated async (open)**; elasticity & fault tolerance carried over unchecked for 3 quarters; fully-async still experimental; sandbox section of uni-agent roadmap literally empty |
| **slime** (v0.3.0) | Agent-first RL: sandboxes (E2B), harnesses for Claude Code/Codex CLIs, fully-async first-class, delta weight sync (disk+NCCL) | Roadmap captive to Z.ai internal needs; sandbox instability = spurious reward noise (GLM-5 paper); FSDP removed to cut maintenance |
| **Miles** (RadixArk, slime fork) | Bit-wise true on-policy (dense done), unified FP8, R3 routing replay, RDMA P2P weight transfer (1T: 53s→7.2s), speculative RL | **MoE zero-mismatch open; "elastic rollout-vs-training scheduling" and "GPU-failure elasticity" are explicit roadmap items, not shipped** |
| **NeMo-RL** (v0.7, arch tracker [#2905](https://github.com/NVIDIA-NeMo/RL/issues/2905)) | Refit (weight sync) war: NCCL m2n reshard, sparse-delta refit, NIXL/RDMA checkpoint engines; TransferQueue data plane; trajectory checkpointing [#2415](https://github.com/NVIDIA-NeMo/RL/issues/2415) | "Resiliency (fault tolerance + auto-scaling)" is planned-not-shipped; refit rollback after partial failure; async weight transitions still synchronous barriers; single-controller "no longer tenable" [#2414](https://github.com/NVIDIA-NeMo/RL/issues/2414) |
| **AReaL 2.0** | Online agent RL micro-services; interruptible rollout (interrupt → discard KV → re-prefill) | Elastic weight-update service deferred; colocation via awex swap |
| **Kimi/Moonshot** | Partial rollout across iterations; [checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine) (1T weights ~20s); K3 resumable microVM sandboxes | Engine-switch latency + failure recovery "significant at 1T scale" (K2 paper) |
| **Others** | PipelineRL: in-flight weight updates mid-decode; ByteDance TensorHub: ownership-free cross-DC weight store; Prime Intellect: 365K sandboxed envs + Environments Hub; Tinker: LoRA-multiplexed training API becoming an interface standard (SkyRL implements it) | HF's 16-library survey: all teams converged on the same macro-architecture; pain has moved down into the shared substrate layers |

## 2. Convergent pain points (every framework, independently)

1. **Trainer→rollout weight transfer.** verl delta-sync (#7060), NeMo refit (692s→14s and still the top theme), Miles P2P RDMA, Moonshot checkpoint-engine, ByteDance TensorHub, [vLLM RFC #31848](https://github.com/vllm-project/vllm/issues/31848) explicitly calling out that every framework reinvents this. Crowded but standardizing.
2. **Elastic GPU reassignment between rollout and training.** The rollout:train compute ratio shifts during training; static splits waste GPUs. Explicitly on-roadmap-but-unshipped at verl ("dynamic trainer↔rollout switch"), Miles ("elastic rollout-vs-training scheduling"), NeMo ("auto-scaling" planned), AReaL (deferred). Blocked everywhere by NCCL communicator rigidity — **verl 26Q3 adopts NCCL 2.29.7 suspend/resume specifically to release communicator memory.**
3. **Rollout lifecycle / long-tail stragglers.** Five incompatible interrupt philosophies: AReaL interrupt-and-re-prefill, slime/SkyRL abort-with-resume, prime-rl soft drain, PipelineRL/Magistral never-stop in-flight swap, Kimi cross-iteration partial rollout. No shared pause/resume/abort abstraction at the engine level; prefix-cache correctness bugs still being discovered (ServiceNow vLLM V0→V1 post).
4. **Colocated sleep/wake memory choreography.** OpenRLHF Hybrid Engine, TRL colocate+sleep, ROLL AutoDeviceMapping, K2 DRAM offload, slime `--colocate`, NeMo colocation escape hatches — all hand-rolled time-sharing of GPU memory between trainer and inference engine against vLLM/SGLang internals. TE/FP8 stacks make it more fragile (benign OOMs kill week-long jobs — Miles's "memory robustness" pitch).
5. **Fault tolerance / episode-level checkpointing.** NeMo: 40-minute rollouts fully recomputed on a crash; K3 built resumable microVM sandboxes; GLM-5: sandbox crashes corrupt rewards. Landscape survey verdict: *"No framework treats episode-level checkpoint/resume of agent + environment + KV state as a first-class primitive; recoverable failures are typically group-fatal."*
6. **Rollout-training numerical mismatch** (TIS/MIS/GSPO/R3 routing replay; MoE zero-mismatch open). Real but algorithm/kernel-layer — less an orchestration play.
7. **Sandbox/environment substrate.** Standards war forming (OpenEnv, Harbor, verifiers, NeMo-Gym); nobody offers sub-second mid-episode env checkpoint/restore, warm pools, or grading-material isolation as standard substrate features.

## 3. The generalization: an accelerator-state substrate under all frameworks

Every framework has converged on the same macro-architecture (disaggregated pools + rollout buffer + async weight push + Ray). The differentiation and the pain have moved *below* the framework: who owns GPU memory, when engines yield it, how state survives interruption. Four of the five "everyone reinvents this" layers are fundamentally **suspend/resume/reassign problems**:

- Elastic rollout↔train GPU switching = suspend one engine's GPU state, hand the device to the other, restore later.
- Sleep/wake colocation = the same operation on a timer.
- Partial rollout / straggler interruption = the same operation plus KV-cache state.
- Fault tolerance / preemptible capacity = the same operation to durable storage.

A framework-agnostic **GPU checkpoint/restore + time-slicing layer** (transparent cuda-checkpoint-style C/R + NCCL suspend + orchestrator) addresses all four with one mechanism, without requiring vLLM/SGLang/Megatron/every-framework to each implement sleep modes, elastic schedulers, and trajectory checkpointing. Validation that the industry is heading here: NCCL upstream shipped communicator suspend/resume (2.29.7) and verl immediately put it on the roadmap — but only for memory release, not full C/R or reassignment. Nobody yet composes it into elastic reassignment or durable engine snapshots.

Ranked opportunities:
1. **Elastic trainer↔rollout GPU reassignment** — universally on-roadmap, universally unshipped; deliverable as an orchestrator + C/R demo on verl fully-async or slime disaggregated mode.
2. **Generic suspend/resume substrate** replacing per-engine sleep/wake (defined wake latency, works for engines with no sleep mode, survives TE/FP8 memory fragility).
3. **Durable rollout state**: snapshot rollout engine + KV + (microVM) sandbox mid-episode → straggler handling, crash recovery, spot/preemptible rollout capacity (RLBoost direction), long-horizon (1M–100M token) episodes.
4. **Weight-transfer standardization** — composable with the above but crowded (checkpoint-engine, TensorHub, vLLM RFC); complement rather than compete.

## 4. Full agent reports

Detailed per-framework findings with all source links are preserved in the conversation transcript (verl, slime/Miles, NeMo-RL, broader landscape). Key primary sources:
- verl 26Q3 roadmap: https://github.com/verl-project/verl/issues/6985 · north star: https://github.com/volcengine/verl/issues/3624
- Miles announcement: https://www.lmsys.org/blog/2025-11-19-miles/ · P2P weight transfer: https://www.lmsys.org/blog/2026-04-29-p2p-update/
- NeMo-RL arch tracker: https://github.com/NVIDIA-NeMo/RL/issues/2905 · refit design docs: https://github.com/NVIDIA-NeMo/RL/tree/main/docs/design-docs
- HF async-RL 16-library survey: https://huggingface.co/blog/async-rl-training-landscape
- Moonshot checkpoint-engine: https://github.com/MoonshotAI/checkpoint-engine · TensorHub: https://arxiv.org/html/2604.09107v1
- GLM-5 infra sections: https://arxiv.org/html/2602.15763v1 · Kimi K3: https://arxiv.org/pdf/2607.24653
- NCCL suspend/resume release: https://github.com/NVIDIA/nccl/releases/tag/v2.29.7-1
