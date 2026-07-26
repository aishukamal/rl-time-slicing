# PoC 2 — Time-Slice a Math-RLVR Trainer with a Code-RLVR Trainer (Copy-Paste Handoff)

Sequel to PoC 1 (`POC1-sweep-trainer-timeslicing.md` — read it first; all environment rules,
swap mechanics, and monitoring discipline carry over verbatim). PoC 1 packs two *identical* sweep
trainers; this PoC packs two *different* RL jobs — the realistic multi-tenant cluster story:
one team training math reasoning, another training code generation, their trainers sharing one GPU
while both generation fleets run hot.

## 1 · The two workloads (both validated, run as-is)

Both use veRL fully-async disagg, R1-Distill-Qwen-1.5B, STALENESS_THRESHOLD=8,
max_response_length=16384, 1 rollout GPU + 1 trainer GPU each. Repo:
https://github.com/aishukamal/rl-time-slicing, directory `async-rl-benchmarks/`.

| | Job A — Math RLVR | Job B — Code RLVR |
|---|---|---|
| Job spec | `async-longcot/k8s-job-async-longcot-16k.yaml` | `code-rlvr/k8s-job-code-rlvr.yaml` |
| Dataset / reward | DAPO-Math-17k, in-tree math_dapo verifier | Eurus-2 code split, in-tree prime_code (live in-pod test execution) |
| Baseline results | `async-longcot/results/run_s8_16k_*` | `code-rlvr/results/run_s8_16k_*` |
| Local copies | `/Users/aishuk/workspaces/GPU-CR/async-longcot/` | `/Users/aishuk/workspaces/GPU-CR/code-rlvr/` |

Known prep bug in the code job (benign but fix before reuse): the token-length prompt filter
no-ops on transformers 4.57 (`apply_chat_template` returns a dict, so `len()==2`); all shipped
prompts measured ≤1027 tokens anyway. See `code-rlvr/REPORT.md` for details.

## 2 · Measured solo baselines (your comparison targets)

| Metric | Math (A) | Code (B) |
|---|---|---|
| Step cadence | 407s (322-513s) | 619s (547-691s) |
| Trainer busy (update) | 185s | 281s |
| Trainer busy fraction | 45% | 45% |
| Trainer idle blocks | 201s mean / 286s max | 245s mean / 399s max |
| Rollout GPU util | 99.5% | 99.6% |
| response_length mean | 8.2K | 11.4K |
| Weight sync | ~1.7s | ~1.7s |

**Packing math**: combined trainer busy fraction 0.45 + 0.45 = 0.91 on one shared GPU, with
UNALIGNED periods (407s vs 619s). Unlike PoC 1's identical pair, collisions WILL occur
(A wants the GPU while B holds it). This is the point of PoC 2: demonstrate that
- the s=8 staleness queues absorb collision-induced delays (a swapped-out trainer's rollouter
  keeps generating into its 8×64-sample budget — watch mq depth rise during collisions, this is
  the shock absorber working),
- a simple scheduling policy (FIFO on batch-ready, or yield-on-update-complete + queue-on-request)
  keeps both jobs within ~10-15% of solo step time at 91% nominal packing.

## 3 · Topology (one a3-megagpu-8g node)

| GPU | Assignment |
|---|---|
| 0 | Math rollout — dedicated, never touched |
| 1 | Code rollout — dedicated, never touched |
| 2 | Math trainer + Code trainer — **time-sliced** |
| 3-7 | free |

## 4 · Success metrics

1. Shared trainer GPU duty cycle ≥ 85% (nominal packing 91%; solo baseline 45%).
2. Each job's mean step time within 15% of its solo baseline (407s / 619s). Report the actual
   collision-induced stretch per job — the distribution matters, not just the mean.
3. Both rollout GPUs ≥ 95% util throughout.
4. Staleness never violated: dropped_stale=0 on both jobs; report max mq depth observed during
   collisions (expect > 0 for the first time in this project — that's the queue absorbing swaps).
5. Rewards stay live on the code job (per-step score means in the 0.1-0.3 band) — proves test
   execution kept overlapping generation while trainers were being swapped.
6. Same artifacts as always: 100ms GPU trace for all 3 GPUs + both training logs, before/after
   charts comparable to `async-rl-benchmarks` baselines.

## 5 · Environment & rules

Identical to PoC 1 §7 (same cluster, same pool h100-mega-8gpu-spot-b ONLY, same do-not-touch
list, same monitoring discipline, spot capacity delays of 4-25 min expected). Both PoC jobs plus
their rollouts fit one 8-GPU node (3 GPUs used); do not spread across nodes — the NCCL weight
sync between each trainer and its rollouter assumes intra-node.

## 6 · Reference artifacts

Everything under `async-rl-benchmarks/` in the repo: full study
(`rl-timeslicing-benchmark-report.html`, rendered at
https://aishukamal.github.io/rl-time-slicing/async-rl-benchmarks/rl-timeslicing-benchmark-report.html),
per-recipe REPORT.md files, traces, `analyze_run.py` (reuse for the shared-GPU trace),
timeline charts (`async-longcot/run_s8_16k_timeline.png`, `code-rlvr/run_s8_16k_timeline.png`)
showing the two solo square waves you are about to interleave.
