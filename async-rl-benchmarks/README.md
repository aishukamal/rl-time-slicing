# veRL Idle-Time Benchmarks — Where Time-Slicing Applies

Four experimental phases measuring GPU duty cycles of veRL RL training across execution mode
(sync colocated / async disaggregated), data policy (on-policy / bounded / unbounded staleness),
and workload balance (gen-heavy / train-heavy). July 2026, 2×H100 per run (GKE spot).

**Start here: [rl-timeslicing-benchmark-report.html](rl-timeslicing-benchmark-report.html)** — the
full self-contained report (all charts embedded).

## Headline result

Realistic RL workloads — especially R1/DAPO-style long chain-of-thought reasoning — leave one GPU
pool idle **28-64% of the time in large periodic blocks** under every on-policy and
bounded-staleness veRL configuration, with zero artificial tuning. On the flagship long-CoT
workload the **trainer GPU idles 46-64% at every staleness setting** (1.6-3.3 min contiguous
blocks) because generation is structurally slower than training and staleness cannot bind.

The law all measurements obey: *the slower pipeline side pins its GPU; the faster side idles in
per-step blocks of duration ≈ |gen − train|; staleness relocates idle onto the faster side but
never below the throughput imbalance.*

## Phases

| Phase | Directory | Workload | Mode | Key result |
|---|---|---|---|---|
| 1 | [benchmark-longtail/](benchmark-longtail/) | GSM8K, Qwen2.5-0.5B, single-turn | async disagg | square wave only via artificial starvation (throttled sampler + batch inflation) |
| 2 | [benchmark-deepresearch/](benchmark-deepresearch/) | MHQA multi-turn search agent (CMU deep-research), Qwen2.5-3B | sync colocated | long-tail is natural (gen 77→190s, ~34% idle) but colocated idle isn't reclaimable |
| 3 | [async-multiturn/](async-multiturn/) | HotpotQA + Wikipedia search tool, Qwen2.5-3B (train-heavy) | async disagg, staleness 0/1/4/8/∞ | s=0: both GPUs anti-phase idle; s≥1: trainer pins 91%, idle moves to rollout GPU (~30s blocks) |
| 4 | [async-longcot/](async-longcot/) | DAPO-Math-17k long CoT, R1-Distill-Qwen-1.5B (gen-heavy) | async disagg, staleness 0/8/∞ | trainer starves at EVERY staleness: 64% (s=0) / 46% (s=8 ≡ s=∞), queue never buffers |
| — | [disagg-deepresearch/](disagg-deepresearch/) | port of the Phase-2 workload into the PoC sync-disagg trainer | — | built (~2.6k lines, PLAN.md), shelved — sync-RL idle is already structural |

Each phase directory contains its own `REPORT.md`, the deployable K8s job spec, 100ms GPU traces
(`*.csv`), training logs, per-run summary JSON, and chart-generation scripts.

## Reproducing

Each `k8s-job-*.yaml` is self-contained (ConfigMap scripts + job): `kubectl apply -f` on a cluster
with 2 free H100s (jobs target a specific spot pool via affinity — edit `nodeSelector`/`affinity`
for your cluster). Phases 2-3 need no external APIs: search runs against a local pyserini BM25
Wikipedia index (~12GB, auto-downloaded). Traces are additionally dumped gzip+base64 to stdout at
completion, so results survive spot preemption via cluster logging.
