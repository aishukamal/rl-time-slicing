# TPU Time-Slicing PoC

Demonstrates checkpoint/restore-based time-slicing of RL training jobs on TPU v5e. Two independent GRPO RL jobs (sampler + trainer each) share the same TPU hardware by interleaving their compute phases via orchestrator-coordinated checkpoint/restore.

## Architecture

```
                    ┌─────────────────┐
                    │   Orchestrator  │  Per-pool locking (sampler, trainer)
                    │   (CPU pod)     │  Coordinates acquire/yield/C-R
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │                             │
     ┌────────▼────────┐          ┌────────▼────────┐
     │   RL Loop A     │          │   RL Loop B     │
     │   (CPU pod)     │          │   (CPU pod)     │
     └───┬────────┬────┘          └───┬────────┬────┘
         │        │                   │        │
    ┌────▼──┐ ┌───▼───┐         ┌────▼──┐ ┌───▼───┐
    │Sampler│ │Trainer│         │Sampler│ │Trainer│
    │  -a   │ │  -a   │         │  -b   │ │  -b   │
    └───────┘ └───────┘         └───────┘ └───────┘
        │          │                │          │
   TPU Node 1  TPU Node 2     TPU Node 1  TPU Node 2
   (spot pool) (gvisor pool)  (spot pool) (gvisor pool)
```

- **Sampler**: vLLM on TPU running GPT-2, generates completions
- **Trainer**: JAX/Flax GRPO trainer, computes policy gradient updates
- **Orchestrator**: Per-pool mutex locks, triggers checkpoint/restore via snapshot-agent
- **Snapshot Agent**: DaemonSet on TPU nodes, calls libtpu-uds gRPC to checkpoint/restore TPU processes
- **RL Loop**: Drives the generate → train cycle, coordinates with orchestrator

## Prerequisites

- GKE cluster with:
  - 2x TPU v5e-8 nodes (one spot pool, one gvisor pool)
  - Default CPU node pool
- `kubectl` configured for the cluster
- `gcloud` authenticated with Cloud Monitoring read access
- Container images built and pushed (see **Building Images** below)

## Building Images

All Dockerfiles are in the repo. Build from **this directory** (`tpu-rl-jax-poc/`):

```bash
# Snapshot Agent (Go gRPC service — checkpoint/restore via libtpu-uds)
docker build -f snapshot-agent/Dockerfile -t snapshot-agent:latest .

# Orchestrator (Python — per-pool locking + C/R coordination)
docker build -f orchestrator/Dockerfile -t tpu-time-slicing-poc:latest orchestrator/

# Sampler (vLLM on TPU)
docker build -f sampler/Dockerfile -t tpu-rl-sampler:latest sampler/

# Trainer (JAX/Flax GRPO)
docker build -f trainer/Dockerfile -t grpo-trainer:latest trainer/
```

The snapshot agent build requires Go 1.25.x (the Dockerfile pins `golang:1.25`). Local `go build` requires Go 1.25 due to go-nvml compatibility.

Note: `libtpu-uds.so` is NOT included in this repo. It must be obtained separately and is copied into the container images that need it.

## Running Experiments

### Quick Start

```bash
cd tpu-rl-jax-poc

# Run baseline (1 job, no C/R) — ~75 min for 5 steps
RUN_ID=myrun MODE=baseline N_RL_STEPS=5 bash deploy.sh

# Run timeslice (2 jobs, interleaved C/R) — ~90 min for 5 steps each
RUN_ID=myrun MODE=timeslice N_RL_STEPS=5 bash deploy.sh
```

### Full Comparison Run

```bash
# Use the same RUN_ID for both modes
export RUN_ID=comparison_$(date +%Y%m%d)

# Baseline first
MODE=baseline N_RL_STEPS=5 bash deploy.sh

# Then timeslice
MODE=timeslice N_RL_STEPS=5 bash deploy.sh

# Generate dashboards
# Cloud Monitoring source (1-min resolution, requires gcloud auth)
python3 telemetry/dashboard_generator.py \
  --baseline-dir runs/${RUN_ID}/baseline \
  --timeslice-dir runs/${RUN_ID}/timeslice \
  --output runs/${RUN_ID}/dashboard.html

# tpu-info scraper source (1s resolution)
python3 telemetry/dashboard_generator.py \
  --baseline-dir runs/${RUN_ID}/baseline \
  --timeslice-dir runs/${RUN_ID}/timeslice \
  --output runs/${RUN_ID}/dashboard_scraper.html \
  --use-scraper
```

### Configuration

All config is passed via environment variables to `deploy.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODE` | `timeslice` | `baseline` (1 job) or `timeslice` (2 jobs with C/R) |
| `RUN_ID` | auto-generated | Groups baseline + timeslice runs together |
| `N_RL_STEPS` | `5` | Number of RL training steps per job |
| `PROMPTS_PER_STEP` | `330` | GSM8K prompts sampled per step |
| `GROUP_SIZE` | `8` | Completions generated per prompt |
| `MAX_NEW_TOKENS` | `1024` | Max tokens per completion |
| `GEN_BATCH_SIZE` | `3` | Prompts per generate batch (controls gen phase duration) |
| `WEIGHT_SYNC_INTERVAL` | `0` | Sync weights every N steps (0 = disabled) |

Phase durations with default config: ~6.5 min generate, ~8 min train per step.

## What deploy.sh Does

1. **Clean slate** — deletes old pods
2. **Deploy infrastructure** — orchestrator, services, snapshot-agent, sampler/trainer pods
3. **Start TPU scrapers** — tpu-info duty cycle + memory scrapers inside TPU pods (1s resolution)
4. **Wait for sampler init** — polls health endpoint until vLLM is ready
5. **Launch RL loops** — 1 loop (baseline) or 2 loops (timeslice)
6. **Monitor** — polls for `[RL_JOB_COMPLETED]` marker (120 min timeout)
7. **Collect metrics** — job JSONL, orchestrator JSONL, tpu-info CSVs, pod logs
8. **Collect Cloud Monitoring** — queries GKE `duty_cycle` + `memory_used` metrics via API
9. **Cleanup** — deletes rl-loop pods after metrics verified

## Output Structure

```
runs/<RUN_ID>/
├── baseline/
│   ├── config.json                    # Run configuration
│   ├── metrics_job-a.jsonl            # Per-step metrics (gen_ms, train_ms, reward, etc.)
│   ├── rl_metrics.jsonl               # Orchestrator events (acquire, yield, checkpoint, restore)
│   ├── tpu_duty_cycle_sampler.csv     # tpu-info scraper: duty cycle + HBM (1s, all chips)
│   ├── tpu_duty_cycle_trainer.csv     # tpu-info scraper: duty cycle + HBM (1s, all chips)
│   ├── cloud_metrics_sampler_a.csv    # Cloud Monitoring: duty_cycle + memory_used (1 min)
│   ├── cloud_metrics_trainer_a.csv    # Cloud Monitoring: duty_cycle + memory_used (1 min)
│   ├── rl-loop-a.log                  # RL loop pod log
│   ├── sampler-a.log, sampler-b.log   # Sampler container logs
│   └── trainer-a.log, trainer-b.log   # Trainer container logs
├── timeslice/
│   ├── (same structure, plus metrics_job-b.jsonl and rl-loop-b.log)
│   └── ...
├── dashboard.html                     # Comparison dashboard (Cloud Monitoring metrics)
└── dashboard_scraper.html             # Comparison dashboard (tpu-info scraper metrics)
```

## Dashboard Generators

| Script | Source | Resolution | Notes |
|--------|--------|------------|-------|
| `telemetry/dashboard_generator.py` | Cloud Monitoring API | 1 min | Requires `gcloud auth`; add `--use-scraper` for tpu-info source |
| `telemetry/scraper_dashboard.py` | tpu-info scraper CSVs | 1 sec | Standalone scraper-only dashboard |
| `telemetry/collect_cloud_metrics.py` | Cloud Monitoring API | 1 min | Standalone metric collection (called by deploy.sh) |

## Directory Layout

```
tpu-rl-jax-poc/
├── main.go                # Standalone TPU checkpoint CLI (direct libtpu-uds gRPC)
├── go.mod, go.sum         # Go module (snapshot agent + CLI)
├── Makefile               # Build targets: build-cli, build-agent
├── cmd/snapshot-agent/    # gRPC Snapshot Agent entry point
├── pkg/snapshot-agent/    # Snapshot Agent implementation
│   ├── api/v1alpha1/      #   Proto definition + generated gRPC bindings
│   ├── backends/          #   CUDA, TPU, and noop checkpoint backends
│   ├── server/            #   gRPC service (Snapshot, Restore, Status)
│   ├── state-machine/     #   Job state tracking (IDLE→RUNNING→SAVED)
│   └── utils/             #   Kubernetes pod discovery, PID extraction
│
├── deploy.sh              # Main entry point
├── deploy/                # Kubernetes manifests
│   ├── samplers-pod.yaml  #   2-container pod (sampler-a, sampler-b)
│   ├── trainers-pod.yaml  #   2-container pod (trainer-a, trainer-b)
│   ├── services-m5.yaml   #   ClusterIP services for all endpoints
│   └── snapshot-agent.yaml#   DaemonSet + RBAC for TPU C/R
├── loop/                  # RL loop driver
│   ├── rl_loop.py         #   GRPO loop: generate → reward → train
│   ├── reward.py          #   GSM8K reward function (3-tier)
│   ├── rl-loop-a.yaml     #   Pod manifest for job-a
│   └── rl-loop-b.yaml     #   Pod manifest for job-b
├── orchestrator/          # Lock-based orchestrator
│   ├── orchestrator.py    #   Per-pool mutex, C/R coordination
│   ├── Dockerfile
│   └── pod.yaml
├── sampler/               # vLLM TPU sampler
│   ├── sampler.py         #   FastAPI server, vLLM on TPU
│   └── Dockerfile
├── trainer/               # JAX/Flax GRPO trainer
│   ├── trainer.py         #   Training server
│   ├── weight_sync.py     #   Flax → safetensors export
│   └── Dockerfile
├── snapshot-agent/        # Snapshot agent image
│   └── Dockerfile
├── telemetry/             # Metrics + dashboards
│   ├── dashboard_generator.py
│   ├── scraper_dashboard.py
│   ├── collect_cloud_metrics.py
│   └── tpu_duty_cycle.py
└── runs/                  # Experiment outputs
```
