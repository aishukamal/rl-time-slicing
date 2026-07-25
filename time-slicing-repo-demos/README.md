# Time-Slicing Replay Demo Generator

Generates the "See Time-Slicing in Action" interactive replay page embedded in the
[llm-d-rl-time-slicing](https://github.com/llm-d-incubation/llm-d-rl-time-slicing) README.
The published artifact is a single self-contained HTML file; this directory holds the
source used to build it from recorded run telemetry.

## Files

- `replay_template.html` — the page template (canvas replay, event feeds, summary card).
  `/*__DATA__*/null` is replaced with the run data at build time.
- `extract_data.py` — parses the raw PoC telemetry exports (per-GPU utilization CSV,
  `rl_metrics.jsonl` lock/swap events, per-step training metrics) into `verl_demo_data.json`.
  Handles the clock alignment between the utilization scraper and the lock-event log
  (their start times differ by ~15 minutes).
- `verl_demo_data.json` — the extracted, aligned dataset for the veRL PoC run
  (baseline 20260427_173147 + timeslice 20260427_092932).
- `build_replay.py` — builds `timeslice-replay.html` from template + dataset.
  To add a run to the dropdown, add an entry to the `runs` array with the same shape.
- `capture_replay_frames.py` — headless-Chrome frame capture (uses the page's
  `?t=SEC&end=1` hook) for GIF/MP4 generation.
- `cloudbuild.yaml` — assembles the captured frames into MP4 + palette-optimized GIF
  with ffmpeg on GCP Cloud Build.

## Regenerating

```bash
python3 extract_data.py     # needs the raw telemetry export dirs (paths at top of file)
python3 build_replay.py     # -> timeslice-replay.html
python3 capture_replay_frames.py
gcloud builds submit --config cloudbuild.yaml .
```

Timeline convention: t=0 is each run's first GPU-busy sample; the replay ends at the
final lock yield. Duty cycle = share of minutes with GPU activity (util >5%) within
each run's active window.
