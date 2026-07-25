# Demo: the full adoption journey — sync colocated replay

Self-contained animated dashboard (`index.html`, press Start, ~100 s, ends
holding on the recap). Story: everything between "a GKE cluster running veRL
jobs" and time-sliced GPUs — helm install, node-pool taint, one shared DRA
ResourceClaim, job adaptation (pip install + `trainer.v1.trainer_mode=sync_timesliced`
+ 3 env vars) — ending in a replay of the recorded 2026-07-20 sync PoC run
(two GRPO jobs alternating on one H100: 23 turns, 22 automatic checkpointed
handoffs, ~14 s switches, both reward curves matching solo; see ../../POC-REPORT.md).

Honest framing baked in: the colocated sync topology has no idle to harvest —
this replay proves the mechanism and the zero-touch adoption path (useful for
multiplexing scarce GPUs); the utilization story belongs to the disaggregated
topology (see ../demo-disagg/).

## Files
- `index.html` — the demo (open in any browser; `?t=<seconds>` freezes a frame)
- `dashboard_template.html` + `build_data.py` + `replay_data.json` — generator:
  `build_data.py` parses the run-3 evidence timeline into `replay_data.json`,
  which gets inlined into the template at the `/*__DATA__*/` marker.
- `capture_frames.sh` + `cloudbuild.yaml` — optional MP4 render: headless-Chrome
  stills at 12 fps, assembled by ffmpeg on Cloud Build.

Regenerate: `python3 build_data.py && python3 -c "import json; t=open('dashboard_template.html').read(); d=json.dumps(json.load(open('replay_data.json')),separators=(',',':')); open('index.html','w').write(t.replace('/*__DATA__*/', d))"`
