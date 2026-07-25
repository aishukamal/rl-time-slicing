# Demo: the full adoption journey — async disaggregated replay

Self-contained animated dashboard (`index.html`, press Start, ~100 s, ends
holding on the recap). Story: the full UX from "a GKE cluster running veRL jobs"
to time-sliced GPU pools — helm install, tainted trainer/sampler pools, one
shared DRA ResourceClaim per pool, job adaptation (pip install +
`trainer.v1.trainer_mode=separate_async_timesliced` + 4 env vars) — ending in a
two-lane replay of the recorded 2026-07-21 async-disagg run dis-a9/dis-b9
(~5 min of measured cross-pipelining: one job training on the trainers pool
while the other generates on the samplers pool; 12/12 steps and solo-matching
rewards for the completing job; see ../../POC-REPORT-DISAGG.md).

Data integrity: lane geometry and the event feed come from orchestrator-side
snapshot/restore timestamps only (job-side logs in this run are flush-batched);
rewards are the logged values; step-marker times are spread across Job A's
measured training tenure (captioned on the chart). Scale note baked in: at 0.5B
generation never starves the trainer, so this proves mechanics, not idle
recovery — the idle-recovery regime is production-scale trainer waits.

## Files
- `index.html` — the demo (open in any browser; `?t=<seconds>` freezes a frame)
- `dashboard_template.html` + `build_data_disagg.py` + `replay_data_disagg.json`
  — generator; data inlined at the `/*__DATA__*/` marker.
- `capture_frames.sh` + `cloudbuild.yaml` — optional MP4 render pipeline.

Regenerate: `python3 build_data_disagg.py && python3 -c "import json; t=open('dashboard_template.html').read(); d=json.dumps(json.load(open('replay_data_disagg.json')),separators=(',',':')); open('index.html','w').write(t.replace('/*__DATA__*/', d))"`
