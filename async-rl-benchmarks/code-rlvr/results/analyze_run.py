import csv, json, re, statistics as st, sys
# usage: analyze_run.py <label> <gpu_csv> <train_log> <out_json> [steady_start_step]
label, gpucsv, trainlog, outjson = sys.argv[1:5]
skip_steps = int(sys.argv[5]) if len(sys.argv) > 5 else 2

# ---- parse train log step metrics ----
step_re = re.compile(r'step:(\d+) - training/global_step')
def grab(line, key):
    m = re.search(re.escape(key) + r':([0-9.eE+-]+)', line)
    return float(m.group(1)) if m else None
steps = []
roll_lines = {}
with open(trainlog, errors='replace') as f:
    for line in f:
        if 'fully_async/rollouter/idle_ratio' in line and 'training/global_step' not in line:
            m = re.search(r'step:(\d+) ', line)
            if m: roll_lines[int(m.group(1))] = grab(line, 'fully_async/rollouter/idle_ratio')
        m = step_re.search(line)
        if m:
            s = int(m.group(1))
            steps.append(dict(
                step=s,
                step_time=grab(line, 'timing_s/step'),
                gen=grab(line, 'timing_s/gen'),
                update=grab(line, 'timing_s/update_actor'),
                param_sync=grab(line, 'timing_s/timing_s/param_sync'),
                wait=grab(line, 'fully_async/total_wait_time'),
                mq=grab(line, 'fully_async/monitor/queue/mq_queue_size'),
                trainer_idle=grab(line, 'fully_async/trainer/idle_ratio'),
                rollouter_idle=roll_lines.get(s),
                num_turns=grab(line, 'num_turns/mean'),
                score=grab(line, 'critic/score/mean'),
                stale_processed=grab(line, 'fully_async/count/stale_trajectory_processed'),
                dropped_stale=grab(line, 'fully_async/count/dropped_stale_samples'),
                partial_ratio=grab(line, 'fully_async/partial/partial_ratio'),
                resp_len=grab(line, 'response_length/mean'),
            ))
ss = [x for x in steps if x['step'] > skip_steps]
def agg(key):
    v = [x[key] for x in ss if x.get(key) is not None]
    if not v: return None
    return dict(mean=round(st.mean(v),3), min=round(min(v),3), max=round(max(v),3))

# steady-state time window from train log timestamps not available; use gpu csv window trimmed
# ---- gpu csv ----
gpus = {}
rowsall = {}
with open(gpucsv) as f:
    for row in csv.DictReader(f):
        rowsall.setdefault(row['gpu_index'], []).append((int(row['timestamp_ms']), int(row['gpu_util_pct']), float(row['power_w'])))
# steady window: drop first 25% and use rest? Better: skip first (startup+2 steps). Estimate startup end
# heuristic: steady = from first time GPU0 util>=90 sustained + 2*median step_time, to end
stt = agg('step_time')
med_step = st.median([x['step_time'] for x in ss if x['step_time']]) if ss else 0
for g, v in sorted(rowsall.items()):
    v.sort()
    ts = [x[0] for x in v]
    # find first busy sample
    first_busy = next((t for t,u,p in v if u>=90), ts[0])
    cut = first_busy + skip_steps*med_step*1000
    sv = [x for x in v if x[0] >= cut]
    us = [x[1] for x in sv]; ps=[x[2] for x in sv]
    blocks=[]; cur=None
    for (t,u,p) in sv:
        if u < 10:
            if cur is None: cur=[t,t]
            else: cur[1]=t
        else:
            if cur is not None: blocks.append((cur[1]-cur[0])/1000); cur=None
    if cur is not None: blocks.append((cur[1]-cur[0])/1000)
    big=[b for b in blocks if b>=2]
    gpus[f'gpu{g}'] = dict(
        window_s=round((sv[-1][0]-sv[0][0])/1000,1),
        util_mean=round(st.mean(us),1),
        pct_idle_lt10=round(100*sum(1 for u in us if u<10)/len(us),1),
        pct_busy_ge90=round(100*sum(1 for u in us if u>=90)/len(us),1),
        power_mean_w=round(st.mean(ps),0),
        idle_blocks_ge2s=dict(count=len(big),
            mean_s=round(st.mean(big),1) if big else 0,
            max_s=round(max(big),1) if big else 0,
            total_s=round(sum(big),0) if big else 0),
    )
out = dict(
    label=label,
    n_steps_total=len(steps),
    n_steps_steady=len(ss),
    skip_steps=skip_steps,
    step_time_s=agg('step_time'), gen_s=agg('gen'), update_actor_s=agg('update'),
    param_sync_s=agg('param_sync'), trainer_collect_wait_s=agg('wait'),
    mq_queue_size=agg('mq'), trainer_idle_ratio=agg('trainer_idle'),
    rollouter_idle_ratio=agg('rollouter_idle'), num_turns_mean=agg('num_turns'),
    score_mean=agg('score'), stale_trajectory_processed=agg('stale_processed'),
    dropped_stale_samples=agg('dropped_stale'), partial_ratio=agg('partial_ratio'),
    response_length_mean=agg('resp_len'),
    gpus=gpus,
)
with open(outjson,'w') as f: json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
