#!/usr/bin/env python3
import json
B = "/Users/aishuk/workspaces/GPU-CR/elastic-rl-poc"
D = json.load(open(f"{B}/dashboard_data.json"))

def line(series, name, color, width=1.4):
    return {"x":[p[0] for p in series], "y":[p[1] for p in series], "name":name, "mode":"lines", "line":{"color":color,"width":width}}

# B: duty cycle traces
m0t = D["m0_trainer"]; m2s = D["m2_shared"]
m0_traces = [line(D["m0"]["series"][g],
                  f"GPU{g}: {'trainer' if g==m0t else 'R1 sampler'} (mean {D['m0']['stats'][g][0]}%, idle {int(D['m0']['stats'][g][1]*100)}%)",
                  "#EA4335" if g==m0t else "#34A853") for g in sorted(D["m0"]["series"])]
m2_traces = [line(D["m2"]["series"][g],
                  f"GPU{g}: {'shared trainer+R2 spare' if g==m2s else 'R1 sampler'} (mean {D['m2']['stats'][g][0]}%, idle {int(D['m2']['stats'][g][1]*100)}%)",
                  "#4285F4" if g==m2s else "#34A853") for g in sorted(D["m2"]["series"])]

# C: anatomy stacked bars
ops = ["switch-to-rollout","switch-to-trainer"]
all_phases = []
for op in ops:
    for ph in D["anatomy"].get(op,{}):
        if ph not in all_phases: all_phases.append(ph)
RAMP = ["#174EA6","#185ABC","#1A73E8","#4285F4","#669DF6","#8AB4F8","#AECBFA","#D2E3FC","#E8F0FE","#FBBC04","#F29900","#EA4335"]
anatomy_traces = [{"x":ops,"y":[D["anatomy"].get(op,{}).get(ph,0) for op in ops],"name":ph,"type":"bar","marker":{"color":RAMP[i%len(RAMP)]}} for i,ph in enumerate(all_phases)]

# D: controller timeline + rt evolution
ev = D["events"]
timeline = [
  {"x":ev["in"], "y":[1]*len(ev["in"]), "name":"switch-in (trainer→spare)", "mode":"markers", "marker":{"symbol":"triangle-down","size":8,"color":"#4285F4"}},
  {"x":ev["back"],"y":[0]*len(ev["back"]),"name":"switch-back (spare→trainer)","mode":"markers","marker":{"symbol":"triangle-up","size":8,"color":"#EA4335"}},
]
rt_traces = [
  {"x":[p[0] for p in ev["rt_in"]], "y":[p[1] for p in ev["rt_in"]], "name":"switch-in duration (s)","mode":"lines+markers","line":{"color":"#4285F4","width":1.6},"marker":{"size":5}},
  {"x":[p[0] for p in ev["rt_out"]],"y":[p[1] for p in ev["rt_out"]],"name":"switch-back duration (s)","mode":"lines+markers","line":{"color":"#EA4335","width":1.6},"marker":{"size":5}},
]

# E: regime shift
rs = D["rs"]
rs_traces = [{"x":rs["in"], "y":[1]*len(rs["in"]), "name":"switch-in events","mode":"markers","marker":{"symbol":"triangle-down","size":8,"color":"#4285F4"}}]
rs_shapes = ([{"type":"line","x0":rs["flip_min"],"x1":rs["flip_min"],"y0":0,"y1":2,"line":{"color":"#5F6368","width":1.5,"dash":"dash"}}] if rs.get("flip_min") else [])
rs_bars = {"x":["pre-flip (16K, switching)","post-flip (8K, switching)","post-flip (8K, no switching)"],
           "y":[rs["steps"]["pre_switching"], rs["steps"]["post_with_switching"], rs["steps"]["post_without"]],
           "type":"bar","marker":{"color":["#9AA0A6","#4285F4","#DADCE0"]}}

# F: no-harm
nh = D["noharm"]
nh_reasons = {"x":list(nh["reasons"].keys()),"y":list(nh["reasons"].values()),"type":"bar","marker":{"color":"#34A853"}}
nh_steps = {"x":["armed controller","control (no controller)"],"y":[nh["armed"],nh["control"]],
            "error_y":{"type":"data","array":[nh["armed_sd"],nh["control_sd"]]},"type":"bar","marker":{"color":["#4285F4","#DADCE0"]}}

# G: run D
RD_ROLES = {"head.0":"trainer","head.1":"sampler","worker.0":"sampler","worker.1":"unused"}
RD_COLORS = {"trainer":"#EA4335","sampler":"#34A853","unused":"#DADCE0"}
rund_traces=[]
seen_s=0
for side in ("head","worker"):
    for g, series in sorted(D["rund"][side].items()):
        r = RD_ROLES[f"{side}.{g}"]
        c = "#188038" if (r=="sampler" and seen_s==1) else RD_COLORS[r]
        if r=="sampler": seen_s+=1
        rund_traces.append(line(series, f"{side} GPU{g} ({r})", c))

L = D["ladder"]
ladder_step = {"x":L["labels"],"y":L["step"],"type":"bar","name":"step time (s)","marker":{"color":["#9AA0A6","#FBBC04","#4285F4","#EA4335"]}}
ladder_eff  = {"x":L["labels"],"y":L["eff"],"type":"bar","name":"samples/GPU-hr","marker":{"color":["#9AA0A6","#FBBC04","#4285F4","#EA4335"]}}

minax = {"title":"minutes since run start","dtick":20,"gridcolor":"#eef0f3","zerolinecolor":"#eef0f3"}
sec = lambda i,t,n: f'<h2>{i}. {t}</h2><p class="note">{n}</p>'

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>verl elastic PoC — results dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>body{{font-family:-apple-system,"Segoe UI",Helvetica,sans-serif;margin:36px auto;background:#ffffff;max-width:1150px;color:#111827;padding:0 24px}}
h1{{font-size:22px;font-weight:600}} h2{{font-size:16px;font-weight:600;margin-top:40px;border-top:1px solid #eef0f3;padding-top:22px}} .note{{color:#4b5563;line-height:1.55;font-size:14px}} .row{{display:flex;gap:16px}} .half{{flex:1}}</style></head><body>
<h1>verl Elastic GPU Scheduling PoC — results dashboard</h1>
<p class="note">All panels generated from the raw run artifacts in <code>elastic-rl-poc/m0..m3-results/</code> (DeepSeek-R1-Distill-Qwen-1.5B, code-RLVR GRPO, verl fully-async, 2x H100 unless noted). Canonical numbers per M0/M2/M3 reports.</p>

{sec("1","The results ladder","Same workload across four provisioning strategies. Elastic (purple) on 2 GPUs beats static 1:1 by 18.8% on step time and reaches 95.8% of the 3-GPU static-1:2 configuration's per-GPU efficiency using one fewer GPU.")}
<div class="row"><div class="half" id="lad1" style="height:330px"></div><div class="half" id="lad2" style="height:330px"></div></div>

{sec("2","Duty cycles: baseline square wave vs elastic fill","Top: M0 static 1:1 — the trainer GPU (red) idles ~half of every step. Bottom: M2 elastic — the same physical GPU (purple) time-shared between trainer and R2 spare; valleys filled, R1 untouched.")}
<div id="m0" style="height:340px"></div><div id="m2" style="height:340px"></div>

{sec("3","Switch anatomy (mean per phase, M2 run)","What a switch actually costs, phase by phase: drain/abort, NCCL suspend signaling + confirmation, cuda-checkpoint snapshot/restore, KV flush, LB re-registration. Totals: " + ", ".join(f"{op} {v[0]}s (n={v[2]})" for op,v in D["op_totals"].items()) + ".")}
<div id="anat" style="height:360px"></div>

{sec("4","Autonomous controller: 26 cycles, timings learned online","Left: every live switch decision over the 4.5h M2 run (26 in / 26 back, zero operator input). Right: per-switch durations across the run — the controller's round-trip estimates track these EMAs.")}
<div class="row"><div class="half" id="tl" style="height:300px"></div><div class="half" id="rt" style="height:300px"></div></div>

{sec("5","Regime shift: adaptation without reconfiguration","Response length flipped 16K→8K mid-run (dashed line). The controller re-learned within ~2 steps: switch rate dropped 1.0→0.65/step as windows shrank below the gate, and switching remained net-positive after the flip (261.8s vs 288.2s without).")}
<div class="row"><div class="half" id="rs1" style="height:300px"></div><div class="half" id="rs2" style="height:300px"></div></div>

{sec("6","No-harm: armed but nothing to harvest","Train-heavy workload (2048 tokens): the armed controller made 1,656 polls, fired zero switches, and every abstention is a logged reason (left). Step time vs the unarmed control differs by +0.8% — noise (right).")}
<div class="row"><div class="half" id="nh1" style="height:300px"></div><div class="half" id="nh2" style="height:300px"></div></div>

{sec("7","Static 1:2 (3 GPUs) — the ratio lens","The true static competitor: near-full utilization everywhere, but it costs a third GPU and freezes the ratio. Elastic on 2 GPUs delivers 95.8% of its per-GPU efficiency and adapts when the workload drifts.")}
<div id="rund" style="height:340px"></div>

<script>
const minax = {json.dumps(minax)};
Plotly.newPlot("lad1",[{json.dumps(ladder_step)}],{{title:"Step time (s, lower better)",yaxis:{{title:"s/step"}}}});
Plotly.newPlot("lad2",[{json.dumps(ladder_eff)}],{{title:"Samples per GPU-hour (higher better)",yaxis:{{title:"samples/GPU-hr"}}}});
Plotly.newPlot("m0",{json.dumps(m0_traces)},{{title:"M0 baseline (static 1:1)",xaxis:minax,yaxis:{{title:"util %",range:[0,102]}},legend:{{orientation:"h"}}}});
Plotly.newPlot("m2",{json.dumps(m2_traces)},{{title:"M2 elastic (26 autonomous cycles)",xaxis:{{...minax,dtick:30}},yaxis:{{title:"util %",range:[0,102]}},legend:{{orientation:"h"}}}});
Plotly.newPlot("anat",{json.dumps(anatomy_traces)},{{barmode:"stack",title:"Mean phase durations per switch operation (s)",yaxis:{{title:"seconds"}}}});
Plotly.newPlot("tl",{json.dumps(timeline)},{{title:"Live switch events over the run",xaxis:{{...minax,dtick:30}},yaxis:{{tickvals:[0,1],ticktext:["back","in"],range:[-0.5,1.5]}}}});
Plotly.newPlot("rt",{json.dumps(rt_traces)},{{title:"Per-switch durations (s)",xaxis:{{...minax,dtick:30}},yaxis:{{title:"seconds"}}}});
Plotly.newPlot("rs1",{json.dumps(rs_traces)},{{title:"Regime shift: switch-ins (dashed = 16K→8K flip)",xaxis:minax,yaxis:{{visible:false,range:[0,2]}},shapes:{json.dumps(rs_shapes)}}});
Plotly.newPlot("rs2",[{json.dumps(rs_bars)}],{{title:"Step time around the flip (s)",yaxis:{{title:"s/step"}}}});
Plotly.newPlot("nh1",[{json.dumps(nh_reasons)}],{{title:"Abstention reasons (1,656 decisions, 0 switches)",yaxis:{{title:"count"}}}});
Plotly.newPlot("nh2",[{json.dumps(nh_steps)}],{{title:"Step time: armed vs control (s)",yaxis:{{title:"s/step"}}}});
Plotly.newPlot("rund",{json.dumps(rund_traces)},{{title:"Run D: static 1:2 on 3 of 4 GPUs",xaxis:{{...minax,dtick:10}},yaxis:{{title:"util %",range:[0,102]}},legend:{{orientation:"h"}}}});
</script></body></html>"""
open(f"{B}/verl-poc-dashboard.html","w").write(html)
print("written", len(html), "bytes")
