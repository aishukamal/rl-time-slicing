#!/usr/bin/env python3
"""Build the self-contained HTML report for the RL time-slicing benchmark project."""
import base64
import os
import re

ROOT = "/Users/aishuk/workspaces/GPU-CR"
OUT = f"{ROOT}/rl-timeslicing-benchmark-report.html"

IMAGES = {
    "P1_SQUARE": f"{ROOT}/benchmark-longtail/iter2_square_wave.png",
    "P1_DUTY": f"{ROOT}/benchmark-longtail/gpu_duty_cycle_iter1_long.png",
    "P2_TIMELINE": f"{ROOT}/benchmark-deepresearch/run4_gpu_timeline.png",
    "P2_PHASES": f"{ROOT}/benchmark-deepresearch/run4_phase_structure.png",
    "P3_TIMELINES": f"{ROOT}/async-multiturn/sweep_timelines.png",
    "P3_IDLE": f"{ROOT}/async-multiturn/sweep_idle_comparison.png",
    "P4_TIMELINES": f"{ROOT}/async-longcot/longcot_timelines.png",
    "P4_REGIME": f"{ROOT}/async-longcot/regime_map.png",
}

def img_tag(key, alt):
    path = IMAGES[key]
    if not os.path.exists(path):
        return f'<p class="muted">[missing image: {os.path.basename(path)}]</p>'
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{alt}"/><figcaption>{alt}</figcaption></figure>'

HTML = open(f"{ROOT}/report_template.html").read()

def repl(m):
    key, alt = m.group(1), m.group(2)
    return img_tag(key, alt)

HTML = re.sub(r"\{\{IMG:([A-Z0-9_]+)\|([^}]*)\}\}", repl, HTML)
open(OUT, "w").write(HTML)
print(f"wrote {OUT} ({os.path.getsize(OUT)//1024} KB)")
