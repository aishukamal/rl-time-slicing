import json, re, sys, datetime as dt

D = "/tmp/tsrun-1784585716"

def parse_orc(path):
    evs = []
    seen = set()
    for line in open(path, errors="replace"):
        i = line.find("{")
        if i < 0: continue
        try: j = json.loads(line[i:])
        except Exception: continue
        msg = j.get("msg",""); t = j.get("time","")
        job = j.get("JobID") or j.get("jobID") or ""
        key = (t, msg, job)
        if key in seen: continue
        seen.add(key)
        if msg in ("Acquire called","Yield called","Acquire succeeded, job loaded and lock held"):
            evs.append((t, "ORC", msg, job))
    return evs

def parse_agent(path):
    evs = []; seen=set()
    for line in open(path, errors="replace"):
        i = line.find("{")
        if i < 0: continue
        try: j = json.loads(line[i:])
        except Exception: continue
        msg = j.get("msg",""); t=j.get("time",""); job=j.get("JobID","")
        key=(t,msg,job)
        if key in seen: continue
        seen.add(key)
        if msg in ("Snapshot called","Restore called"):
            evs.append((t,"AGT",msg,job))
        elif msg == "cuda-checkpoint action took":
            evs.append((t,"AGT",f"snapshot cuda-checkpoint done ({j['duration']/1e9:.1f}s)",job))
        elif msg == "cuda-checkpoint toggle took":
            evs.append((t,"AGT",f"restore cuda-checkpoint done ({j['duration']/1e9:.1f}s) pids={j.get('pids')}",job))
        elif msg in ("Snapshotting PIDs","Restoring PIDs"):
            evs.append((t,"AGT",f"{msg} {j.get('pids')}",job))
    return evs

def parse_steps(path, job):
    evs=[]; seen=set()
    pat = re.compile(r"^(\S+)\s.*step:(\d+) - global_seqlen")
    met = re.compile(r"critic/rewards/mean:([\d.eE+-]+)")
    loss = re.compile(r"actor/loss:([\-\d.eE+]+)")
    for line in open(path, errors="replace"):
        m = pat.match(line)
        if not m: continue
        step = int(m.group(2))
        if step in seen: continue
        seen.add(step)
        r = met.search(line); l = loss.search(line)
        evs.append((m.group(1), "JOB", f"step {step} done reward_mean={r.group(1) if r else '?'} loss={l.group(1) if l else '?'}", job))
    return evs

def norm(t):
    t = t.rstrip("Z")
    if "." in t:
        base, frac = t.split("."); frac = (frac + "000000")[:6]
        t = f"{base}.{frac}"
    else:
        t += ".000000"
    return dt.datetime.fromisoformat(t)

evs = parse_orc(f"{D}/orchestrator.log") + parse_agent(f"{D}/snapshot-agent.log")
evs += parse_steps(f"{D}/final-verl-job-a3.log", "verl-job-a3")
evs += parse_steps(f"{D}/final-verl-job-b3.log", "verl-job-b3")
evs += parse_steps(f"{D}/run1-verl-job-a.log", "verl-job-a")
evs = [(norm(t), src, msg, job) for t, src, msg, job in evs if t]
evs.sort(key=lambda e: e[0])

start = dt.datetime(2026,7,20,22,15,0)
with open(f"{D}/timeline-full.txt","w") as f:
    for t,src,msg,job in evs:
        if t < start: continue
        f.write(f"{t.isoformat()}  {src:3s}  {job:12s}  {msg}\n")

# run3 window
r3 = [e for e in evs if e[0] >= dt.datetime(2026,7,20,23,27,0)]
with open(f"{D}/timeline-run3.txt","w") as f:
    for t,src,msg,job in r3:
        f.write(f"{t.isoformat()}  {src:3s}  {job:12s}  {msg}\n")

# turn/switch accounting run3: grants and yields from ORC
def turns(evs, jobs):
    grants = [(t,j) for t,s,m,j in evs if m.startswith("Acquire succeeded") and j in jobs]
    yields_ = [(t,j) for t,s,m,j in evs if m=="Yield called" and j in jobs]
    seq = sorted([(t,"G",j) for t,j in grants] + [(t,"Y",j) for t,j in yields_])
    out=[]
    cur=None
    for t,k,j in seq:
        if k=="G": cur=(t,j)
        elif k=="Y" and cur and cur[1]==j:
            out.append((j, cur[0], t, (t-cur[0]).total_seconds()))
            cur=None
    return out, seq

r3turns, seq3 = turns(r3, {"verl-job-a3","verl-job-b3"})
print("=== RUN3 turns (grant->yield) ===")
for j,t0,t1,d in r3turns:
    print(f"{j}  {t0.time()} -> {t1.time()}  {d:.1f}s")
print("\n=== RUN3 switch overhead (yield X -> grant Y) ===")
for i in range(len(seq3)-1):
    t,k,j = seq3[i]; t2,k2,j2 = seq3[i+1]
    if k=="Y" and k2=="G" and j!=j2:
        print(f"{j} yield {t.time()} -> {j2} grant {t2.time()}  overhead {(t2-t).total_seconds():.1f}s")
# run1 solo turns
r1 = [e for e in evs if dt.datetime(2026,7,20,22,36,0) <= e[0] <= dt.datetime(2026,7,20,22,46,0)]
r1turns,_ = turns(r1, {"verl-job-a"})
print("\n=== RUN1 solo turns (verl-job-a) ===")
for j,t0,t1,d in r1turns:
    print(f"{j}  {t0.time()} -> {t1.time()}  {d:.1f}s")
