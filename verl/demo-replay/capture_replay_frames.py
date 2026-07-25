#!/usr/bin/env python3
"""Capture frames of the unified replay page for GIF/MP4 generation."""
import json, os, shutil, subprocess

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BASE = "/Users/aishuk/workspaces/GPU-CR/timeslice-demos"
OUT = f"{BASE}/frames-replay"
N = 200
DUR = 11330  # 188:50, final yield

os.makedirs(OUT, exist_ok=True)
for i in range(N):
    t = DUR * i / (N - 1)
    out = f"{OUT}/frame_{i:04d}.png"
    if os.path.exists(out):
        continue
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--window-size=1380,900",
                    f"--screenshot={out}",
                    f"file://{BASE}/timeslice-replay.html?t={t:.0f}" + ("&end=1" if i == N-1 else "")],
                   capture_output=True, timeout=30)
    if i % 40 == 0:
        print(f"{i}/{N}")
# hold last frame 2s at 10fps
for j in range(20):
    shutil.copy(f"{OUT}/frame_{N-1:04d}.png", f"{OUT}/frame_{N+j:04d}.png")
print("done")
