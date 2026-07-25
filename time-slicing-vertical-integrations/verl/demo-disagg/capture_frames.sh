#!/bin/bash
# Capture the dashboard at 12 fps into frames/ for MP4 assembly.
# Each frame is a deterministic still (?t=N with transitions frozen).
set -u
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/frames"
mkdir -p "$OUT"
FPS=12
DUR=106
N=$((DUR * FPS))
for ((i = 0; i < N; i++)); do
  f=$(printf "%s/f%05d.png" "$OUT" "$i")
  [ -s "$f" ] && continue   # resumable
  t=$(python3 -c "print($i / $FPS)")
  "$CHROME" --headless --disable-gpu --hide-scrollbars \
    --window-size=1920,1080 --screenshot="$f" \
    "file://$DIR/index.html?t=$t" >/dev/null 2>&1
done
echo "captured $(ls "$OUT" | wc -l | tr -d ' ') frames"
