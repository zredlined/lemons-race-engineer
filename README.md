# Race Engineer (Driver Coach)

This repo helps you **record, understand, and replay track notes** while you train in iRacing, with a clear path to reuse those same notes in a real car later.

**Why this exists**
1. Learn a track faster with **in‑sim cues**, not just offline analysis.
2. Keep a **single source of truth** for track notes that won’t get overwritten.
3. Make it easy to iterate as your lap times improve.

---

## Core Workflow (KISS)

**Step 1 — Record a reference lap in iRacing**
Export an `.ibt` file from iRacing (any filename is fine).

**Step 2 — Generate the track map + cues**
> **Cues** = short coaching callouts tied to a specific point on track (usually a turn entry, apex, or exit).
```bash
uv run --with pyirsdk --with matplotlib python3 tools/track_notes.py \
  --ibt "/path/to/file.ibt" \
  --track sonoma_lemons \
  --out outputs \
  --optimal
```
Review `outputs/sonoma_lemons/track.html` to confirm turn labels look right.

**Step 3 — Add human/AI notes (safe to edit)**
Edit:
```
notes/sonoma_lemons_notes.json
```
These notes are **never overwritten** when you regenerate from a faster lap.

**Step 4 — Train with replay or live mode**
Replay (no iRacing required):
```bash
./replay.sh sonoma_lemons
```

Live (iRacing running on Windows):
```bash
./live.sh sonoma_lemons
```

Open the UI at `http://localhost:5000` and toggle audio on/off.

---

## What This Generates
1. `outputs/<track_id>/turns.json`  
Turn boundaries and apex points.
2. `outputs/<track_id>/cues.json`  
Trigger points plus per‑turn telemetry stats (brake, throttle, target gear).
3. `outputs/<track_id>/track.html` and `track.png`  
Track map with labeled turns and target gears.
4. `outputs/<track_id>/reference_lap.csv`  
Reference lap for replay mode.
5. `outputs/<track_id>/optimal_segments.csv`  
Best per‑turn segments across all laps in the session.

---

## Notes File (Human‑Editable)
Use `notes/<track>_notes.json` to add coaching cues. Timing is defined in **meters before** the anchor point.
```json
{
  "defaults": {
    "announce_m_before_by_type": { "apex": 40, "brake": 80, "throttle": 20 },
    "anchor_by_type": { "apex": "apex", "brake": "brake", "throttle": "throttle" }
  },
  "turns": {
    "T11": [
      { "type": "brake", "text": "Brake early downhill" },
      { "type": "apex", "text": "Very late apex" },
      { "type": "throttle", "text": "Squeeze throttle on exit", "announce_m_before": 10 }
    ]
  }
}
```

---

## Current Scope (and what’s coming)
**Now**
1. Cues are derived from your **reference lap**.
2. Track maps are generated from that lap.
3. Notes can be augmented by AI or transcripts and stay editable.

**Later**
1. Full track boundaries and line‑distance metrics.
2. Real‑car GPS ingestion.
3. Optional LCD race display mode.

---

## Key Options
1. `--track`  
Sets the track id and output subfolder (e.g., `outputs/sonoma_lemons/`).
2. `--steer-threshold`, `--min-sep`  
Tune turn detection based on steering peaks.
3. `--min-gear`, `--min-speed`  
Filter out shift artifacts for target gear labeling.
4. `--map`  
Optional reference lap CSV for the track map (defaults to `reference_lap.csv` next to `cues.json`).

---

## Notes
1. Live mode only works on Windows (iRacing shared memory is Windows‑only).
2. TTS output uses PowerShell on Windows, `say` on macOS, and `espeak` on Linux.
