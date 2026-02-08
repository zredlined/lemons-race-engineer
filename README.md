# Driver Coach (Race Engineer)

Lightweight tooling to turn iRacing telemetry into track maps, cues, and a live "Race Engineer" assistant with audio callouts and a small debug UI.

**Quick Start**
1. Generate cues from an `.ibt` file:
```bash
uv run --with pyirsdk --with matplotlib python3 tools/track_notes.py \
  --ibt "/path/to/file.ibt" \
  --track sonoma_lemons \
  --out outputs \
  --optimal
```
2. Replay mode (debug without iRacing):
```bash
uv run --with pyirsdk python3 tools/race_engineer.py \
  --cues outputs/sonoma_lemons/cues.json \
  --turns outputs/sonoma_lemons/turns.json \
  --notes notes/sonoma_lemons_notes.json \
  --replay outputs/sonoma_lemons/reference_lap.csv
```
3. Live mode (iRacing running on Windows):
```bash
uv run --with pyirsdk python3 tools/race_engineer.py \
  --cues outputs/sonoma_lemons/cues.json \
  --turns outputs/sonoma_lemons/turns.json \
  --notes notes/sonoma_lemons_notes.json \
  --live
```

Open the UI at `http://localhost:5000`.

**What Gets Generated**
1. `outputs/<track_id>/turns.json`  
Turn boundaries and apex points.
2. `outputs/<track_id>/cues.json`  
Trigger points plus per-turn telemetry stats (brake, throttle, target gear).
3. `outputs/<track_id>/track.png` and `track.html`  
Track map with labeled turns and target gears.
4. `outputs/<track_id>/reference_lap.csv`  
Reference lap for replay mode.
5. `outputs/<track_id>/optimal_segments.csv`  
Best per-turn segments across all laps in the session.

**Key Options**
1. `--track`  
Sets the track id and output subfolder (e.g., `outputs/sonoma_lemons/`).
2. `--steer-threshold`, `--min-sep`  
Tune turn detection based on steering peaks.
3. `--min-gear`, `--min-speed`  
Filter out shift artifacts for target gear labeling.
4. `--map`  
Optional reference lap CSV for the track map (defaults to `reference_lap.csv` next to `cues.json`).

**Notes File (Human-Editable)**
Use `notes/<track>_notes.json` to add coaching cues. Timing is defined in **meters before** the anchor point (recommended).
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
Legacy support: `lead_s_pct` still works but is discouraged.

**Notes**
1. Live mode only works on Windows because iRacing shared memory is Windows-only.
2. TTS output uses PowerShell on Windows, `say` on macOS, and `espeak` on Linux.

**Sharing This Repo**
If this folder isn’t a git repo yet, you can initialize and push it:
```bash
git init
git add .
git commit -m "Add race engineer tooling"
git remote add origin <your_repo_url>
git push -u origin main
```
