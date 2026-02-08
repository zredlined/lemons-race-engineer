#!/usr/bin/env python3
"""
Live cue player for iRacing with a small web UI.

Run (live):
  uv run --with pyirsdk python3 tools/race_engineer.py \
    --cues outputs/sonoma_lemons/cues.json \
    --turns outputs/sonoma_lemons/turns.json \
    --live

Run (replay):
  uv run --with pyirsdk python3 tools/race_engineer.py \
    --cues outputs/sonoma_lemons/cues.json \
    --turns outputs/sonoma_lemons/turns.json \
    --replay outputs/sonoma_lemons/reference_lap.csv
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import platform
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
from typing import Dict, List, Optional

import irsdk


@dataclass
class Cue:
    turn_id: str
    trigger_s_pct: float
    data: Dict
    text: str | None = None
    cue_type: str | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Race Engineer: iRacing cue player + web UI")
    p.add_argument("--cues", required=True, help="Path to cues.json")
    p.add_argument("--turns", required=True, help="Path to turns.json")
    p.add_argument("--notes", default="", help="Optional notes JSON to overlay spoken cues")
    p.add_argument("--live", action="store_true", help="Use live iRacing telemetry")
    p.add_argument("--replay", default="", help="Replay from reference_lap.csv")
    p.add_argument("--map", default="", help="Reference lap CSV for track map (optional)")
    p.add_argument("--audio", default="tts", choices=["tts", "off"], help="Audio mode")
    p.add_argument("--port", type=int, default=5000, help="Web UI port")
    p.add_argument("--tick", type=float, default=0.10, help="Update interval (seconds)")
    p.add_argument("--replay-speed", type=float, default=1.0, help="Replay speed multiplier")
    return p.parse_args()


def load_cues(path: str) -> List[Cue]:
    with open(path, "r") as f:
        data = json.load(f)
    cues = []
    for c in data.get("cues", []):
        cues.append(
            Cue(
                turn_id=c["turn_id"],
                trigger_s_pct=float(c["trigger_s_pct"]),
                data=c.get("data", {}),
                text=None,
                cue_type="auto",
            )
        )
    return sorted(cues, key=lambda c: c.trigger_s_pct)


def load_turns(path: str) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("turns", [])


def load_notes(path: str) -> Optional[Dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def in_segment(s_pct: float, start: float, end: float) -> bool:
    seg_len = (end - start) % 1.0
    d = (s_pct - start) % 1.0
    return d <= seg_len


def find_turn(turns: List[Dict], s_pct: float) -> Optional[Dict]:
    for t in turns:
        if "start_s_pct" in t and "end_s_pct" in t:
            if in_segment(s_pct, t["start_s_pct"], t["end_s_pct"]):
                return t
    return None


def build_phrase(cue: Cue, turn_label: str) -> str:
    if cue.text:
        return f"Turn {turn_label}. {cue.text}"
    gear = cue.data.get("target_gear")
    if gear is not None:
        return f"Turn {turn_label}. Gear {int(gear)}."
    return f"Turn {turn_label}."


class TTSEngine(threading.Thread):
    def __init__(self, enabled: bool = True):
        super().__init__(daemon=True)
        self.enabled = enabled
        self.q: "queue.Queue[str]" = queue.Queue()

    def run(self) -> None:
        while True:
            text = self.q.get()
            if text is None:
                return
            if not self.enabled:
                continue
            self.speak(text)

    def speak(self, text: str) -> None:
        system = platform.system()
        try:
            if system == "Windows":
                cmd = [
                    "powershell",
                    "-Command",
                    ("Add-Type -AssemblyName System.Speech; "
                     f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"),
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            elif system == "Darwin":
                subprocess.run(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            else:
                subprocess.run(["espeak", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception:
            pass

    def say(self, text: str) -> None:
        if self.enabled:
            self.q.put(text)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled


class State:
    def __init__(self) -> None:
        self.s_pct = 0.0
        self.speed = 0.0
        self.gear = 0
        self.lap = 0
        self.mode = ""
        self.audio_enabled = True
        self.current_turn = ""
        self.next_turn = ""
        self.next_cue_in_pct = 0.0
        self.last_cue = ""
        self.updated_at = time.time()


def serve_state(state: State, port: int, map_data: Optional[Dict], tts: TTSEngine) -> None:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, content: str, ctype: str = "text/html") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))

        def do_GET(self) -> None:
            if self.path == "/state":
                payload = json.dumps(
                    {
                        "s_pct": state.s_pct,
                        "speed": state.speed,
                        "gear": state.gear,
                        "lap": state.lap,
                        "mode": state.mode,
                        "audio_enabled": state.audio_enabled,
                        "current_turn": state.current_turn,
                        "next_turn": state.next_turn,
                        "next_cue_in_pct": state.next_cue_in_pct,
                        "last_cue": state.last_cue,
                        "updated_at": state.updated_at,
                    }
                )
                self._send(200, payload, "application/json")
                return
            if self.path.startswith("/audio"):
                q = parse_qs(urlparse(self.path).query)
                if "enabled" in q:
                    val = q["enabled"][0].lower()
                    enabled = val in ("1", "true", "yes", "on")
                    tts.set_enabled(enabled)
                    state.audio_enabled = enabled
                self._send(200, json.dumps({"audio_enabled": state.audio_enabled}), "application/json")
                return
            if self.path == "/map":
                if map_data is None:
                    self._send(200, json.dumps({"available": False}), "application/json")
                else:
                    self._send(200, json.dumps(map_data), "application/json")
                return

            html = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Race Engineer</title>
    <style>
      body { font-family: sans-serif; margin: 20px; }
      .box { border: 1px solid #ccc; padding: 12px; width: 380px; }
      h2 { margin: 0 0 10px 0; }
      .row { margin: 6px 0; }
      .label { color: #666; }
      .value { font-weight: 600; }
      canvas { border: 1px solid #eee; margin-top: 10px; }
    </style>
  </head>
  <body>
    <div class="box">
      <h2>Race Engineer</h2>
      <div class="row"><span class="label">Mode:</span> <span class="value" id="mode"></span></div>
      <div class="row">
        <span class="label">Audio:</span>
        <span class="value" id="audio_state"></span>
        <button id="audio_btn" style="margin-left:8px;">Toggle</button>
      </div>
      <div class="row"><span class="label">Lap:</span> <span class="value" id="lap"></span></div>
      <div class="row"><span class="label">s_pct:</span> <span class="value" id="s_pct"></span></div>
      <div class="row"><span class="label">Speed:</span> <span class="value" id="speed"></span></div>
      <div class="row"><span class="label">Gear:</span> <span class="value" id="gear"></span></div>
      <div class="row"><span class="label">Current:</span> <span class="value" id="current_turn"></span></div>
      <div class="row"><span class="label">Next:</span> <span class="value" id="next_turn"></span></div>
      <div class="row"><span class="label">Next Cue In:</span> <span class="value" id="next_cue"></span></div>
      <div class="row"><span class="label">Last Cue:</span> <span class="value" id="last_cue"></span></div>
      <canvas id="map" width="340" height="340"></canvas>
    </div>
    <script>
      let mapData = null;
      const canvas = document.getElementById('map');
      const ctx = canvas.getContext('2d');

      function toCanvas(x, y) {
        const pad = 10;
        const w = canvas.width, h = canvas.height;
        const dx = mapData.max_x - mapData.min_x || 1;
        const dy = mapData.max_y - mapData.min_y || 1;
        const scale = Math.min((w - 2*pad)/dx, (h - 2*pad)/dy);
        const cx = pad + (x - mapData.min_x) * scale;
        const cy = h - (pad + (y - mapData.min_y) * scale);
        return {cx, cy};
      }

      function drawTrackDot(s_pct) {
        if (!mapData || !mapData.available) return;
        const xs = mapData.x, ys = mapData.y, ss = mapData.s_pct;
        // find closest index
        let best = 0;
        let bestD = 10;
        for (let i = 0; i < ss.length; i++) {
          const d = Math.abs(ss[i] - s_pct);
          if (d < bestD) { bestD = d; best = i; }
        }
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
        for (let i = 0; i < xs.length; i++) {
          const p = toCanvas(xs[i], ys[i]);
          if (i === 0) ctx.moveTo(p.cx, p.cy);
          else ctx.lineTo(p.cx, p.cy);
        }
        ctx.strokeStyle = "#222";
        ctx.lineWidth = 1.5;
        ctx.stroke();
        const p = toCanvas(xs[best], ys[best]);
        ctx.beginPath();
        ctx.arc(p.cx, p.cy, 4, 0, Math.PI*2);
        ctx.fillStyle = "#d62728";
        ctx.fill();
      }

      async function loadMap() {
        const r = await fetch('/map');
        const data = await r.json();
        mapData = data;
        if (!mapData.available) {
          ctx.fillStyle = "#999";
          ctx.fillText("Map not available", 90, 170);
        }
      }

      async function tick() {
        const r = await fetch('/state');
        const s = await r.json();
        document.getElementById('mode').textContent = s.mode;
        document.getElementById('audio_state').textContent = s.audio_enabled ? 'On' : 'Off';
        document.getElementById('lap').textContent = s.lap;
        document.getElementById('s_pct').textContent = s.s_pct.toFixed(4);
        document.getElementById('speed').textContent = s.speed.toFixed(1);
        const g = s.gear;
        const gText = (g === 0) ? 'N' : (g < 0 ? 'R' : String(g));
        document.getElementById('gear').textContent = gText;
        document.getElementById('current_turn').textContent = s.current_turn;
        document.getElementById('next_turn').textContent = s.next_turn;
        document.getElementById('next_cue').textContent = (s.next_cue_in_pct * 100).toFixed(1) + '%';
        document.getElementById('last_cue').textContent = s.last_cue;
        if (mapData && mapData.available) {
          drawTrackDot(s.s_pct);
        }
      }
      document.getElementById('audio_btn').addEventListener('click', async () => {
        const r = await fetch('/state');
        const s = await r.json();
        const next = s.audio_enabled ? '0' : '1';
        await fetch('/audio?enabled=' + next);
        tick();
      });
      loadMap();
      setInterval(tick, 200);
      tick();
    </script>
  </body>
</html>
"""
            self._send(200, html)

        def log_message(self, format: str, *args) -> None:
            return

    server = HTTPServer(("0.0.0.0", port), Handler)
    server.serve_forever()


def replay_rows(path: str):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return []
    return rows


def project_latlon(lat: List[float], lon: List[float]) -> Dict[str, List[float]]:
    if not lat or not lon:
        return {"x": [], "y": []}
    lat0 = sum(lat) / len(lat)
    lon0 = sum(lon) / len(lon)
    r = 6371000.0
    x = [math.radians(lon[i] - lon0) * math.cos(math.radians(lat0)) * r for i in range(len(lat))]
    y = [math.radians(lat[i] - lat0) * r for i in range(len(lat))]
    return {"x": x, "y": y}


def load_map_data(map_path: str) -> Optional[Dict]:
    if not map_path or not os.path.exists(map_path):
        return None
    rows = replay_rows(map_path)
    if not rows:
        return None
    lat = [float(r["Lat"]) for r in rows]
    lon = [float(r["Lon"]) for r in rows]
    s_pct = [float(r["LapDistPct"]) for r in rows]
    speed = [float(r["Speed"]) for r in rows]
    lapdist = [float(r["LapDist"]) for r in rows]
    track_length_m = max(lapdist) if lapdist else None
    xy = project_latlon(lat, lon)
    x = xy["x"]
    y = xy["y"]
    step = max(1, len(x) // 800)
    x = x[::step]
    y = y[::step]
    s_pct = s_pct[::step]
    speed = speed[::step]
    if not x or not y:
        return None
    return {
        "available": True,
        "x": x,
        "y": y,
        "s_pct": s_pct,
        "speed": speed,
        "track_length_m": track_length_m,
        "min_x": min(x),
        "max_x": max(x),
        "min_y": min(y),
        "max_y": max(y),
    }


def main() -> None:
    args = parse_args()
    cues = load_cues(args.cues)
    turns = load_turns(args.turns)
    turn_labels = {t["id"]: t.get("label", t["id"]) for t in turns}
    map_path = args.map
    if not map_path:
        # assume reference_lap.csv next to cues
        base = os.path.dirname(os.path.abspath(args.cues))
        candidate = os.path.join(base, "reference_lap.csv")
        if os.path.exists(candidate):
            map_path = candidate

    # start web server (map data needed for optional note timing)
    map_data = load_map_data(map_path) if map_path else None

    # overlay notes if provided
    notes = load_notes(args.notes) if args.notes else None
    if notes:
        defaults = notes.get("defaults", {})
        announce_m_by_type = defaults.get(
            "announce_m_before_by_type",
            {
                "setup": 80,
                "line": 80,
                "apex": 40,
                "brake": 80,
                "throttle": 20,
                "caution": 80,
                "landmark": 80,
            },
        )
        # legacy (backward compatible)
        lead_by_type = defaults.get(
            "lead_s_pct_by_type",
            {
                "setup": 0.02,
                "line": 0.02,
                "apex": 0.01,
                "brake": 0.02,
                "throttle": 0.005,
                "caution": 0.02,
                "landmark": 0.02,
            },
        )
        anchor_by_type = defaults.get(
            "anchor_by_type",
            {
                "setup": "start",
                "line": "start",
                "apex": "apex",
                "brake": "brake",
                "throttle": "throttle",
                "caution": "start",
                "landmark": "start",
            },
        )
        base_by_turn = {c.turn_id: c for c in cues}
        for turn_id, items in notes.get("turns", {}).items():
            base = base_by_turn.get(turn_id)
            if not base:
                continue
            for item in items:
                ctype = item.get("type", "line")
                text = item.get("text", "")
                anchor = item.get("anchor", anchor_by_type.get(ctype, "start"))
                if anchor == "apex":
                    anchor_s = base.data.get("apex_s_pct") or base.trigger_s_pct
                elif anchor == "brake":
                    anchor_s = base.data.get("brake_start_s_pct") or base.trigger_s_pct
                elif anchor == "throttle":
                    anchor_s = base.data.get("throttle_on_s_pct") or base.trigger_s_pct
                else:
                    anchor_s = base.trigger_s_pct

                lead_s_pct = None
                # Preferred: meters before anchor
                announce_m = item.get("announce_m_before")
                if announce_m is None:
                    announce_m = announce_m_by_type.get(ctype)
                if announce_m is not None and map_data and map_data.get("track_length_m"):
                    lead_s_pct = float(announce_m) / float(map_data["track_length_m"])
                # Optional: seconds before (based on reference speed at anchor)
                announce_s = item.get("announce_s_before")
                if announce_s is not None and map_data and map_data.get("speed") and map_data.get("s_pct"):
                    s_list = map_data["s_pct"]
                    spd_list = map_data["speed"]
                    best_i = min(range(len(s_list)), key=lambda i: abs((s_list[i] - anchor_s + 0.5) % 1.0 - 0.5))
                    ref_speed = spd_list[best_i] or 0.1
                    lead_s_pct = (float(announce_s) * float(ref_speed)) / float(map_data["track_length_m"] or 1.0)
                if lead_s_pct is None:
                    lead_s_pct = float(item.get("lead_s_pct", lead_by_type.get(ctype, 0.02)))

                trigger = (float(anchor_s) - float(lead_s_pct)) % 1.0
                cues.append(
                    Cue(
                        turn_id=turn_id,
                        trigger_s_pct=trigger,
                        data=base.data,
                        text=text,
                        cue_type=ctype,
                    )
                )
        cues = sorted(cues, key=lambda c: c.trigger_s_pct)

    state = State()
    tts = TTSEngine(enabled=(args.audio == "tts"))
    tts.start()

    # start web server
    state.audio_enabled = (args.audio == "tts")
    threading.Thread(target=serve_state, args=(state, args.port, map_data, tts), daemon=True).start()
    print(f"Race Engineer UI: http://localhost:{args.port}")

    cue_triggers = [c.trigger_s_pct for c in cues]
    next_idx = 0
    last_s_pct = None

    if args.replay:
        state.mode = "replay"
        rows = replay_rows(args.replay)
        if not rows:
            raise SystemExit("Replay file empty.")
        i = 0
        base_t = float(rows[0]["SessionTime"])
        start_wall = time.time()
        while True:
            row = rows[i]
            session_t = float(row["SessionTime"])
            target_wall = start_wall + (session_t - base_t) / args.replay_speed
            now = time.time()
            if target_wall > now:
                time.sleep(min(0.05, target_wall - now))
                continue

            s_pct = float(row["LapDistPct"])
            speed = float(row["Speed"])
            gear = int(float(row["Gear"]))
            lap = int(float(row.get("Lap", 0)))

            # cue logic
            if last_s_pct is None:
                next_idx = bisect.bisect_right(cue_triggers, s_pct)
            else:
                if s_pct >= last_s_pct:
                    while next_idx < len(cues) and cue_triggers[next_idx] <= s_pct:
                        cue = cues[next_idx]
                        phrase = build_phrase(cue, turn_labels.get(cue.turn_id, cue.turn_id))
                        tts.say(phrase)
                        state.last_cue = phrase
                        next_idx += 1
                else:
                    while next_idx < len(cues):
                        cue = cues[next_idx]
                        phrase = build_phrase(cue, turn_labels.get(cue.turn_id, cue.turn_id))
                        tts.say(phrase)
                        state.last_cue = phrase
                        next_idx += 1
                    next_idx = 0
                    while next_idx < len(cues) and cue_triggers[next_idx] <= s_pct:
                        cue = cues[next_idx]
                        phrase = build_phrase(cue, turn_labels.get(cue.turn_id, cue.turn_id))
                        tts.say(phrase)
                        state.last_cue = phrase
                        next_idx += 1

            last_s_pct = s_pct
            current_turn = find_turn(turns, s_pct)
            next_turn = cues[next_idx].turn_id if next_idx < len(cues) else cues[0].turn_id
            next_trigger = cue_triggers[next_idx] if next_idx < len(cues) else cue_triggers[0]
            state.s_pct = s_pct
            state.speed = speed
            state.gear = gear
            state.lap = lap
            state.current_turn = (current_turn or {}).get("label", "")
            state.next_turn = turn_labels.get(next_turn, next_turn)
            state.next_cue_in_pct = (next_trigger - s_pct) % 1.0
            state.updated_at = time.time()

            i += 1
            if i >= len(rows):
                i = 0
                base_t = float(rows[0]["SessionTime"])
                start_wall = time.time()

    # live mode
    if not args.live:
        raise SystemExit("Use --live or --replay.")

    if platform.system() != "Windows":
        raise SystemExit("Live mode requires Windows iRacing telemetry. Use --replay on this OS.")

    state.mode = "live"
    ir = irsdk.IRSDK()
    if not ir.startup():
        raise SystemExit("iRacing not running or telemetry not available.")

    while True:
        s_pct = float(ir["LapDistPct"])
        speed = float(ir["Speed"])
        gear = int(ir["Gear"])
        lap = int(ir["Lap"])

        if last_s_pct is None:
            next_idx = bisect.bisect_right(cue_triggers, s_pct)
        else:
            if s_pct >= last_s_pct:
                while next_idx < len(cues) and cue_triggers[next_idx] <= s_pct:
                    cue = cues[next_idx]
                    phrase = build_phrase(cue, turn_labels.get(cue.turn_id, cue.turn_id))
                    tts.say(phrase)
                    state.last_cue = phrase
                    next_idx += 1
            else:
                while next_idx < len(cues):
                    cue = cues[next_idx]
                    phrase = build_phrase(cue, turn_labels.get(cue.turn_id, cue.turn_id))
                    tts.say(phrase)
                    state.last_cue = phrase
                    next_idx += 1
                next_idx = 0
                while next_idx < len(cues) and cue_triggers[next_idx] <= s_pct:
                    cue = cues[next_idx]
                    phrase = build_phrase(cue, turn_labels.get(cue.turn_id, cue.turn_id))
                    tts.say(phrase)
                    state.last_cue = phrase
                    next_idx += 1

        last_s_pct = s_pct
        current_turn = find_turn(turns, s_pct)
        next_turn = cues[next_idx].turn_id if next_idx < len(cues) else cues[0].turn_id
        next_trigger = cue_triggers[next_idx] if next_idx < len(cues) else cue_triggers[0]
        state.s_pct = s_pct
        state.speed = speed
        state.gear = gear
        state.lap = lap
        state.current_turn = (current_turn or {}).get("label", "")
        state.next_turn = turn_labels.get(next_turn, next_turn)
        state.next_cue_in_pct = (next_trigger - s_pct) % 1.0
        state.updated_at = time.time()

        time.sleep(args.tick)


if __name__ == "__main__":
    main()
