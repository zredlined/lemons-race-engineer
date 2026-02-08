#!/usr/bin/env python3
"""
Track notes utility for iRacing .ibt telemetry.

Outputs:
  - lap_summary.csv
  - reference_lap.csv
  - turns.json
  - cues.json
  - track.png / track.svg / track.html
  - optimal_segments.csv (optional)

Run:
  uv run --with pyirsdk --with matplotlib python3 tools/track_notes.py --ibt "path/to/file.ibt" --track sonoma_lemons
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import irsdk  # from pyirsdk
import matplotlib.pyplot as plt


SONOMA_LEMONS_TURNS = [
    "1",
    "2",
    "3",
    "3A",
    "4",
    "5",
    "6",
    "7",
    "7A",
    "8",
    "8A",
    "9A",
    "10",
    "11",
    "12",
]


@dataclass
class LapInfo:
    lap: int
    time_s: float
    idxs: List[int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="iRacing .ibt track notes utility")
    p.add_argument("--ibt", required=True, help="Path to .ibt file")
    p.add_argument("--out", default="outputs", help="Base output directory")
    p.add_argument("--track", default="", help="Track preset / track id (e.g., sonoma_lemons)")
    p.add_argument("--lap", default="fastest", help="Lap number or 'fastest'")
    p.add_argument("--turns-json", default="", help="Turn list JSON (labels only or with apex_s_pct)")
    p.add_argument("--turns-count", type=int, default=0, help="Auto-detect N turns (if no labels)")
    p.add_argument("--steer-threshold", type=float, default=0.10, help="Steering peak threshold (abs)")
    p.add_argument("--min-sep", type=float, default=0.02, help="Min separation between peaks (s_pct)")
    p.add_argument(
        "--min-gear",
        type=int,
        default=2,
        help="Minimum gear to consider for targets (default: 2)",
    )
    p.add_argument(
        "--min-speed",
        type=float,
        default=5.0,
        help="Minimum speed (m/s) to consider for gear stats (default: 5.0)",
    )
    p.add_argument("--optimal", action="store_true", help="Compute optimal segments across laps")
    return p.parse_args()


def load_ibt(path: str) -> Dict[str, List[float]]:
    ibt = irsdk.IBT()
    ibt.open(path)
    keys = [
        "SessionTime",
        "Lap",
        "LapDistPct",
        "LapDist",
        "Speed",
        "Brake",
        "Throttle",
        "Gear",
        "SteeringWheelAngle",
        "Lat",
        "Lon",
        "Alt",
        "IsOnTrack",
    ]
    data = {k: ibt.get_all(k) for k in keys}
    ibt.close()
    return data


def collect_laps(data: Dict[str, List[float]]) -> List[LapInfo]:
    lap = data["Lap"]
    lapdistpct = data["LapDistPct"]
    t = data["SessionTime"]
    on = data["IsOnTrack"]

    by_lap: Dict[int, List[int]] = {}
    for i, (l, ot) in enumerate(zip(lap, on)):
        if not ot:
            continue
        by_lap.setdefault(int(l), []).append(i)

    laps: List[LapInfo] = []
    for l, idxs in sorted(by_lap.items()):
        if len(idxs) < 10:
            continue
        ld = [lapdistpct[i] for i in idxs]
        if min(ld) > 0.05 or max(ld) < 0.95:
            continue
        time_s = t[idxs[-1]] - t[idxs[0]]
        laps.append(LapInfo(lap=l, time_s=time_s, idxs=idxs))
    return laps


def choose_lap(laps: List[LapInfo], lap_arg: str) -> LapInfo:
    if not laps:
        raise ValueError("No full laps found.")
    if lap_arg == "fastest":
        return min(laps, key=lambda x: x.time_s)
    try:
        target = int(lap_arg)
    except ValueError as e:
        raise ValueError(f"Invalid --lap: {lap_arg}") from e
    for li in laps:
        if li.lap == target:
            return li
    raise ValueError(f"Lap {target} not found in full laps.")


def circular_delta(a: float, b: float) -> float:
    # delta from b to a on [0,1)
    return (a - b) % 1.0


def find_steering_peaks(
    s_pct: List[float],
    steer: List[float],
    count: int,
    min_sep: float,
    threshold: float,
) -> List[float]:
    peaks: List[Tuple[float, float]] = []
    for i in range(1, len(steer) - 1):
        if steer[i] < threshold:
            continue
        if steer[i] > steer[i - 1] and steer[i] > steer[i + 1]:
            peaks.append((steer[i], s_pct[i]))

    peaks.sort(reverse=True)  # by magnitude
    selected: List[Tuple[float, float]] = []
    for mag, sp in peaks:
        if all(min(circular_delta(sp, s), circular_delta(s, sp)) > min_sep for _, s in selected):
            selected.append((mag, sp))
        if len(selected) >= count:
            break
    selected.sort(key=lambda x: x[1])
    return [s for _, s in selected]


def load_turn_labels(args: argparse.Namespace) -> List[str]:
    if args.track == "sonoma_lemons":
        return SONOMA_LEMONS_TURNS[:]
    if args.turns_json:
        with open(args.turns_json, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x) for x in data]
        if isinstance(data, dict) and "turns" in data:
            return [str(t.get("label") or t.get("id")) for t in data["turns"]]
    if args.turns_count:
        return [f"T{i+1}" for i in range(args.turns_count)]
    return []


def slugify(name: str) -> str:
    keep = []
    for ch in name.lower():
        if ch.isalnum():
            keep.append(ch)
        else:
            keep.append("_")
    s = "".join(keep).strip("_")
    while "__" in s:
        s = s.replace("__", "_")
    return s or "track"


def build_turns(
    labels: List[str],
    s_pct: List[float],
    steer: List[float],
    min_sep: float,
    threshold: float,
    provided_apex: Dict[str, float] | None = None,
) -> List[Dict[str, float]]:
    turns = []
    if provided_apex:
        apexes = [provided_apex.get(lbl) for lbl in labels]
        if any(a is None for a in apexes):
            raise ValueError("Provided apex list missing some labels.")
    else:
        apexes = find_steering_peaks(s_pct, steer, len(labels), min_sep, threshold)
        if len(apexes) != len(labels):
            raise ValueError(
                f"Detected {len(apexes)} apexes but need {len(labels)}. "
                "Adjust --steer-threshold or --min-sep."
            )

    for i, lbl in enumerate(labels):
        turns.append({"id": f"T{lbl}", "label": lbl, "order": i + 1, "apex_s_pct": apexes[i]})

    # compute segment boundaries
    for i, t in enumerate(turns):
        prev_apex = turns[i - 1]["apex_s_pct"]
        next_apex = turns[(i + 1) % len(turns)]["apex_s_pct"]
        start = (prev_apex + t["apex_s_pct"]) / 2.0
        end = (t["apex_s_pct"] + next_apex) / 2.0
        if prev_apex > t["apex_s_pct"]:
            start = (prev_apex + (t["apex_s_pct"] + 1.0)) / 2.0
            start %= 1.0
        if t["apex_s_pct"] > next_apex:
            end = (t["apex_s_pct"] + (next_apex + 1.0)) / 2.0
            end %= 1.0
        t["start_s_pct"] = start
        t["end_s_pct"] = end
    return turns


def segment_indices(s_pct: List[float], start: float, end: float) -> List[int]:
    seg_len = (end - start) % 1.0
    idxs = []
    for i, sp in enumerate(s_pct):
        d = (sp - start) % 1.0
        if d <= seg_len:
            idxs.append(i)
    # sort by circular delta so we can find "first" in segment
    idxs.sort(key=lambda i: (s_pct[i] - start) % 1.0)
    return idxs


def compute_turn_stats(
    turns: List[Dict[str, float]],
    data: Dict[str, List[float]],
    idxs: List[int],
    min_gear: int,
    min_speed: float,
) -> None:
    s_pct = [data["LapDistPct"][i] for i in idxs]
    speed = [data["Speed"][i] for i in idxs]
    brake = [data["Brake"][i] for i in idxs]
    throttle = [data["Throttle"][i] for i in idxs]
    gear = [data["Gear"][i] for i in idxs]

    for t in turns:
        seg_idx = segment_indices(s_pct, t["start_s_pct"], t["end_s_pct"])
        if not seg_idx:
            continue
        # min speed and apex position
        min_i = min(seg_idx, key=lambda i: speed[i])
        t["min_speed"] = speed[min_i]
        t["min_speed_s_pct"] = s_pct[min_i]
        # filter gears to avoid neutral/low-gear artifacts during shifts
        valid_idx = [i for i in seg_idx if speed[i] >= min_speed and int(gear[i]) >= min_gear]
        if not valid_idx:
            valid_idx = [i for i in seg_idx if int(gear[i]) >= min_gear]
        if not valid_idx:
            valid_idx = seg_idx
        gears = [int(gear[i]) for i in valid_idx]

        t["min_gear"] = int(min(gears))

        # gear at min speed (apex-ish)
        def nearest_valid_gear(index: int) -> int:
            if index in valid_idx:
                return int(gear[index])
            # search outward for nearest valid
            pos = seg_idx.index(index)
            for d in range(1, max(pos + 1, len(seg_idx) - pos)):
                for j in (pos - d, pos + d):
                    if 0 <= j < len(seg_idx):
                        idx = seg_idx[j]
                        if idx in valid_idx:
                            return int(gear[idx])
            return int(gear[index])

        t["gear_at_apex"] = int(nearest_valid_gear(min_i))

        gear_counts: Dict[int, int] = {}
        for g in gears:
            gear_counts[g] = gear_counts.get(g, 0) + 1
        gear_mode = max(gear_counts.items(), key=lambda x: (x[1], -x[0]))[0]

        # brake onset
        brake_start = None
        for i in seg_idx:
            if brake[i] > 0.1:
                brake_start = s_pct[i]
                break
        t["brake_start_s_pct"] = brake_start
        # target gear heuristic:
        # - if braking occurs in the segment and we downshift, use the lowest gear
        # - otherwise use the modal gear (typical straight/flowing section)
        if brake_start is not None and t["min_gear"] < gear_mode:
            t["target_gear"] = t["min_gear"]
        else:
            t["target_gear"] = gear_mode

        # throttle pickup after apex
        apex_delta = circular_delta(t["apex_s_pct"], t["start_s_pct"])
        throttle_on = None
        for i in seg_idx:
            d = circular_delta(s_pct[i], t["start_s_pct"])
            if d < apex_delta:
                continue
            if throttle[i] > 0.8:
                throttle_on = s_pct[i]
                break
        t["throttle_on_s_pct"] = throttle_on


def write_lap_summary(laps: List[LapInfo], out_dir: str) -> None:
    path = os.path.join(out_dir, "lap_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lap", "time_s", "samples"])
        for li in laps:
            w.writerow([li.lap, f"{li.time_s:.3f}", len(li.idxs)])


def write_reference_lap(data: Dict[str, List[float]], idxs: List[int], out_dir: str) -> None:
    path = os.path.join(out_dir, "reference_lap.csv")
    keys = ["SessionTime", "LapDistPct", "LapDist", "Speed", "Brake", "Throttle", "Gear", "Lat", "Lon", "Alt"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index"] + keys)
        for n, i in enumerate(idxs):
            w.writerow([n] + [data[k][i] for k in keys])


def write_turns(turns: List[Dict[str, float]], out_dir: str) -> None:
    path = os.path.join(out_dir, "turns.json")
    with open(path, "w") as f:
        json.dump({"turns": turns}, f, indent=2)


def write_cues(turns: List[Dict[str, float]], out_dir: str) -> None:
    cues = []
    for t in turns:
        cues.append(
            {
                "turn_id": t["id"],
                "trigger_s_pct": t["start_s_pct"],
                "data": {
                    "apex_s_pct": t.get("apex_s_pct"),
                    "brake_start_s_pct": t.get("brake_start_s_pct"),
                    "throttle_on_s_pct": t.get("throttle_on_s_pct"),
                    "min_speed": t.get("min_speed"),
                    "min_gear": t.get("min_gear"),
                    "gear_at_apex": t.get("gear_at_apex"),
                    "target_gear": t.get("target_gear"),
                },
                "text": {},
            }
        )
    path = os.path.join(out_dir, "cues.json")
    with open(path, "w") as f:
        json.dump({"cues": cues}, f, indent=2)


def project_latlon(lat: List[float], lon: List[float]) -> Tuple[List[float], List[float]]:
    if not lat or not lon:
        return [], []
    lat0 = sum(lat) / len(lat)
    lon0 = sum(lon) / len(lon)
    r = 6371000.0
    x = [math.radians(lon[i] - lon0) * math.cos(math.radians(lat0)) * r for i in range(len(lat))]
    y = [math.radians(lat[i] - lat0) * r for i in range(len(lat))]
    return x, y


def plot_track(
    data: Dict[str, List[float]],
    idxs: List[int],
    turns: List[Dict[str, float]],
    out_dir: str,
) -> None:
    lat = [data["Lat"][i] for i in idxs]
    lon = [data["Lon"][i] for i in idxs]
    s_pct = [data["LapDistPct"][i] for i in idxs]
    x, y = project_latlon(lat, lon)
    if not x or not y:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, color="#222222", linewidth=1.5)

    # turn labels at closest s_pct
    for t in turns:
        apex = t["apex_s_pct"]
        # find closest index
        best_i = min(range(len(s_pct)), key=lambda i: abs((s_pct[i] - apex + 0.5) % 1.0 - 0.5))
        ax.scatter([x[best_i]], [y[best_i]], color="#d62728", s=12)
        ax.text(x[best_i], y[best_i], t["label"], fontsize=8, ha="left", va="bottom")
        # gear label offset (perpendicular to local direction)
        gi = t.get("target_gear")
        if gi is not None:
            anchor = t.get("throttle_on_s_pct") or t.get("brake_start_s_pct") or apex
            gear_i = min(range(len(s_pct)), key=lambda i: abs((s_pct[i] - anchor + 0.5) % 1.0 - 0.5))
            i0 = max(0, gear_i - 5)
            i1 = min(len(x) - 1, gear_i + 5)
            dx = x[i1] - x[i0]
            dy = y[i1] - y[i0]
            # perpendicular vector
            nx, ny = -dy, dx
            norm = math.hypot(nx, ny) or 1.0
            nx /= norm
            ny /= norm
            offset = 25.0  # meters-ish in projected space
            ax.text(
                x[gear_i] + nx * offset,
                y[gear_i] + ny * offset,
                f"G{int(gi)}",
                fontsize=8,
                color="#1f77b4",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Track Map (Reference Lap)")

    png_path = os.path.join(out_dir, "track.png")
    svg_path = os.path.join(out_dir, "track.svg")
    html_path = os.path.join(out_dir, "track.html")

    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    fig.savefig(svg_path)
    plt.close(fig)

    # simple HTML wrapper
    with open(html_path, "w") as f:
        f.write(
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Track Map</title></head><body>"
            "<img src='track.svg' alt='Track Map' />"
            "</body></html>"
        )


def compute_optimal_segments(
    laps: List[LapInfo],
    turns: List[Dict[str, float]],
    data: Dict[str, List[float]],
) -> List[Dict[str, float]]:
    results = []
    for t in turns:
        start = t["start_s_pct"]
        end = t["end_s_pct"]
        best = None
        for li in laps:
            idxs = li.idxs
            s_pct = [data["LapDistPct"][i] for i in idxs]
            seg_idx = segment_indices(s_pct, start, end)
            if len(seg_idx) < 2:
                continue
            t0 = data["SessionTime"][idxs[seg_idx[0]]]
            t1 = data["SessionTime"][idxs[seg_idx[-1]]]
            seg_time = t1 - t0
            if best is None or seg_time < best["time_s"]:
                best = {"turn_id": t["id"], "time_s": seg_time, "lap": li.lap}
        if best:
            results.append(best)
    return results


def write_optimal_segments(optimal: List[Dict[str, float]], out_dir: str) -> None:
    path = os.path.join(out_dir, "optimal_segments.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "time_s", "lap"])
        for row in optimal:
            w.writerow([row["turn_id"], f"{row['time_s']:.3f}", row["lap"]])


def main() -> None:
    args = parse_args()
    base_out = args.out
    track_id = args.track.strip()
    if not track_id:
        track_id = slugify(os.path.splitext(os.path.basename(args.ibt))[0])
    out_dir = os.path.join(base_out, track_id)
    os.makedirs(out_dir, exist_ok=True)
    data = load_ibt(args.ibt)
    laps = collect_laps(data)
    if not laps:
        raise SystemExit("No full laps found.")

    write_lap_summary(laps, out_dir)
    ref_lap = choose_lap(laps, args.lap)
    ref_idxs = ref_lap.idxs
    write_reference_lap(data, ref_idxs, out_dir)

    labels = load_turn_labels(args)
    if not labels:
        raise SystemExit("No turns configured. Use --track, --turns-json, or --turns-count.")

    s_pct = [data["LapDistPct"][i] for i in ref_idxs]
    steer = [abs(data["SteeringWheelAngle"][i]) for i in ref_idxs]

    turns = build_turns(labels, s_pct, steer, args.min_sep, args.steer_threshold)
    compute_turn_stats(turns, data, ref_idxs, args.min_gear, args.min_speed)
    write_turns(turns, out_dir)
    write_cues(turns, out_dir)
    plot_track(data, ref_idxs, turns, out_dir)

    if args.optimal:
        optimal = compute_optimal_segments(laps, turns, data)
        write_optimal_segments(optimal, out_dir)

    print(f"Reference lap: {ref_lap.lap} ({ref_lap.time_s:.3f}s)")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
