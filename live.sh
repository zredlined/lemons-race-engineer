#!/usr/bin/env bash
set -euo pipefail

TRACK_ID="${1:-sonoma_lemons}"
PORT="${PORT:-5000}"

CUES="outputs/${TRACK_ID}/cues.json"
TURNS="outputs/${TRACK_ID}/turns.json"
NOTES="notes/${TRACK_ID}_notes.json"

uv run --with pyirsdk python3 tools/race_engineer.py \
  --cues "${CUES}" \
  --turns "${TURNS}" \
  --notes "${NOTES}" \
  --live \
  --port "${PORT}"
