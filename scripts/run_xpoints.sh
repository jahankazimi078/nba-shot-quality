#!/usr/bin/env bash
set -euo pipefail
SEASON="${1:-2024-25}"
PY="${PY:-.venv/bin/python}"
"$PY" -m nba_shot_quality.cli ingest   --season "$SEASON"
"$PY" -m nba_shot_quality.cli features --season "$SEASON"
"$PY" -m nba_shot_quality.cli train    --season "$SEASON"
"$PY" -m nba_shot_quality.cli eval     --season "$SEASON"
