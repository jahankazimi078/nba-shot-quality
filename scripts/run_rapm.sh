#!/usr/bin/env bash
set -euo pipefail
SEASON_NEW="${1:-2024-25}"
SEASON_PREV="${2:-2023-24}"
PY="${PY:-.venv/bin/python}"

# Per-game rotations (slow, ~1,230 calls each; resumable via per-game shard cache).
"$PY" -m nba_shot_quality.cli ingest-rotations --season "$SEASON_PREV"
"$PY" -m nba_shot_quality.cli ingest-rotations --season "$SEASON_NEW"

# Reconstruct on-floor 5v5 per shot.
"$PY" -m nba_shot_quality.cli lineups          --season "$SEASON_PREV"
"$PY" -m nba_shot_quality.cli lineups          --season "$SEASON_NEW"

# Tracking defended-FG metric for face validity (single cheap request each).
"$PY" -m nba_shot_quality.cli ingest-def       --season "$SEASON_PREV"
"$PY" -m nba_shot_quality.cli ingest-def       --season "$SEASON_NEW"

# Pooled fit (primary ratings) + per-season fits (for the stability check).
"$PY" -m nba_shot_quality.cli rapm             --seasons "$SEASON_PREV" "$SEASON_NEW"
"$PY" -m nba_shot_quality.cli rapm             --seasons "$SEASON_PREV"
"$PY" -m nba_shot_quality.cli rapm             --seasons "$SEASON_NEW"

# Stability + face-validity diagnostics.
"$PY" -m nba_shot_quality.cli rapm-eval        --season-a "$SEASON_PREV" --season-b "$SEASON_NEW" --season "$SEASON_NEW"
