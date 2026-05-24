#!/usr/bin/env bash
set -euo pipefail
SEASON_NEW="${1:-2024-25}"
SEASON_PREV="${2:-2023-24}"
PY="${PY:-.venv/bin/python}"

# xPoints artifacts for SEASON_NEW are assumed to exist; the previous season needs ingest+features.
"$PY" -m nba_shot_quality.cli ingest       --season "$SEASON_PREV"
"$PY" -m nba_shot_quality.cli features      --season "$SEASON_PREV"

# True TS% baseline for both seasons (single cheap request each).
"$PY" -m nba_shot_quality.cli ingest-stats  --season "$SEASON_NEW"
"$PY" -m nba_shot_quality.cli ingest-stats  --season "$SEASON_PREV"

# Out-of-fold xPoints over the full season (each season self-scored).
"$PY" -m nba_shot_quality.cli score         --season "$SEASON_NEW"
"$PY" -m nba_shot_quality.cli score         --season "$SEASON_PREV"

# Per-player-season POE with bootstrap CIs.
"$PY" -m nba_shot_quality.cli poe           --season "$SEASON_NEW"
"$PY" -m nba_shot_quality.cli poe           --season "$SEASON_PREV"

# Stability + raw-efficiency comparison.
"$PY" -m nba_shot_quality.cli stability     --season-a "$SEASON_PREV" --season-b "$SEASON_NEW"
"$PY" -m nba_shot_quality.cli poe-vs-rts    --season "$SEASON_NEW"
