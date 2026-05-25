#!/usr/bin/env bash
set -euo pipefail
PY="${PY:-.venv/bin/python}"
# All positional args are seasons in chronological order; default to the two most recent.
if [ "$#" -ge 1 ]; then SEASONS=("$@"); else SEASONS=("2023-24" "2024-25"); fi

# Per-game rotations via fast PlayByPlayV3 reconstruction (~1s/call sequential, resumable shard cache).
# Use `--source gamerotation` for the slow exact-times endpoint instead.
for S in "${SEASONS[@]}"; do "$PY" -m nba_shot_quality.cli ingest-rotations --season "$S"; done

# Reconstruct on-floor 5v5 per shot, + the tracking defended-FG metric for face validity.
for S in "${SEASONS[@]}"; do "$PY" -m nba_shot_quality.cli lineups   --season "$S"; done
for S in "${SEASONS[@]}"; do "$PY" -m nba_shot_quality.cli ingest-def --season "$S"; done

# Pooled fit (primary ratings) + per-season fits (for stability/reliability).
"$PY" -m nba_shot_quality.cli rapm --seasons "${SEASONS[@]}"
for S in "${SEASONS[@]}"; do "$PY" -m nba_shot_quality.cli rapm --seasons "$S"; done

# Diagnostics on each adjacent season pair (yoy + face validity + cross-pipeline + reliability).
for ((i = 0; i < ${#SEASONS[@]} - 1; i++)); do
    "$PY" -m nba_shot_quality.cli rapm-eval \
        --season-a "${SEASONS[$i]}" --season-b "${SEASONS[$((i + 1))]}" --season "${SEASONS[$((i + 1))]}"
done
