# NBA Shot Quality

LightGBM-based **xPoints** model for NBA shots: estimate the expected points of any shot from its context, then measure who beats expectation. Two pipelines are implemented:

- **xPoints** — a per-shot expected-points model with a Streamlit shot-map dashboard.
- **Shooter skill (POE)** — aggregates out-of-fold **POE** (points over expected) to player-season rankings with bootstrap confidence intervals, year-over-year stability, and a true-TS%/rTS% comparison.

Defender impact and a coaching-change event study are possible future additions.

## Install

```bash
python3 -m venv .venv          # or use the existing .venv
.venv/bin/pip install -e .
.venv/bin/pip install -r requirements.txt
```

Tested on Python 3.14 with the pinned versions in `requirements.txt`. If a wheel is missing on your Python version, recreate the venv with Python 3.12.

## Run the pipeline

```bash
.venv/bin/python -m nba_shot_quality.cli ingest   --season 2024-25
.venv/bin/python -m nba_shot_quality.cli features --season 2024-25
.venv/bin/python -m nba_shot_quality.cli train    --season 2024-25
.venv/bin/python -m nba_shot_quality.cli eval     --season 2024-25
```

Or all at once: `bash scripts/run_xpoints.sh 2024-25`.

### Shooter-skill (POE) pipeline

```bash
.venv/bin/python -m nba_shot_quality.cli ingest-stats --season 2024-25
.venv/bin/python -m nba_shot_quality.cli score        --season 2024-25
.venv/bin/python -m nba_shot_quality.cli poe          --season 2024-25
.venv/bin/python -m nba_shot_quality.cli stability    --season-a 2023-24 --season-b 2024-25
.venv/bin/python -m nba_shot_quality.cli poe-vs-rts   --season 2024-25
```

Or all at once (also ingests + builds features for the prior season): `bash scripts/run_poe.sh 2024-25 2023-24`.

Outputs land in:
- `data/raw/shots_{season}.parquet` — cached nba_api shot detail (skip-on-hit; pass `--force` to re-pull).
- `data/processed/shots_features_{season}.parquet` — engineered features.
- `data/processed/xpoints_model.joblib` — trained LightGBM classifier.
- `data/processed/holdout_predictions_{season}.parquet` — per-shot holdout predictions for eval and the app.
- `reports/calibration_{season}.png` — reliability diagram + per-zone calibration bar chart.
- `data/processed/shots_scored_{season}.parquet` — out-of-fold per-shot xPoints/POE over the full season.
- `data/processed/poe_player_season_{season}.parquet` — player-season POE per 100 with bootstrap CIs + TS%/rTS%.
- `data/raw/player_stats_{season}.parquet` — league player true-shooting totals.
- `reports/poe_stability_*.png`, `reports/poe_vs_rts_{season}.png` — year-over-year stability + efficiency-comparison plots.

## Dashboard

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

Sidebar: season (2024-25 / 2023-24) + minimum-attempts filter. Tabs: player shot map (colored by per-shot POE), top-20 and bottom-20 POE leaderboards (with confidence intervals and TS%/rTS%), and a stability tab showing the year-over-year and POE-vs-rTS% plots.

## Current model — 2024-25 (1 season, regular season only)

- Train / holdout split: chronological, last 15% of games held out.
- CV: 5-fold `GroupKFold` on `game_id`.
- Holdout log-loss **0.6394**, Brier **0.2251** (constant-predictor baseline ≈ 0.248).
- Per-zone calibration within ~1.5pp on every zone.
- Player-season PPS MAE on the top-100 attempt players: **0.111**.

Features: shot distance, angle, zone (5 categories), action type, period, seconds remaining in period, three-point flag. No tracking data (defender distance, catch-and-shoot) yet.

## Shooter skill (POE) — 2024-25 + 2023-24

- Out-of-fold scoring (GroupKFold on `game_id`): Brier **0.226** (matches the holdout), mean per-shot POE ≈ 0.
- Year-over-year POE-per-100 correlation **r = 0.58** across 256 players qualified in both seasons — the metric is a persistent skill, not noise.
- POE-per-100 vs relative TS% **r = 0.66** — correlated with raw efficiency but diverging where shot difficulty matters most.
