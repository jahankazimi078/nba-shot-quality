# NBA Shot Quality

LightGBM-based xPoints model for NBA shots. **Layer 1** is implemented: per-shot expected points + a Streamlit dashboard with shot maps and POE (points over expected) leaderboards. Layers 2-4 (POE aggregation polish, ridge RAPM defender impact, coaching-change event study) are planned per `nba_shot_quality_spec.md`.

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

Or all at once: `bash scripts/run_layer1.sh 2024-25`.

Outputs land in:
- `data/raw/shots_{season}.parquet` — cached nba_api shot detail (skip-on-hit; pass `--force` to re-pull).
- `data/processed/shots_features_{season}.parquet` — engineered features.
- `data/processed/xpoints_model.joblib` — trained LightGBM classifier.
- `data/processed/holdout_predictions_{season}.parquet` — per-shot holdout predictions for eval and the app.
- `reports/calibration_{season}.png` — reliability diagram + per-zone calibration bar chart.

## Dashboard

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

Sidebar: season + minimum-attempts filter. Tabs: player shot map (colored by per-shot POE), top-20 POE leaderboard, bottom-20 POE leaderboard.

## Current model — 2024-25 (1 season, regular season only)

- Train / holdout split: chronological, last 15% of games held out.
- CV: 5-fold `GroupKFold` on `game_id`.
- Holdout log-loss **0.6394**, Brier **0.2251** (constant-predictor baseline ≈ 0.248).
- Per-zone calibration within ~1.5pp on every zone.
- Player-season PPS MAE on the top-100 attempt players: **0.111**.

Features: shot distance, angle, zone (5 categories), action type, period, seconds remaining in period, three-point flag. No tracking data (defender distance, catch-and-shoot) yet.
