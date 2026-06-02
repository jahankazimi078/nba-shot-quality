"""Pull per-team-per-game box-score totals via nba_api LeagueGameLog (one call per season).

The coaching-change DiD event study (analysis/coaching_event_study.py) needs two things the
shot-chart data lacks: **possessions** (to express defense per-100-possessions, not per-shot) and
**actual points allowed** (the ground-truth defensive-rating robustness check). LeagueGameLog
returns one row per team per game with the box-score components, from which possessions are
estimated by the standard formula `FGA + 0.44*FTA - OREB + TOV`.

One cheap request per season (~2,460 rows = 1,230 games × 2 teams). `GAME_ID` is already the same
zero-padded 10-char string used in `shots_scored_{season}.parquet`, so the panel builder merges on
`(game_id, team_id)` directly.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from tenacity import retry, stop_after_attempt, wait_exponential

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"
REQUEST_SLEEP_SEC = 0.6
REQUEST_TIMEOUT_SEC = 60


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    reraise=True,
)
def _fetch_team_game_logs(season: str) -> pd.DataFrame:
    resp = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="T",
        timeout=REQUEST_TIMEOUT_SEC,
    )
    return resp.get_data_frames()[0]


def ingest_team_game_logs(season: str, force: bool = False) -> Path:
    """Pull per-team-game box-score totals; cache the columns the DiD panel needs."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"team_game_logs_{season}.parquet"

    if out_path.exists() and not force:
        print(f"[team_logs] cache hit: {out_path}")
        return out_path

    print(f"[team_logs] pulling team game logs for season {season}")
    raw = _fetch_team_game_logs(season)
    time.sleep(REQUEST_SLEEP_SEC)
    print(f"[team_logs] received {len(raw):,} rows, columns: {list(raw.columns)}")

    required = {"TEAM_ID", "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "PTS", "FGA", "FTA", "OREB", "TOV"}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"[team_logs] expected columns missing from API response: {sorted(missing)}")

    df = raw[sorted(required)].rename(columns=str.lower)
    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    df["team_id"] = df["team_id"].astype("int64")
    df["game_date"] = pd.to_datetime(df["game_date"])
    for c in ("pts", "fga", "fta", "oreb", "tov"):
        df[c] = df[c].astype("float64")

    # Standard offensive-possessions estimate (a team's own possessions in that game).
    df["possessions"] = df["fga"] + 0.44 * df["fta"] - df["oreb"] + df["tov"]

    per_game = df.groupby("game_id").size()
    if not (per_game == 2).all():
        bad = per_game[per_game != 2]
        raise ValueError(f"[team_logs] expected exactly 2 teams per game; offenders: {bad.to_dict()}")

    df.to_parquet(out_path, index=False)
    print(f"[team_logs] league-avg possessions/team-game: {df['possessions'].mean():.1f} (expect ~99-101)")
    print(f"[team_logs] {df['game_id'].nunique():,} games -> {out_path}")
    return out_path
