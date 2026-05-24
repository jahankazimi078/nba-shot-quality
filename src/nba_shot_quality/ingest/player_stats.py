"""Pull league player totals via nba_api; compute true TS% and an attempt-weighted league baseline.

TS% here is free-throw-inclusive (the external reference metric), computed from Base totals
(PTS, FGA, FTA) so we never depend on the Advanced endpoint's server-defined column names.
This is a different FGA universe than the shot-chart, so it correlates with but does not equal
our shot-only POE — that's the intended comparison in the POE aggregation step.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats
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
def _fetch_player_totals(season: str) -> pd.DataFrame:
    resp = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Base",
        per_mode_detailed="Totals",
        timeout=REQUEST_TIMEOUT_SEC,
    )
    return resp.get_data_frames()[0]


def ingest_player_stats(season: str, force: bool = False) -> Path:
    """Pull season player totals and cache per-player TS% + a league-average baseline."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"player_stats_{season}.parquet"

    if out_path.exists() and not force:
        print(f"[player_stats] cache hit: {out_path}")
        return out_path

    print(f"[player_stats] pulling league player totals for season {season}")
    raw = _fetch_player_totals(season)
    time.sleep(REQUEST_SLEEP_SEC)
    print(f"[player_stats] received {len(raw):,} rows, columns: {list(raw.columns)}")

    required = {"PLAYER_ID", "PLAYER_NAME", "PTS", "FGA", "FTA"}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"[player_stats] expected columns missing from API response: {sorted(missing)}")

    df = raw[["PLAYER_ID", "PLAYER_NAME", "PTS", "FGA", "FTA"]].copy()
    df = df.rename(columns={"PLAYER_ID": "player_id", "PLAYER_NAME": "player_name"})

    # True Shooting %: PTS / (2 * (FGA + 0.44 * FTA)). Guard against zero-shot players.
    tsa = 2.0 * (df["FGA"] + 0.44 * df["FTA"])
    df["ts_pct"] = (df["PTS"] / tsa).where(tsa > 0)

    # Attempt-weighted league TS% (the true league baseline, not a mean of per-player rates).
    total_pts = df["PTS"].sum()
    total_tsa = 2.0 * (df["FGA"].sum() + 0.44 * df["FTA"].sum())
    league_ts = float(total_pts / total_tsa) if total_tsa > 0 else float("nan")
    df["league_avg_ts_pct"] = league_ts

    df = df[["player_id", "player_name", "ts_pct", "league_avg_ts_pct"]]
    df.to_parquet(out_path, index=False)
    print(f"[player_stats] league-average TS%: {league_ts:.4f}")
    print(f"[player_stats] -> {out_path}")
    return out_path
