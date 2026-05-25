"""Pull season player-vs-player matchup tracking via nba_api LeagueSeasonMatchups.

For every (offensive player, defensive player) pair this gives how much the defender guarded the
shooter over the season — `PARTIAL_POSS` (partial possessions matched up) and `MATCHUP_FGA` (FGAs
faced). The RAPM weighting step uses these to concentrate each shot's defensive credit on the player
who actually guarded the shooter (restricted to the 5 on the floor), instead of splitting it equally.

One cheap request per season (~1s, ~140k rows). Tracking matchups exist from 2017-18 onward.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leagueseasonmatchups
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
def _fetch_matchups(season: str) -> pd.DataFrame:
    resp = leagueseasonmatchups.LeagueSeasonMatchups(
        season=season,
        season_type_playoffs="Regular Season",
        per_mode_simple="Totals",
        timeout=REQUEST_TIMEOUT_SEC,
    )
    return resp.get_data_frames()[0]


def ingest_matchups(season: str, force: bool = False) -> Path:
    """Pull season offense-vs-defense matchup totals (who guarded whom, how much)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"matchups_{season}.parquet"
    if out_path.exists() and not force:
        print(f"[matchups] cache hit: {out_path}")
        return out_path

    print(f"[matchups] pulling season matchups for {season}")
    raw = _fetch_matchups(season)
    time.sleep(REQUEST_SLEEP_SEC)
    print(f"[matchups] received {len(raw):,} (off,def) pairs")

    required = {"OFF_PLAYER_ID", "DEF_PLAYER_ID", "PARTIAL_POSS", "MATCHUP_FGA"}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"[matchups] expected columns missing from API response: {sorted(missing)}")

    df = raw[["OFF_PLAYER_ID", "DEF_PLAYER_ID", "PARTIAL_POSS", "MATCHUP_FGA"]].rename(
        columns={
            "OFF_PLAYER_ID": "off_player_id",
            "DEF_PLAYER_ID": "def_player_id",
            "PARTIAL_POSS": "partial_poss",
            "MATCHUP_FGA": "matchup_fga",
        }
    )
    df["off_player_id"] = df["off_player_id"].astype("int64")
    df["def_player_id"] = df["def_player_id"].astype("int64")
    df["partial_poss"] = df["partial_poss"].astype("float64")
    df["matchup_fga"] = df["matchup_fga"].astype("float64")
    df.to_parquet(out_path, index=False)
    print(f"[matchups] -> {out_path}")
    return out_path
