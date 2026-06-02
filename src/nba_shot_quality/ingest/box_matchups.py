"""Pull per-game player-vs-player matchup tracking via nba_api BoxScoreMatchupsV3.

This is the per-game analogue of `ingest/matchups.py` (the season `LeagueSeasonMatchups` totals).
For each game it returns every (offensive player, defensive player) pair with `partialPossessions`
(how much that defender guarded that shooter *in that game*) and `matchupFieldGoalsAttempted`. The
RAPM matchup weighting then concentrates each shot's defensive credit on whoever actually guarded the
shooter in that specific game — finer-grained than the season share, which averages over every
meeting all year (the season shares proved too coarse to beat uniform; this tests whether per-game
assignments extract cleaner per-defender signal).

Like the PBP rotation pull this is one request per game, so it pulls SEQUENTIALLY with a small sleep
and a resumable per-game shard cache (`box_matchups_{season}_parts/`) — concurrency triggers nba_api
429 throttling. Re-run until failures stop to fill gaps. Tracking matchups exist from 2017-18 onward.

Output schema: `game_id, off_player_id, def_player_id, partial_poss, matchup_fga` in
`data/raw/box_matchups_{season}.parquet`.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import boxscorematchupsv3
from tenacity import retry, stop_after_attempt, wait_exponential

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REQUEST_SLEEP_SEC = 0.6  # sequential pacing; concurrency triggers nba_api 429 throttling (see pbp_rotations)
REQUEST_TIMEOUT_SEC = 30

KEEP_COLS = ["game_id", "off_player_id", "def_player_id", "partial_poss", "matchup_fga"]


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _fetch_box_matchups(game_id: str) -> pd.DataFrame:
    return boxscorematchupsv3.BoxScoreMatchupsV3(game_id=game_id, timeout=REQUEST_TIMEOUT_SEC).get_data_frames()[0]


def _normalize(raw: pd.DataFrame, game_id: str) -> pd.DataFrame:
    """Slim the API response to the (off, def, partial_poss, matchup_fga) columns RAPM needs."""
    df = raw[["personIdOff", "personIdDef", "partialPossessions", "matchupFieldGoalsAttempted"]].rename(
        columns={
            "personIdOff": "off_player_id",
            "personIdDef": "def_player_id",
            "partialPossessions": "partial_poss",
            "matchupFieldGoalsAttempted": "matchup_fga",
        }
    )
    df = df[df["def_player_id"] > 0]  # drop the "matched up against no one" rollup rows
    df["game_id"] = str(game_id).zfill(10)
    df["off_player_id"] = df["off_player_id"].astype("int64")
    df["def_player_id"] = df["def_player_id"].astype("int64")
    df["partial_poss"] = df["partial_poss"].astype("float64")
    df["matchup_fga"] = df["matchup_fga"].astype("float64")
    return df[KEEP_COLS]


def _pull_one(game_id: str, parts_dir: Path) -> str:
    """Fetch + normalize + cache one game's matchup shard; return 'ok' | 'empty' | 'failed'."""
    try:
        raw = _fetch_box_matchups(game_id)
    except Exception:
        return "failed"
    if raw is None or raw.empty:
        return "empty"
    df = _normalize(raw, game_id)
    if df.empty:
        return "empty"
    df.to_parquet(parts_dir / f"{game_id}.parquet", index=False)
    return "ok"


def ingest_box_matchups(season: str, force: bool = False) -> Path:
    """Pull per-game offense-vs-defense matchup totals for every game in the season (sequential, resumable)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"box_matchups_{season}.parquet"
    parts_dir = RAW_DIR / f"box_matchups_{season}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    scored_path = PROCESSED_DIR / f"shots_scored_{season}.parquet"
    if not scored_path.exists():
        raise FileNotFoundError(f"{scored_path} not found — run score --season {season} first")
    game_ids = sorted(pd.read_parquet(scored_path, columns=["game_id"])["game_id"].astype(str).unique())

    todo = [g for g in game_ids if force or not (parts_dir / f"{g}.parquet").exists()]
    print(f"[box_matchups] {len(game_ids):,} games for {season}: {len(todo):,} to pull, {len(game_ids)-len(todo):,} cached")

    counts = {"ok": 0, "empty": 0, "failed": 0}
    for done, g in enumerate(todo, 1):
        counts[_pull_one(g, parts_dir)] += 1
        time.sleep(REQUEST_SLEEP_SEC)
        if done % 100 == 0:
            print(f"  {done}/{len(todo)}  ok={counts['ok']} empty={counts['empty']} failed={counts['failed']}", flush=True)

    shards = sorted(parts_dir.glob("*.parquet"))
    all_mu = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    all_mu.to_parquet(out_path, index=False)
    missing = len(game_ids) - len(shards)
    print(f"[box_matchups] this run: ok={counts['ok']} empty={counts['empty']} failed={counts['failed']}")
    if missing:
        print(f"[box_matchups] {missing:,} games still missing — re-run to retry (resumes from shard cache)")
    print(f"[box_matchups] {len(all_mu):,} (off,def) rows across {len(shards):,}/{len(game_ids):,} games -> {out_path}")
    return out_path
