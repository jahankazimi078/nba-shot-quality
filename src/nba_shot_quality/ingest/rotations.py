"""Pull per-game player rotations (check-in/check-out times) via nba_api GameRotation.

These stints let us reconstruct which 5 players per team were on the floor at the moment of
each shot — the input to the defender-impact RAPM model.

GameRotation is slow and aggressively rate-limited: individual calls take ~13-30s server-side
(vs ~0.4s for shotchartdetail) and the server returns empty bodies under load. Measured throughput:
~6 concurrent workers is the sweet spot (~6.7s/game wall, ~58% per-pass success); 10 workers is
*worse* (the server throttles harder). So this uses a small thread pool, generous retries, and a
resumable per-game shard cache — re-run the command until `failed=0` to fill throttled gaps.
A full season is still ~hours; size the pull accordingly.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import gamerotation
from tenacity import retry, stop_after_attempt, wait_exponential

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REQUEST_TIMEOUT_SEC = 40
MAX_WORKERS = 6  # measured sweet spot; more workers trigger heavier server-side throttling

KEEP_COLS = ["game_id", "team_id", "person_id", "player_name", "in_time", "out_time"]


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _fetch_game_rotation(game_id: str) -> pd.DataFrame:
    """Return both teams' rotation stints for one game as a single tidy frame (retried on empty body)."""
    resp = gamerotation.GameRotation(game_id=game_id, league_id="00", timeout=REQUEST_TIMEOUT_SEC)
    raw = pd.concat(resp.get_data_frames(), ignore_index=True)
    return pd.DataFrame(
        {
            "game_id": str(game_id).zfill(10),
            "team_id": raw["TEAM_ID"].astype("int64"),
            "person_id": raw["PERSON_ID"].astype("int64"),
            "player_name": (raw["PLAYER_FIRST"].fillna("") + " " + raw["PLAYER_LAST"].fillna("")).str.strip(),
            "in_time": raw["IN_TIME_REAL"].astype("int64"),
            "out_time": raw["OUT_TIME_REAL"].astype("int64"),
        }
    )[KEEP_COLS]


def _pull_one(game_id: str, parts_dir: Path) -> str:
    """Fetch + cache one game's shard; return a status: 'ok', 'empty', or 'failed'."""
    try:
        df = _fetch_game_rotation(game_id)
    except Exception:
        return "failed"
    if df.empty:
        return "empty"
    df.to_parquet(parts_dir / f"{game_id}.parquet", index=False)
    return "ok"


def ingest_rotations(season: str, force: bool = False) -> Path:
    """Pull GameRotation for every game in the season (concurrent, resumable), then concat."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"rotations_{season}.parquet"
    parts_dir = RAW_DIR / f"rotations_{season}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    scored_path = PROCESSED_DIR / f"shots_scored_{season}.parquet"
    if not scored_path.exists():
        raise FileNotFoundError(f"{scored_path} not found — run score --season {season} first")
    game_ids = sorted(pd.read_parquet(scored_path, columns=["game_id"])["game_id"].astype(str).unique())

    todo = [g for g in game_ids if force or not (parts_dir / f"{g}.parquet").exists()]
    skipped = len(game_ids) - len(todo)
    print(f"[rotations] {len(game_ids):,} games for {season}: {len(todo):,} to pull, {skipped:,} cached")

    counts = {"ok": 0, "empty": 0, "failed": 0}
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_pull_one, g, parts_dir): g for g in todo}
        for fut in as_completed(futures):
            counts[fut.result()] += 1
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(todo)}  ok={counts['ok']} empty={counts['empty']} failed={counts['failed']}", flush=True)

    shards = sorted(parts_dir.glob("*.parquet"))
    all_rot = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    all_rot.to_parquet(out_path, index=False)
    missing = len(game_ids) - len(shards)
    print(f"[rotations] this run: ok={counts['ok']} empty={counts['empty']} failed={counts['failed']}")
    if missing:
        print(f"[rotations] {missing:,} games still missing — re-run to retry (resumes from shard cache)")
    print(f"[rotations] {len(all_rot):,} stint rows across {len(shards):,}/{len(game_ids):,} games -> {out_path}")
    return out_path
