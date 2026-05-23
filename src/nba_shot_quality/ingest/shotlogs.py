"""Pull NBA shot-detail data via nba_api, cache to Parquet."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import teams
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
def _fetch_team_shots(team_id: int, season: str) -> pd.DataFrame:
    resp = shotchartdetail.ShotChartDetail(
        team_id=team_id,
        player_id=0,
        season_nullable=season,
        season_type_all_star="Regular Season",
        context_measure_simple="FGA",
        timeout=REQUEST_TIMEOUT_SEC,
    )
    return resp.get_data_frames()[0]


def ingest_season(season: str, force: bool = False) -> Path:
    """Pull all shots for a season, one team at a time, and cache to Parquet."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"shots_{season}.parquet"

    if out_path.exists() and not force:
        print(f"[ingest] cache hit: {out_path}")
        df = pd.read_parquet(out_path)
        _print_summary(df, season, out_path)
        return out_path

    all_teams = teams.get_teams()
    frames: list[pd.DataFrame] = []
    print(f"[ingest] pulling {len(all_teams)} teams for season {season}")
    for i, team in enumerate(all_teams, start=1):
        print(f"  ({i:2d}/{len(all_teams)}) {team['full_name']}", flush=True)
        frames.append(_fetch_team_shots(team["id"], season))
        time.sleep(REQUEST_SLEEP_SEC)

    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out_path, index=False)
    _print_summary(df, season, out_path)
    return out_path


def _print_summary(df: pd.DataFrame, season: str, out_path: Path) -> None:
    print(f"[ingest] season={season} rows={len(df):,} cols={df.shape[1]}")
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        print("[ingest] null counts:")
        for col, n in nulls.items():
            print(f"  {col}: {n:,}")
    else:
        print("[ingest] no nulls")

    if "SHOT_DISTANCE" in df.columns:
        hist_path = out_path.with_name(out_path.stem + "_dist.png")
        _save_distance_histogram(df["SHOT_DISTANCE"], hist_path, season)
        print(f"[ingest] distance histogram saved to {hist_path}")


def _save_distance_histogram(distances: pd.Series, path: Path, season: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(distances.clip(upper=40), bins=40, edgecolor="black")
    ax.set_xlabel("Shot distance (ft)")
    ax.set_ylabel("Attempts")
    ax.set_title(f"Shot distance distribution — {season}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
