"""Shooter skill: aggregate out-of-fold POE to player-season with bootstrap CIs.

POE = Σ(actual_points − xPoints) over a player's shots; POE_per_100 = 100 * mean(per-shot poe).
Confidence intervals come from a shot-level bootstrap. Players are aggregated at the
player-season level (team is ignored, so a traded player is a single summed row). True TS%/rTS%
from ingest.player_stats is merged as an external reference — it uses a different (FT-inclusive,
full-season) shot universe than our shot chart, so it correlates with but does not equal POE.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_DIR = REPO_ROOT / "data" / "raw"
N_BOOT = 1000
DEFAULT_MIN_ATTEMPTS = 200

OUTPUT_COLS = [
    "player_id", "player_name", "team_id", "season",
    "attempts", "made", "points", "xpoints",
    "poe", "poe_per_100", "poe_ci_low", "poe_ci_high",
    "pps", "xpps", "efg_pct",
    "ts_pct", "league_avg_ts_pct", "rel_ts_pct",
]


def summarize_player_poe(
    scored_shots: pd.DataFrame,
    season: str,
    min_attempts: int = DEFAULT_MIN_ATTEMPTS,
    stats: pd.DataFrame | None = None,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> pd.DataFrame:
    """Aggregate scored shots to player-season POE metrics.

    This pure helper contains the arithmetic used by the file-based pipeline, which makes the metric
    behavior easy to unit test without reading or writing parquet files.
    """
    required = {"player_id", "player_name", "team_id", "shot_made", "points", "xpoints"}
    missing = sorted(required - set(scored_shots.columns))
    if missing:
        raise ValueError(f"scored shots missing required columns: {missing}")

    df = scored_shots.copy()
    if "game_date" in df.columns:
        df = df.sort_values("game_date")
    if "poe" not in df.columns:
        df["poe"] = df["points"] - df["xpoints"]
    if "is_three" not in df.columns:
        if "shot_value" not in df.columns:
            raise ValueError("scored shots must include is_three or shot_value")
        df["is_three"] = (df["shot_value"] == 3).astype("int8")
    df["made_three"] = df["shot_made"] * df["is_three"]

    agg = (
        df.groupby("player_id")
        .agg(
            player_name=("player_name", "last"),
            team_id=("team_id", "last"),
            attempts=("points", "size"),
            made=("shot_made", "sum"),
            points=("points", "sum"),
            xpoints=("xpoints", "sum"),
            made_three=("made_three", "sum"),
        )
        .reset_index()
    )
    agg = agg[agg["attempts"] >= min_attempts].copy()
    agg["season"] = season
    agg["poe"] = agg["points"] - agg["xpoints"]
    agg["poe_per_100"] = agg["poe"] / agg["attempts"] * 100.0
    agg["pps"] = agg["points"] / agg["attempts"]
    agg["xpps"] = agg["xpoints"] / agg["attempts"]
    agg["efg_pct"] = (agg["made"] + 0.5 * agg["made_three"]) / agg["attempts"]

    if len(agg) and n_boot > 0:
        rng = np.random.default_rng(seed)
        qualified = df[df["player_id"].isin(agg["player_id"])]
        poe_by_player = {pid: g["poe"].to_numpy() for pid, g in qualified.groupby("player_id")}
        ci = {pid: bootstrap_poe_ci(poe_by_player[pid], rng, n_boot=n_boot) for pid in agg["player_id"]}
        agg["poe_ci_low"] = agg["player_id"].map(lambda p: ci[p][0])
        agg["poe_ci_high"] = agg["player_id"].map(lambda p: ci[p][1])
    else:
        agg["poe_ci_low"] = np.nan
        agg["poe_ci_high"] = np.nan

    if stats is not None:
        stats_cols = [c for c in ("player_id", "ts_pct", "league_avg_ts_pct") if c in stats.columns]
        stats = stats[stats_cols].copy()
        for col in ("ts_pct", "league_avg_ts_pct"):
            if col not in stats.columns:
                stats[col] = np.nan
        agg = agg.merge(stats[["player_id", "ts_pct", "league_avg_ts_pct"]], on="player_id", how="left")
        agg["rel_ts_pct"] = agg["ts_pct"] - agg["league_avg_ts_pct"]
    else:
        agg["ts_pct"] = np.nan
        agg["league_avg_ts_pct"] = np.nan
        agg["rel_ts_pct"] = np.nan

    return agg[OUTPUT_COLS].sort_values("poe_per_100", ascending=False).reset_index(drop=True)


def bootstrap_poe_ci(per_shot_poe: np.ndarray, rng: np.random.Generator,
                     n_boot: int = N_BOOT, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    """Percentile CI on POE_per_100 via shot-level resampling with replacement."""
    n = len(per_shot_poe)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_per_100 = per_shot_poe[idx].mean(axis=1) * 100.0
    ci_low, ci_high = np.percentile(boot_per_100, [lo, hi])
    return float(ci_low), float(ci_high)


def aggregate_player_season(season: str, min_attempts: int = DEFAULT_MIN_ATTEMPTS) -> Path:
    """Aggregate OOF-scored shots to a per-player-season POE table with bootstrap CIs."""
    scored_path = PROCESSED_DIR / f"shots_scored_{season}.parquet"
    if not scored_path.exists():
        raise FileNotFoundError(f"scored shots not found at {scored_path} — run score first")
    df = pd.read_parquet(scored_path)
    print(f"[poe] loaded {len(df):,} scored shots from {scored_path.name}")

    before = df["player_id"].nunique()

    # Merge external TS% / rTS%.
    stats_path = RAW_DIR / f"player_stats_{season}.parquet"
    if stats_path.exists():
        stats = pd.read_parquet(stats_path)[["player_id", "ts_pct", "league_avg_ts_pct"]]
    else:
        print(f"[poe] WARNING: {stats_path.name} missing — ts_pct/rel_ts_pct will be NaN")
        stats = None

    out = summarize_player_poe(df, season, min_attempts=min_attempts, stats=stats)
    print(f"[poe] {len(out):,}/{before:,} players meet min_attempts={min_attempts}")
    if stats is not None:
        n_matched = out["ts_pct"].notna().sum()
        print(f"[poe] TS% merged for {n_matched:,}/{len(out):,} players")

    # Sanity assertions.
    bad_ci = ~((out["poe_ci_low"] <= out["poe_per_100"]) & (out["poe_per_100"] <= out["poe_ci_high"]))
    if bad_ci.any():
        raise AssertionError(f"[poe] {int(bad_ci.sum())} rows have poe_per_100 outside their CI")

    out_path = PROCESSED_DIR / f"poe_player_season_{season}.parquet"
    out.to_parquet(out_path, index=False)

    top = out.head(5)
    print("[poe] top 5 by POE/100:")
    for _, r in top.iterrows():
        print(
            f"  {r['player_name']:<24} {r['poe_per_100']:+6.2f} "
            f"[{r['poe_ci_low']:+.2f}, {r['poe_ci_high']:+.2f}]  ({int(r['attempts'])} FGA)"
        )
    print(f"[poe] -> {out_path}")
    return out_path
