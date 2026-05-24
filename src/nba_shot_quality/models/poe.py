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

    df = df.sort_values("game_date")
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

    before = len(agg)
    agg = agg[agg["attempts"] >= min_attempts].copy()
    print(f"[poe] {len(agg):,}/{before:,} players meet min_attempts={min_attempts}")

    agg["season"] = season
    agg["poe"] = agg["points"] - agg["xpoints"]
    agg["poe_per_100"] = agg["poe"] / agg["attempts"] * 100.0
    agg["pps"] = agg["points"] / agg["attempts"]
    agg["xpps"] = agg["xpoints"] / agg["attempts"]
    agg["efg_pct"] = (agg["made"] + 0.5 * agg["made_three"]) / agg["attempts"]

    # Bootstrap CIs on the qualified players only (deterministic via a single shared rng).
    rng = np.random.default_rng(0)
    poe_by_player = {pid: g["poe"].to_numpy() for pid, g in df[df["player_id"].isin(agg["player_id"])].groupby("player_id")}
    ci = {pid: bootstrap_poe_ci(poe_by_player[pid], rng) for pid in agg["player_id"]}
    agg["poe_ci_low"] = agg["player_id"].map(lambda p: ci[p][0])
    agg["poe_ci_high"] = agg["player_id"].map(lambda p: ci[p][1])

    # Merge external TS% / rTS%.
    stats_path = RAW_DIR / f"player_stats_{season}.parquet"
    if stats_path.exists():
        stats = pd.read_parquet(stats_path)[["player_id", "ts_pct", "league_avg_ts_pct"]]
        agg = agg.merge(stats, on="player_id", how="left")
        agg["rel_ts_pct"] = agg["ts_pct"] - agg["league_avg_ts_pct"]
        n_matched = agg["ts_pct"].notna().sum()
        print(f"[poe] TS% merged for {n_matched:,}/{len(agg):,} players")
    else:
        print(f"[poe] WARNING: {stats_path.name} missing — ts_pct/rel_ts_pct will be NaN")
        agg["ts_pct"] = np.nan
        agg["league_avg_ts_pct"] = np.nan
        agg["rel_ts_pct"] = np.nan

    # Sanity assertions.
    bad_ci = ~((agg["poe_ci_low"] <= agg["poe_per_100"]) & (agg["poe_per_100"] <= agg["poe_ci_high"]))
    if bad_ci.any():
        raise AssertionError(f"[poe] {int(bad_ci.sum())} rows have poe_per_100 outside their CI")

    out = agg[OUTPUT_COLS].sort_values("poe_per_100", ascending=False).reset_index(drop=True)
    out_path = PROCESSED_DIR / f"poe_player_season_{season}.parquet"
    out.to_parquet(out_path, index=False)

    top = out.head(5)
    print("[poe] top 5 by POE/100:")
    for _, r in top.iterrows():
        print(f"  {r['player_name']:<24} {r['poe_per_100']:+6.2f} [{r['poe_ci_low']:+.2f}, {r['poe_ci_high']:+.2f}]  ({int(r['attempts'])} FGA)")
    print(f"[poe] -> {out_path}")
    return out_path
