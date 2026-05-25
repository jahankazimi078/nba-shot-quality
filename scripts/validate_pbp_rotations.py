"""Validate PBP-V3 rotation reconstruction against cached GameRotation ground truth.

For every game that has BOTH a cached GameRotation shard (`data/raw/rotations_{season}_parts/`,
produced by `ingest-rotations --source gamerotation`) and a fresh PlayByPlayV3 reconstruction,
compare the on-floor 5-per-team at each shot's elapsed time. Reports the 5v5 reconstruction rate and
the exact 10-player match rate, and exits non-zero if the match rate falls below MIN_MATCH — a guard
against silent regressions in `pbp_rotations.reconstruct_stints`.

No-op (exit 0) when no GameRotation ground-truth shards are present, since they are an optional,
gitignored cache. Usage: `python scripts/validate_pbp_rotations.py --season 2023-24`.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from nba_shot_quality.features.shot_lineups import _elapsed_tenths
from nba_shot_quality.ingest import pbp_rotations as P

MIN_MATCH = 0.97   # fail below this exact-10-player agreement among shots both sources call 5v5
MIN_5V5 = 0.90     # warn below this PBP 5v5 reconstruction rate


def _oncourt(stints: pd.DataFrame, t: int, team: int) -> frozenset:
    s = stints[stints["team_id"] == team]
    return frozenset(s["person_id"][(s["in_time"] <= t) & (t < s["out_time"])])


def _pbp_stints(gid: str):
    try:
        return gid, P.reconstruct_stints(P._fetch_pbp(gid), gid)
    except Exception:
        return gid, None


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate PBP rotations vs GameRotation ground truth")
    ap.add_argument("--season", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    gr_parts = P.RAW_DIR / f"rotations_{args.season}_parts"
    gr_games = sorted(p.stem for p in gr_parts.glob("*.parquet")) if gr_parts.exists() else []
    if not gr_games:
        print(f"[validate] no GameRotation ground-truth shards for {args.season} — skipping (exit 0)")
        return 0
    print(f"[validate] {len(gr_games)} ground-truth GameRotation games for {args.season}")

    shots = pd.read_parquet(
        P.PROCESSED_DIR / f"shots_scored_{args.season}.parquet",
        columns=["game_id", "team_id", "period", "seconds_remaining_period"],
    )
    shots["game_id"] = shots["game_id"].astype(str)
    shots = shots[shots["game_id"].isin(gr_games)].copy()
    shots["elapsed"] = _elapsed_tenths(shots["period"].to_numpy(), shots["seconds_remaining_period"].to_numpy())

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        pbp_by_game = dict(ex.map(_pbp_stints, gr_games))

    n_shot = both5 = match = gr5 = pbp5 = pbp_fail = 0
    for gid, g_shots in shots.groupby("game_id"):
        gr = pd.read_parquet(gr_parts / f"{gid}.parquet")
        pbp = pbp_by_game.get(gid)
        if pbp is None or pbp.empty:
            pbp_fail += 1
            continue
        teams = list(pbp["team_id"].unique())
        for _, r in g_shots.iterrows():
            t, off = int(r["elapsed"]), int(r["team_id"])
            deff = next((x for x in teams if x != off), off)
            n_shot += 1
            gr_ok = len(_oncourt(gr, t, off)) == 5 and len(_oncourt(gr, t, deff)) == 5
            pb_off, pb_def = _oncourt(pbp, t, off), _oncourt(pbp, t, deff)
            pb_ok = len(pb_off) == 5 and len(pb_def) == 5
            gr5 += gr_ok
            pbp5 += pb_ok
            if gr_ok and pb_ok:
                both5 += 1
                if _oncourt(gr, t, off) == pb_off and _oncourt(gr, t, deff) == pb_def:
                    match += 1

    if not n_shot or not both5:
        print("[validate] no comparable shots — skipping (exit 0)")
        return 0
    pbp5_rate, match_rate = pbp5 / n_shot, match / both5
    print(f"[validate] shots compared: {n_shot:,}  (pbp_fail games={pbp_fail})")
    print(f"[validate] GameRotation 5v5 rate: {gr5 / n_shot:.3%}")
    print(f"[validate] PBP          5v5 rate: {pbp5_rate:.3%}")
    print(f"[validate] EXACT 10-player match (among both-5v5): {match_rate:.3%}  (n={both5:,})")

    ok = match_rate >= MIN_MATCH
    if pbp5_rate < MIN_5V5:
        print(f"[validate] WARNING: PBP 5v5 rate {pbp5_rate:.1%} < {MIN_5V5:.0%}")
    if not ok:
        print(f"[validate] FAIL: match rate {match_rate:.1%} < {MIN_MATCH:.0%} — reconstruction regressed")
    else:
        print(f"[validate] PASS: match rate >= {MIN_MATCH:.0%}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
