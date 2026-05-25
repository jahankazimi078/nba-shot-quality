"""Reconstruct the 5 on-floor players per team at each shot, from per-game rotation stints.

Maps every shot's elapsed game clock to the rotation [in, out) windows so each shot is tagged
with its 5 offensive (shooter's team) and 5 defensive (the other team) players. Output is a
long/melted table (10 rows per shot) that feeds directly into the RAPM sparse design matrix.

GameRotation IN/OUT times are in tenths of a second of elapsed game time; ELAPSED_TO_ROTATION_SCALE
encodes that. It is the one assumption that must hold — if it is wrong, almost no shot reconstructs
to exactly 5v5 and the drop-rate guard below fires loudly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

ELAPSED_TO_ROTATION_SCALE = 10  # GameRotation times are in tenths of a second of elapsed game time
REG_PERIOD_SEC = 720
OT_PERIOD_SEC = 300
MAX_DROP_RATE = 0.02  # warn loudly above this — signals a wrong time scale or missing rotations


def _elapsed_tenths(period: np.ndarray, sec_remaining: np.ndarray) -> np.ndarray:
    """Elapsed game time at a shot, in tenths of a second (GameRotation scale)."""
    # Widen first: the features stage stores period as int8, which overflows on `* 720`.
    period = period.astype(np.int64)
    sec_remaining = sec_remaining.astype(np.int64)
    regulation_done = np.minimum(period - 1, 4) * REG_PERIOD_SEC
    ot_done = np.maximum(period - 5, 0) * OT_PERIOD_SEC
    period_len = np.where(period <= 4, REG_PERIOD_SEC, OT_PERIOD_SEC)
    elapsed_sec = regulation_done + ot_done + (period_len - sec_remaining)
    return elapsed_sec * ELAPSED_TO_ROTATION_SCALE


def build_shot_lineups(season: str) -> Path:
    """Tag each shot with its 5 offensive + 5 defensive on-floor players; write a long table."""
    scored_path = PROCESSED_DIR / f"shots_scored_{season}.parquet"
    rot_path = RAW_DIR / f"rotations_{season}.parquet"
    if not scored_path.exists():
        raise FileNotFoundError(f"{scored_path} not found — run score --season {season} first")
    if not rot_path.exists():
        raise FileNotFoundError(f"{rot_path} not found — run ingest-rotations --season {season} first")

    shots = pd.read_parquet(scored_path, columns=["game_id", "team_id", "period", "seconds_remaining_period", "poe"])
    shots["game_id"] = shots["game_id"].astype(str)
    shots = shots.reset_index(drop=True)
    shots["shot_uid"] = np.arange(len(shots), dtype="int64")
    shots["elapsed"] = _elapsed_tenths(
        shots["period"].to_numpy(), shots["seconds_remaining_period"].to_numpy()
    )
    rot = pd.read_parquet(rot_path)
    rot["game_id"] = rot["game_id"].astype(str)
    print(f"[shot_lineups] {len(shots):,} shots, {rot['game_id'].nunique():,} games with rotations")

    rot_by_game = {gid: g for gid, g in rot.groupby("game_id")}
    uid_parts, poe_parts, person_parts, side_parts, gid_parts = [], [], [], [], []
    n_off_all, n_def_all = [], []

    for gid, g_shots in shots.groupby("game_id"):
        g_rot = rot_by_game.get(gid)
        sh_t = g_shots["elapsed"].to_numpy()
        sh_team = g_shots["team_id"].to_numpy()
        sh_uid = g_shots["shot_uid"].to_numpy()
        sh_poe = g_shots["poe"].to_numpy()

        if g_rot is None or len(g_rot) == 0:
            n_off_all.append(np.zeros(len(g_shots), dtype=int))
            n_def_all.append(np.zeros(len(g_shots), dtype=int))
            continue

        s_team = g_rot["team_id"].to_numpy()
        s_person = g_rot["person_id"].to_numpy()
        s_in = g_rot["in_time"].to_numpy()
        s_out = g_rot["out_time"].to_numpy()

        on = (s_in[None, :] <= sh_t[:, None]) & (sh_t[:, None] < s_out[None, :])  # (n_shots, n_stints)
        same_team = s_team[None, :] == sh_team[:, None]
        n_off = (on & same_team).sum(axis=1)
        n_def = (on & ~same_team).sum(axis=1)
        n_off_all.append(n_off)
        n_def_all.append(n_def)

        keep = (n_off == 5) & (n_def == 5)
        if not keep.any():
            continue
        rows, cols = np.nonzero(on[keep])  # restricted to valid shots → exactly 10 per shot
        kept_uid = sh_uid[keep]
        kept_team = sh_team[keep]
        kept_poe = sh_poe[keep]
        uid_parts.append(kept_uid[rows])
        poe_parts.append(kept_poe[rows])
        gid_parts.append(np.full(len(rows), gid))
        person_parts.append(s_person[cols])
        # side: 0 = offense (stint team == shooter's team), 1 = defense
        side_parts.append((s_team[cols] != kept_team[rows]).astype("int8"))

    n_off_all = np.concatenate(n_off_all)
    n_def_all = np.concatenate(n_def_all)
    valid = (n_off_all == 5) & (n_def_all == 5)
    kept, total = int(valid.sum()), len(valid)
    drop_rate = 1 - kept / total
    print(f"[shot_lineups] reconstructed 5v5 for {kept:,}/{total:,} shots ({100*kept/total:.2f}%), dropped {total-kept:,}")
    print(f"[shot_lineups] offense on-floor counts: {dict(zip(*np.unique(n_off_all, return_counts=True)))}")
    print(f"[shot_lineups] defense on-floor counts: {dict(zip(*np.unique(n_def_all, return_counts=True)))}")
    print(f"[shot_lineups] max elapsed (tenths): {int(shots['elapsed'].max())} (expect ~28800 for a 48-min game)")
    if drop_rate > MAX_DROP_RATE:
        print(f"[shot_lineups] WARNING: drop rate {drop_rate:.1%} > {MAX_DROP_RATE:.0%} — "
              "check ELAPSED_TO_ROTATION_SCALE / rotation coverage")

    out = pd.DataFrame(
        {
            "shot_uid": np.concatenate(uid_parts),
            "game_id": np.concatenate(gid_parts),
            "season": season,
            "poe": np.concatenate(poe_parts),
            "person_id": np.concatenate(person_parts).astype("int64"),
            "side": np.concatenate(side_parts),
        }
    )
    out_path = PROCESSED_DIR / f"shot_lineups_{season}.parquet"
    out.to_parquet(out_path, index=False)
    print(f"[shot_lineups] {len(out):,} on-floor rows ({len(out)//10:,} shots × 10) -> {out_path}")
    return out_path
