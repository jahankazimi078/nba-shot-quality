"""Defender-impact ridge RAPM on per-shot POE.

Each shot is a row; each player gets two indicator columns (offense + defense), set to 1 when the
player is on the floor for that shot on that side. Target = the shot's POE (offense perspective,
+ = offense beat expectation). Fitting offense and defense jointly lets the offensive columns absorb
shot-maker quality, so the defensive coefficients isolate defender impact; ridge tames the severe
multicollinearity of players who always share the floor. Defensive coefficients are sign-flipped and
scaled to per-100 shots so that **+ = good defense** (opponent under-performed expectation).

Caveats (on-floor attribution, not closest-defender; FGA-only) are documented in build_shot_lineups
and surfaced in the dashboard. Reads `shot_lineups_{season}.parquet`; pooling multiple seasons shares
each player's two columns across seasons for more shots per player.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_DIR = REPO_ROOT / "data" / "raw"

ALPHA_GRID = np.logspace(2, 5, 13)
N_BOOT = 300
N_FOLDS = 5
SOLVER = "sparse_cg"

OUTPUT_COLS = [
    "player_id", "player_name", "season",
    "def_rapm", "def_ci_low", "def_ci_high", "off_rapm", "def_shots", "off_shots",
]


def _name_map(seasons: list[str]) -> dict[int, str]:
    names: dict[int, str] = {}
    for s in seasons:
        path = RAW_DIR / f"rotations_{s}.parquet"
        if path.exists():
            r = pd.read_parquet(path, columns=["person_id", "player_name"])
            names.update(dict(zip(r["person_id"].astype("int64"), r["player_name"])))
    return names


def fit_rapm(seasons: list[str], n_boot: int = N_BOOT, alpha_grid: np.ndarray = ALPHA_GRID) -> Path:
    """Fit pooled (or single-season) defender RAPM with game-level bootstrap CIs."""
    frames = []
    for s in seasons:
        path = PROCESSED_DIR / f"shot_lineups_{s}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run lineups --season {s} first")
        frames.append(pd.read_parquet(path))
    long = pd.concat(frames, ignore_index=True)
    long["shot_key"] = long["season"] + ":" + long["shot_uid"].astype(str)  # unique across seasons
    print(f"[rapm] {len(long):,} on-floor rows from seasons {seasons}")

    # Column map: each player -> (offense col 2i, defense col 2i+1)
    players = np.sort(long["person_id"].unique())
    col_of = {pid: i for i, pid in enumerate(players)}
    n_cols = 2 * len(players)

    # Row map: one matrix row per shot
    row_codes, shot_keys = pd.factorize(long["shot_key"], sort=False)
    n_rows = len(shot_keys)
    cols = long["person_id"].map(col_of).to_numpy() * 2 + long["side"].to_numpy()
    X = sparse.coo_matrix(
        (np.ones(len(long), dtype=np.float64), (row_codes, cols)), shape=(n_rows, n_cols)
    ).tocsr()

    shot_tbl = long.drop_duplicates("shot_key").set_index("shot_key")
    y = shot_tbl.loc[shot_keys, "poe"].to_numpy()
    groups = shot_tbl.loc[shot_keys, "game_id"].to_numpy()

    assert X.shape == (n_rows, n_cols), "matrix shape mismatch"
    nnz_per_row = np.diff(X.indptr)
    assert (nnz_per_row == 10).all(), f"expected 10 nonzeros/row, got {np.unique(nnz_per_row)}"
    print(f"[rapm] design matrix {n_rows:,} shots × {n_cols:,} cols ({len(players):,} players), 10 nnz/row")

    best_alpha = _select_alpha(X, y, groups, alpha_grid)

    model = Ridge(alpha=best_alpha, solver=SOLVER, fit_intercept=True)
    model.fit(X, y)
    coef = model.coef_
    off_rapm = coef[0::2] * 100.0
    def_rapm = -coef[1::2] * 100.0  # sign-flip: + = good defense

    side_counts = long.groupby(["person_id", "side"]).size().unstack(fill_value=0)
    off_shots = side_counts.get(0, pd.Series(0, index=side_counts.index)).reindex(players, fill_value=0)
    def_shots = side_counts.get(1, pd.Series(0, index=side_counts.index)).reindex(players, fill_value=0)

    def_boot = _bootstrap(X, y, groups, best_alpha, n_boot, n_players=len(players))
    ci_low = np.nanpercentile(def_boot, 2.5, axis=0)
    ci_high = np.nanpercentile(def_boot, 97.5, axis=0)

    names = _name_map(seasons)
    pooled = len(seasons) > 1
    out = pd.DataFrame(
        {
            "player_id": players,
            "player_name": [names.get(int(p), str(p)) for p in players],
            "season": "pooled" if pooled else seasons[0],
            "def_rapm": def_rapm,
            "def_ci_low": ci_low,
            "def_ci_high": ci_high,
            "off_rapm": off_rapm,
            "def_shots": def_shots.to_numpy().astype("int64"),
            "off_shots": off_shots.to_numpy().astype("int64"),
        }
    )[OUTPUT_COLS].sort_values("def_rapm", ascending=False).reset_index(drop=True)

    out_path = PROCESSED_DIR / ("rapm_pooled.parquet" if pooled else f"rapm_{seasons[0]}.parquet")
    out.to_parquet(out_path, index=False)

    shown = out[out["def_shots"] >= 1500]
    print(f"[rapm] best_alpha={best_alpha:.1f}  (top/bottom defenders, ≥1500 def shots)")
    for _, r in shown.head(5).iterrows():
        print(f"  + {r['player_name']:<24} {r['def_rapm']:+6.2f} [{r['def_ci_low']:+.2f}, {r['def_ci_high']:+.2f}]  ({int(r['def_shots'])} def shots)")
    for _, r in shown.tail(5).iterrows():
        print(f"  - {r['player_name']:<24} {r['def_rapm']:+6.2f} [{r['def_ci_low']:+.2f}, {r['def_ci_high']:+.2f}]  ({int(r['def_shots'])} def shots)")
    print(f"[rapm] -> {out_path}")
    return out_path


def _select_alpha(X, y, groups, alpha_grid) -> float:
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits = list(gkf.split(X, y, groups=groups))
    best_alpha, best_mse = alpha_grid[0], np.inf
    for alpha in alpha_grid:
        fold_mse = []
        for tr, va in splits:
            m = Ridge(alpha=alpha, solver=SOLVER, fit_intercept=True)
            m.fit(X[tr], y[tr])
            pred = m.predict(X[va])
            fold_mse.append(float(np.mean((pred - y[va]) ** 2)))
        mse = float(np.mean(fold_mse))
        print(f"[rapm]   alpha={alpha:10.1f}  cv_mse={mse:.5f}")
        if mse < best_mse:
            best_alpha, best_mse = alpha, mse
    if best_alpha in (alpha_grid[0], alpha_grid[-1]):
        print(f"[rapm] WARNING: best_alpha={best_alpha:.1f} at grid edge — widen ALPHA_GRID")
    return float(best_alpha)


def _bootstrap(X, y, groups, alpha, n_boot, n_players) -> np.ndarray:
    rng = np.random.default_rng(0)
    unique_games = np.unique(groups)
    rows_by_game = {g: np.flatnonzero(groups == g) for g in unique_games}
    out = np.full((n_boot, n_players), np.nan)
    for b in range(n_boot):
        sampled = rng.choice(unique_games, size=len(unique_games), replace=True)
        sel = np.concatenate([rows_by_game[g] for g in sampled])
        m = Ridge(alpha=alpha, solver=SOLVER, fit_intercept=True)
        m.fit(X[sel], y[sel])
        out[b] = -m.coef_[1::2] * 100.0
        if (b + 1) % 50 == 0:
            print(f"[rapm]   bootstrap {b+1}/{n_boot}", flush=True)
    return out
