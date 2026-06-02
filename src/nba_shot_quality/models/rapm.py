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
    "def_rapm", "def_ci_low", "def_ci_high",
    "off_rapm", "off_ci_low", "off_ci_high",
    "net_rapm", "net_ci_low", "net_ci_high",
    "def_shots", "off_shots",
]


def _name_map(seasons: list[str]) -> dict[int, str]:
    names: dict[int, str] = {}
    for s in seasons:
        path = RAW_DIR / f"rotations_{s}.parquet"
        if path.exists():
            r = pd.read_parquet(path, columns=["person_id", "player_name"])
            names.update(dict(zip(r["person_id"].astype("int64"), r["player_name"])))
    return names


def _matchup_suffix(weighting: str, matchup_source: str) -> str:
    """Output-parquet suffix: '' (uniform), '_matchup' (season shares), '_matchup_game' (per-game)."""
    if weighting != "matchup":
        return ""
    return "_matchup_game" if matchup_source == "game" else "_matchup"


def fit_rapm(seasons: list[str], n_boot: int = N_BOOT, alpha_grid: np.ndarray = ALPHA_GRID,
             weighting: str = "uniform", lam: float = 1.0, matchup_source: str = "season") -> Path:
    """Fit pooled (or single-season) defender RAPM with game-level bootstrap CIs.

    weighting="uniform" splits each shot's defensive credit equally across the 5 on-floor defenders;
    "matchup" weights it by who guarded the shooter (see apply_weighting). matchup_source selects the
    matchup tracking granularity: "season" (LeagueSeasonMatchups, averaged over the year) or "game"
    (BoxScoreMatchupsV3, that game's actual assignments). Each variant writes a separate suffixed
    parquet (`_matchup` / `_matchup_game`) so the uniform ratings are untouched.
    """
    frames = []
    for s in seasons:
        path = PROCESSED_DIR / f"shot_lineups_{s}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run lineups --season {s} first")
        frames.append(pd.read_parquet(path))
    long = pd.concat(frames, ignore_index=True)
    print(f"[rapm] {len(long):,} on-floor rows from seasons {seasons}  weighting={weighting} source={matchup_source} lam={lam}")
    if weighting == "matchup":
        long["weight"] = apply_weighting(long, lam, _load_matchups(seasons, matchup_source))

    X, y, groups, players = _build_design(long)
    print(f"[rapm] design matrix {X.shape[0]:,} shots × {X.shape[1]:,} cols ({len(players):,} players), 10 nnz/row")

    best_alpha = _select_alpha(X, y, groups, alpha_grid)
    off_rapm, def_rapm = _fit_sides(X, y, best_alpha)

    side_counts = long.groupby(["person_id", "side"]).size().unstack(fill_value=0)
    off_shots = side_counts.get(0, pd.Series(0, index=side_counts.index)).reindex(players, fill_value=0)
    def_shots = side_counts.get(1, pd.Series(0, index=side_counts.index)).reindex(players, fill_value=0)

    # Game-level bootstrap CIs for all three views (offense, defense, net = off + def).
    off_b, def_b, net_b = _bootstrap(X, y, groups, best_alpha, n_boot, n_players=len(players))

    names = _name_map(seasons)
    pooled = len(seasons) > 1
    out = pd.DataFrame(
        {
            "player_id": players,
            "player_name": [names.get(int(p), str(p)) for p in players],
            "season": "pooled" if pooled else seasons[0],
            "def_rapm": def_rapm,
            "def_ci_low": np.nanpercentile(def_b, 2.5, axis=0),
            "def_ci_high": np.nanpercentile(def_b, 97.5, axis=0),
            "off_rapm": off_rapm,
            "off_ci_low": np.nanpercentile(off_b, 2.5, axis=0),
            "off_ci_high": np.nanpercentile(off_b, 97.5, axis=0),
            "net_rapm": off_rapm + def_rapm,
            "net_ci_low": np.nanpercentile(net_b, 2.5, axis=0),
            "net_ci_high": np.nanpercentile(net_b, 97.5, axis=0),
            "def_shots": def_shots.to_numpy().astype("int64"),
            "off_shots": off_shots.to_numpy().astype("int64"),
        }
    )[OUTPUT_COLS].sort_values("def_rapm", ascending=False).reset_index(drop=True)

    suffix = _matchup_suffix(weighting, matchup_source)
    stem = "rapm_pooled" if pooled else f"rapm_{seasons[0]}"
    out_path = PROCESSED_DIR / f"{stem}{suffix}.parquet"
    out.to_parquet(out_path, index=False)

    shown = out[out["def_shots"] >= 1500]
    print(f"[rapm] best_alpha={best_alpha:.1f}  (top/bottom defenders, ≥1500 def shots)")
    for _, r in shown.head(5).iterrows():
        print(f"  + {r['player_name']:<24} {r['def_rapm']:+6.2f} [{r['def_ci_low']:+.2f}, {r['def_ci_high']:+.2f}]  ({int(r['def_shots'])} def shots)")
    for _, r in shown.tail(5).iterrows():
        print(f"  - {r['player_name']:<24} {r['def_rapm']:+6.2f} [{r['def_ci_low']:+.2f}, {r['def_ci_high']:+.2f}]  ({int(r['def_shots'])} def shots)")
    print(f"[rapm] -> {out_path}")
    return out_path


def _load_matchups(seasons: list[str], source: str = "season") -> dict[str, pd.DataFrame]:
    """Load matchup tracking per season. source="season" -> LeagueSeasonMatchups totals;
    "game" -> per-game BoxScoreMatchupsV3 (carries a game_id column the weighting merges on)."""
    stem = "box_matchups" if source == "game" else "matchups"
    cmd = "ingest-box-matchups" if source == "game" else "ingest-matchups"
    out = {}
    for s in seasons:
        p = RAW_DIR / f"{stem}_{s}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — run {cmd} --season {s} first")
        out[s] = pd.read_parquet(p)
    return out


def apply_weighting(long: pd.DataFrame, lam: float, matchups: dict[str, pd.DataFrame]) -> np.ndarray:
    """Per-row design weight: offense rows -> 1.0; defense rows -> 5 * blended matchup share.

    For each shot, a defender's share is its partial-possessions guarding that shot's shooter,
    normalized over the shot's 5 on-floor defenders (uniform 1/5 fallback when no matchup data exists).
    w_d = 5 * [(1-lam)*(1/5) + lam*share5_d], so the total defensive mass per shot stays 5 — making
    uniform attribution the exact lam=0 special case and leaving the offense/defense scale unchanged.

    The matchup frames may be season totals (merge key: shooter+defender) or per-game
    BoxScoreMatchupsV3 (frames carry `game_id`, so the merge also keys on the shot's game) — the
    per-game variant pins credit to the defender who actually guarded the shooter *in that game*.
    """
    if "shooter_id" not in long.columns:
        raise KeyError("shot_lineups lacks shooter_id — re-run `lineups` to add it")
    per_game = all("game_id" in m.columns for m in matchups.values())
    w = np.ones(len(long), dtype=np.float64)
    defmask = long["side"].to_numpy() == 1
    cols = ["season", "shot_uid", "shooter_id", "person_id"] + (["game_id"] if per_game else [])
    dd = long.loc[defmask, cols].reset_index()
    mu_cols = ["off_player_id", "def_player_id", "partial_poss"] + (["game_id"] if per_game else [])
    mu = pd.concat(
        [m[mu_cols].assign(season=s) for s, m in matchups.items()], ignore_index=True
    ).rename(columns={"off_player_id": "shooter_id", "def_player_id": "person_id"})
    on_keys = ["season", "shooter_id", "person_id"] + (["game_id"] if per_game else [])
    dd = dd.merge(mu, how="left", on=on_keys)
    dd["pp"] = dd["partial_poss"].fillna(0.0)
    dd["sk"] = dd["season"] + ":" + dd["shot_uid"].astype(str)
    tot = dd.groupby("sk")["pp"].transform("sum").to_numpy()
    share5 = np.full(len(tot), 0.2)  # uniform fallback when a shot has no matchup data
    nz = tot > 0
    share5[nz] = dd["pp"].to_numpy()[nz] / tot[nz]
    w[dd["index"].to_numpy()] = 5.0 * ((1.0 - lam) * 0.2 + lam * share5)
    return w


def _build_design(long: pd.DataFrame):
    """Build the sparse off+def design from a long shot_lineups frame.

    Returns (X_csr, y, groups, players): X has 2 columns per player (offense col 2i, defense col
    2i+1) and exactly 10 nonzeros per shot row; y is per-shot POE; groups is per-shot game_id.
    Reusable so diagnostics can fit on game subsets without duplicating this logic.
    """
    long = long.copy()
    long["shot_key"] = long["season"] + ":" + long["shot_uid"].astype(str)  # unique across seasons
    players = np.sort(long["person_id"].unique())
    col_of = {pid: i for i, pid in enumerate(players)}
    n_cols = 2 * len(players)

    row_codes, shot_keys = pd.factorize(long["shot_key"], sort=False)
    n_rows = len(shot_keys)
    cols = long["person_id"].map(col_of).to_numpy() * 2 + long["side"].to_numpy()
    data = long["weight"].to_numpy(dtype=np.float64) if "weight" in long.columns else np.ones(len(long))
    X = sparse.coo_matrix((data, (row_codes, cols)), shape=(n_rows, n_cols)).tocsr()

    shot_tbl = long.drop_duplicates("shot_key").set_index("shot_key")
    y = shot_tbl.loc[shot_keys, "poe"].to_numpy()
    groups = shot_tbl.loc[shot_keys, "game_id"].to_numpy()

    assert X.shape == (n_rows, n_cols), "matrix shape mismatch"
    nnz_per_row = np.diff(X.indptr)
    assert (nnz_per_row == 10).all(), f"expected 10 nonzeros/row, got {np.unique(nnz_per_row)}"
    return X, y, groups, players


def _fit_sides(X, y, alpha):
    """Ridge fit -> (off_rapm, def_rapm) per-100 arrays; defense sign-flipped so + = good defense."""
    m = Ridge(alpha=alpha, solver=SOLVER, fit_intercept=True)
    m.fit(X, y)
    return m.coef_[0::2] * 100.0, -m.coef_[1::2] * 100.0


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


def _bootstrap(X, y, groups, alpha, n_boot, n_players):
    """Game-level bootstrap; returns (off_b, def_b, net_b), each (n_boot, n_players) per-100.

    The defense draws are identical to the previous def-only bootstrap (same seed/resample order),
    so existing def CIs are unchanged; offense and net = off + def are collected alongside.
    """
    rng = np.random.default_rng(0)
    unique_games = np.unique(groups)
    rows_by_game = {g: np.flatnonzero(groups == g) for g in unique_games}
    off_b = np.full((n_boot, n_players), np.nan)
    def_b = np.full((n_boot, n_players), np.nan)
    for b in range(n_boot):
        sampled = rng.choice(unique_games, size=len(unique_games), replace=True)
        sel = np.concatenate([rows_by_game[g] for g in sampled])
        off_b[b], def_b[b] = _fit_sides(X[sel], y[sel], alpha)
        if (b + 1) % 50 == 0:
            print(f"[rapm]   bootstrap {b+1}/{n_boot}", flush=True)
    return off_b, def_b, off_b + def_b
