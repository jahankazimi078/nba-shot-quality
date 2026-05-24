"""xPoints model: LightGBM make-probability classifier; xPoints = p_make * shot_value."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold

from nba_shot_quality.features.shot_features import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    TARGET_COL,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
HOLDOUT_FRACTION = 0.15
N_FOLDS = 5
LGB_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    learning_rate=0.05,
    num_leaves=63,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    min_data_in_leaf=200,
    verbosity=-1,
)


@dataclass
class TrainResult:
    cv_logloss: list[float]
    holdout_logloss: float
    holdout_brier: float
    holdout_pps_mae: float
    n_train: int
    n_holdout: int
    model_path: Path
    holdout_predictions_path: Path


def train(season: str) -> TrainResult:
    features_path = PROCESSED_DIR / f"shots_features_{season}.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"features not found at {features_path} — run features first")
    df = pd.read_parquet(features_path)
    print(f"[train] loaded {len(df):,} rows from {features_path.name}")

    df = df.sort_values("game_date").reset_index(drop=True)
    train_df, holdout_df = _chronological_split(df, HOLDOUT_FRACTION)
    print(f"[train] split: train={len(train_df):,}  holdout={len(holdout_df):,}")

    X_train, y_train = _xy(train_df)
    X_holdout, y_holdout = _xy(holdout_df)

    cv_logloss = _cv_eval(train_df)
    print(f"[train] CV log-loss (per fold): {[round(v, 4) for v in cv_logloss]}")
    print(f"[train] CV log-loss mean: {np.mean(cv_logloss):.4f}")

    model = lgb.LGBMClassifier(n_estimators=600, **LGB_PARAMS)
    model.fit(
        X_train,
        y_train,
        categorical_feature=CATEGORICAL_COLS,
        eval_set=[(X_holdout, y_holdout)],
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)],
    )

    p_holdout = model.predict_proba(X_holdout)[:, 1]
    holdout_logloss = log_loss(y_holdout, p_holdout, labels=[0, 1])
    holdout_brier = brier_score_loss(y_holdout, p_holdout)
    print(f"[train] holdout log-loss: {holdout_logloss:.4f}")
    print(f"[train] holdout Brier:    {holdout_brier:.4f}")

    holdout_df = holdout_df.copy()
    holdout_df["p_make"] = p_holdout
    holdout_df["xpoints"] = p_holdout * holdout_df["shot_value"]
    holdout_df["poe"] = holdout_df["points"] - holdout_df["xpoints"]

    pps_mae = _player_pps_mae(holdout_df)
    print(f"[train] player-season PPS MAE (top 100 by attempts): {pps_mae:.4f}")

    model_path = PROCESSED_DIR / "xpoints_model.joblib"
    joblib.dump(model, model_path)

    holdout_path = PROCESSED_DIR / f"holdout_predictions_{season}.parquet"
    holdout_df.to_parquet(holdout_path, index=False)

    print(f"[train] model -> {model_path}")
    print(f"[train] holdout predictions -> {holdout_path}")

    return TrainResult(
        cv_logloss=cv_logloss,
        holdout_logloss=holdout_logloss,
        holdout_brier=holdout_brier,
        holdout_pps_mae=pps_mae,
        n_train=len(train_df),
        n_holdout=len(holdout_df),
        model_path=model_path,
        holdout_predictions_path=holdout_path,
    )


def score_oof(season: str) -> Path:
    """Score every shot in a season out-of-fold (GroupKFold on game_id).

    Each shot is predicted by a model that never saw its game, so the full-season
    per-shot poe is free of in-sample optimism — this is the leaderboard foundation
    for the POE aggregation step. Each season is scored independently (self-calibrated), so there is
    no unseen-category issue across seasons. This NEVER writes the canonical
    xpoints_model.joblib or holdout_predictions artifacts from train().
    """
    features_path = PROCESSED_DIR / f"shots_features_{season}.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"features not found at {features_path} — run features first")
    df = pd.read_parquet(features_path).reset_index(drop=True)
    print(f"[score] loaded {len(df):,} rows from {features_path.name}")

    X, y = _xy(df)
    groups = df["game_id"].to_numpy()
    p_oof = np.full(len(df), np.nan)

    gkf = GroupKFold(n_splits=N_FOLDS)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        m = lgb.LGBMClassifier(n_estimators=600, **LGB_PARAMS)
        m.fit(X.iloc[tr_idx], y[tr_idx], categorical_feature=CATEGORICAL_COLS)
        p_oof[va_idx] = m.predict_proba(X.iloc[va_idx])[:, 1]
        print(f"[score] fold {fold}: scored {len(va_idx):,} shots")

    if np.isnan(p_oof).any():
        raise RuntimeError("[score] some shots were never scored — check GroupKFold coverage")

    df["p_make"] = p_oof
    df["xpoints"] = p_oof * df["shot_value"]
    df["poe"] = df["points"] - df["xpoints"]

    oof_logloss = log_loss(y, p_oof, labels=[0, 1])
    oof_brier = brier_score_loss(y, p_oof)
    mean_poe = float(df["poe"].mean())
    print(f"[score] OOF log-loss: {oof_logloss:.4f}  Brier: {oof_brier:.4f}")
    print(f"[score] mean per-shot poe: {mean_poe:+.5f}  total poe: {df['poe'].sum():+.1f} (expect ~0)")

    out_path = PROCESSED_DIR / f"shots_scored_{season}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[score] scored shots -> {out_path}")
    return out_path


def predict_xpoints(model: lgb.LGBMClassifier, df: pd.DataFrame) -> np.ndarray:
    X, _ = _xy(df, require_target=False)
    p = model.predict_proba(X)[:, 1]
    return p * df["shot_value"].to_numpy()


def _xy(df: pd.DataFrame, require_target: bool = True):
    X = df[FEATURE_COLS].copy()
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype("category")
    y = df[TARGET_COL].to_numpy() if require_target else None
    return X, y


def _chronological_split(df: pd.DataFrame, holdout_fraction: float):
    games = df[["game_id", "game_date"]].drop_duplicates().sort_values("game_date")
    n_holdout = max(1, int(len(games) * holdout_fraction))
    holdout_games = set(games.tail(n_holdout)["game_id"])
    is_holdout = df["game_id"].isin(holdout_games)
    return df[~is_holdout].reset_index(drop=True), df[is_holdout].reset_index(drop=True)


def _cv_eval(train_df: pd.DataFrame) -> list[float]:
    X, y = _xy(train_df)
    groups = train_df["game_id"].to_numpy()
    gkf = GroupKFold(n_splits=N_FOLDS)
    losses: list[float] = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        m = lgb.LGBMClassifier(n_estimators=400, **LGB_PARAMS)
        m.fit(
            X.iloc[tr_idx],
            y[tr_idx],
            categorical_feature=CATEGORICAL_COLS,
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)],
        )
        p = m.predict_proba(X.iloc[va_idx])[:, 1]
        loss = log_loss(y[va_idx], p, labels=[0, 1])
        losses.append(loss)
        print(f"[train] fold {fold}: logloss={loss:.4f}  n_val={len(va_idx):,}")
    return losses


def _player_pps_mae(holdout: pd.DataFrame, top_n: int = 100) -> float:
    agg = (
        holdout.groupby("player_id")
        .agg(attempts=("points", "size"), pps=("points", "mean"), x_pps=("xpoints", "mean"))
        .sort_values("attempts", ascending=False)
        .head(top_n)
    )
    return float((agg["pps"] - agg["x_pps"]).abs().mean())
