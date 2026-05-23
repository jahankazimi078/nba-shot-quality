"""Calibration + reliability diagnostics for the xPoints model."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"


def evaluate(season: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    holdout_path = PROCESSED_DIR / f"holdout_predictions_{season}.parquet"
    if not holdout_path.exists():
        raise FileNotFoundError(f"holdout predictions not found at {holdout_path} — run train first")

    df = pd.read_parquet(holdout_path)
    print(f"[eval] loaded {len(df):,} holdout shots from {holdout_path.name}")

    overall_logloss = log_loss(df["shot_made"], df["p_make"], labels=[0, 1])
    overall_brier = brier_score_loss(df["shot_made"], df["p_make"])
    print(f"[eval] overall log-loss: {overall_logloss:.4f}")
    print(f"[eval] overall Brier:    {overall_brier:.4f}")

    print("[eval] per-zone observed vs predicted make rate:")
    zone_stats = (
        df.groupby("shot_zone", observed=True)
        .agg(
            n=("shot_made", "size"),
            observed=("shot_made", "mean"),
            predicted=("p_make", "mean"),
        )
        .assign(diff_pp=lambda d: (d["predicted"] - d["observed"]) * 100)
        .sort_values("n", ascending=False)
    )
    print(zone_stats.round(4).to_string())

    plot_path = REPORTS_DIR / f"calibration_{season}.png"
    _plot_calibration(df, zone_stats, plot_path, season, overall_brier, overall_logloss)
    print(f"[eval] calibration plot -> {plot_path}")
    return plot_path


def _plot_calibration(
    df: pd.DataFrame,
    zone_stats: pd.DataFrame,
    out_path: Path,
    season: str,
    brier: float,
    logloss: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    bins = np.linspace(0.0, 1.0, 11)
    df = df.assign(bin=pd.cut(df["p_make"], bins=bins, include_lowest=True))
    bin_stats = df.groupby("bin", observed=True).agg(
        predicted=("p_make", "mean"),
        observed=("shot_made", "mean"),
        n=("shot_made", "size"),
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    ax.plot(
        bin_stats["predicted"],
        bin_stats["observed"],
        marker="o",
        linewidth=2,
        label="model",
    )
    for _, row in bin_stats.iterrows():
        ax.annotate(
            f"n={int(row['n']):,}",
            (row["predicted"], row["observed"]),
            textcoords="offset points",
            xytext=(5, -10),
            fontsize=7,
            alpha=0.7,
        )
    ax.set_xlabel("Predicted P(make)")
    ax.set_ylabel("Observed make rate")
    ax.set_title(f"Reliability diagram — {season}\nBrier={brier:.4f}  logloss={logloss:.4f}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    ax = axes[1]
    zones = zone_stats.index.tolist()
    x = np.arange(len(zones))
    width = 0.35
    ax.bar(x - width / 2, zone_stats["observed"], width, label="observed")
    ax.bar(x + width / 2, zone_stats["predicted"], width, label="predicted")
    ax.set_xticks(x)
    ax.set_xticklabels(zones, rotation=20, ha="right")
    ax.set_ylabel("Make rate")
    ax.set_title("Per-zone calibration")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
