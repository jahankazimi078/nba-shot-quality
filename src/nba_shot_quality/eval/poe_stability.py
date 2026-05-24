"""Stability diagnostics: year-over-year POE reliability and POE-vs-rTS% divergence.

YoY correlation answers "is POE real skill, not noise?" — a player good in one season should
be good the next. POE-vs-rTS% shows where difficulty-adjusted skill agrees/disagrees with raw
true-shooting efficiency. Correlations use numpy/pandas only (no scipy dependency).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: pd.Series, y: pd.Series) -> float:
    return float(np.corrcoef(x.rank(), y.rank())[0, 1])


def _load_poe(season: str, min_attempts: int) -> pd.DataFrame:
    path = PROCESSED_DIR / f"poe_player_season_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run poe --season {season} first")
    df = pd.read_parquet(path)
    return df[df["attempts"] >= min_attempts].copy()


def yoy_stability(season_a: str, season_b: str, min_attempts: int = 200) -> Path:
    """Correlate POE_per_100 across two seasons for players qualified in both."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    a = _load_poe(season_a, min_attempts)
    b = _load_poe(season_b, min_attempts)
    merged = a.merge(b, on="player_id", suffixes=("_a", "_b"))
    print(f"[poe_stability] {season_a}: {len(a):,} qualified, {season_b}: {len(b):,} qualified, "
          f"matched in both: {len(merged):,}")

    xa = merged["poe_per_100_a"].to_numpy()
    yb = merged["poe_per_100_b"].to_numpy()
    r_p = _pearson(xa, yb)
    r_s = _spearman(merged["poe_per_100_a"], merged["poe_per_100_b"])
    print(f"[poe_stability] YoY POE/100  Pearson r={r_p:.3f}  Spearman r={r_s:.3f}  (n={len(merged):,})")

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(xa, yb, s=20, alpha=0.6, edgecolor="black", linewidths=0.2)
    lim = [min(xa.min(), yb.min()) - 1, max(xa.max(), yb.max()) + 1]
    ax.plot(lim, lim, "k--", alpha=0.5, label="y = x")
    # Label a few high-volume standouts (by combined attempts).
    merged["vol"] = merged["attempts_a"] + merged["attempts_b"]
    for _, r in merged.nlargest(8, "vol").iterrows():
        ax.annotate(r["player_name_b"], (r["poe_per_100_a"], r["poe_per_100_b"]),
                    fontsize=7, alpha=0.8, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel(f"POE per 100 — {season_a}")
    ax.set_ylabel(f"POE per 100 — {season_b}")
    ax.set_title(f"Year-over-year POE stability\nPearson r={r_p:.3f}  Spearman r={r_s:.3f}  n={len(merged):,}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = REPORTS_DIR / f"poe_stability_{season_a}_vs_{season_b}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[poe_stability] -> {out_path}")
    return out_path


def poe_vs_rts(season: str, min_attempts: int = 200) -> Path:
    """Scatter POE_per_100 vs relative TS% to show where the two metrics disagree."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_poe(season, min_attempts)
    df = df[df["rel_ts_pct"].notna()].copy()
    if df.empty:
        raise ValueError(f"[poe_stability] no rows with rel_ts_pct for {season} — was ingest-stats run?")

    x = df["poe_per_100"].to_numpy()
    y = (df["rel_ts_pct"] * 100).to_numpy()  # scale to percentage points
    r_p = _pearson(x, y)
    r_s = _spearman(df["poe_per_100"], df["rel_ts_pct"])
    print(f"[poe_stability] POE/100 vs rTS%  Pearson r={r_p:.3f}  Spearman r={r_s:.3f}  (n={len(df):,})")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(x, y, s=20, alpha=0.6, edgecolor="black", linewidths=0.2)
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8)
    # Label the biggest disagreements: residual from the best-fit line.
    coef = np.polyfit(x, y, 1)
    resid = y - np.polyval(coef, x)
    df = df.assign(_resid=resid)
    standouts = pd.concat([df.nlargest(5, "_resid"), df.nsmallest(5, "_resid")])
    for _, r in standouts.iterrows():
        ax.annotate(r["player_name"], (r["poe_per_100"], r["rel_ts_pct"] * 100),
                    fontsize=7, alpha=0.8, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("POE per 100 (shot-quality adjusted)")
    ax.set_ylabel("Relative TS% (pp vs league)")
    ax.set_title(f"POE vs rTS% — {season}\nPearson r={r_p:.3f}  Spearman r={r_s:.3f}  n={len(df):,}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = REPORTS_DIR / f"poe_vs_rts_{season}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[poe_stability] -> {out_path}")
    return out_path
