"""Defender-RAPM diagnostics: year-over-year stability and face validity vs a tracking metric.

YoY stability asks whether defensive RAPM persists across seasons (real signal, not noise).
Face validity compares it to nba_api's tracking-based defended-FG metric (LeagueDashPtDefend):
elite rim protectors and perimeter stoppers should rank near the top. Correlations use
numpy/pandas only (no scipy stats), matching the POE stability module.
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
RAW_DIR = REPO_ROOT / "data" / "raw"
REPORTS_DIR = REPO_ROOT / "reports"


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: pd.Series, y: pd.Series) -> float:
    return float(np.corrcoef(x.rank(), y.rank())[0, 1])


def _load_rapm(season: str, min_def_shots: int) -> pd.DataFrame:
    path = PROCESSED_DIR / f"rapm_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run rapm --seasons {season} first")
    df = pd.read_parquet(path)
    return df[df["def_shots"] >= min_def_shots].copy()


def yoy_rapm_stability(season_a: str, season_b: str, min_def_shots: int = 1500) -> Path:
    """Correlate per-season defensive RAPM for players qualified in both seasons."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    a = _load_rapm(season_a, min_def_shots)
    b = _load_rapm(season_b, min_def_shots)
    merged = a.merge(b, on="player_id", suffixes=("_a", "_b"))
    print(f"[rapm_eval] {season_a}: {len(a):,} qualified, {season_b}: {len(b):,}, matched: {len(merged):,}")

    x = merged["def_rapm_a"].to_numpy()
    y = merged["def_rapm_b"].to_numpy()
    r_p = _pearson(x, y)
    r_s = _spearman(merged["def_rapm_a"], merged["def_rapm_b"])
    print(f"[rapm_eval] YoY def_rapm  Pearson r={r_p:.3f}  Spearman r={r_s:.3f}  (n={len(merged):,})")

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(x, y, s=20, alpha=0.6, edgecolor="black", linewidths=0.2)
    lim = [min(x.min(), y.min()) - 1, max(x.max(), y.max()) + 1]
    ax.plot(lim, lim, "k--", alpha=0.5, label="y = x")
    merged["vol"] = merged["def_shots_a"] + merged["def_shots_b"]
    for _, r in merged.nlargest(8, "def_rapm_b").iterrows():
        ax.annotate(r["player_name_b"], (r["def_rapm_a"], r["def_rapm_b"]),
                    fontsize=7, alpha=0.8, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel(f"Defensive RAPM — {season_a}")
    ax.set_ylabel(f"Defensive RAPM — {season_b}")
    ax.set_title(f"Year-over-year defender RAPM stability\nPearson r={r_p:.3f}  Spearman r={r_s:.3f}  n={len(merged):,}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = REPORTS_DIR / f"rapm_stability_{season_a}_vs_{season_b}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[rapm_eval] -> {out_path}")
    return out_path


def rapm_face_validity(season: str, min_def_shots: int = 1500) -> Path:
    """Compare defensive RAPM to nba_api's defended-FG tracking metric for the season."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rapm = _load_rapm(season, min_def_shots)
    print(f"[rapm_eval] {season} top-10 defenders by RAPM (≥{min_def_shots} def shots):")
    for _, r in rapm.sort_values("def_rapm", ascending=False).head(10).iterrows():
        print(f"  + {r['player_name']:<24} {r['def_rapm']:+6.2f}")
    print(f"[rapm_eval] {season} bottom-10 defenders by RAPM:")
    for _, r in rapm.sort_values("def_rapm").head(10).iterrows():
        print(f"  - {r['player_name']:<24} {r['def_rapm']:+6.2f}")

    pt_path = RAW_DIR / f"pt_defend_{season}.parquet"
    if not pt_path.exists():
        raise FileNotFoundError(f"{pt_path} not found — run ingest-def --season {season} first")
    pt = pd.read_parquet(pt_path)
    # def_fg_suppression: higher = holds opponents further below their normal FG% = good defense
    pt["def_fg_suppression"] = -pt["pct_plusminus"]
    merged = rapm.merge(pt[["player_id", "def_fg_suppression"]], on="player_id", how="inner").dropna(
        subset=["def_fg_suppression"]
    )
    x = merged["def_rapm"].to_numpy()
    y = merged["def_fg_suppression"].to_numpy()
    r_p = _pearson(x, y)
    r_s = _spearman(merged["def_rapm"], merged["def_fg_suppression"])
    print(f"[rapm_eval] RAPM vs defended-FG suppression  Pearson r={r_p:.3f}  Spearman r={r_s:.3f}  (n={len(merged):,})")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(x, y, s=20, alpha=0.6, edgecolor="black", linewidths=0.2)
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8)
    for _, r in merged.nlargest(6, "def_rapm").iterrows():
        ax.annotate(r["player_name"], (r["def_rapm"], r["def_fg_suppression"]),
                    fontsize=7, alpha=0.8, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("Defensive RAPM (+ = good)")
    ax.set_ylabel("Defended-FG suppression (−PCT_PLUSMINUS, + = good)")
    ax.set_title(f"Defender RAPM vs tracking defense — {season}\nPearson r={r_p:.3f}  Spearman r={r_s:.3f}  n={len(merged):,}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = REPORTS_DIR / f"rapm_face_validity_{season}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[rapm_eval] -> {out_path}")
    return out_path
