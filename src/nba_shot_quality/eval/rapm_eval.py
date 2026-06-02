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
    df = _load_rapm_full(season)
    return df[df["def_shots"] >= min_def_shots].copy()


def _load_rapm_full(season: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"rapm_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run rapm --seasons {season} first")
    return pd.read_parquet(path)


def _yoy_panel(ax, merged, metric, shots_col, min_shots, season_a, season_b, label):
    """Draw one YoY scatter (offense or defense) and return (pearson, spearman, n)."""
    sub = merged[(merged[f"{shots_col}_a"] >= min_shots) & (merged[f"{shots_col}_b"] >= min_shots)]
    x, y = sub[f"{metric}_a"].to_numpy(), sub[f"{metric}_b"].to_numpy()
    r_p = _pearson(x, y) if len(sub) > 2 else float("nan")
    r_s = _spearman(sub[f"{metric}_a"], sub[f"{metric}_b"]) if len(sub) > 2 else float("nan")
    ax.scatter(x, y, s=20, alpha=0.6, edgecolor="black", linewidths=0.2)
    if len(sub):
        lim = [min(x.min(), y.min()) - 1, max(x.max(), y.max()) + 1]
        ax.plot(lim, lim, "k--", alpha=0.5, label="y = x")
        for _, r in sub.nlargest(8, f"{metric}_b").iterrows():
            ax.annotate(r["player_name_b"], (r[f"{metric}_a"], r[f"{metric}_b"]),
                        fontsize=7, alpha=0.8, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel(f"{label} RAPM — {season_a}")
    ax.set_ylabel(f"{label} RAPM — {season_b}")
    ax.set_title(f"{label}: Pearson r={r_p:.3f}  Spearman r={r_s:.3f}  n={len(sub):,}")
    ax.legend()
    ax.grid(alpha=0.3)
    return r_p, r_s, len(sub)


def yoy_rapm_stability(season_a: str, season_b: str, min_def_shots: int = 1500) -> Path:
    """Year-over-year RAPM stability, defense vs offense side-by-side (same fits, same players).

    Offense being far more stable than defense is the headline diagnostic: it shows the fit is sound
    and that defense is inherently the noisy side, not a pipeline bug.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    merged = _load_rapm_full(season_a).merge(_load_rapm_full(season_b), on="player_id", suffixes=("_a", "_b"))

    print(f"[rapm_eval] YoY stability, offense vs defense (same fits) — {season_a} vs {season_b}:")
    for thr in (1500, 2500, 3500):
        d = merged[(merged["def_shots_a"] >= thr) & (merged["def_shots_b"] >= thr)]
        o = merged[(merged["off_shots_a"] >= thr) & (merged["off_shots_b"] >= thr)]
        rd = _pearson(d["def_rapm_a"].to_numpy(), d["def_rapm_b"].to_numpy()) if len(d) > 2 else float("nan")
        ro = _pearson(o["off_rapm_a"].to_numpy(), o["off_rapm_b"].to_numpy()) if len(o) > 2 else float("nan")
        print(f"  shots>={thr}:  DEF r={rd:+.3f} (n={len(d)})   |   OFF r={ro:+.3f} (n={len(o)})")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    rd_p, rd_s, nd = _yoy_panel(axes[0], merged, "def_rapm", "def_shots", min_def_shots, season_a, season_b, "Defensive")
    ro_p, ro_s, no = _yoy_panel(axes[1], merged, "off_rapm", "off_shots", min_def_shots, season_a, season_b, "Offensive")
    print(f"[rapm_eval] @>= {min_def_shots} shots:  DEF Pearson r={rd_p:.3f} (n={nd})   OFF Pearson r={ro_p:.3f} (n={no})")
    fig.suptitle(f"Year-over-year RAPM stability — {season_a} vs {season_b}")
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


def rapm_off_vs_poe(season: str, min_shots: int = 1500) -> Path | None:
    """Cross-pipeline check: does offensive RAPM track the independent POE/100 shooter metric?

    A strong positive correlation confirms the RAPM fit recovers real signal (the offensive columns
    re-derive shooting skill estimated by a completely separate pipeline)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    poe_path = PROCESSED_DIR / f"poe_player_season_{season}.parquet"
    if not poe_path.exists():
        print(f"[rapm_eval] skip off_vs_poe: {poe_path} missing (run poe --season {season})")
        return None
    rapm = _load_rapm_full(season)
    poe = pd.read_parquet(poe_path)[["player_id", "poe_per_100"]]
    merged = rapm[rapm["off_shots"] >= min_shots].merge(poe, on="player_id", how="inner").dropna(
        subset=["poe_per_100"]
    )
    x, y = merged["off_rapm"].to_numpy(), merged["poe_per_100"].to_numpy()
    r_p, r_s = _pearson(x, y), _spearman(merged["off_rapm"], merged["poe_per_100"])
    print(f"[rapm_eval] off_rapm vs POE/100 (cross-pipeline)  Pearson r={r_p:.3f}  Spearman r={r_s:.3f}  (n={len(merged):,})")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(x, y, s=20, alpha=0.6, edgecolor="black", linewidths=0.2)
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Offensive RAPM (+ = lifts shot quality)")
    ax.set_ylabel("POE per 100 (independent shooter metric)")
    ax.set_title(f"Offensive RAPM vs POE/100 — {season}\nPearson r={r_p:.3f}  Spearman r={r_s:.3f}  n={len(merged):,}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = REPORTS_DIR / f"rapm_off_vs_poe_{season}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[rapm_eval] -> {out_path}")
    return out_path


def _sb(r: float) -> float:  # Spearman-Brown: half-length r -> full-length reliability
    return 2 * r / (1 + r) if np.isfinite(r) else float("nan")


def splithalf_reliability(season: str, min_def_shots: int = 1500,
                          weighting: str = "uniform", lam: float = 1.0,
                          matchup_source: str = "season") -> dict:
    """Within-season split-half reliability — the measurement-noise ceiling (no roster change).

    Splits the season's games into two random halves, fits RAPM on each at a shared alpha, and
    correlates per-player coefficients across halves. Spearman-Brown adjusts the half-length
    correlation up to the full-season reliability. With weighting="matchup" the defensive design is
    matchup-weighted (matchup_source picks season vs per-game tracking), so this directly measures
    whether matchup attribution lifts the ceiling. Returns {def_r, def_sb, off_r, off_sb, n}.
    """
    from nba_shot_quality.models import rapm as R

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    long = pd.read_parquet(PROCESSED_DIR / f"shot_lineups_{season}.parquet")
    if weighting == "matchup":
        long["weight"] = R.apply_weighting(long, lam, R._load_matchups([season], matchup_source))
    games = np.sort(long["game_id"].unique())
    perm = np.random.default_rng(0).permutation(len(games))
    half_a = set(games[perm[: len(games) // 2]])

    X_full, y_full, g_full, _ = R._build_design(long)
    alpha = R._select_alpha(X_full, y_full, g_full, R.ALPHA_GRID)

    def fit_half(lh: pd.DataFrame) -> pd.DataFrame:
        X, y, _g, players = R._build_design(lh)
        off, deff = R._fit_sides(X, y, alpha)
        sc = lh.groupby(["person_id", "side"]).size().unstack(fill_value=0)
        defsh = sc.get(1, pd.Series(0, index=sc.index)).reindex(players, fill_value=0).to_numpy()
        return pd.DataFrame({"player_id": players, "off": off, "deff": deff, "def_shots": defsh})

    fa = fit_half(long[long["game_id"].isin(half_a)])
    fb = fit_half(long[~long["game_id"].isin(half_a)])
    m = fa.merge(fb, on="player_id", suffixes=("_a", "_b"))
    m = m[(m["def_shots_a"] >= min_def_shots // 2) & (m["def_shots_b"] >= min_def_shots // 2)]

    r_def = _pearson(m["deff_a"].to_numpy(), m["deff_b"].to_numpy())
    r_off = _pearson(m["off_a"].to_numpy(), m["off_b"].to_numpy())
    variant = weighting if weighting != "matchup" else f"matchup/{matchup_source}"
    print(f"[rapm_eval] split-half reliability {season} [{variant}] (alpha={alpha:.0f}, n={len(m):,}):")
    print(f"  DEF  half-half r={r_def:.3f}  ->  full-season reliability (Spearman-Brown) {_sb(r_def):.3f}")
    print(f"  OFF  half-half r={r_off:.3f}  ->  full-season reliability (Spearman-Brown) {_sb(r_off):.3f}")

    from nba_shot_quality.models.rapm import _matchup_suffix
    suffix = _matchup_suffix(weighting, matchup_source)
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(m["deff_a"], m["deff_b"], s=20, alpha=0.6, edgecolor="black", linewidths=0.2, label="defense")
    ax.scatter(m["off_a"], m["off_b"], s=20, alpha=0.4, edgecolor="black", linewidths=0.2, label="offense", color="orange")
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("RAPM — random half A")
    ax.set_ylabel("RAPM — random half B")
    ax.set_title(f"Split-half reliability — {season} [{variant}]\n"
                 f"DEF r={r_def:.3f} (SB {_sb(r_def):.3f})  OFF r={r_off:.3f} (SB {_sb(r_off):.3f})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = REPORTS_DIR / f"rapm_splithalf_{season}{suffix}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[rapm_eval] -> {out_path}")
    return {"def_r": r_def, "def_sb": _sb(r_def), "off_r": r_off, "off_sb": _sb(r_off), "n": len(m)}


def _yoy_def_r(season_a: str, season_b: str, suffix: str, min_def_shots: int) -> float:
    """Pearson r of def_rapm across two seasons for a given weighting suffix ('' or '_matchup')."""
    pa = PROCESSED_DIR / f"rapm_{season_a}{suffix}.parquet"
    pb = PROCESSED_DIR / f"rapm_{season_b}{suffix}.parquet"
    if not (pa.exists() and pb.exists()):
        return float("nan")
    a, b = pd.read_parquet(pa), pd.read_parquet(pb)
    m = a.merge(b, on="player_id", suffixes=("_a", "_b"))
    m = m[(m["def_shots_a"] >= min_def_shots) & (m["def_shots_b"] >= min_def_shots)]
    return _pearson(m["def_rapm_a"].to_numpy(), m["def_rapm_b"].to_numpy()) if len(m) > 2 else float("nan")


def _facevalidity_r(season: str, suffix: str, min_def_shots: int) -> float:
    """Pearson r of def_rapm vs PtDefend suppression for a given weighting suffix."""
    rp = PROCESSED_DIR / f"rapm_{season}{suffix}.parquet"
    pt = RAW_DIR / f"pt_defend_{season}.parquet"
    if not (rp.exists() and pt.exists()):
        return float("nan")
    r = pd.read_parquet(rp)
    r = r[r["def_shots"] >= min_def_shots]
    p = pd.read_parquet(pt)
    p["sup"] = -p["pct_plusminus"]
    m = r.merge(p[["player_id", "sup"]], on="player_id").dropna(subset=["sup"])
    return _pearson(m["def_rapm"].to_numpy(), m["sup"].to_numpy()) if len(m) > 2 else float("nan")


def weighting_compare(season_a: str, season_b: str, min_def_shots: int = 1500, lam: float = 1.0,
                      sources: list[str] | None = None) -> Path:
    """Decision table: does matchup-weighting beat uniform on defensive reliability/stability?

    Compares uniform against each matchup `sources` granularity ("season" = LeagueSeasonMatchups,
    "game" = per-game BoxScoreMatchupsV3). For each variant it runs split-half reliability per season
    plus YoY def_rapm correlation and PtDefend face validity from the (separately fit) per-season
    parquets, then writes a grouped bar chart so the comparison is read at a glance. The matchup
    parquets must already exist for the requested sources (rapm --weighting matchup --matchup-source ...).
    """
    from nba_shot_quality.models.rapm import _matchup_suffix

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    sources = sources or ["season"]
    # (label, weighting, matchup_source, parquet suffix), uniform first as the baseline.
    variants = [("uniform", "uniform", "season", "")]
    variants += [(f"matchup-{src}", "matchup", src, _matchup_suffix("matchup", src)) for src in sources]

    rel = {lbl: {s: splithalf_reliability(s, min_def_shots, weighting=w, lam=lam, matchup_source=src)
                 for s in (season_a, season_b)} for lbl, w, src, _suf in variants}
    yoy = {lbl: _yoy_def_r(season_a, season_b, suf, min_def_shots) for lbl, _w, _src, suf in variants}
    face = {lbl: _facevalidity_r(season_b, suf, min_def_shots) for lbl, _w, _src, suf in variants}

    labels_v = [v[0] for v in variants]
    print("\n[rapm_eval] ===== WEIGHTING COMPARISON (defense) =====")
    header = f"  {'metric':<34}" + "".join(f"{lbl:>14}" for lbl in labels_v)
    print(header)
    for s in (season_a, season_b):
        print(f"  {'split-half reliability '+s:<34}" + "".join(f"{rel[lbl][s]['def_sb']:>14.3f}" for lbl in labels_v))
    print(f"  {'YoY def_rapm '+season_a+'->'+season_b:<34}" + "".join(f"{yoy[lbl]:>14.3f}" for lbl in labels_v))
    print(f"  {'face validity vs PtDefend '+season_b:<34}" + "".join(f"{face[lbl]:>14.3f}" for lbl in labels_v))

    metrics = [f"reliab\n{season_a}", f"reliab\n{season_b}", f"YoY\n{season_a[-5:]}->{season_b[-5:]}", f"face\n{season_b}"]
    series = {lbl: [rel[lbl][season_a]["def_sb"], rel[lbl][season_b]["def_sb"], yoy[lbl], face[lbl]] for lbl in labels_v}
    x = np.arange(len(metrics))
    n = len(labels_v)
    width = 0.8 / n
    colors = ["steelblue", "darkorange", "seagreen", "crimson"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, lbl in enumerate(labels_v):
        off = (i - (n - 1) / 2) * width
        vals = series[lbl]
        ax.bar(x + off, vals, width, label=lbl, color=colors[i % len(colors)])
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi + off, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Pearson r (defense)")
    ax.set_title(f"Defensive RAPM: uniform vs matchup-weighted (lam={lam})")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_path = REPORTS_DIR / "rapm_weighting_compare.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[rapm_eval] -> {out_path}")
    return out_path


def ptdefend_yoy(season_a: str, season_b: str, min_d_fga: int = 300) -> float | None:
    """Benchmark: YoY stability of nba_api's tracking defended-FG suppression metric.

    Contextualizes our def_rapm YoY number — even the closest-defender tracking stat tops out
    around r≈0.45-0.56, showing shot-defense is a noisy domain regardless of method.
    """
    pa_path, pb_path = RAW_DIR / f"pt_defend_{season_a}.parquet", RAW_DIR / f"pt_defend_{season_b}.parquet"
    if not (pa_path.exists() and pb_path.exists()):
        print(f"[rapm_eval] skip ptdefend_yoy: missing pt_defend parquet(s) (run ingest-def)")
        return None
    m = pd.read_parquet(pa_path).merge(pd.read_parquet(pb_path), on="player_id", suffixes=("_a", "_b"))
    m = m[(m["d_fga_a"] >= min_d_fga) & (m["d_fga_b"] >= min_d_fga)]
    x, y = (-m["pct_plusminus_a"]).to_numpy(), (-m["pct_plusminus_b"]).to_numpy()
    r_p, r_s = _pearson(x, y), _spearman(-m["pct_plusminus_a"], -m["pct_plusminus_b"])
    print(f"[rapm_eval] BENCHMARK PtDefend suppression YoY  Pearson r={r_p:.3f}  Spearman r={r_s:.3f}  (n={len(m):,}, d_fga>={min_d_fga})")
    return r_p
