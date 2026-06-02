"""Layer 4 — coaching-change difference-in-differences (DiD) event study.

Question: when a team fires its head coach mid-season, does its defense actually improve, or is the
post-firing "bounce" just regression to the mean? We compare a firing team's allowed efficiency
before vs after the change, *netted against* the rest of the league over the same calendar window
(DiD), so league-wide trends don't masquerade as a coaching effect.

Pipeline:
  build_def_panel(season)  -> per-(game, defending team) xPoints/points/possessions allowed.
  run_coaching_study(...)  -> per-event DiD across 10/20/30-game windows and three outcome metrics,
                              event-clustered + game-level bootstrap CIs, and a pre-trend /
                              mean-reversion diagnostic + event-study plot.

Honest framing: only a handful of *true mid-season* firings exist across the cached seasons, so the
event-clustered CI is wide and likely spans zero. The deliverable is a clearly-stated estimate with
a CI and the pre-trend caveat — not a "firings work / don't work" verdict. Sign convention
throughout: **negative DiD = defense improved** (allowed efficiency fell beyond league drift).

Outcome metrics (all "pool-then-ratio": 100 * sum(numerator) / sum(denominator) over a window):
  xpts_100poss  — allowed xPoints per 100 possessions   (PRIMARY; shot-quality, possession-paced)
  pts_100poss   — allowed actual points per 100 poss     (true defensive rating; robustness)
  xpts_100shots — allowed xPoints per 100 shots faced     (pure shot-quality lens; robustness)
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
CHANGES_CSV = Path(__file__).resolve().parent / "coaching_changes.csv"

SEASONS = ["2022-23", "2023-24", "2024-25"]
W_MIN = 6  # minimum games required on each side of a change for an event to count at a given window

# metric key -> (numerator column, denominator column, human label, per-100 unit)
METRICS = {
    "xpts_100poss": ("xpoints_allowed", "def_possessions", "Allowed xPoints / 100 poss"),
    "pts_100poss": ("pts_allowed", "def_possessions", "Allowed points / 100 poss (Drtg)"),
    "xpts_100shots": ("xpoints_allowed", "def_fga_faced", "Allowed xPoints / 100 shots"),
}


# --------------------------------------------------------------------------------------------------
# A. Per-team-per-game defense panel
# --------------------------------------------------------------------------------------------------
def build_def_panel(season: str) -> Path:
    """Build one row per (game_id, defending team) of what that team ALLOWED.

    xPoints/points/FGA allowed come from shots_scored (each game has exactly 2 teams, so a defender's
    allowed total = game total - its own offense). Possessions and ground-truth points allowed come
    from the team game logs (same self-flip). Writes data/processed/def_game_panel_{season}.parquet.
    """
    scored_path = PROCESSED_DIR / f"shots_scored_{season}.parquet"
    logs_path = RAW_DIR / f"team_game_logs_{season}.parquet"
    if not scored_path.exists():
        raise FileNotFoundError(f"{scored_path} not found — run score --season {season} first")
    if not logs_path.exists():
        raise FileNotFoundError(f"{logs_path} not found — run ingest-team-logs --season {season} first")

    shots = pd.read_parquet(scored_path, columns=["game_id", "game_date", "team_id", "xpoints", "points"])
    shots["game_id"] = shots["game_id"].astype(str).str.zfill(10)

    # Offense per (game, team), then game totals; allowed = game total - own offense.
    off = shots.groupby(["game_id", "team_id"]).agg(
        off_xpoints=("xpoints", "sum"), off_points=("points", "sum"), off_fga=("xpoints", "size"),
        game_date=("game_date", "first"),
    ).reset_index()
    tot = off.groupby("game_id")[["off_xpoints", "off_points", "off_fga"]].transform("sum")
    panel = pd.DataFrame({
        "season": season,
        "game_id": off["game_id"],
        "game_date": off["game_date"],
        "def_team_id": off["team_id"].astype("int64"),
        "xpoints_allowed": tot["off_xpoints"] - off["off_xpoints"],
        "points_allowed_fg": tot["off_points"] - off["off_points"],  # FG-only; FT-exclusive cross-check
        "def_fga_faced": (tot["off_fga"] - off["off_fga"]).astype("float64"),
    })

    # Possessions + ground-truth points allowed from the game logs (same self-flip).
    logs = pd.read_parquet(logs_path, columns=["game_id", "team_id", "pts", "possessions"])
    logs["game_id"] = logs["game_id"].astype(str).str.zfill(10)
    ltot = logs.groupby("game_id")[["pts", "possessions"]].transform("sum")
    logs_def = pd.DataFrame({
        "game_id": logs["game_id"],
        "def_team_id": logs["team_id"].astype("int64"),
        "pts_allowed": ltot["pts"] - logs["pts"],
        "def_possessions": ltot["possessions"] - logs["possessions"],
    })

    n_before = len(panel)
    panel = panel.merge(logs_def, on=["game_id", "def_team_id"], how="left")
    unmatched = int(panel["def_possessions"].isna().sum())
    if unmatched > 0.01 * n_before:
        raise ValueError(f"[def_panel] {unmatched}/{n_before} panel rows missing game-log possessions "
                         f"(>1%) — team_game_logs_{season} likely incomplete")
    panel = panel.dropna(subset=["def_possessions"])
    if (panel["def_possessions"] <= 0).any():
        raise ValueError("[def_panel] non-positive possessions encountered")

    # Per-game ratio columns for diagnostics/plots only (analysis pools numerators, see _pool_rate).
    panel["allowed_xpts_100poss"] = 100 * panel["xpoints_allowed"] / panel["def_possessions"]
    panel["allowed_pts_100poss"] = 100 * panel["pts_allowed"] / panel["def_possessions"]
    panel["allowed_xpts_100shots"] = 100 * panel["xpoints_allowed"] / panel["def_fga_faced"]

    out_path = PROCESSED_DIR / f"def_game_panel_{season}.parquet"
    panel.to_parquet(out_path, index=False)

    g = panel.groupby("game_id").size()
    print(f"[def_panel] {season}: {len(panel):,} team-games, {g.size:,} games, {unmatched} unmatched")
    print(f"  league-avg allowed pts/100poss (Drtg) : {_pool_rate(panel, 'pts_allowed', 'def_possessions'):.1f}  (expect ~113)")
    print(f"  league-avg allowed xPts/100poss        : {_pool_rate(panel, 'xpoints_allowed', 'def_possessions'):.1f}  (lower: FT-excluded)")
    print(f"  league-avg allowed xPts/100shots       : {_pool_rate(panel, 'xpoints_allowed', 'def_fga_faced'):.1f}  (expect ~108-109)")
    if not (g == 2).all():
        print(f"  WARNING: {(g != 2).sum()} games without exactly 2 defending rows")
    print(f"[def_panel] -> {out_path}")
    return out_path


def load_def_panel(season: str, rebuild: bool = True) -> pd.DataFrame:
    path = PROCESSED_DIR / f"def_game_panel_{season}.parquet"
    if rebuild or not path.exists():
        build_def_panel(season)
    df = pd.read_parquet(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


# --------------------------------------------------------------------------------------------------
# Curated coaching changes
# --------------------------------------------------------------------------------------------------
def load_coaching_changes() -> pd.DataFrame:
    """Read the curated in-season coaching-change table (one row per change)."""
    if not CHANGES_CSV.exists():
        raise FileNotFoundError(f"{CHANGES_CSV} not found — the curated coaching-changes table is missing")
    df = pd.read_csv(CHANGES_CSV, comment="#")
    df["change_date"] = pd.to_datetime(df["change_date"])
    df["team_id"] = df["team_id"].astype("int64")
    # Keep only the first change per (season, team) for the headline; flag any dropped.
    df = df.sort_values("change_date")
    dup = df.duplicated(subset=["season", "team_id"], keep="first")
    if dup.any():
        for _, r in df[dup].iterrows():
            print(f"[coaching] dropping 2nd same-season change: {r['season']} {r['team_abbr']} {r['change_date'].date()}")
    return df[~dup].reset_index(drop=True)


# --------------------------------------------------------------------------------------------------
# B. DiD estimator
# --------------------------------------------------------------------------------------------------
def _pool_rate(df: pd.DataFrame, num: str, den: str) -> float:
    """Possession-weighted (pool-then-ratio) rate over a set of team-games."""
    d = df[den].sum()
    return float(100.0 * df[num].sum() / d) if d > 0 else float("nan")


def _event_cells(panel: pd.DataFrame, team_id: int, change_date, W: int,
                 control_team_ids: set[int]) -> dict | None:
    """The 4 DiD cells for one event at window W, calendar-aligned. Returns None if invalid.

    Treated = team_id's last W games before / first W games on-or-after change_date. Control = the
    same calendar window for all never-treated-that-season teams. Requires >= W_MIN games per side.
    """
    t = panel[panel["def_team_id"] == team_id].sort_values("game_date")
    pre = t[t["game_date"] < change_date].tail(W)
    post = t[t["game_date"] >= change_date].head(W)
    if len(pre) < W_MIN or len(post) < W_MIN:
        return None

    pre_start, post_end = pre["game_date"].min(), post["game_date"].max()
    ctrl = panel[panel["def_team_id"].isin(control_team_ids)]
    ctrl_pre = ctrl[(ctrl["game_date"] >= pre_start) & (ctrl["game_date"] < change_date)]
    ctrl_post = ctrl[(ctrl["game_date"] >= change_date) & (ctrl["game_date"] <= post_end)]
    return {
        "treated_pre": pre, "treated_post": post,
        "control_pre": ctrl_pre, "control_post": ctrl_post,
        "n_pre": len(pre), "n_post": len(post), "n_ctrl_teams": len(control_team_ids),
    }


def _did_from_cells(cells: dict, num: str, den: str) -> dict:
    """2x2 DiD for one metric: (treated_post - treated_pre) - (control_post - control_pre)."""
    tp = _pool_rate(cells["treated_pre"], num, den)
    tq = _pool_rate(cells["treated_post"], num, den)
    cp = _pool_rate(cells["control_pre"], num, den)
    cq = _pool_rate(cells["control_post"], num, den)
    return {"treated_pre": tp, "treated_post": tq, "control_pre": cp, "control_post": cq,
            "treated_delta": tq - tp, "control_delta": cq - cp, "did": (tq - tp) - (cq - cp)}


def _events_for_window(panel: pd.DataFrame, changes: pd.DataFrame, W: int, verbose: bool = False) -> list[dict]:
    """Build valid event cells for every change at window W (per season, with clean controls).

    Called repeatedly (per metric/bootstrap), so skip messages only print when verbose=True (once,
    from the orchestrator) to avoid flooding the log.
    """
    events = []
    for season, sch in changes.groupby("season"):
        sp = panel[panel["season"] == season]
        treated_ids = set(sch["team_id"])
        control_ids = set(sp["def_team_id"].unique()) - treated_ids
        for _, r in sch.iterrows():
            cells = _event_cells(sp, int(r["team_id"]), r["change_date"], W, control_ids)
            if cells is None:
                if verbose:
                    print(f"[coaching]   W={W}: skip {season} {r['team_abbr']} (only "
                          f"<{W_MIN} games one side)")
                continue
            events.append({"meta": r, "cells": cells})
    return events


def compute_did_table(panel: pd.DataFrame, changes: pd.DataFrame, W: int, metric: str) -> pd.DataFrame:
    """Per-event DiD table (one row per valid event) for a given window and metric."""
    num, den, _ = METRICS[metric]
    rows = []
    for ev in _events_for_window(panel, changes, W):
        r, c = ev["meta"], ev["cells"]
        d = _did_from_cells(c, num, den)
        rows.append({
            "season": r["season"], "team_abbr": r["team_abbr"], "coach_out": r["coach_out"],
            "change_date": r["change_date"].date(), "window": W, "metric": metric,
            "n_pre": c["n_pre"], "n_post": c["n_post"], **d,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# C. Bootstrap CIs
# --------------------------------------------------------------------------------------------------
def _percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    s = samples[np.isfinite(samples)]
    if len(s) < 2:
        return float("nan"), float("nan")
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def bootstrap_event_ci(per_event_did: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    """PRIMARY (design-honest): resample EVENTS with replacement (cluster bootstrap). Wide at small N."""
    n = len(per_event_did)
    if n < 2:
        return float("nan"), float("nan")
    draws = np.array([np.mean(per_event_did[rng.integers(0, n, n)]) for _ in range(n_boot)])
    return _percentile_ci(draws)


def bootstrap_game_ci(panel: pd.DataFrame, changes: pd.DataFrame, W: int, metric: str,
                      rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    """SECONDARY (events fixed): resample games WITHIN each cell with replacement; narrower."""
    num, den, _ = METRICS[metric]
    events = _events_for_window(panel, changes, W)
    if len(events) < 2:
        return float("nan"), float("nan")
    cell_keys = ["treated_pre", "treated_post", "control_pre", "control_post"]
    draws = np.empty(n_boot)
    for b in range(n_boot):
        dids = []
        for ev in events:
            rs = {k: ev["cells"][k].sample(frac=1.0, replace=True, random_state=int(rng.integers(0, 2**31)))
                  for k in cell_keys}
            dids.append(_did_from_cells(rs, num, den)["did"])
        draws[b] = np.mean(dids)
    return _percentile_ci(draws)


# --------------------------------------------------------------------------------------------------
# D. Pre-trend / mean-reversion diagnostic
# --------------------------------------------------------------------------------------------------
def _slope_per_day(df: pd.DataFrame, ratio_col: str, origin) -> float:
    """OLS slope of a per-game ratio vs days-since-origin (metric units per day)."""
    if len(df) < 2:
        return float("nan")
    x = (df["game_date"] - origin).dt.total_seconds().to_numpy() / 86400.0
    y = df[ratio_col].to_numpy()
    if np.ptp(x) == 0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def pretrend_diagnostic(panel: pd.DataFrame, changes: pd.DataFrame, W: int, metric: str,
                        rng: np.random.Generator, n_boot: int) -> dict:
    """Excess pre-trend slope (treated minus control) over each event's PRE window.

    Positive excess slope = treated defenses worsening faster than the league right before the firing
    = mean-reversion risk (a post-firing improvement could be regression to the mean, not coaching).
    """
    ratio_col = {"xpts_100poss": "allowed_xpts_100poss", "pts_100poss": "allowed_pts_100poss",
                 "xpts_100shots": "allowed_xpts_100shots"}[metric]
    excess = []
    for ev in _events_for_window(panel, changes, W):
        c = ev["cells"]
        origin = c["treated_pre"]["game_date"].min()
        b_t = _slope_per_day(c["treated_pre"], ratio_col, origin)
        b_c = _slope_per_day(c["control_pre"], ratio_col, origin)
        if np.isfinite(b_t) and np.isfinite(b_c):
            excess.append(b_t - b_c)
    excess = np.array(excess)
    lo, hi = bootstrap_event_ci(excess, rng, n_boot) if len(excess) else (float("nan"), float("nan"))
    return {"mean_excess_slope_per_day": float(np.mean(excess)) if len(excess) else float("nan"),
            "ci_low": lo, "ci_high": hi, "n": len(excess)}


# --------------------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------------------
def _event_study_plot(panel: pd.DataFrame, changes: pd.DataFrame, W: int, metric: str, out_path: Path) -> None:
    """Leads/lags: per-event de-meaned treated metric by game index around the firing, averaged."""
    ratio_col = {"xpts_100poss": "allowed_xpts_100poss", "pts_100poss": "allowed_pts_100poss",
                 "xpts_100shots": "allowed_xpts_100shots"}[metric]
    label = METRICS[metric][2]
    rel_pre = {k: [] for k in range(-W, 0)}
    rel_post = {k: [] for k in range(1, W + 1)}
    ctrl_shift = []
    for ev in _events_for_window(panel, changes, W):
        c = ev["cells"]
        pre, post = c["treated_pre"], c["treated_post"]
        base = pre[ratio_col].mean()  # de-mean per event so events are comparable
        pv = pre[ratio_col].to_numpy() - base
        for i, v in enumerate(pv):  # last value is the game just before the change -> index -1
            rel_pre[-(len(pv) - i)].append(v)
        qv = post[ratio_col].to_numpy() - base
        for i, v in enumerate(qv):
            rel_post[i + 1].append(v)
        ctrl_shift.append(c["control_post"][ratio_col].mean() - c["control_pre"][ratio_col].mean())

    xs = list(range(-W, 0)) + list(range(1, W + 1))
    means = [np.mean(rel_pre[k]) if rel_pre.get(k) else np.nan for k in range(-W, 0)]
    means += [np.mean(rel_post[k]) if rel_post.get(k) else np.nan for k in range(1, W + 1)]

    # Centered rolling mean to reveal the leads/lags trend through the per-game-index noise (N events
    # is small, so the raw event-average is jumpy). Plotted within pre and post separately so the
    # smoother doesn't bridge across the change.
    means_arr = np.array(means, dtype=float)
    split = W  # first `W` entries are pre (-W..-1), rest are post
    def _roll(a, k=5):
        s = pd.Series(a)
        return s.rolling(k, center=True, min_periods=1).mean().to_numpy()
    smooth = np.concatenate([_roll(means_arr[:split]), _roll(means_arr[split:])])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvline(0, color="crimson", lw=1.2, ls="--", label="coaching change")
    ax.axhline(0, color="grey", lw=0.8)
    ax.plot(xs, means, "o", ms=3, color="steelblue", alpha=0.35, label="treated per-game (event-avg)")
    ax.plot(xs[:split], smooth[:split], "-", lw=2.2, color="steelblue", label="treated 5-game rolling mean")
    ax.plot(xs[split:], smooth[split:], "-", lw=2.2, color="steelblue")
    if ctrl_shift:
        ax.axhline(float(np.mean(ctrl_shift)), color="darkorange", lw=1.0, ls=":",
                   label="control post-pre shift (league drift)")
    ax.set_xlabel("game relative to coaching change")
    ax.set_ylabel(f"{label}\n(de-meaned vs pre-window)")
    ax.set_title(f"Coaching-change event study (W={W}, {label})\nlook for: declining-before-0 = mean reversion")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[coaching] -> {out_path}")


def _did_summary_plot(summary: pd.DataFrame, out_path: Path) -> None:
    """Pooled DiD by window x metric with event-clustered 95% CI error bars."""
    metrics = list(METRICS.keys())
    windows = sorted(summary["window"].unique())
    x = np.arange(len(windows))
    width = 0.8 / len(metrics)
    colors = ["steelblue", "darkorange", "seagreen"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(metrics):
        sub = summary[summary["metric"] == m].set_index("window").reindex(windows)
        vals = sub["pooled_did"].to_numpy()
        lo = vals - sub["event_ci_low"].to_numpy()
        hi = sub["event_ci_high"].to_numpy() - vals
        ax.bar(x + (i - 1) * width, vals, width, yerr=[lo, hi], capsize=3,
               label=METRICS[m][2], color=colors[i % len(colors)])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"W={w}" for w in windows])
    ax.set_ylabel("Pooled DiD  (negative = defense improved)")
    ax.set_title("Coaching-change DiD by window & metric (event-clustered 95% CI)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[coaching] -> {out_path}")


# --------------------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------------------
def run_coaching_study(windows: tuple[int, ...] = (10, 20, 30), n_boot: int = 1000,
                       metric: str = "xpts_100poss") -> Path:
    """Full DiD study across all cached seasons: per-event tables, pooled DiD + CIs, pre-trend, plots."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.concat([load_def_panel(s) for s in SEASONS], ignore_index=True)
    changes = load_coaching_changes()

    print("\n[coaching] ===== curated in-season coaching changes =====")
    for season, sch in changes.groupby("season"):
        print(f"  {season}: {len(sch)} change(s) — " + ", ".join(f"{r.team_abbr}({r.change_date.date()})" for r in sch.itertuples()))
    print(f"  TOTAL events in table: {len(changes)}  (true mid-season firings; most NBA turnover is offseason)")
    print(f"[coaching] headline metric: {metric} ({METRICS[metric][2]});  sign: NEGATIVE DiD = defense IMPROVED")

    for W in windows:  # report dropped events once per window (the loops below run silently)
        _events_for_window(panel, changes, W, verbose=True)

    rng = np.random.default_rng(0)
    per_event_rows, summary_rows = [], []
    for W in windows:
        for m in METRICS:
            tbl = compute_did_table(panel, changes, W, m)
            if tbl.empty:
                continue
            per_event_rows.append(tbl)
            dids = tbl["did"].to_numpy()
            ev_lo, ev_hi = bootstrap_event_ci(dids, rng, n_boot)
            gm_lo, gm_hi = bootstrap_game_ci(panel, changes, W, m, rng, n_boot)
            summary_rows.append({
                "window": W, "metric": m, "n_events": len(tbl), "pooled_did": float(np.mean(dids)),
                "event_ci_low": ev_lo, "event_ci_high": ev_hi, "game_ci_low": gm_lo, "game_ci_high": gm_hi,
            })

    per_event = pd.concat(per_event_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    # Per-event detail for the headline metric across windows.
    print(f"\n[coaching] ===== per-event DiD ({METRICS[metric][2]}) =====")
    hdr = per_event[per_event["metric"] == metric]
    for W in windows:
        sub = hdr[hdr["window"] == W]
        if sub.empty:
            continue
        print(f"  --- window W={W} ({len(sub)} events) ---")
        for _, r in sub.iterrows():
            print(f"    {r['season']} {r['team_abbr']:<4} ({r['coach_out']:<16} out {r['change_date']}, "
                  f"n={int(r['n_pre'])}/{int(r['n_post'])}):  "
                  f"treated Δ{r['treated_delta']:+5.1f}  control Δ{r['control_delta']:+5.1f}  -> DiD {r['did']:+5.1f}")

    print(f"\n[coaching] ===== pooled DiD (mean of per-event), all metrics =====")
    print(f"  {'metric':<28}{'W':>4}{'n':>4}{'pooledDiD':>11}{'  event-cluster 95% CI':>26}{'  game-level 95% CI':>24}")
    for _, r in summary.iterrows():
        print(f"  {r['metric']:<28}{int(r['window']):>4}{int(r['n_events']):>4}{r['pooled_did']:>11.2f}"
              f"   [{r['event_ci_low']:+6.2f},{r['event_ci_high']:+6.2f}]   [{r['game_ci_low']:+6.2f},{r['game_ci_high']:+6.2f}]")

    print(f"\n[coaching] ===== pre-trend / mean-reversion diagnostic ({METRICS[metric][2]}) =====")
    for W in windows:
        pt = pretrend_diagnostic(panel, changes, W, metric, rng, n_boot)
        flag = "  <-- treated worsening pre-firing (mean-reversion risk)" if pt["mean_excess_slope_per_day"] > 0 else ""
        print(f"  W={W}: mean excess pre-trend slope = {pt['mean_excess_slope_per_day']:+.3f}/day "
              f"[{pt['ci_low']:+.3f}, {pt['ci_high']:+.3f}]  (n={pt['n']}){flag}")

    headline_W = max(w for w in windows if not hdr[hdr["window"] == w].empty) if not hdr.empty else windows[0]
    _event_study_plot(panel, changes, headline_W, metric, REPORTS_DIR / "coaching_event_study.png")
    _did_summary_plot(summary, REPORTS_DIR / "coaching_did_summary.png")

    out_path = PROCESSED_DIR / "coaching_did_results.parquet"
    per_event.to_parquet(out_path, index=False)
    summary.to_parquet(PROCESSED_DIR / "coaching_did_summary.parquet", index=False)
    print(f"\n[coaching] {len(changes)} events, windows {windows}, n_boot={n_boot}")
    print(f"[coaching] -> {out_path}")
    print("[coaching] NOTE: with so few mid-season firings the event-clustered CI is wide and likely "
          "spans zero — read this as directional evidence, not a significant causal effect.")
    return out_path
