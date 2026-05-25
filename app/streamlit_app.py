"""Streamlit dashboard: player shot map colored by POE + POE leaderboard with CIs.

Consumes out-of-fold scored shots and the per-player-season POE table
(with bootstrap CIs and true TS%/rTS%) rather than scoring in-sample on the fly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"
SEASONS = ["2024-25", "2023-24", "2022-23"]

LEADERBOARD_COLS = [
    "player_name", "attempts", "points", "xpoints", "poe", "poe_per_100",
    "poe_ci_low", "poe_ci_high", "pps", "xpps", "efg_pct", "ts_pct", "rel_ts_pct",
]
ROUND_MAP = {
    "xpoints": 1, "poe": 1, "poe_per_100": 2, "poe_ci_low": 2, "poe_ci_high": 2,
    "pps": 3, "xpps": 3, "efg_pct": 3, "ts_pct": 3, "rel_ts_pct": 3,
}


@st.cache_data(show_spinner=False)
def load_scored_shots(season: str) -> pd.DataFrame | None:
    path = PROCESSED_DIR / f"shots_scored_{season}.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_leaderboard(season: str) -> pd.DataFrame | None:
    path = PROCESSED_DIR / f"poe_player_season_{season}.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_rapm() -> tuple[pd.DataFrame, str] | None:
    """Best available RAPM table: prefer pooled, else fall back to the newest per-season file.

    Returns (table, source) where source is "pooled" or a season tag; None if nothing exists.
    """
    pooled = PROCESSED_DIR / "rapm_pooled.parquet"
    if pooled.exists():
        return pd.read_parquet(pooled), "pooled"
    singles = sorted(p for p in PROCESSED_DIR.glob("rapm_*.parquet") if p.name != "rapm_pooled.parquet")
    if singles:
        df = pd.read_parquet(singles[-1])
        source = str(df["season"].iloc[0]) if len(df) else singles[-1].stem.removeprefix("rapm_")
        return df, source
    return None


# label -> (value col, ci-low col, ci-high col, shot-volume col used for the min-shots filter)
RAPM_METRICS = {
    "Defense (+ = suppresses scoring)": ("def_rapm", "def_ci_low", "def_ci_high", "def_shots"),
    "Offense (+ = lifts shot quality)": ("off_rapm", "off_ci_low", "off_ci_high", "off_shots"),
    "Net (offense + defense)": ("net_rapm", "net_ci_low", "net_ci_high", "def_shots"),
}


def draw_court(ax) -> None:
    """Half-court outline (NBA dimensions in feet, basket at origin)."""
    ax.add_patch(mpatches.Circle((0, 0), 0.75, lw=2, fill=False, color="black"))
    ax.add_patch(mpatches.Rectangle((-3, -0.75), 6, 0.1, color="black"))
    ax.add_patch(mpatches.Rectangle((-8, -4), 16, 19, lw=2, fill=False, color="black"))
    ax.add_patch(mpatches.Rectangle((-6, -4), 12, 19, lw=2, fill=False, color="black"))
    ax.add_patch(mpatches.Arc((0, 15), 12, 12, theta1=0, theta2=180, lw=2, color="black"))
    ax.add_patch(mpatches.Arc((0, 15), 12, 12, theta1=180, theta2=360, lw=2, ls="--", color="black"))
    ax.add_patch(mpatches.Arc((0, 0), 8, 8, theta1=0, theta2=180, lw=2, color="black"))
    ax.add_patch(mpatches.Rectangle((-22, -4), 0, 14, lw=2, color="black"))
    ax.add_patch(mpatches.Rectangle((22, -4), 0, 14, lw=2, color="black"))
    ax.add_patch(mpatches.Arc((0, 0), 47.5, 47.5, theta1=22, theta2=158, lw=2, color="black"))
    ax.add_patch(mpatches.Rectangle((-25, -4), 50, 51, lw=2, fill=False, color="black"))


def render_shot_map(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(7, 6.5))
    draw_court(ax)
    if len(df):
        scatter = ax.scatter(
            df["loc_x_ft"], df["loc_y_ft"], c=df["poe"], cmap="RdYlGn",
            vmin=-2, vmax=2, s=18, alpha=0.75, edgecolor="black", linewidths=0.2,
        )
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("POE (points over expected)")
    ax.set_xlim(-27, 27)
    ax.set_ylim(-6, 49)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="NBA Shot Quality", layout="wide")
    st.title("NBA Shot Quality — xPoints & POE")
    st.caption(
        "POE = actual points − expected points, from an out-of-fold LightGBM make-probability model. "
        "TS%/rTS% are FT-inclusive league references (different shot universe than POE)."
    )

    season = st.sidebar.selectbox("Season", SEASONS)
    min_attempts = st.sidebar.slider("Min shot attempts (leaderboard)", 200, 1500, 400, step=50)

    shots = load_scored_shots(season)
    board = load_leaderboard(season)
    if shots is None or board is None:
        st.warning(
            f"Missing POE artifacts for {season}. Run `bash scripts/run_poe.sh` "
            "(or the `score` and `poe` CLI commands) for this season first."
        )
        return

    st.sidebar.write(f"**Total shots:** {len(shots):,}")
    st.sidebar.write(f"**Players (≥200 FGA):** {len(board):,}")

    qualified = board[board["attempts"] >= min_attempts].sort_values("poe_per_100", ascending=False)

    tab_map, tab_top, tab_bottom, tab_stability, tab_def = st.tabs(
        ["Player shot map", "Top 20 POE", "Bottom 20 POE", "Stability", "RAPM impact"]
    )

    with tab_map:
        names = qualified["player_name"].tolist()
        if not names:
            st.warning(f"No players have ≥{min_attempts} attempts at the current threshold.")
        else:
            default_idx = names.index("Stephen Curry") if "Stephen Curry" in names else 0
            player = st.selectbox("Player", names, index=default_idx)
            stats = qualified[qualified["player_name"] == player].iloc[0]
            pdf = shots[shots["player_id"] == stats["player_id"]]
            col_map, col_stats = st.columns([3, 1])
            with col_map:
                fig = render_shot_map(pdf, f"{player} — {season} ({len(pdf)} shots)")
                st.pyplot(fig)
            with col_stats:
                st.metric("Attempts", f"{int(stats['attempts']):,}")
                st.metric("Points", f"{int(stats['points']):,}")
                st.metric("POE per 100", f"{stats['poe_per_100']:+.2f}")
                st.caption(f"95% CI: [{stats['poe_ci_low']:+.2f}, {stats['poe_ci_high']:+.2f}]")
                st.metric("POE (total)", f"{stats['poe']:+.0f}")
                st.metric("PPS / xPPS", f"{stats['pps']:.3f} / {stats['xpps']:.3f}")
                if pd.notna(stats["ts_pct"]):
                    st.metric("TS% (rTS%)", f"{stats['ts_pct']:.3f} ({stats['rel_ts_pct']:+.3f})")

    with tab_top:
        st.subheader(f"Top 20 by POE per 100 (min {min_attempts} attempts)")
        st.dataframe(
            qualified.head(20)[LEADERBOARD_COLS].round(ROUND_MAP),
            hide_index=True, use_container_width=True,
        )

    with tab_bottom:
        st.subheader(f"Bottom 20 by POE per 100 (min {min_attempts} attempts)")
        st.dataframe(
            qualified.tail(20).iloc[::-1][LEADERBOARD_COLS].round(ROUND_MAP),
            hide_index=True, use_container_width=True,
        )

    with tab_stability:
        st.subheader("Is POE real skill?")
        yoy = sorted(REPORTS_DIR.glob("poe_stability_*.png"))
        pvr = REPORTS_DIR / f"poe_vs_rts_{season}.png"
        if yoy:
            st.markdown("**Year-over-year POE correlation** — a stable diagonal means POE persists across seasons.")
            st.image(str(yoy[-1]), use_container_width=True)
        else:
            st.info("Run `stability` to generate the year-over-year plot.")
        if pvr.exists():
            st.markdown("**POE vs relative TS%** — points off the trend are where difficulty-adjustment matters most.")
            st.image(str(pvr), use_container_width=True)
        else:
            st.info(f"Run `poe-vs-rts --season {season}` to generate the comparison plot.")

    with tab_def:
        st.subheader("Player impact — ridge RAPM")
        st.caption(
            "Coefficient on per-shot POE, per 100 shots. Defense + = suppresses opponent scoring; "
            "offense + = lifts shot quality; net = offense + defense. On-floor attribution (credit "
            "shared by all 5) and FGA-only — shot impact, not total value. Offense validates at "
            "r≈0.73 vs the independent POE metric; defense is noisier (within-season reliability ~0.2)."
        )
        loaded = load_rapm()
        if loaded is None:
            st.info("Run `bash scripts/run_rapm.sh` to generate RAPM ratings.")
        else:
            rapm, source = loaded
            # Back-compat for older parquet schemas without off/net CIs.
            if "net_rapm" not in rapm.columns:
                rapm = rapm.assign(net_rapm=rapm["off_rapm"] + rapm["def_rapm"])
            for c in ("off_ci_low", "off_ci_high", "net_ci_low", "net_ci_high"):
                if c not in rapm.columns:
                    rapm[c] = float("nan")
            if source != "pooled":
                st.warning(f"⚠️ Limited data — single-season partial sample (`{source}`); rankings may not be meaningful.")

            metric_label = st.radio("Metric", list(RAPM_METRICS), horizontal=True)
            val_col, lo_col, hi_col, shots_col = RAPM_METRICS[metric_label]
            max_shots = int(rapm[shots_col].max()) if len(rapm) else 0
            slider_max = max(500, max_shots)
            step = 250 if slider_max > 1500 else 25
            default = ((min(1500, max_shots // 2)) // step) * step
            ctrl1, ctrl2 = st.columns(2)
            with ctrl1:
                min_shots = st.slider(f"Min {shots_col.replace('_', ' ')}", 0, slider_max, default, step=step)
            with ctrl2:
                query = st.text_input("Search player name", "")
            show_cols = ["player_name", val_col, lo_col, hi_col, shots_col]
            ranked = rapm[rapm[shots_col] >= min_shots].sort_values(val_col, ascending=False)
            if query:
                hits = ranked[ranked["player_name"].str.contains(query, case=False, na=False)]
                st.markdown(f"**Search '{query}'** — {len(hits)} match")
                st.dataframe(hits[show_cols].round(2), hide_index=True, use_container_width=True)
            else:
                c_top, c_bot = st.columns(2)
                with c_top:
                    st.markdown(f"**Top 15** (n={len(ranked)})")
                    st.dataframe(ranked.head(15)[show_cols].round(2), hide_index=True, use_container_width=True)
                with c_bot:
                    st.markdown("**Bottom 15**")
                    st.dataframe(ranked.tail(15).iloc[::-1][show_cols].round(2), hide_index=True, use_container_width=True)

            for pat, title in [
                ("rapm_stability_*.png", "Year-over-year stability — defense vs offense"),
                ("rapm_off_vs_poe_*.png", "Offensive RAPM vs independent POE/100 (cross-pipeline)"),
                ("rapm_splithalf_*.png", "Within-season split-half reliability"),
                ("rapm_face_validity_*.png", "Face validity vs tracking defended-FG"),
            ]:
                imgs = sorted(REPORTS_DIR.glob(pat))
                if imgs:
                    st.markdown(f"**{title}**")
                    st.image(str(imgs[-1]), use_container_width=True)


if __name__ == "__main__":
    main()
