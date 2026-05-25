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
SEASONS = ["2024-25", "2023-24"]

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


RAPM_COLS = ["player_name", "def_rapm", "def_ci_low", "def_ci_high", "off_rapm", "def_shots"]
RAPM_ROUND = {"def_rapm": 2, "def_ci_low": 2, "def_ci_high": 2, "off_rapm": 2}


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
        ["Player shot map", "Top 20 POE", "Bottom 20 POE", "Stability", "Defender impact"]
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
        st.subheader("Defender impact — ridge RAPM")
        st.caption(
            "Defensive coefficient on per-shot POE, per 100 defensive shots, signed so + = suppresses "
            "opponent scoring vs expectation. On-floor attribution (all 5 defenders share credit, not "
            "closest-defender) and FGA-only (excludes free throws, turnovers, non-shot defense)."
        )
        loaded = load_rapm()
        if loaded is None:
            st.info("Run `bash scripts/run_rapm.sh` to generate defender-impact ratings.")
        else:
            rapm, source = loaded
            max_def = int(rapm["def_shots"].max()) if len(rapm) else 0
            if source != "pooled":
                st.warning(
                    f"⚠️ Limited data — single-season partial sample (`{source}`, max "
                    f"{max_def:,} defensive shots/player). This is a mechanics preview; the "
                    "rankings are **not yet meaningful**. Run the full two-season pull for real ratings."
                )
            # Adapt the threshold to the available sample so the table isn't empty on partial data.
            slider_max = max(500, max_def)
            step = 250 if slider_max > 1500 else 25
            default = ((min(1500, max_def // 2)) // step) * step
            min_def = st.slider("Min defensive shots", 0, slider_max, default, step=step)
            ranked = rapm[rapm["def_shots"] >= min_def].sort_values("def_rapm", ascending=False)
            c_top, c_bot = st.columns(2)
            with c_top:
                st.markdown(f"**Top 15 defenders** (n={len(ranked)})")
                st.dataframe(ranked.head(15)[RAPM_COLS].round(RAPM_ROUND), hide_index=True, use_container_width=True)
            with c_bot:
                st.markdown("**Bottom 15 defenders**")
                st.dataframe(ranked.tail(15).iloc[::-1][RAPM_COLS].round(RAPM_ROUND), hide_index=True, use_container_width=True)
            stab = sorted(REPORTS_DIR.glob("rapm_stability_*.png"))
            face = sorted(REPORTS_DIR.glob("rapm_face_validity_*.png"))
            if stab:
                st.markdown("**Year-over-year RAPM stability**")
                st.image(str(stab[-1]), use_container_width=True)
            if face:
                st.markdown("**Face validity vs tracking defended-FG metric**")
                st.image(str(face[-1]), use_container_width=True)


if __name__ == "__main__":
    main()
