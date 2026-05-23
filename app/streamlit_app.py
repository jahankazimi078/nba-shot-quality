"""Streamlit dashboard: player shot map colored by POE + POE leaderboard."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from nba_shot_quality.features.shot_features import FEATURE_COLS, CATEGORICAL_COLS
from nba_shot_quality.models.xpoints import predict_xpoints

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


@st.cache_data(show_spinner=False)
def load_shots(season: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"shots_features_{season}.parquet"
    return pd.read_parquet(path)


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(PROCESSED_DIR / "xpoints_model.joblib")


@st.cache_data(show_spinner=False)
def scored_shots(season: str) -> pd.DataFrame:
    df = load_shots(season).copy()
    model = load_model()
    df["xpoints"] = predict_xpoints(model, df)
    df["poe"] = df["points"] - df["xpoints"]
    return df


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
            df["loc_x_ft"],
            df["loc_y_ft"],
            c=df["poe"],
            cmap="RdYlGn",
            vmin=-2,
            vmax=2,
            s=18,
            alpha=0.75,
            edgecolor="black",
            linewidths=0.2,
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
    st.caption("Layer 1: LightGBM make-probability model. POE = actual points − expected points.")

    season = st.sidebar.selectbox("Season", ["2024-25"])
    min_attempts = st.sidebar.slider("Min shot attempts (leaderboard)", 100, 1500, 400, step=50)

    df = scored_shots(season)
    st.sidebar.write(f"**Total shots:** {len(df):,}")
    st.sidebar.write(f"**Players:** {df['player_id'].nunique():,}")

    leaderboard = (
        df.groupby(["player_id", "player_name"])
        .agg(
            attempts=("points", "size"),
            points=("points", "sum"),
            xpoints=("xpoints", "sum"),
            pps=("points", "mean"),
            x_pps=("xpoints", "mean"),
        )
        .reset_index()
    )
    leaderboard["poe"] = leaderboard["points"] - leaderboard["xpoints"]
    leaderboard["poe_per_100"] = leaderboard["poe"] / leaderboard["attempts"] * 100
    qualified = leaderboard[leaderboard["attempts"] >= min_attempts].copy()
    qualified = qualified.sort_values("poe", ascending=False)

    tab_map, tab_top, tab_bottom = st.tabs(["Player shot map", "Top 20 POE", "Bottom 20 POE"])

    with tab_map:
        names = sorted(qualified["player_name"].tolist())
        if not names:
            st.warning(f"No players have ≥{min_attempts} attempts at the current threshold.")
            return
        default_idx = names.index("Stephen Curry") if "Stephen Curry" in names else 0
        player = st.selectbox("Player", names, index=default_idx)
        pdf = df[df["player_name"] == player]
        col_map, col_stats = st.columns([3, 1])
        with col_map:
            fig = render_shot_map(pdf, f"{player} — {season} ({len(pdf)} shots)")
            st.pyplot(fig)
        with col_stats:
            stats = qualified[qualified["player_name"] == player].iloc[0]
            st.metric("Attempts", f"{int(stats['attempts']):,}")
            st.metric("Points", f"{int(stats['points']):,}")
            st.metric("xPoints", f"{stats['xpoints']:.0f}")
            st.metric("POE (total)", f"{stats['poe']:+.0f}")
            st.metric("POE per 100 shots", f"{stats['poe_per_100']:+.2f}")
            st.metric("PPS", f"{stats['pps']:.3f}")
            st.metric("xPPS", f"{stats['x_pps']:.3f}")

    with tab_top:
        st.subheader(f"Top 20 by POE (min {min_attempts} attempts)")
        st.dataframe(
            qualified.head(20)[
                ["player_name", "attempts", "points", "xpoints", "poe", "poe_per_100", "pps", "x_pps"]
            ].round({"xpoints": 1, "poe": 1, "poe_per_100": 2, "pps": 3, "x_pps": 3}),
            hide_index=True,
            use_container_width=True,
        )

    with tab_bottom:
        st.subheader(f"Bottom 20 by POE (min {min_attempts} attempts)")
        st.dataframe(
            qualified.tail(20).iloc[::-1][
                ["player_name", "attempts", "points", "xpoints", "poe", "poe_per_100", "pps", "x_pps"]
            ].round({"xpoints": 1, "poe": 1, "poe_per_100": 2, "pps": 3, "x_pps": 3}),
            hide_index=True,
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
