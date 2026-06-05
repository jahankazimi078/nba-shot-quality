"""Build CSV artifacts and the dashboard data package.

This module converts processed public-derived parquet files into CSVs and local report assets used
by the browser dashboard.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"
STATIC_DIR = REPO_ROOT / "static"
STATIC_DATA_DIR = STATIC_DIR / "data"
STATIC_REPORTS_DIR = STATIC_DIR / "assets" / "reports"

DEFAULT_SEASONS = ("2022-23", "2023-24", "2024-25")
SHOT_SAMPLE_PER_PLAYER = 220
ZONE_ORDER = ("restricted_area", "paint_non_ra", "mid_range", "corner_3", "above_break_3")
ZONE_SHARE_COLS = tuple(f"{zone}_share" for zone in ZONE_ORDER)
PROFILE_FEATURE_COLS = (
    "restricted_area_share",
    "paint_non_ra_share",
    "mid_range_share",
    "corner_3_share",
    "above_break_3_share",
    "avg_distance_ft",
    "three_pa_rate",
)
SHOT_APP_COLS = (
    "season",
    "player_id",
    "player_name",
    "loc_x_ft",
    "loc_y_ft",
    "shot_distance_ft",
    "shot_zone",
    "is_three",
    "shot_value",
    "shot_made",
    "points",
    "xpoints",
    "poe",
)
LEADERBOARD_COLS = (
    "player_id",
    "player_name",
    "team_id",
    "season",
    "attempts",
    "made",
    "points",
    "xpoints",
    "poe",
    "poe_per_100",
    "poe_ci_low",
    "poe_ci_high",
    "pps",
    "xpps",
    "efg_pct",
    "ts_pct",
    "league_avg_ts_pct",
    "rel_ts_pct",
)
OPTIONAL_TABLES = (
    ("rapm_pooled.parquet", "rapm_pooled.csv"),
    ("coaching_did_results.parquet", "coaching_did_results.csv"),
    ("coaching_did_summary.parquet", "coaching_did_summary.csv"),
)


def make_shots_app(scored_shots: pd.DataFrame, season: str) -> pd.DataFrame:
    """Return a slim per-shot table with only the columns needed for shot maps and profile fallback."""
    missing = [c for c in SHOT_APP_COLS if c != "season" and c not in scored_shots.columns]
    if missing:
        raise ValueError(f"scored shots missing required columns: {missing}")

    out = scored_shots.copy()
    out["season"] = season
    out = out[list(SHOT_APP_COLS)].copy()
    out["shot_zone"] = out["shot_zone"].astype(str)
    for col in ("loc_x_ft", "loc_y_ft", "shot_distance_ft", "xpoints", "poe"):
        out[col] = out[col].astype("float32")
    for col in ("player_id", "is_three", "shot_value", "shot_made", "points"):
        out[col] = out[col].astype("int64")
    return out


def sample_shots_for_maps(shots: pd.DataFrame, max_per_player: int = SHOT_SAMPLE_PER_PLAYER) -> pd.DataFrame:
    """Deterministically cap shot-map rows per player for browser rendering."""
    if shots.empty:
        return shots.copy()
    parts = []
    ordered = shots.sort_values(["season", "player_id", "shot_distance_ft", "loc_x_ft", "loc_y_ft"])
    for (_, player_id), group in ordered.groupby(["season", "player_id"], sort=False):
        if len(group) <= max_per_player:
            parts.append(group)
        else:
            parts.append(group.sample(max_per_player, random_state=int(player_id) % 99991))
    sampled = pd.concat(parts, ignore_index=True)
    return sampled.sort_values(["season", "player_name", "player_id"]).reset_index(drop=True)


def build_player_profiles(
    scored_shots: pd.DataFrame,
    leaderboard: pd.DataFrame,
    season: str,
    n_clusters: int = 5,
) -> pd.DataFrame:
    """Create player-season profile rows with shot-diet features and deterministic archetypes.

    Clustering uses shot-profile features only. Outcome metrics such as POE and TS% are merged after
    the shot-diet features are built so archetypes describe how players shoot, not whether they make.
    """
    required_shots = {"player_id", "player_name", "team_id", "shot_zone", "shot_distance_ft", "is_three"}
    required_board = {"player_id", *LEADERBOARD_COLS}
    missing_shots = sorted(required_shots - set(scored_shots.columns))
    missing_board = sorted(required_board - set(leaderboard.columns))
    if missing_shots:
        raise ValueError(f"scored shots missing required columns: {missing_shots}")
    if missing_board:
        raise ValueError(f"leaderboard missing required columns: {missing_board}")

    shots = scored_shots.copy()
    shots["shot_zone"] = shots["shot_zone"].astype(str)
    counts = pd.crosstab(shots["player_id"], shots["shot_zone"])
    counts = counts.reindex(columns=ZONE_ORDER, fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)
    shares.columns = ZONE_SHARE_COLS

    shot_profile = (
        shots.groupby("player_id")
        .agg(
            profile_player_name=("player_name", "last"),
            profile_team_id=("team_id", "last"),
            avg_distance_ft=("shot_distance_ft", "mean"),
            three_pa_rate=("is_three", "mean"),
        )
        .join(shares, how="left")
        .reset_index()
    )
    for col in ZONE_SHARE_COLS:
        shot_profile[col] = shot_profile[col].fillna(0.0)
    shot_profile["rim_rate"] = shot_profile["restricted_area_share"]
    shot_profile["midrange_rate"] = shot_profile["mid_range_share"]

    profiles = leaderboard[list(LEADERBOARD_COLS)].merge(shot_profile, on="player_id", how="left")
    profiles["season"] = season
    profiles["player_name"] = profiles["player_name"].fillna(profiles["profile_player_name"])
    profiles["team_id"] = profiles["team_id"].fillna(profiles["profile_team_id"])
    profiles = profiles.drop(columns=["profile_player_name", "profile_team_id"])

    for col in (*PROFILE_FEATURE_COLS, "rim_rate", "midrange_rate"):
        profiles[col] = profiles[col].astype("float64").fillna(0.0)

    profiles = assign_shooter_archetypes(profiles, n_clusters=n_clusters)
    ordered = [
        "season",
        "player_id",
        "player_name",
        "team_id",
        "archetype_id",
        "archetype",
        "attempts",
        "points",
        "xpoints",
        "poe",
        "poe_per_100",
        "poe_ci_low",
        "poe_ci_high",
        "pps",
        "xpps",
        "efg_pct",
        "ts_pct",
        "rel_ts_pct",
        "avg_distance_ft",
        "three_pa_rate",
        "rim_rate",
        "midrange_rate",
        *ZONE_SHARE_COLS,
    ]
    return profiles[ordered].sort_values(["archetype", "poe_per_100"], ascending=[True, False]).reset_index(drop=True)


def assign_shooter_archetypes(profiles: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """Assign deterministic shot-diet archetypes using standardized profile features and KMeans."""
    out = profiles.copy()
    if out.empty:
        out["archetype_id"] = pd.Series(dtype="int64")
        out["archetype"] = pd.Series(dtype="object")
        return out

    features = out[list(PROFILE_FEATURE_COLS)].fillna(0.0).astype("float64")
    unique_rows = np.unique(features.round(12).to_numpy(), axis=0).shape[0]
    k = min(n_clusters, len(out), unique_rows)
    if k < 2:
        out["archetype_id"] = 0
        out["archetype"] = "Balanced Shot Diet"
        return out

    scaled = StandardScaler().fit_transform(features)
    labels = KMeans(n_clusters=k, random_state=0, n_init=20).fit_predict(scaled)
    temp = out.assign(_cluster=labels)
    centers = temp.groupby("_cluster")[list(PROFILE_FEATURE_COLS)].mean()
    centers = centers.assign(
        _sort_distance=centers["avg_distance_ft"],
        _sort_three=centers["three_pa_rate"],
        _sort_mid=centers["mid_range_share"],
    )
    ordered_clusters = centers.sort_values(
        ["_sort_distance", "_sort_three", "_sort_mid"],
        ascending=[False, False, False],
    ).index.tolist()
    cluster_id_map = {old_id: new_id for new_id, old_id in enumerate(ordered_clusters)}
    out["archetype_id"] = pd.Series(labels).map(cluster_id_map).to_numpy(dtype="int64")
    out["archetype"] = out["archetype_id"].map(_name_archetypes(out, k))
    return out


def _name_archetypes(profiles: pd.DataFrame, k: int) -> dict[int, str]:
    centers = profiles.groupby("archetype_id")[list(PROFILE_FEATURE_COLS)].mean()
    remaining = set(range(k))
    names: dict[int, str] = {}

    def assign_best(metric: str, label: str) -> None:
        if not remaining:
            return
        ranked = centers.loc[list(remaining), metric].sort_values(ascending=False)
        archetype_id = int(ranked.index[0])
        names[archetype_id] = label
        remaining.remove(archetype_id)

    assign_best("three_pa_rate", "Perimeter Spacers")
    assign_best("restricted_area_share", "Rim Pressure")
    assign_best("mid_range_share", "Midrange Creators")
    assign_best("corner_3_share", "Corner Spacers")

    fallback_labels = ["Balanced Shot Diet", "Paint Touches", "Low-Volume Mix"]
    for archetype_id, label in zip(sorted(remaining), fallback_labels, strict=False):
        names[int(archetype_id)] = label
    return names


def _write_csv(df: pd.DataFrame, path: Path, written: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    written.append(path)


def _clear_generated_static_files(data_dir: Path, reports_dir: Path) -> None:
    for path in data_dir.glob("*.csv"):
        path.unlink()
    for path in reports_dir.glob("*.png"):
        path.unlink()


def _csv_description(path: Path) -> str:
    name = path.name
    if name.startswith("shots_"):
        return "Full per-shot xPoints and POE export for one season."
    if name.startswith("shot_map_sample_"):
        return "Deterministic per-player shot sample used by browser shot maps."
    if name.startswith("leaderboard"):
        return "POE leaderboard with points, xPoints, confidence intervals, TS%, and rTS%."
    if name.startswith("player_profiles"):
        return "Player profile metrics, shot-zone shares, and shot-diet archetypes."
    if name == "season_summary.csv":
        return "Season-level totals and headline POE leaders."
    if name == "archetype_summary.csv":
        return "Per-season archetype aggregates."
    if name.startswith("rapm"):
        return "Pooled RAPM shot-quality impact ratings."
    if name.startswith("coaching"):
        return "Coaching-change difference-in-differences estimates."
    if name == "model_evidence.csv":
        return "Report image index bundled with the dashboard."
    return "Dashboard data export."


def _build_season_summary(profiles: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, season_profiles in profiles.groupby("season"):
        season_shots = shots[shots["season"] == season]
        qualified = season_profiles[season_profiles["attempts"] >= 400]
        top = qualified.sort_values("poe_per_100", ascending=False).head(1)
        bottom = qualified.sort_values("poe_per_100").head(1)
        rows.append(
            {
                "season": season,
                "shots": len(season_shots),
                "players": len(season_profiles),
                "qualified_players_400_fga": len(qualified),
                "avg_pps": season_profiles["pps"].mean(),
                "avg_xpps": season_profiles["xpps"].mean(),
                "top_player": top["player_name"].iloc[0] if len(top) else "",
                "top_poe_per_100": top["poe_per_100"].iloc[0] if len(top) else np.nan,
                "bottom_player": bottom["player_name"].iloc[0] if len(bottom) else "",
                "bottom_poe_per_100": bottom["poe_per_100"].iloc[0] if len(bottom) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _build_archetype_summary(profiles: pd.DataFrame) -> pd.DataFrame:
    return (
        profiles.groupby(["season", "archetype"])
        .agg(
            players=("player_id", "size"),
            avg_attempts=("attempts", "mean"),
            avg_poe_per_100=("poe_per_100", "mean"),
            avg_pps=("pps", "mean"),
            avg_xpps=("xpps", "mean"),
            avg_rel_ts_pct=("rel_ts_pct", "mean"),
            avg_distance_ft=("avg_distance_ft", "mean"),
            avg_three_pa_rate=("three_pa_rate", "mean"),
            avg_rim_rate=("rim_rate", "mean"),
            avg_midrange_rate=("midrange_rate", "mean"),
        )
        .reset_index()
        .sort_values(["season", "players"], ascending=[True, False])
    )


def _build_model_evidence_index(reports_dir: Path = STATIC_REPORTS_DIR) -> pd.DataFrame:
    rows = []
    for path in sorted(reports_dir.glob("*.png")):
        name = path.name
        if name.startswith("calibration_"):
            section = "xPoints calibration"
        elif name.startswith("poe_"):
            section = "POE validation"
        elif name.startswith("rapm_"):
            section = "RAPM validation"
        elif name.startswith("coaching_"):
            section = "Coaching study"
        else:
            section = "Other evidence"
        rows.append({"asset": f"assets/reports/{name}", "name": name, "section": section})
    return pd.DataFrame(rows)


def build_app_artifacts(
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    processed_dir: Path = PROCESSED_DIR,
    reports_dir: Path = REPORTS_DIR,
    app_data_dir: Path = STATIC_DATA_DIR,
    n_clusters: int = 5,
) -> list[Path]:
    """Write deploy-ready CSV artifacts from existing processed outputs."""
    app_data_dir.mkdir(parents=True, exist_ok=True)
    STATIC_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _clear_generated_static_files(app_data_dir, STATIC_REPORTS_DIR)
    written: list[Path] = []
    all_profiles: list[pd.DataFrame] = []
    all_leaderboards: list[pd.DataFrame] = []
    all_shots: list[pd.DataFrame] = []
    for season in seasons:
        shots_path = processed_dir / f"shots_scored_{season}.parquet"
        leaderboard_path = processed_dir / f"poe_player_season_{season}.parquet"
        if not shots_path.exists():
            raise FileNotFoundError(f"{shots_path} not found; run score --season {season} first")
        if not leaderboard_path.exists():
            raise FileNotFoundError(f"{leaderboard_path} not found; run poe --season {season} first")

        shots = pd.read_parquet(shots_path)
        leaderboard = pd.read_parquet(leaderboard_path)

        shots_app = make_shots_app(shots, season)
        profiles = build_player_profiles(shots, leaderboard, season, n_clusters=n_clusters)
        leaderboard = leaderboard[list(LEADERBOARD_COLS)].copy()

        all_shots.append(shots_app)
        all_profiles.append(profiles)
        all_leaderboards.append(leaderboard)

        _write_csv(shots_app, app_data_dir / f"shots_{season}.csv", written)
        _write_csv(sample_shots_for_maps(shots_app), app_data_dir / f"shot_map_sample_{season}.csv", written)
        _write_csv(leaderboard, app_data_dir / f"leaderboard_{season}.csv", written)
        _write_csv(profiles, app_data_dir / f"player_profiles_{season}.csv", written)

    combined_profiles = pd.concat(all_profiles, ignore_index=True)
    combined_leaderboard = pd.concat(all_leaderboards, ignore_index=True)
    combined_shots = pd.concat(all_shots, ignore_index=True)

    _write_csv(combined_profiles, app_data_dir / "player_profiles.csv", written)
    _write_csv(combined_leaderboard, app_data_dir / "leaderboard.csv", written)
    _write_csv(_build_season_summary(combined_profiles, combined_shots), app_data_dir / "season_summary.csv", written)
    _write_csv(_build_archetype_summary(combined_profiles), app_data_dir / "archetype_summary.csv", written)

    for source_name, csv_name in OPTIONAL_TABLES:
        source = processed_dir / source_name
        if source.exists():
            _write_csv(pd.read_parquet(source), app_data_dir / csv_name, written)

    report_dest = STATIC_REPORTS_DIR
    report_dest.mkdir(parents=True, exist_ok=True)
    for source in sorted(reports_dir.glob("*.png")):
        dest = report_dest / source.name
        shutil.copy2(source, dest)
        written.append(dest)

    _write_csv(_build_model_evidence_index(report_dest), app_data_dir / "model_evidence.csv", written)
    csv_rows = []
    for path in sorted(p for p in written if p.suffix == ".csv"):
        csv_rows.append(
            {
                "file": path.relative_to(STATIC_DIR).as_posix(),
                "bytes": path.stat().st_size,
                "description": _csv_description(path),
            }
        )
    _write_csv(pd.DataFrame(csv_rows), app_data_dir / "data_manifest.csv", written)

    total_mb = sum(path.stat().st_size for path in written) / 1_000_000
    print(f"[app-artifacts] wrote {len(written)} files to {STATIC_DIR} ({total_mb:.1f} MB)")
    for path in written:
        print(f"  {path.relative_to(REPO_ROOT)}  {path.stat().st_size / 1_000_000:.2f} MB")
    return written
