"""Feature engineering on raw nba_api shot detail."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

CATEGORICAL_COLS = ["shot_zone", "shot_type_action"]
FEATURE_COLS = [
    "shot_distance_ft",
    "shot_angle_deg",
    "shot_zone",
    "shot_type_action",
    "period",
    "seconds_remaining_period",
    "is_three",
]
TARGET_COL = "shot_made"


def build_features(season: str) -> Path:
    """Read raw shots parquet, derive features, write processed parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    in_path = RAW_DIR / f"shots_{season}.parquet"
    out_path = PROCESSED_DIR / f"shots_features_{season}.parquet"

    if not in_path.exists():
        raise FileNotFoundError(f"raw shots not found at {in_path} — run ingest first")

    raw = pd.read_parquet(in_path)
    print(f"[features] loaded {len(raw):,} raw rows from {in_path.name}")

    df = _transform(raw)
    df.to_parquet(out_path, index=False)

    print(f"[features] wrote {len(df):,} rows to {out_path.name}")
    print("[features] shot_zone counts:")
    print(df["shot_zone"].value_counts().to_string())
    print("[features] feature null counts:")
    nulls = df[FEATURE_COLS + [TARGET_COL]].isna().sum()
    nulls = nulls[nulls > 0]
    print(nulls.to_string() if len(nulls) else "  (none)")
    return out_path


def _transform(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.upper() for c in df.columns]

    # LOC_X / LOC_Y are in 0.1 ft units, basket at (0, 0).
    df["loc_x_ft"] = df["LOC_X"] / 10.0
    df["loc_y_ft"] = df["LOC_Y"] / 10.0
    df["shot_distance_ft"] = df["SHOT_DISTANCE"].astype("float32")
    df["shot_angle_deg"] = np.degrees(np.arctan2(df["loc_x_ft"], df["loc_y_ft"].clip(lower=0.01)))

    df["is_three"] = (df["SHOT_TYPE"] == "3PT Field Goal").astype("int8")
    df["shot_value"] = np.where(df["is_three"] == 1, 3, 2).astype("int8")
    df["shot_made"] = df["SHOT_MADE_FLAG"].astype("int8")
    df["points"] = (df["shot_made"] * df["shot_value"]).astype("int8")

    df["period"] = df["PERIOD"].astype("int8")
    df["seconds_remaining_period"] = (
        df["MINUTES_REMAINING"].astype("int16") * 60 + df["SECONDS_REMAINING"].astype("int16")
    ).astype("int16")

    df["shot_type_action"] = df["ACTION_TYPE"].astype("category")
    df["shot_zone"] = _derive_zone(df).astype("category")

    df["game_id"] = df["GAME_ID"]
    df["game_date"] = pd.to_datetime(df["GAME_DATE"], format="%Y%m%d", errors="coerce")
    df["player_id"] = df["PLAYER_ID"].astype("int64")
    df["player_name"] = df["PLAYER_NAME"]
    df["team_id"] = df["TEAM_ID"].astype("int64")

    keep = [
        "game_id",
        "game_date",
        "player_id",
        "player_name",
        "team_id",
        "loc_x_ft",
        "loc_y_ft",
        "shot_distance_ft",
        "shot_angle_deg",
        "shot_zone",
        "shot_type_action",
        "period",
        "seconds_remaining_period",
        "is_three",
        "shot_value",
        "shot_made",
        "points",
    ]
    df = df[keep]

    before = len(df)
    df = df[df["shot_distance_ft"].between(0, 50)]
    df = df[df["loc_y_ft"].between(-5, 50)]
    dropped = before - len(df)
    if dropped:
        print(f"[features] dropped {dropped:,} rows with bad coords")
    return df.reset_index(drop=True)


def _derive_zone(df: pd.DataFrame) -> pd.Series:
    """Map to 5 mutually-exclusive shot zones using API zone fields + coords."""
    is_three = df["is_three"] == 1
    is_corner = df["SHOT_ZONE_BASIC"].isin(
        ["Left Corner 3", "Right Corner 3"]
    )
    is_restricted = df["SHOT_ZONE_BASIC"] == "Restricted Area"
    is_paint_non_ra = df["SHOT_ZONE_BASIC"] == "In The Paint (Non-RA)"

    zone = pd.Series("mid_range", index=df.index, dtype="object")
    zone[is_three & is_corner] = "corner_3"
    zone[is_three & ~is_corner] = "above_break_3"
    zone[is_restricted] = "restricted_area"
    zone[is_paint_non_ra] = "paint_non_ra"
    return zone
