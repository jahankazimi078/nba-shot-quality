from pathlib import Path

import pandas as pd

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
APP_DATA_DIR = STATIC_DIR / "data"
SEASONS = ("2022-23", "2023-24", "2024-25")


def test_committed_static_csvs_have_required_columns():
    profile_cols = {
        "season",
        "player_id",
        "player_name",
        "archetype",
        "attempts",
        "poe_per_100",
        "pps",
        "xpps",
        "rel_ts_pct",
        "avg_distance_ft",
        "three_pa_rate",
        "rim_rate",
        "midrange_rate",
    }
    shot_cols = {"season", "player_id", "loc_x_ft", "loc_y_ft", "shot_zone", "xpoints", "poe"}
    leaderboard_cols = {"player_id", "player_name", "attempts", "poe_per_100", "ts_pct", "rel_ts_pct"}

    for season in SEASONS:
        profiles = pd.read_csv(APP_DATA_DIR / f"player_profiles_{season}.csv")
        shots = pd.read_csv(APP_DATA_DIR / f"shots_{season}.csv")
        shot_sample = pd.read_csv(APP_DATA_DIR / f"shot_map_sample_{season}.csv")
        leaderboard = pd.read_csv(APP_DATA_DIR / f"leaderboard_{season}.csv")

        assert profile_cols <= set(profiles.columns)
        assert shot_cols <= set(shots.columns)
        assert shot_cols <= set(shot_sample.columns)
        assert leaderboard_cols <= set(leaderboard.columns)
        assert len(profiles) > 250
        assert len(shots) > 200_000
        assert len(shot_sample) < len(shots)


def test_committed_model_evidence_assets_exist():
    required = [
        APP_DATA_DIR / "coaching_did_results.csv",
        APP_DATA_DIR / "coaching_did_summary.csv",
        APP_DATA_DIR / "rapm_pooled.csv",
        APP_DATA_DIR / "model_evidence.csv",
        APP_DATA_DIR / "data_manifest.csv",
        STATIC_DIR / "assets" / "reports" / "poe_stability_2023-24_vs_2024-25.png",
        STATIC_DIR / "assets" / "reports" / "coaching_event_study.png",
    ]

    for path in required:
        assert path.exists()


def test_static_data_manifest_lists_csv_exports():
    manifest = pd.read_csv(APP_DATA_DIR / "data_manifest.csv")

    assert {"file", "bytes", "description"} <= set(manifest.columns)
    assert "data/player_profiles.csv" in set(manifest["file"])
    assert all((STATIC_DIR / path).exists() for path in manifest["file"])
