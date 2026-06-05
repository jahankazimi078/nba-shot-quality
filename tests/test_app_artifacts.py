import pandas as pd

from nba_shot_quality.app_artifacts import build_player_profiles


def _sample_shots() -> pd.DataFrame:
    rows = []
    specs = [
        (1, "Rim A", "restricted_area", 2.0, 0, [2, 2, 0, 2]),
        (2, "Arc B", "above_break_3", 25.0, 1, [3, 0, 3, 0]),
        (3, "Mid C", "mid_range", 17.0, 0, [2, 0, 2, 0]),
        (4, "Corner D", "corner_3", 23.0, 1, [3, 3, 0, 0]),
    ]
    for player_id, name, zone, distance, is_three, points in specs:
        for idx, pts in enumerate(points):
            rows.append(
                {
                    "player_id": player_id,
                    "player_name": name,
                    "team_id": 100 + player_id,
                    "shot_zone": zone,
                    "shot_distance_ft": distance + idx * 0.1,
                    "is_three": is_three,
                    "loc_x_ft": float(idx),
                    "loc_y_ft": distance,
                    "shot_value": 3 if is_three else 2,
                    "shot_made": int(pts > 0),
                    "points": pts,
                    "xpoints": 1.0,
                    "poe": pts - 1.0,
                }
            )
    return pd.DataFrame(rows)


def _sample_leaderboard(shots: pd.DataFrame) -> pd.DataFrame:
    board = (
        shots.groupby("player_id")
        .agg(
            player_name=("player_name", "last"),
            team_id=("team_id", "last"),
            attempts=("points", "size"),
            made=("shot_made", "sum"),
            points=("points", "sum"),
            xpoints=("xpoints", "sum"),
        )
        .reset_index()
    )
    board["season"] = "2024-25"
    board["poe"] = board["points"] - board["xpoints"]
    board["poe_per_100"] = board["poe"] / board["attempts"] * 100
    board["poe_ci_low"] = board["poe_per_100"] - 1
    board["poe_ci_high"] = board["poe_per_100"] + 1
    board["pps"] = board["points"] / board["attempts"]
    board["xpps"] = board["xpoints"] / board["attempts"]
    board["efg_pct"] = 0.5
    board["ts_pct"] = 0.58
    board["league_avg_ts_pct"] = 0.56
    board["rel_ts_pct"] = 0.02
    return board


def test_build_player_profiles_adds_shot_mix_and_outcomes():
    shots = _sample_shots()
    board = _sample_leaderboard(shots)

    profiles = build_player_profiles(shots, board, "2024-25", n_clusters=3)
    rim = profiles[profiles["player_name"] == "Rim A"].iloc[0]
    arc = profiles[profiles["player_name"] == "Arc B"].iloc[0]

    assert len(profiles) == 4
    assert rim["restricted_area_share"] == 1.0
    assert rim["rim_rate"] == 1.0
    assert arc["above_break_3_share"] == 1.0
    assert arc["three_pa_rate"] == 1.0
    assert "poe_per_100" in profiles.columns
    assert "archetype" in profiles.columns


def test_archetype_assignment_is_deterministic():
    shots = _sample_shots()
    board = _sample_leaderboard(shots)

    first = build_player_profiles(shots, board, "2024-25", n_clusters=3).sort_values("player_id")
    second = build_player_profiles(shots, board, "2024-25", n_clusters=3).sort_values("player_id")

    assert first["archetype_id"].tolist() == second["archetype_id"].tolist()
    assert first["archetype"].tolist() == second["archetype"].tolist()
