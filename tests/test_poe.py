import pandas as pd

from nba_shot_quality.models.poe import summarize_player_poe


def test_summarize_player_poe_math_without_file_io():
    shots = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 2],
            "player_name": ["A", "A", "A", "B"],
            "team_id": [10, 10, 10, 20],
            "shot_made": [1, 0, 1, 0],
            "is_three": [0, 1, 1, 0],
            "points": [2, 0, 3, 0],
            "xpoints": [1.1, 0.9, 1.2, 1.0],
        }
    )
    stats = pd.DataFrame({"player_id": [1], "ts_pct": [0.62], "league_avg_ts_pct": [0.58]})

    out = summarize_player_poe(shots, "2024-25", min_attempts=2, stats=stats, n_boot=0)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["player_id"] == 1
    assert row["attempts"] == 3
    assert row["points"] == 5
    assert row["xpoints"] == 3.2
    assert round(row["poe"], 6) == 1.8
    assert round(row["poe_per_100"], 6) == 60.0
    assert round(row["pps"], 6) == round(5 / 3, 6)
    assert round(row["xpps"], 6) == round(3.2 / 3, 6)
    assert round(row["efg_pct"], 6) == round((2 + 0.5 * 1) / 3, 6)
    assert round(row["rel_ts_pct"], 6) == 0.04
