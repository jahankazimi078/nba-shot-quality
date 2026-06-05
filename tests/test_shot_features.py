import pandas as pd

from nba_shot_quality.features.shot_features import _derive_zone


def test_derive_zone_uses_api_zone_and_three_point_flag():
    df = pd.DataFrame(
        {
            "is_three": [1, 1, 0, 0, 0],
            "SHOT_ZONE_BASIC": [
                "Left Corner 3",
                "Above the Break 3",
                "Restricted Area",
                "In The Paint (Non-RA)",
                "Mid-Range",
            ],
        }
    )

    zones = _derive_zone(df).tolist()

    assert zones == [
        "corner_3",
        "above_break_3",
        "restricted_area",
        "paint_non_ra",
        "mid_range",
    ]
