import pandas as pd

from nba_shot_quality.analysis.coaching_event_study import _did_from_cells, _pool_rate


def test_pool_rate_and_did_from_cells():
    treated_pre = pd.DataFrame({"num": [100, 120], "den": [100, 100]})
    treated_post = pd.DataFrame({"num": [90, 90], "den": [100, 100]})
    control_pre = pd.DataFrame({"num": [100, 100], "den": [100, 100]})
    control_post = pd.DataFrame({"num": [105, 105], "den": [100, 100]})
    cells = {
        "treated_pre": treated_pre,
        "treated_post": treated_post,
        "control_pre": control_pre,
        "control_post": control_post,
    }

    assert _pool_rate(treated_pre, "num", "den") == 110.0

    did = _did_from_cells(cells, "num", "den")

    assert did["treated_delta"] == -20.0
    assert did["control_delta"] == 5.0
    assert did["did"] == -25.0
