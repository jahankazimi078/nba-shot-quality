# NBA Shot Quality Case Study

## Objective

The project asks a practical basketball analytics question: who creates points above expectation
after accounting for shot difficulty, and how can that skill be made visible to coaches, analysts,
and non-technical reviewers?

The final product is a browser dashboard plus reproducible pipelines for xPoints, POE, RAPM impact,
and a coaching-change event study.

Live app: <https://jahankazimi078.github.io/nba-shot-quality/>

Supporting docs: [portfolio brief](portfolio_brief.md) and [data dictionary](data_dictionary.md).

## Data

The pipeline uses public NBA regular-season data:

- Shot detail for field-goal attempts, locations, clock context, action type, and shot value.
- Player-season totals for TS% and league-average TS%.
- Rotation and lineup-derived shot contexts for RAPM.
- Team game logs for possessions and actual points allowed in the coaching study.

Dashboard data is exported into CSVs, with report images bundled beside the app. The Data tab lists
the files behind each view so the metrics can be audited or reused outside the dashboard.

## xPoints Model

xPoints predicts made-shot probability and multiplies by shot value. The feature set is intentionally
public and reproducible: shot distance, angle, zone, action type, period, seconds remaining, and
3-point flag. Validation uses game-grouped splits so shots from the same game do not leak across
train and validation folds.

The model is not meant to be a tracking-grade shot-quality system. It is a transparent baseline that
captures the largest public-context effects and creates a clean residual: actual points minus expected
points.

## POE Shooter Skill

POE is the player-season sum of:

```text
actual field-goal points - xPoints
```

The leaderboard reports POE per 100 attempts, total POE, PPS, xPPS, eFG%, TS%, rTS%, and bootstrap
confidence intervals. Shots are scored out of fold, which keeps the player-level residuals from
benefiting from in-sample model fit.

Evidence that POE is useful:

- 2023-24 to 2024-25 POE/100 stability is **r = 0.58** among qualified players.
- 2024-25 POE/100 vs relative TS% is **r = 0.66**. That correlation is strong enough to validate the
  direction but weak enough to show that shot-difficulty adjustment adds information.
- 2024-25 leaders by POE/100 include Ty Jerome (**+23.6**), Nikola Jokic (**+22.2**), and Payton
  Pritchard (**+21.1**).

## Player Profiles And Archetypes

Player profiles merge outcome metrics with shot-diet features:

- Zone shares: rim, paint non-RA, midrange, corner 3, above-break 3.
- Average distance, 3PA rate, rim rate, and midrange rate.
- Attempts, PPS, xPPS, POE/100, TS%, and rTS%.

Archetypes are assigned with standardized KMeans features and `random_state=0`. The clustering uses
shot-profile features only, not POE or TS%. This matters because it separates style from performance:
users can compare the best and worst shot makers inside the same shot diet.

## RAPM Impact

RAPM estimates on-floor shot-quality impact with ridge regularization. The app separates:

- Defensive RAPM: positive means suppressing opponent shot quality.
- Offensive RAPM: positive means lifting team shot quality.
- Net RAPM: offensive plus defensive shot impact.

The model is intentionally described as shot-quality impact, not total player value. It excludes
turnovers, rebounds, free throws, and broader possession outcomes.

## Coaching-Change Study

The coaching analysis estimates whether mid-season firings changed defensive outcomes beyond league
drift. For each event and window, it compares:

```text
(treated post - treated pre) - (control post - control pre)
```

Controls are teams in the same season without a mid-season firing, aligned to the same calendar
window. The sign convention is negative DiD = defense improved.

Headline results across seven events:

| Window | Metric | Pooled DiD | Event-clustered 95% CI |
| --- | --- | ---: | ---: |
| 10 | Allowed xPoints / 100 poss | -1.30 | [-3.58, +1.27] |
| 20 | Allowed xPoints / 100 poss | -0.00 | [-2.45, +1.93] |
| 30 | Allowed xPoints / 100 poss | -0.19 | [-2.43, +1.86] |
| 10 | Allowed points / 100 poss | -2.95 | [-6.79, +0.62] |
| 20 | Allowed points / 100 poss | -2.43 | [-5.37, +0.34] |
| 30 | Allowed points / 100 poss | -2.36 | [-5.00, +0.17] |

The interpretation is deliberately conservative. Actual defensive rating moves in the expected
direction, but xPoints effects are small and event-clustered intervals include zero. With only seven
events, the study is better evidence of analytical design judgment than a definitive causal claim.

## Business Interpretation

The app supports common front-office and coaching questions:

- Which players beat shot difficulty, not just raw efficiency?
- Which players share a similar shot diet but have different outcomes?
- Which high-efficiency players are thriving because of shot quality versus shot making?
- Which RAPM names are directionally consistent with independent POE evidence?
- How should analysts communicate uncertainty when a causal sample is small?

## Limitations And Next Steps

- Add tracking features such as defender distance, touch time, catch-and-shoot, and shot contest.
- Add possession-level outcomes to extend RAPM beyond field-goal shot quality.
- Add playoff and multi-year rolling views.
- Improve archetype labels with analyst review once more seasons are included.
- Add tracking-data features and richer mobile interactions when those data sources are available.
