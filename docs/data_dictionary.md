# Metric Glossary And Data Dictionary

This dictionary covers the CSV exports in `static/data/` and the dashboard labels a reviewer is
most likely to inspect.

## Metric Glossary

| Metric | Definition | How to read it |
| --- | --- | --- |
| `xpoints` | Expected points for a shot, equal to model make probability times shot value. | Higher means the shot was easier or worth more points. |
| `poe` | Points over expected, equal to actual field-goal points minus `xpoints`. | Positive means the shooter beat the model expectation. |
| `poe_per_100` | POE scaled to 100 field-goal attempts. | Main player comparison metric for shot-making. |
| `pps` | Actual points per field-goal attempt. | Raw scoring output per shot. |
| `xpps` | Expected points per field-goal attempt. | Shot quality per shot before shooter residual. |
| `efg_pct` | Effective field goal percentage, with made threes weighted by 1.5. | Shooting efficiency on field goals only. |
| `ts_pct` | True shooting percentage from player-season stats. | Broader efficiency including free throws. |
| `rel_ts_pct` | Player TS% minus league-average TS% for the same season. | Positive means above league average. |
| Zone shares | Share of a player's attempts from each shot zone. | Shot diet context, not performance by itself. |
| Archetype labels | KMeans shot-diet groups from profile features only. | Style buckets, not rankings. |
| RAPM columns | Regularized adjusted plus-minus for shot-quality impact. | Directional on-floor shot-quality signal. |
| DiD columns | Difference-in-differences estimates for coaching changes. | Negative defensive values mean allowed outcomes improved. |
| CI columns | Bootstrap confidence interval bounds. | If the interval crosses zero, evidence is uncertain. |

## Common Identifier Columns

| Column | Definition |
| --- | --- |
| `season` | NBA season label, such as `2024-25`. |
| `player_id` | NBA player identifier. |
| `player_name` | Player display name. |
| `team_id` | NBA team identifier attached to the player row. |
| `team_abbr` | Three-letter team abbreviation in coaching-study rows. |

## Shot Exports

Files: `shots_YYYY-YY.csv` and `shot_map_sample_YYYY-YY.csv`.

| Column | Definition |
| --- | --- |
| `loc_x_ft` | Horizontal shot coordinate in feet. |
| `loc_y_ft` | Vertical shot coordinate in feet. |
| `shot_distance_ft` | Distance from basket in feet. |
| `shot_zone` | Derived shot zone: restricted area, paint non-RA, midrange, corner three, or above-break three. |
| `is_three` | `1` for a three-point attempt, `0` otherwise. |
| `shot_value` | Point value of the attempt, usually `2` or `3`. |
| `shot_made` | `1` if made, `0` if missed. |
| `points` | Actual points scored on the shot. |
| `xpoints` | Model-expected points for that shot. |
| `poe` | Actual points minus expected points for that shot. |

The full shot exports are meant for audit and reuse. The app uses `shot_map_sample_YYYY-YY.csv`
for shot maps so browser rendering stays responsive.

## Leaderboard And Player Profile Exports

Files: `leaderboard.csv`, `leaderboard_YYYY-YY.csv`, `player_profiles.csv`,
`player_profiles_YYYY-YY.csv`.

| Column | Definition |
| --- | --- |
| `attempts` | Field-goal attempts included in the xPoints/POE universe. |
| `made` | Made field goals. |
| `points` | Actual field-goal points. |
| `xpoints` | Sum of expected points across attempts. |
| `poe` | Sum of points over expected. |
| `poe_per_100` | POE per 100 field-goal attempts. |
| `poe_ci_low` | Lower bootstrap confidence bound for POE/100. |
| `poe_ci_high` | Upper bootstrap confidence bound for POE/100. |
| `pps` | Actual points per shot. |
| `xpps` | Expected points per shot. |
| `efg_pct` | Effective field goal percentage. |
| `ts_pct` | True shooting percentage from player totals. |
| `league_avg_ts_pct` | League-average TS% for the season. |
| `rel_ts_pct` | Player TS% minus league-average TS%. |
| `archetype_id` | Deterministic numeric shot-diet cluster id. |
| `archetype` | Human-readable shot-diet label. |
| `avg_distance_ft` | Average shot distance in feet. |
| `three_pa_rate` | Share of attempts that were threes. |
| `rim_rate` | Share of attempts in the restricted area. |
| `midrange_rate` | Share of attempts from midrange. |
| `restricted_area_share` | Share of attempts at the rim. |
| `paint_non_ra_share` | Share of attempts in the paint outside the restricted area. |
| `mid_range_share` | Share of attempts from midrange. |
| `corner_3_share` | Share of attempts from corner three. |
| `above_break_3_share` | Share of attempts from above-break three. |

## Summary Exports

Files: `season_summary.csv`, `archetype_summary.csv`, `model_evidence.csv`, `data_manifest.csv`.

| Column | Definition |
| --- | --- |
| `shots` | Count of season shot attempts in the app export. |
| `players` | Count of player-season profile rows. |
| `qualified_players_400_fga` | Players with at least 400 field-goal attempts. |
| `avg_pps` | Average player PPS in that season summary. |
| `avg_xpps` | Average player xPPS in that season summary. |
| `top_player` | Qualified player with the highest POE/100. |
| `top_poe_per_100` | Highest qualified POE/100 in the season. |
| `bottom_player` | Qualified player with the lowest POE/100. |
| `bottom_poe_per_100` | Lowest qualified POE/100 in the season. |
| `archetype` | Shot-diet group label. |
| `avg_attempts` | Average attempts for players in the archetype. |
| `avg_poe_per_100` | Average POE/100 for players in the archetype. |
| `avg_rel_ts_pct` | Average relative TS% for players in the archetype. |
| `avg_three_pa_rate` | Average three-point attempt rate for the archetype. |
| `avg_rim_rate` | Average rim rate for the archetype. |
| `avg_midrange_rate` | Average midrange rate for the archetype. |
| `asset` | Dashboard-relative report image path. |
| `name` | Report image filename. |
| `section` | Evidence section label for the report image. |
| `file` | Dashboard-relative CSV path. |
| `bytes` | File size in bytes. |
| `description` | Short manifest description. |

## RAPM Export

File: `rapm_pooled.csv`.

| Column | Definition |
| --- | --- |
| `def_rapm` | Defensive shot-quality impact estimate. Positive means suppressing opponent shot quality. |
| `def_ci_low` | Lower bootstrap confidence bound for defensive RAPM. |
| `def_ci_high` | Upper bootstrap confidence bound for defensive RAPM. |
| `off_rapm` | Offensive shot-quality impact estimate. Positive means improving team shot quality. |
| `off_ci_low` | Lower bootstrap confidence bound for offensive RAPM. |
| `off_ci_high` | Upper bootstrap confidence bound for offensive RAPM. |
| `net_rapm` | Offensive plus defensive shot-quality impact. |
| `net_ci_low` | Lower bootstrap confidence bound for net RAPM. |
| `net_ci_high` | Upper bootstrap confidence bound for net RAPM. |
| `def_shots` | Defensive shot possessions included for the player. |
| `off_shots` | Offensive shot possessions included for the player. |

What to conclude: RAPM is useful directional evidence about shot-quality impact after regularizing
lineup context. What not to conclude: it is not total player value, because it excludes turnovers,
rebounds, free throws, and many possession outcomes.

## Coaching DiD Exports

Files: `coaching_did_results.csv`, `coaching_did_summary.csv`.

| Column | Definition |
| --- | --- |
| `coach_out` | Coach replaced during the season. |
| `change_date` | Date of the coaching change. |
| `window` | Number of games before and after the change. |
| `metric` | Outcome being tested, such as `xpts_100poss` or `pts_100poss`. |
| `n_pre` | Treated-team games before the change in the event window. |
| `n_post` | Treated-team games after the change in the event window. |
| `treated_pre` | Treated-team pre-change outcome average. |
| `treated_post` | Treated-team post-change outcome average. |
| `control_pre` | Control-team pre-window outcome average. |
| `control_post` | Control-team post-window outcome average. |
| `treated_delta` | Treated post minus treated pre. |
| `control_delta` | Control post minus control pre. |
| `did` | Difference-in-differences estimate: treated delta minus control delta. |
| `n_events` | Number of coaching-change events in the pooled summary. |
| `pooled_did` | Average DiD across events. |
| `event_ci_low` | Event-clustered lower confidence bound. |
| `event_ci_high` | Event-clustered upper confidence bound. |
| `game_ci_low` | Game-bootstrap lower confidence bound. |
| `game_ci_high` | Game-bootstrap upper confidence bound. |

What to conclude: negative defensive estimates suggest improvement relative to league drift, but the
intervals show large uncertainty. What not to conclude: the current seven-event sample does not prove
that coaching firings caused defensive improvement.
