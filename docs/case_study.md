# NBA Shot Quality: the long version

Site: <https://jahankazimi078.github.io/nba-shot-quality/> · Column definitions:
[data dictionary](data_dictionary.md)

## The question

Who actually makes tough shots? Raw efficiency can't answer that, because it blends shot selection with shot
making. A center who only takes dunks and a guard who lives on pull-up jumpers can post the same
field-goal percentage while doing completely different things. I wanted a number for the second part (how
much better a player scores than their shots are worth) and then to see whether that number holds up.

## Data

Everything comes from the public NBA stats API, regular season only, 2022–23 through 2024–25:

- Shot-level detail: location, result, action type, period, game clock, and 2 vs. 3.
- Player-season totals, for true shooting and the league average.
- Play-by-play substitutions, to rebuild who was on the floor for each shot.
- Team game logs, for possessions and points allowed in the coaching study.

The API rate-limits aggressively, so every pull is cached to Parquet and each stage reads the previous
stage's file. That makes it possible to rerun one step without re-downloading anything.

## Expected points

The model is a LightGBM classifier that predicts whether a shot goes in, using distance, angle, zone,
action type, period, seconds remaining, and whether it's a three. Multiply the make probability by the
shot's value and you get its expected points.

Two choices matter most here:

- **Folds are grouped by game.** Shots from the same game are correlated (same defense, same night), so
  letting them sit on both sides of a split would make the model look better than it is.
- **Every shot is scored out of fold.** When I compute a player's POE, their shots were priced by a model
  that never saw those games. Otherwise a player's hot shooting would pull their own baseline up and shrink
  their measured skill.

On held-out games the Brier score is 0.225, against 0.248 for a constant guess, and every shot zone is
calibrated to within about 1.5 percentage points. This isn't a tracking-grade model (it doesn't know where
the defender was), but it's transparent and it captures the big effects.

## Points over expected

For each player-season:

```text
POE = actual field-goal points − expected points
```

I report it per 100 attempts, with a 1,000-sample bootstrap interval, alongside PPS, xPPS, eFG%, TS%, and
relative TS%.

Does it measure something real?

- **It repeats.** Among qualified players, POE/100 correlates at r = 0.58 between 2023–24 and 2024–25.
  Pure noise would sit near zero, and leakage would push it suspiciously high.
- **It agrees with true shooting, but not too much.** POE/100 vs. relative TS% is r = 0.66: same
  direction, with enough disagreement that adjusting for shot difficulty is clearly adding something.
- **The leaders make sense.** In 2024–25 (200+ attempts): Ty Jerome (+23.6), Nikola Jokić (+22.2),
  Payton Pritchard (+21.1).

## Shot diets

Player profiles combine outcome stats with where the shots come from: rim, paint, midrange, corner three,
above-the-break three, plus average distance and three-point rate. I cluster players with k-means on the
location features only. POE and TS% are deliberately left out, so the groups describe style and not
quality. That lets you ask fairer questions, like who the best-shooting spacer is, rather than ranking
spacers against rim-running centers.

## RAPM

A shot's quality also depends on the other nine players on the court. To get at that, I rebuilt every
lineup from play-by-play substitution events. That turned out to be about 20× faster than the NBA's
rotation endpoint, and on fully reconstructed shots it matches that endpoint 99% of the time. Then I fit a
ridge regression that splits each shot's POE across all ten players.

- **Offense** is solid. It lines up with each player's POE (r ≈ 0.7) even though the two come from
  separate models.
- **Defense** is the weak spot. Year-to-year correlation is only about 0.12. I dug into why: defense
  repeats three to four times less than offense within the same model, its split-half reliability tops out
  around 0.2–0.5, and the per-player uncertainty is bigger than the spread between players. That's a
  limit of measuring defense from field-goal attempts with credit shared five ways, not a bug. I also
  tried weighting credit toward each shooter's primary defender, using both season-level and per-game
  matchup data. It helped in one season and hurt in another, so I kept the simpler version.

The ratings on the site are pooled across all three seasons. Even at this noise level, the top defenders
are the rim protectors you'd expect (Jaren Jackson Jr., Wembanyama, Holmgren), and defensive RAPM
correlates at r = 0.39 with the NBA's own defended-FG tracking data.

## Firing the coach

Teams that fire their coach mid-season often look better afterward. The catch is that coaches usually get
fired after a bad stretch, and bad stretches tend to end on their own. So for each of the 7 in-season
firings I compared the team's defense before vs. after against every team that didn't change coaches, over
the same dates:

```text
(fired team after − before) − (other teams after − before)
```

Negative means the defense improved. I measured it two ways: points allowed per 100 possessions, and the
*expected* points of the shots allowed (the quality of looks the defense gave up).

| Window | Measure | Estimate | 95% interval (clustered by firing) |
| --- | --- | ---: | ---: |
| 10 games | Shot quality allowed / 100 poss | −1.30 | [−3.58, +1.27] |
| 20 games | Shot quality allowed / 100 poss | −0.00 | [−2.45, +1.93] |
| 30 games | Shot quality allowed / 100 poss | −0.19 | [−2.43, +1.86] |
| 10 games | Points allowed / 100 poss | −2.95 | [−6.79, +0.62] |
| 20 games | Points allowed / 100 poss | −2.43 | [−5.37, +0.34] |
| 30 games | Points allowed / 100 poss | −2.36 | [−5.00, +0.17] |

Points allowed fell by about 2.5 per 100 possessions, but the quality of shots allowed barely moved. Whatever
the new coaches changed, it wasn't mainly shot selection. It could be free throws, turnovers, or luck. The
teams also weren't sliding before the firing, so it isn't simple regression to the mean. With only seven
events, every interval crosses zero. It's a lean, not a finding, and I'd want more seasons before saying
more.

## What I'd do next

- Add tracking features (closest defender, touch time, catch-and-shoot) if that data becomes available.
- Extend RAPM past field goals to full possessions, so turnovers and free throws count.
- Pull more seasons, mostly to give the coaching study real statistical power.
- Add playoff and rolling multi-year views.
