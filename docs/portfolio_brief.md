# NBA Shot Quality: one-page summary

**Site:** <https://jahankazimi078.github.io/nba-shot-quality/> · **Full write-up:** [case_study.md](case_study.md)

## The problem

Shooting efficiency mixes two things: how good a player's shots are, and how well they make them. A guard
hitting tough pull-ups and a center finishing lobs can have the same percentages for very different
reasons. I wanted to separate them.

## What I built

- An expected-points model (LightGBM) that prices all 654,609 NBA shots from 2022–23 through 2024–25, with
  game-grouped validation and out-of-fold scoring so no player's own games leak into their baseline.
- **Points over expected (POE)** for 983 player-seasons, with bootstrap confidence intervals.
- Shot-diet groups (k-means on shot location only), so players get compared to others with a similar
  style.
- A RAPM model built on lineups I reconstructed from play-by-play data, to estimate each player's effect
  on shot quality at both ends.
- A difference-in-differences study of whether firing a coach mid-season improves the defense.
- A static website that shows all of it, backed by downloadable CSVs.

## What I found

- POE is mostly skill: r = 0.58 from one season to the next.
- It agrees with relative true shooting (r = 0.66) but adds information where shot difficulty differs.
- Offensive RAPM independently lines up with POE (r ≈ 0.7). Defensive RAPM is much noisier, and I
  documented why instead of tuning it until it looked better.
- After coach firings, points allowed dropped about 2.5 per 100 possessions but shot quality allowed didn't
  change. With 7 events the intervals all cross zero, so it's a lean, not a claim.

## Tools

Python, pandas, scikit-learn, LightGBM, NumPy/SciPy (ridge regression, bootstrap), the public NBA stats
API, plain HTML/CSS/JavaScript, GitHub Actions, and GitHub Pages.
